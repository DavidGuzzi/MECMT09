# Script para replicar Tablas 1, 2, 3 y Figura 1 del paper de Sadoon et al. (2019)

import numpy as np
import pandas as pd
from pydynpd import regression
import contextlib
import os
import sys
from scipy import stats
from datetime import datetime
import multiprocessing as mp

# ------------------- Silenciar prints -------------------
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ------------------- Configuración global -------------------
seed = int('03649')
np.random.seed(seed)
T_total_default = 20 
T_drop = 13
rho_values = [0.25, 0.5, 0.75]
N_values = [500, 5000]
selection_models = ['A', 'B']
reps = 500

# ------------------- Generador de datos -------------------
def generate_panel_data(N, rho, selection_type, T_total=T_total_default,
                        selection_rate=0.85,
                        sigma_alpha_ratio=1,
                        correlation=0.447,
                        non_stationary=False,
                        experiment_label='',
                        seed_offset=0):
    
    # Usar semilla única para cada simulación
    np.random.seed(seed + seed_offset)
    
    # Parámetros base
    sigma_eta = 1
    sigma_u = 1
    sigma_alpha0 = sigma_alpha_ratio
    sigma_epsilon0 = 1

    # Generar componentes base independientes
    eta_i = np.random.normal(0, sigma_eta, N)
    alpha0_i = np.random.normal(0, sigma_alpha0, N)
    
    # Variables exógenas y errores base
    z = np.random.normal(0, 1, (N, T_total))
    u = np.random.normal(0, sigma_u, (N, T_total))
    epsilon0 = np.random.normal(0, sigma_epsilon0, (N, T_total))
    
    # Correlaciones
    if correlation > 0:
        # αi = α0i + θ0ηi, εit = ε0it + ϑ0uit con θ0 = ϑ0 = 0.5 para corr = 0.447
        correlation_param = 0.5 if abs(correlation - 0.447) < 0.01 else 0.25  # 0.25 para corr ≈ 0.242
        alpha_i = alpha0_i + correlation_param * eta_i
        epsilon = epsilon0 + correlation_param * u
    else:
        alpha_i = alpha0_i
        epsilon = epsilon0
    
    # No estacionario
    if non_stationary:
        scaling = np.random.binomial(1, 0.5, size=(N, T_total)) + 1  # 1 o 2
        epsilon *= scaling
        u *= scaling

    # Generar y*it
    y = np.zeros((N, T_total))
    # Condición inicial (t=1): y*i1 = (2 + αi + εi1)/(1 - ρ)
    y[:, 0] = (2 + alpha_i + epsilon[:, 0]) / (1 - rho)
    # Proceso AR(1): y*it = 2 + ρy*it-1 + αi + εit
    for t in range(1, T_total):
        y[:, t] = 2 + rho * y[:, t - 1] + alpha_i + epsilon[:, t]

    # Calibrar constante 'a' para lograr la tasa de selección deseada # P(d*it > 0) = selection_rate
    if selection_type == 'A':
        a = stats.norm.ppf(selection_rate) * 1.732  # sqrt(3)
    elif selection_type == 'B':
        a = stats.norm.ppf(selection_rate) * 1.85  
    
    # Proceso de selección
    d = np.zeros((N, T_total))
    d_star = np.zeros((N, T_total))
    
    if selection_type == 'A':
        # Modelo estático: d*it = a - zit - ηi - uit (ecuación 40)
        d_star = a - z - eta_i[:, None] - u
        d = (d_star > 0).astype(int)
        
    elif selection_type == 'B':
        # Modelo dinámico: d*it = a - 0.5dit-1 + zit - ηi - uit (ecuación 41)
        for t in range(T_total):
            if t == 0:
                # Primera observación sin rezago
                d_star[:, t] = a + z[:, t] - eta_i - u[:, t]
            else:
                # Con rezago dinámico
                d_star[:, t] = a - 0.5 * d[:, t-1] + z[:, t] - eta_i - u[:, t]
            d[:, t] = (d_star[:, t] > 0).astype(int)
    else:
        raise ValueError("Tipo de selección debe ser 'A' o 'B'")

    # Ajuste específico para short_T
    if experiment_label == 'short_T':
        start_t = 1
        T_drop_adj = 14 
        min_consecutive = 2 
    else:
        start_t = 2
        T_drop_adj = T_drop  
        min_consecutive = 3

    # Descartar primeras T_drop_adj observaciones
    y_final = y[:, T_drop_adj:]
    d_final = d[:, T_drop_adj:]
    T_effective = T_total - T_drop_adj

    panel = []
    for i in range(N):
        for t in range(start_t, T_effective):
            if min_consecutive == 2:
                # Solo 2 períodos consecutivos
                if d_final[i, t] == d_final[i, t-1] == 1:
                    panel.append({
                        'id': i + 1,
                        'year': t + 1,
                        'n': y_final[i, t],
                        'L1.n': y_final[i, t-1],
                        'L2.n': y_final[i, t-2] if t >= 2 else y_final[i, t-1],
                    })
            else:
                # 3 períodos consecutivos (caso normal)
                if d_final[i, t] == d_final[i, t-1] == d_final[i, t-2] == 1:
                    panel.append({
                        'id': i + 1,
                        'year': t + 1,
                        'n': y_final[i, t],
                        'L1.n': y_final[i, t-1],
                        'L2.n': y_final[i, t-2],
                    })

    return pd.DataFrame(panel)

# ------------------- Extraer coeficientes -------------------
def get_coefs(model, variable='L1.n'):
    """Extrae coeficiente y error estándar de la regresión"""
    try:
        table = model.models[0].regression_table
        row = table[table['variable'] == variable]
        if not row.empty:
            coef = row['coefficient'].values[0]
            se = row['std_err'].values[0]
            return coef, se
        else:
            return np.nan, np.nan
    except Exception:
        return np.nan, np.nan
    

def get_timestamp():
    """Retorna timestamp formateado para logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------- Función auxiliar para una replicación -------------------
def run_single_replication(args):
    """Ejecuta una sola replicación - diseñada para paralelización"""
    (N, rho, sel_model, expt_label, T_total, selection_rate, 
     sigma_alpha_ratio, correlation, non_stationary, rep_id) = args
    
    try:
        # Generar datos con semilla única
        df = generate_panel_data(N, rho, sel_model,
                                 T_total=T_total,
                                 selection_rate=selection_rate,
                                 sigma_alpha_ratio=sigma_alpha_ratio,
                                 correlation=correlation,
                                 non_stationary=non_stationary,
                                 experiment_label=expt_label,
                                 seed_offset=rep_id * 1000)  # Semilla única
        
        min_obs = 10 if expt_label != 'short_T' else 3
        
        if len(df) < min_obs:
            return None
            
        with suppress_output():
            model_ab = regression.abond('n L1.n | gmm(n, 2:99) | nolevel', 
                                      df, ['id', 'year'])
            model_sys = regression.abond('n L1.n | gmm(n, 2:99)', 
                                       df, ['id', 'year'])
        
        ab_coef, ab_se = get_coefs(model_ab, 'L1.n')
        sys_coef, sys_se = get_coefs(model_sys, 'L1.n')
        
        result = {
            'ab_bias': ab_coef - rho if not np.isnan(ab_coef) else np.nan,
            'ab_se': ab_se if not np.isnan(ab_coef) else np.nan,
            'sys_bias': sys_coef - rho if not np.isnan(sys_coef) else np.nan,
            'sys_se': sys_se if not np.isnan(sys_coef) else np.nan,
            'ab_valid': not np.isnan(ab_coef),
            'sys_valid': not np.isnan(sys_coef)
        }
        
        return result
        
    except Exception as e:
        return None

# ------------------- Función de simulación PARALELIZADA -------------------
def run_simulation(N, rho, sel_model, expt_label='',
                   T_total=T_total_default,
                   selection_rate=0.85,
                   sigma_alpha_ratio=1,
                   correlation=0.447,
                   non_stationary=False,
                   n_reps=None):  # Nuevo parámetro

    # Usar n_reps si se proporciona, sino usar global reps
    num_reps = n_reps if n_reps is not None else reps
    
    # Solo mostrar detalles para experimentos principales
    verbose = expt_label != 'figure1'
    
    if verbose:
        print(f"  Simulando: {expt_label} N={N} ρ={rho} Modelo={sel_model} ({num_reps} reps) [{get_timestamp()}]")
    
    # Crear argumentos para cada replicación
    args_list = []
    for rep in range(num_reps):
        args = (N, rho, sel_model, expt_label, T_total, selection_rate,
                sigma_alpha_ratio, correlation, non_stationary, rep)
        args_list.append(args)
    
    # Determinar número de cores a usar
    cores = min(mp.cpu_count(), 18)
    
    # Ejecutar en paralelo
    with mp.Pool(processes=cores) as pool:
        results = pool.map(run_single_replication, args_list)
        
        # Mostrar progreso durante la ejecución para experimentos verbosos
        if verbose:
            completed = sum(1 for r in results if r is not None)
            if completed > 0:
                print(f"    Progreso: {completed}/{num_reps} replicaciones completadas")
    
    # Procesar resultados
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return {
            'N': N, 'rho': rho, 'selection': sel_model, 'expt': expt_label,
            'AB_bias': np.nan, 'AB_se': np.nan, 'SYS_bias': np.nan, 'SYS_se': np.nan,
            'AB_reps': 0, 'SYS_reps': 0, 'total_reps': num_reps
        }
    
    # Separar resultados válidos para AB y SYS
    ab_results = [r for r in valid_results if r['ab_valid']]
    sys_results = [r for r in valid_results if r['sys_valid']]
    
    # Calcular estadísticas finales
    result = {
        'N': N,
        'rho': rho,
        'selection': sel_model,
        'expt': expt_label,
        'AB_bias': np.mean([r['ab_bias'] for r in ab_results]) if ab_results else np.nan,
        'AB_se': np.mean([r['ab_se'] for r in ab_results]) if ab_results else np.nan,
        'SYS_bias': np.mean([r['sys_bias'] for r in sys_results]) if sys_results else np.nan,
        'SYS_se': np.mean([r['sys_se'] for r in sys_results]) if sys_results else np.nan,
        'AB_reps': len(ab_results),
        'SYS_reps': len(sys_results),
        'total_reps': num_reps
    }
    
    if verbose:
        print(f"    ✓ AB: {result['AB_bias']:.4f} ({result['AB_reps']}/{num_reps} válidas)")
        print(f"    ✓ SYS: {result['SYS_bias']:.4f} ({result['SYS_reps']}/{num_reps} válidas)")
    
    return result

# ------------------- Configuración de experimentos -------------------
experiments = [
    # TABLA 1 - Parte 1: Sin selección endógena
    {
        'label': 'no_endogenous', 
        'T_total': 20,
        'selection_rate': 0.85, 
        'sigma_alpha_ratio': 1, 
        'correlation': 0.0,  # SIN correlación
        'non_stationary': False
    },
    # TABLA 1 - Parte 2: Con selección endógena (baseline)
    {
        'label': 'baseline', 
        'T_total': 20,
        'selection_rate': 0.85, 
        'sigma_alpha_ratio': 1, 
        'correlation': 0.447,  # CON correlación
        'non_stationary': False
    },
    # TABLAS 2-3: Experimentos de sensibilidad
    {
        'label': 'short_T', 
        'T_total': 20,  # (Experimento I)
        'selection_rate': 0.85,  
        'sigma_alpha_ratio': 1, 
        'correlation': 0.447, 
        'non_stationary': False
    },
    {
        'label': 'more_selection', 
        'T_total': 20, 
        'selection_rate': 0.75,  # 25% selección (Experimento II)
        'sigma_alpha_ratio': 1, 
        'correlation': 0.447, 
        'non_stationary': False
    },
    {
        'label': 'high_alpha_ratio', 
        'T_total': 20, 
        'selection_rate': 0.85, 
        'sigma_alpha_ratio': 2,  # ση/σε = 2 (Experimento III)
        'correlation': 0.447, 
        'non_stationary': False
    },
    {
        'label': 'low_corr', 
        'T_total': 20, 
        'selection_rate': 0.85, 
        'sigma_alpha_ratio': 1, 
        'correlation': 0.242,  # ρ = 0.25 (Experimento IV)
        'non_stationary': False
    },
    {
        'label': 'nonstationary', 
        'T_total': 20, 
        'selection_rate': 0.85, 
        'sigma_alpha_ratio': 1, 
        'correlation': 0.447, 
        'non_stationary': True  # Experimento V
    },
]

# ------------------- Experimento especial para Figura 1 -------------------
def run_figure1_simulation():
    """Simulación especial para Figura 1 con varios tamaños de muestra."""
    
    # Tamaños de muestra para la figura
    N_figure = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000]
    rho_figure = [0.25, 0.5, 0.75]
    reps_figure = 500 
    
    figure_results = []
    total_sims = len(N_figure) * len(rho_figure) * 2  # 2 modelos de selección
    current_sim = 0
    
    print(f"Simulando {total_sims} configuraciones...")
    print(f"Usando {reps_figure} replicaciones por configuración")
    
    for N in N_figure:
        for rho in rho_figure:
            for sel_model in ['A', 'B']:
                current_sim += 1
                
                # Mostrar progreso cada 10 simulaciones
                if current_sim % 10 == 0 or current_sim == 1:
                    print(f"  [{current_sim}/{total_sims}] N={N}, ρ={rho}, Modelo={sel_model} [{get_timestamp()}]")

                res = run_simulation(N, rho, sel_model, 
                                   expt_label='figure1',
                                   T_total=20,
                                   selection_rate=0.85,
                                   sigma_alpha_ratio=1,
                                   correlation=0.447,
                                   non_stationary=False,
                                   n_reps=reps_figure) 
                
                # Cambiar label para identificar
                res['expt'] = 'figure1'
                figure_results.append(res)
    
    print(f"Figura 1 completada con {reps_figure} reps cada una [{get_timestamp()}]")
    return figure_results

# ------------------- EJECUTAR TODO -------------------
if __name__ == "__main__":

    # ------------------- Ejecutar simulaciones -------------------
    print("="*80)
    print(f"REPLICACIÓN SADOON ET AL. (2019) - INICIO: {get_timestamp()}")
    print(f"Semilla: {seed} | Replicaciones por experimento: {reps}")
    print(f"Cores disponibles: {mp.cpu_count()}")
    print("="*80)

    all_results = []

    # 1. EXPERIMENTOS PRINCIPALES (Tablas 1-3) con N = [500, 5000]
    total_experiments = len(experiments) * len(N_values) * len(rho_values) * len(selection_models)
    current_exp = 0

    for expt in experiments:
        print(f"\n--- EXPERIMENTO: {expt['label'].upper()} --- [{get_timestamp()}]")
        for N in N_values:
            for rho in rho_values:
                for sel_model in selection_models:
                    current_exp += 1
                    print(f"[{current_exp}/{total_experiments}]", end=" ")
                    
                    res = run_simulation(N, rho, sel_model, 
                                    expt_label=expt['label'],
                                    T_total=expt['T_total'],
                                    selection_rate=expt['selection_rate'],
                                    sigma_alpha_ratio=expt['sigma_alpha_ratio'],
                                    correlation=expt['correlation'],
                                    non_stationary=expt['non_stationary'])
                    all_results.append(res)

    # 2. SIMULACIÓN ESPECIAL PARA FIGURA 1
    print(f"\n{'='*80}")
    print(f"EJECUTANDO SIMULACIÓN ESPECIAL PARA FIGURA 1 - INICIO: {get_timestamp()}")
    print("(Varios tamaños de muestra para obtener líneas suaves)")
    print(f"{'='*80}")

    figure_results = run_figure1_simulation()
    all_results.extend(figure_results)

    # ------------------- Guardar resultados -------------------
    df_all = pd.DataFrame(all_results)
    df_all = df_all.round(5)  # Redondear para prolijidad
    df_all.to_csv('resultados_v2.csv', index=False)

    print("\n" + "="*80)
    print(f"SIMULACIONES COMPLETADAS - FIN: {get_timestamp()}")
    print(f"Resultados guardados en 'resultados_v2.csv'")
    print(f"Total de experimentos: {len(df_all)}")
    print("Experimentos incluidos:")
    for exp_name in df_all['expt'].unique():
        count = len(df_all[df_all['expt'] == exp_name])
        print(f"   - {exp_name}: {count} configuraciones")
    print("="*80)