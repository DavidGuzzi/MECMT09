# Script maestro corregido para replicar Tablas 1, 2, 3 y Figura 1 del paper de Sadoon et al. (2019)

import numpy as np
import pandas as pd
from pydynpd import regression
import matplotlib.pyplot as plt
import contextlib
import os
import sys
from scipy import stats

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
np.random.seed(3649)
T_total_default = 20  # Corregido: 20 total - 13 drop = 7 efectivo (como el paper)
T_drop = 13
rho_values = [0.25, 0.5, 0.75]
N_values = [500, 5000]
selection_models = ['A', 'B']
reps = 500

# ------------------- Generador de datos corregido -------------------
def generate_panel_data(N, rho, selection_type, T_total=T_total_default,
                        selection_rate=0.85,
                        sigma_alpha_ratio=1,
                        correlation=0.447,
                        non_stationary=False,
                        experiment_label=''):  # ✅ Nuevo parámetro
    
    # Parámetros base (como en el paper)
    sigma_eta = 1
    sigma_u = 1
    sigma_alpha0 = sigma_alpha_ratio  # Ajustado
    sigma_epsilon0 = 1

    # Generar componentes base independientes
    eta_i = np.random.normal(0, sigma_eta, N)
    alpha0_i = np.random.normal(0, sigma_alpha0, N)
    
    # Variables exógenas y errores base
    z = np.random.normal(0, 1, (N, T_total))
    u = np.random.normal(0, sigma_u, (N, T_total))
    epsilon0 = np.random.normal(0, sigma_epsilon0, (N, T_total))
    
    # Correlaciones correctas según el paper (sección 3.1)
    if correlation > 0:
        # αi = α0i + θ0ηi, εit = ε0it + ϑ0uit con θ0 = ϑ0 = 0.5 para corr = 0.447
        correlation_param = 0.5 if abs(correlation - 0.447) < 0.01 else 0.25  # 0.25 para corr ≈ 0.242
        alpha_i = alpha0_i + correlation_param * eta_i
        epsilon = epsilon0 + correlation_param * u
    else:
        alpha_i = alpha0_i
        epsilon = epsilon0
    
    # No estacionario (opcional)
    if non_stationary:
        scaling = np.random.binomial(1, 0.5, size=(N, T_total)) + 1  # 1 o 2
        epsilon *= scaling
        u *= scaling

    # Generar y*it (ecuación 43-44 del paper)
    y = np.zeros((N, T_total))
    # Condición inicial (t=1): y*i1 = (2 + αi + εi1)/(1 - ρ)
    y[:, 0] = (2 + alpha_i + epsilon[:, 0]) / (1 - rho)
    # Proceso AR(1): y*it = 2 + ρy*it-1 + αi + εit
    for t in range(1, T_total):
        y[:, t] = 2 + rho * y[:, t - 1] + alpha_i + epsilon[:, t]

    # Calibrar constante 'a' para lograr la tasa de selección deseada
    a = stats.norm.ppf(selection_rate)  # P(d*it > 0) = selection_rate
    
    # Proceso de selección corregido
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

    # Descartar primeras T_drop observaciones
    y_final = y[:, T_drop:]
    d_final = d[:, T_drop:]
    T_effective = T_total - T_drop

    # ✅ CORRECCIÓN PARA EXPERIMENTOS CON T CORTO
    # Ajustar requisitos según el experimento
    if experiment_label == 'short_T':
        min_consecutive = 2  # Solo 2 períodos consecutivos para T corto
        start_t = 1  # Empezar desde t=1 en lugar de t=2
    else:
        min_consecutive = 3  # 3 períodos consecutivos (normal)
        start_t = 2  # Empezar desde t=2 (normal)
    
    # Crear panel balanceado para pydynpd
    panel = []
    for i in range(N):
        for t in range(start_t, T_effective):
            # ✅ VERIFICACIÓN FLEXIBLE según experimento
            if experiment_label == 'short_T':
                # Para T corto: solo verificar 2 períodos consecutivos
                if t >= 1 and d_final[i, t] == d_final[i, t-1] == 1:
                    panel.append({
                        'id': i + 1,
                        'year': t + 1,
                        'n': y_final[i, t],
                        'L1.n': y_final[i, t-1],
                        'L2.n': y_final[i, t-2] if t >= 2 else y_final[i, t-1],  # Duplicar si no hay L2
                    })
            else:
                # Para experimentos normales: verificar 3 períodos consecutivos
                if d_final[i, t] == d_final[i, t-1] == d_final[i, t-2] == 1:
                    panel.append({
                        'id': i + 1,
                        'year': t + 1,
                        'n': y_final[i, t],
                        'L1.n': y_final[i, t-1],
                        'L2.n': y_final[i, t-2],
                    })
    
    return pd.DataFrame(panel)

# ------------------- Extraer coeficientes corregido -------------------
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

# ------------------- Función de simulación corregida -------------------
def run_simulation(N, rho, sel_model, expt_label='',
                   T_total=T_total_default,
                   selection_rate=0.85,
                   sigma_alpha_ratio=1,
                   correlation=0.447,
                   non_stationary=False,
                   n_reps=None):  # Nuevo parámetro

    # Usar n_reps si se proporciona, sino usar global reps
    num_reps = n_reps if n_reps is not None else reps
    
    biases_ab, biases_sys = [], []
    ses_ab, ses_sys = [], []
    
    # Solo mostrar detalles para experimentos principales
    verbose = expt_label != 'figure1'
    
    if verbose:
        print(f"  Simulando: {expt_label} N={N} ρ={rho} Modelo={sel_model} ({num_reps} reps)")
    
    for rep in range(num_reps):
        if verbose and rep % 100 == 0 and rep > 0:
            print(f"    Progreso: {rep}/{num_reps}")
            
        # Generar datos
        df = generate_panel_data(N, rho, sel_model,
                                 T_total=T_total,
                                 selection_rate=selection_rate,
                                 sigma_alpha_ratio=sigma_alpha_ratio,
                                 correlation=correlation,
                                 non_stationary=non_stationary)
        
        # Ajustar umbral según el experimento
        min_obs = 10 if expt_label != 'short_T' else 5  # Menos restrictivo para T corto
        
        if len(df) < min_obs:  # Necesitamos datos suficientes
            continue
            
        try:
            with suppress_output():
                # Arellano-Bond: AR(1) puro en primeras diferencias
                model_ab = regression.abond('n L1.n | gmm(n, 2:6) | nolevel', 
                                          df, ['id', 'year'])
                
                # System GMM: AR(1) con ecuaciones en niveles y diferencias
                model_sys = regression.abond('n L1.n | gmm(n, 2:6)', 
                                           df, ['id', 'year'])
            
            # Extraer resultados
            ab_coef, ab_se = get_coefs(model_ab, 'L1.n')
            sys_coef, sys_se = get_coefs(model_sys, 'L1.n')
            
            # Guardar si son válidos
            if not np.isnan(ab_coef):
                biases_ab.append(ab_coef - rho)
                ses_ab.append(ab_se)
            if not np.isnan(sys_coef):
                biases_sys.append(sys_coef - rho)
                ses_sys.append(sys_se)
                
        except Exception as e:
            continue

    # Calcular estadísticas finales
    result = {
        'N': N,
        'rho': rho,
        'selection': sel_model,
        'expt': expt_label,
        'AB_bias': np.mean(biases_ab) if biases_ab else np.nan,
        'AB_se': np.mean(ses_ab) if ses_ab else np.nan,  # Error estándar promedio
        'SYS_bias': np.mean(biases_sys) if biases_sys else np.nan,
        'SYS_se': np.mean(ses_sys) if ses_sys else np.nan,
        'AB_reps': len(biases_ab),
        'SYS_reps': len(biases_sys),
        'total_reps': num_reps  # Para tracking
    }
    
    if verbose:
        print(f"    ✓ AB: {result['AB_bias']:.4f} ({result['AB_reps']}/{num_reps} válidas)")
        print(f"    ✓ SYS: {result['SYS_bias']:.4f} ({result['SYS_reps']}/{num_reps} válidas)")
    
    return result

# ------------------- Configuración de experimentos corregida -------------------
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
    # TABLAS 2-3: Experimentos de sensibilidad (ajustados para evitar NaN)
    {
        'label': 'short_T', 
        'T_total': 17,  # 4 efectivo (Experimento I)
        'selection_rate': 0.90,  # ✅ Menos selección para compensar T corto
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
    """Simulación especial para Figura 1 con muchos tamaños de muestra"""
    print("\n--- SIMULACIÓN ESPECIAL PARA FIGURA 1 ---")
    
    # Tamaños de muestra para la figura (como en el paper)
    N_figure = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000]
    rho_figure = [0.25, 0.5, 0.75]
    reps_figure = 500  # ✅ AHORA SÍ se usa - menos replicaciones para acelerar
    
    figure_results = []
    total_sims = len(N_figure) * len(rho_figure) * 2  # 2 modelos de selección
    current_sim = 0
    
    print(f"Simulando {total_sims} configuraciones para líneas suaves...")
    print(f"⚡ Usando {reps_figure} replicaciones por configuración (vs {reps} para tablas)")
    
    for N in N_figure:
        for rho in rho_figure:
            for sel_model in ['A', 'B']:
                current_sim += 1
                
                # Mostrar progreso cada 10 simulaciones
                if current_sim % 10 == 0 or current_sim == 1:
                    print(f"  [{current_sim}/{total_sims}] N={N}, ρ={rho}, Modelo={sel_model}")
                
                # ✅ CORRECCIÓN: Usar n_reps=reps_figure (100)
                res = run_simulation(N, rho, sel_model, 
                                   expt_label='figure1',
                                   T_total=20,
                                   selection_rate=0.85,
                                   sigma_alpha_ratio=1,
                                   correlation=0.447,
                                   non_stationary=False,
                                   n_reps=reps_figure)  # ✅ Pasar 100 replicaciones
                
                # Cambiar label para identificar
                res['expt'] = 'figure1'
                figure_results.append(res)
    
    print(f"✅ Figura 1 completada con {reps_figure} reps cada una (más rápido)")
    return figure_results

# ------------------- Ejecutar simulaciones -------------------
print("="*80)
print("REPLICACIÓN SADOON ET AL. (2019) - VERSIÓN CORREGIDA")
print(f"Semilla: {3649} | Replicaciones por experimento: {reps}")
print("="*80)

all_results = []

# 1. EXPERIMENTOS PRINCIPALES (Tablas 1-3) con N = [500, 5000]
total_experiments = len(experiments) * len(N_values) * len(rho_values) * len(selection_models)
current_exp = 0

for expt in experiments:
    print(f"\n--- EXPERIMENTO: {expt['label'].upper()} ---")
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
print("EJECUTANDO SIMULACIÓN ESPECIAL PARA FIGURA 1")
print("(Muchos tamaños de muestra para líneas suaves)")
print(f"{'='*80}")

figure_results = run_figure1_simulation()
all_results.extend(figure_results)

# ------------------- Guardar resultados -------------------
df_all = pd.DataFrame(all_results)
df_all = df_all.round(5)  # Redondear para prolijidad
df_all.to_csv('resultados_sadoon_corregido.csv', index=False)

print("\n" + "="*80)
print("✅ SIMULACIONES COMPLETADAS")
print(f"✅ Resultados guardados en 'resultados_sadoon_corregido.csv'")
print(f"✅ Total de experimentos: {len(df_all)}")
print("✅ Experimentos incluidos:")
for exp_name in df_all['expt'].unique():
    count = len(df_all[df_all['expt'] == exp_name])
    print(f"   - {exp_name}: {count} configuraciones")
print("="*80)

# Mostrar resumen de resultados principales
print("\nRESUMEN - EXPERIMENTOS PRINCIPALES:")
main_experiments = df_all[df_all['expt'].isin(['no_endogenous', 'baseline'])]
summary_cols = ['expt', 'N', 'rho', 'selection', 'AB_bias', 'SYS_bias', 'AB_reps', 'SYS_reps', 'total_reps']
print(main_experiments[summary_cols].to_string(index=False))

print(f"\nRESUMEN DE REPLICACIONES:")
print(f"✅ Experimentos principales (Tablas 1-3): {reps} replicaciones por configuración")

figure_data = df_all[df_all['expt'] == 'figure1']
if len(figure_data) > 0:
    avg_reps = figure_data['total_reps'].iloc[0] if 'total_reps' in figure_data.columns else 100
    print(f"⚡ Experimento Figura 1: {avg_reps} replicaciones por configuración (acelerado)")
    time_saved_pct = (1 - avg_reps/reps) * 100
    print(f"💾 Tiempo ahorrado en Figura 1: ~{time_saved_pct:.0f}%")

import subprocess

# ------------------- Subida automática a Cloud Storage -------------------
try:
    bucket_name = "mecmt09-bucket"
    output_file = "resultados_sadoon_corregido.csv"
    destino = f"gs://{bucket_name}/{output_file}"
    
    print(f"\n🚀 Subiendo {output_file} a {destino}...")
    subprocess.run(["gsutil", "cp", output_file, destino], check=True)
    print("✅ Archivo subido exitosamente a Cloud Storage.")
except Exception as e:
    print("⚠ Error al subir el archivo al bucket:", e)

# ------------------- Apagado automático de la VM -------------------
try:
    print("⏻ Apagando la instancia de Compute Engine...")
    subprocess.run([
        "gcloud", "compute", "instances", "stop", 
        "mecmt09-vm", "--zone", "us-central1-c"
    ], check=True)
except Exception as e:
    print("⚠ Error al intentar apagar la VM:", e)