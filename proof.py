#%%
import numpy as np
import pandas as pd
from pydynpd import regression
import matplotlib.pyplot as plt
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

    # Descartar primeras T_drop observaciones
    y_final = y[:, T_drop:]
    d_final = d[:, T_drop:]
    T_effective = T_total - T_drop

    # Crear panel balanceado
    start_t = 2  # Para TODOS los experimentos

    panel = []
    for i in range(N):
        for t in range(start_t, T_effective):
            # Verificar 3 períodos consecutivos para TODOS los experimentos
            if d_final[i, t] == d_final[i, t-1] == d_final[i, t-2] == 1:
                panel.append({
                    'id': i + 1,
                    'year': t + 1,
                    'n': y_final[i, t],
                    'L1.n': y_final[i, t-1],
                    'L2.n': y_final[i, t-2],
                })
    
    return pd.DataFrame(panel)

df = generate_panel_data(500, 0.75, "A", T_total=T_total_default,
                        selection_rate=0.85,
                        sigma_alpha_ratio=1,
                        correlation=0.447,
                        non_stationary=False,
                        experiment_label='',
                        seed_offset=0)