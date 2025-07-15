# Script para construir Tablas 1, 2, 3 y Figura 1 con formato exacto del paper Sadoon et al. (2019)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------- Cargar resultados -------------------
print("Cargando resultados de simulaciones...")
try:
    results = pd.read_csv("resultados.csv")
    print(f"Cargados {len(results)} resultados")
except FileNotFoundError:
    print("No se encontró 'resultados.csv'")
    print("Ejecuta primero el script de generación de datos")
    exit()

# ------------------- TABLA 1 - FORMATO DEL PAPER -------------------
def make_table1_paper_format(df):
    """Genera Tabla 1 con formato del paper"""
    print("\n" + "="*85)
    print("Table 1: Average bias in the AR(1) model (T = 7, 500 replications)")
    print("="*85)
    
    # Datos
    no_endog = df[df['expt'] == 'no_endogenous'].copy()
    endog = df[df['expt'] == 'baseline'].copy()
    
    # Header de la tabla con alineación perfecta
    print("                           No endogenous        Endogenous")
    print("                             selection          selection")
    print("Select.         ρ         (1)      (2)       (3)      (4)")
    print("Model                     AB      SYS        AB      SYS")
    print("                                N = 500")
    
    # Modelo A, N=500
    for rho in [0.25, 0.5, 0.75]:
        no_end_row = no_endog.query("N == 500 & selection == 'A' & rho == @rho")
        end_row = endog.query("N == 500 & selection == 'A' & rho == @rho")
        
        # Línea de bias
        if len(no_end_row) > 0 and len(end_row) > 0:
            no_ab_bias = no_end_row['AB_bias'].values[0]
            no_sys_bias = no_end_row['SYS_bias'].values[0]
            end_ab_bias = end_row['AB_bias'].values[0]
            end_sys_bias = end_row['SYS_bias'].values[0]
            
            print(f"A         {rho:4.2f}  bias  {no_ab_bias:8.5f} {no_sys_bias:8.5f}  {end_ab_bias:8.5f} {end_sys_bias:8.5f}")
            
            # Línea de s.e.
            no_ab_se = no_end_row['AB_se'].values[0]
            no_sys_se = no_end_row['SYS_se'].values[0]
            end_ab_se = end_row['AB_se'].values[0]
            end_sys_se = end_row['SYS_se'].values[0]
            
            print(f"               s.e.  {no_ab_se:8.5f} {no_sys_se:8.5f}  {end_ab_se:8.5f} {end_sys_se:8.5f}")
    
    print("                                N = 5000")
    
    # Modelo A, N=5000
    for rho in [0.25, 0.5, 0.75]:
        no_end_row = no_endog.query("N == 5000 & selection == 'A' & rho == @rho")
        end_row = endog.query("N == 5000 & selection == 'A' & rho == @rho")
        
        if len(no_end_row) > 0 and len(end_row) > 0:
            no_ab_bias = no_end_row['AB_bias'].values[0]
            no_sys_bias = no_end_row['SYS_bias'].values[0]
            end_ab_bias = end_row['AB_bias'].values[0]
            end_sys_bias = end_row['SYS_bias'].values[0]
            
            print(f"A         {rho:4.2f}  bias  {no_ab_bias:8.5f} {no_sys_bias:8.5f}  {end_ab_bias:8.5f} {end_sys_bias:8.5f}")
            
            no_ab_se = no_end_row['AB_se'].values[0]
            no_sys_se = no_end_row['SYS_se'].values[0]
            end_ab_se = end_row['AB_se'].values[0]
            end_sys_se = end_row['SYS_se'].values[0]
            
            print(f"               s.e.  {no_ab_se:8.5f} {no_sys_se:8.5f}  {end_ab_se:8.5f} {end_sys_se:8.5f}")
    
    print("                                N = 500")
    
    # Modelo B, N=500
    for rho in [0.25, 0.5, 0.75]:
        no_end_row = no_endog.query("N == 500 & selection == 'B' & rho == @rho")
        end_row = endog.query("N == 500 & selection == 'B' & rho == @rho")
        
        if len(no_end_row) > 0 and len(end_row) > 0:
            no_ab_bias = no_end_row['AB_bias'].values[0]
            no_sys_bias = no_end_row['SYS_bias'].values[0]
            end_ab_bias = end_row['AB_bias'].values[0]
            end_sys_bias = end_row['SYS_bias'].values[0]
            
            print(f"B         {rho:4.2f}  bias  {no_ab_bias:8.5f} {no_sys_bias:8.5f}  {end_ab_bias:8.5f} {end_sys_bias:8.5f}")
            
            no_ab_se = no_end_row['AB_se'].values[0]
            no_sys_se = no_end_row['SYS_se'].values[0]
            end_ab_se = end_row['AB_se'].values[0]
            end_sys_se = end_row['SYS_se'].values[0]
            
            print(f"               s.e.  {no_ab_se:8.5f} {no_sys_se:8.5f}  {end_ab_se:8.5f} {end_sys_se:8.5f}")
    
    print("                                N = 5000")
    
    # Modelo B, N=5000
    for rho in [0.25, 0.5, 0.75]:
        no_end_row = no_endog.query("N == 5000 & selection == 'B' & rho == @rho")
        end_row = endog.query("N == 5000 & selection == 'B' & rho == @rho")
        
        if len(no_end_row) > 0 and len(end_row) > 0:
            no_ab_bias = no_end_row['AB_bias'].values[0]
            no_sys_bias = no_end_row['SYS_bias'].values[0]
            end_ab_bias = end_row['AB_bias'].values[0]
            end_sys_bias = end_row['SYS_bias'].values[0]
            
            print(f"B         {rho:4.2f}  bias  {no_ab_bias:8.5f} {no_sys_bias:8.5f}  {end_ab_bias:8.5f} {end_sys_bias:8.5f}")
            
            no_ab_se = no_end_row['AB_se'].values[0]
            no_sys_se = no_end_row['SYS_se'].values[0]
            end_ab_se = end_row['AB_se'].values[0]
            end_sys_se = end_row['SYS_se'].values[0]
            
            print(f"               s.e.  {no_ab_se:8.5f} {no_sys_se:8.5f}  {end_ab_se:8.5f} {end_sys_se:8.5f}")

# ------------------- TABLAS 2 y 3 - FORMATO DEL PAPER -------------------
def make_sensitivity_tables_paper_format(df):
    """Genera Tablas 2 y 3 con formato del paper"""
    
    experiments = {
        'short_T': 'Experiment I: Very short T (T = 4)',
        'more_selection': 'Experiment II: More sample selection (25%)',
        'high_alpha_ratio': 'Experiment III: Increasing the ratio of variances: ση/σε = 2',
        'low_corr': 'Experiment IV: Reducing the correlation of the errors: ρ = 0.25',
        'nonstationary': 'Experiment V: Non-stationary time-varying error components'
    }
    
    for table_n, N in enumerate([500, 5000], 2):
        print(f"\n" + "="*90)
        print(f"Table {table_n}: Average bias in the AR(1) model. Sensitivity analysis for {'small' if N==500 else 'large'} N")
        print("="*90)
        
        # Header
        print("Model               ρ = 0.25      ρ = 0.5       ρ = 0.75")
        print("                   AB    SYS    AB    SYS    AB    SYS")
        
        sens_data = df[df['N'] == N].copy()
        
        for exp_key, exp_name in experiments.items():
            if exp_key not in sens_data['expt'].values:
                continue
                
            print(f"                   {exp_name}")
            
            for sel_model in ['A', 'B']:
                # Fila de bias
                bias_values = []
                se_values = []
                
                for rho in [0.25, 0.5, 0.75]:
                    row_data = sens_data.query("expt == @exp_key & selection == @sel_model & rho == @rho")
                    if len(row_data) > 0:
                        ab_bias = row_data['AB_bias'].values[0]
                        ab_se = row_data['AB_se'].values[0]
                        sys_bias = row_data['SYS_bias'].values[0]
                        sys_se = row_data['SYS_se'].values[0]
                        
                        bias_values.extend([ab_bias, sys_bias])
                        se_values.extend([ab_se, sys_se])
                    else:
                        bias_values.extend([np.nan, np.nan])
                        se_values.extend([np.nan, np.nan])
                
                # Imprimir filas
                bias_str = "  ".join([f"{v:7.5f}" if not np.isnan(v) else "    NaN" for v in bias_values])
                se_str = "  ".join([f"{v:7.5f}" if not np.isnan(v) else "    NaN" for v in se_values])
                
                print(f"{sel_model}        bias  {bias_str}")
                print(f"         s.e.  {se_str}")

# ------------------- FIGURA 1 - COLORES Y ESTILOS DEL PAPER -------------------
def make_figure1_paper_format(df):
    """Genera Figura 1 con colores y estilos del paper"""
    print("\nGenerando Figura 1 con formato del paper...")
    
    # Usar datos especiales de figure1
    figure_data = df[df['expt'] == 'figure1'].copy()
    
    if len(figure_data) == 0:
        print("No se encontraron datos de 'figure1'. Usando datos baseline...")
        figure_data = df[df['expt'] == 'baseline'].copy()
    else:
        print(f"Usando {len(figure_data)} datos especiales para Figura 1")
    
    # Configurar la figura con más espacio para el título y fondo celeste
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#f0f8ff')  # Fondo celeste para toda la figura
    axes = axes.flatten()
    
    # Colores y estilos según el paper (líneas más finas)
    styles = {
        ('AB', 'A'): {'color': 'blue', 'linestyle': '-', 'marker': '', 'linewidth': 1.5, 'markersize': 0, 'label': 'A&B all'},
        ('AB', 'B'): {'color': 'green', 'linestyle': '-', 'marker': '^', 'linewidth': 1.5, 'markersize': 4, 'label': 'A&B select'},
        ('SYS', 'A'): {'color': 'purple', 'linestyle': '-', 'marker': 's', 'linewidth': 1.5, 'markersize': 3, 'label': 'system all'},
        ('SYS', 'B'): {'color': 'orange', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.5, 'markersize': 3, 'label': 'system select'}
    }
    
    rho_values = [0.25, 0.50, 0.75]
    
    for i, rho in enumerate(rho_values):
        ax = axes[i]
        
        # Datos para este valor de rho
        rho_data = figure_data[figure_data['rho'] == rho].copy()
        
        # Plotear cada combinación
        for sel_model in ['A', 'B']:
            model_data = rho_data[rho_data['selection'] == sel_model].sort_values('N')
            
            if len(model_data) > 0:
                # AB estimator
                ab_style = styles[('AB', sel_model)]
                if ab_style['marker']:
                    ax.plot(model_data['N'], model_data['AB_bias'], 
                           color=ab_style['color'], linestyle=ab_style['linestyle'],
                           marker=ab_style['marker'], linewidth=ab_style['linewidth'],
                           markersize=ab_style['markersize'], label=ab_style['label'])
                else:
                    ax.plot(model_data['N'], model_data['AB_bias'], 
                           color=ab_style['color'], linestyle=ab_style['linestyle'],
                           linewidth=ab_style['linewidth'], label=ab_style['label'])
                
                # System estimator
                sys_style = styles[('SYS', sel_model)]
                ax.plot(model_data['N'], model_data['SYS_bias'], 
                       color=sys_style['color'], linestyle=sys_style['linestyle'],
                       marker=sys_style['marker'], linewidth=sys_style['linewidth'],
                       markersize=sys_style['markersize'], label=sys_style['label'])
        
        # Línea en cero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Formato
        ax.set_xlabel('sample size', fontsize=10)
        ax.set_ylabel('mean bias', fontsize=10)
        ax.set_title(f'mean bias, alpha={rho:.2f}', fontsize=11, fontweight='normal')
        # Leyenda siempre abajo a la derecha
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2)
        
        # Límites
        if len(figure_data) > 0:
            max_n = figure_data['N'].max()
            ax.set_xlim(0, max_n * 1.05)
            ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        
        # Ajustar límites del eje y
        if len(rho_data) > 0:
            all_biases = np.concatenate([rho_data['AB_bias'].values, rho_data['SYS_bias'].values])
            all_biases = all_biases[~np.isnan(all_biases)]
            if len(all_biases) > 0:
                y_range = np.max(all_biases) - np.min(all_biases)
                y_margin = max(0.01, y_range * 0.1)
                y_min, y_max = np.min(all_biases) - y_margin, np.max(all_biases) + y_margin
                ax.set_ylim(y_min, y_max)
    
    # Ocultar el cuarto subplot
    axes[3].set_visible(False)
    
    # Ajustar layout con más espacio arriba para el título
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Título sin solapamiento
    plt.suptitle('Figure 1: Average bias of the AB and system estimators in the full sample\n' + 
                 '(NxT observations) and the endogenously selected sample', 
                 fontsize=12, fontweight='normal')
    
    # Guardar figura
    plt.savefig('figura1_sadoon_paper_format.png', dpi=300, bbox_inches='tight')
    print("Figura 1 guardada como 'figura1_sadoon_paper_format.png'")
    plt.show()

# ------------------- EJECUTAR TODO -------------------
if __name__ == "__main__":
    print("="*80)
    print("GENERACIÓN DE TABLAS Y FIGURAS - FORMATO SADOON ET AL. (2019)")
    print("="*80)
    
    # Verificar datos
    print(f"Datos cargados: {len(results)} resultados")
    print(f"Experimentos: {sorted(results['expt'].unique())}")
    
    # Generar outputs con formato del paper
    make_table1_paper_format(results)
    make_sensitivity_tables_paper_format(results)
    make_figure1_paper_format(results)
    
    print("\n" + "="*80)
    print("TODAS LAS TABLAS Y FIGURAS GENERADAS CON FORMATO DEL PAPER")
    print("Archivos creados:")
    print("   - Tabla 1: Formato del paper")
    print("   - Tablas 2-3: Formato del paper") 
    print("   - figura1_sadoon_paper_format.png: Colores y estilos del paper")
    print("="*80)