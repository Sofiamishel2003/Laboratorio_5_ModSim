import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from pathlib import Path
from datetime import datetime

outdir = Path("./output")
outdir.mkdir(parents=True, exist_ok=True)
run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

# Inciso 2: Generación de redes
n = 1000

# Barabási–Albert (libre de escala). Elegimos m tal que grado medio 2m.
m = 3  # promedio 6
G_ba = nx.barabasi_albert_graph(n=n, m=m, seed=42)

# Erdős–Rényi con grado promedio similar
avg_k_target = 2 * m  # 6
p = avg_k_target / (n - 1)
G_er = nx.erdos_renyi_graph(n=n, p=p, seed=42)

# Inciso 3: Análisis de las redes

def degree_sequence(G):
    return np.array([d for _, d in G.degree()], dtype=np.int64)

deg_ba = degree_sequence(G_ba)
deg_er = degree_sequence(G_er)

def hist_and_save(degrees, title, filename, bins='auto', log_scale=False):
    plt.figure()
    plt.hist(degrees, bins=bins)
    plt.xlabel("Grado (k)")
    plt.ylabel("Frecuencia")
    plt.title(title)
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    outpath = outdir / filename
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath



# a) Histogramas en escala lineal (dos figuras separadas por restricciones de la herramienta)

def joined_hist(deg_ba, deg_er, title, filename, log_scale=False):
    bins = np.linspace(0,deg_ba.max(), 40)
    bins=range(0, deg_ba.max()+2)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    fig.suptitle(title)
    ax[0].hist(deg_ba, bins)
    ax[0].set_title("Barabási–Albert")
    ax[1].hist(deg_er, bins)
    ax[1].set_title("Erdős–Rényi")
    if log_scale:
        ax[0].set_xscale("log")
        ax[1].set_yscale("log")
    
    outpath = outdir / filename
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    
joined_hist(deg_ba, deg_er, 
            "Distribución de grados (lineal)", 
            f"histograma_joined_{run_tag}.png")

path_hist_ba_linear = hist_and_save(
    deg_ba,
    title="Distribución de grados (lineal) — Barabási–Albert (n=1000, m=3)",
    filename=f"histograma_ba_lineal_{run_tag}.png",
    bins=range(0, deg_ba.max()+2)
)

path_hist_er_linear = hist_and_save(
    deg_er,
    title=f"Distribución de grados (lineal) — Erdős–Rényi (n=1000, p≈{p:.4f})",
    filename=f"histograma_er_lineal_{run_tag}.png",
    bins=range(0, deg_er.max()+2)
)

# b) Distribución de grados de BA en escala logarítmica (log-log de P(k))
# Construimos conteo de grados y graficamos k vs frecuencia (o densidad) en log-log
counts_ba = Counter(deg_ba)
k_vals = np.array(sorted(counts_ba.keys()))
freqs = np.array([counts_ba[k] for k in k_vals], dtype=float)
pk = freqs / freqs.sum()  # distribución empírica

plt.figure()
# Para visualizar mejor, usamos scatter de k vs pk en log-log
plt.scatter(k_vals, pk, s=10)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("k (log)")
plt.ylabel("P(k) (log)")
plt.title("Distribución de grados — BA log–log (P(k) vs k)")
path_ba_loglog = outdir / f"ba_loglog_pk_{run_tag}.png"
plt.tight_layout()
plt.savefig(path_ba_loglog, dpi=200, bbox_inches="tight")
plt.close()

# c) Media y varianza de los grados en ambas redes
def mean_var(x):
    return float(np.mean(x)), float(np.var(x, ddof=0))

mean_ba, var_ba = mean_var(deg_ba)
mean_er, var_er = mean_var(deg_er)

metrics_df = pd.DataFrame(
    {
        "red": ["Barabási–Albert", "Erdős–Rényi"],
        "n": [n, n],
        "param": [f"m={m}", f"p≈{p:.4f}"],
        "grado_medio": [mean_ba, mean_er],
        "varianza_grado": [var_ba, var_er],
        "grado_máximo": [int(deg_ba.max()), int(deg_er.max())],
        "grado_mínimo": [int(deg_ba.min()), int(deg_er.min())],
    }
)

# Guardar métricas
metrics_path = outdir / f"metrics_{run_tag}.csv"
metrics_df.to_csv(metrics_path, index=False)