import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def main():
    n_nodes = 200
    prob_vector = np.linspace(0, 0.1, 100).tolist()

    norm_size = []
    
    for p in prob_vector:
        net_er = nx.erdos_renyi_graph(n=n_nodes, p=p)
        connected = nx.connected_components(net_er)
        largest_size = max(len(x) for x in connected)
        
        norm_size.append(
            largest_size/n_nodes
        )
    
    plt.title("Diagrama de Percolacion (Erdős–Rényi) N = 200")
    plt.plot(prob_vector, norm_size, label="Percolacion")
    plt.xlabel("Probabilidad")
    plt.ylabel("Grado de componente gigante")
    umbral_teorico = 1/n_nodes
    plt.axvline(umbral_teorico, linestyle='--', color='r', label=f"Umbral Teorico: {umbral_teorico}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()