import networkx as nx
import matplotlib.pyplot as plt

n = 500
k = 6
p = [0, 0.0001, 0.001, 0.01, 0.1, 1]

clustering_results = []
path_lengths = []

for prob in p:
    net = nx.watts_strogatz_graph(n, k, prob)
    
    coef_avg = nx.average_clustering(net)
    clustering_results.append(coef_avg)

    try:
        avg_length = nx.average_shortest_path_length(net)
    except:
        avg_length = float('inf')

    path_lengths.append(avg_length)

base_clustering = clustering_results[0]
base_path_lenght = path_lengths[0]

clustering_norm = [c / base_clustering for c in clustering_results]
path_length_norm = [i / base_path_lenght for i in path_lengths]

plt.figure(figsize=(8, 5))
plt.plot(p, clustering_norm, "o-", label="Coeficiente de agrupamiento normalizado")
plt.plot(p, path_length_norm, 's-', label="Longitud de camino promedio normalizado")
plt.xscale('log')
plt.xlabel("Probabilidad de recableado (p)")
plt.ylabel("Medida normalizada")
plt.title("Small World (Watts-Strogatz)")
plt.legend()
plt.grid(True)
plt.savefig("wattz_strogatz_resultados.png", dpi=300)
