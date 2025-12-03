'''
import networkx as nx
import itertools
import time
import sys
from tabulate import tabulate

#recursion depth for hashing of large tuples
sys.setrecursionlimit(2000)

def atomicType(graph, kTuple):
    k = len(kTuple)
    adj = graph.adj
    kMatrix = []

    for i in range(k):
        row = []
        for j in range(k):
            vi, vj = kTuple[i], kTuple[j]
            if vi == vj:
                row.append(2)
            elif vj in adj[vi]:
                row.append(1)
            else:
                row.append(0)
        kMatrix.append(tuple(row))

    return tuple(kMatrix)

def getColours(C, kTuple, neighbours):
    k = len(kTuple)
    multisets = []

    if isinstance(neighbours, list): # Standard k-WL, neighbors is V(G)
        allNodes = neighbours
        for j in range(k):
            jMultiset = []
            vList = list(kTuple)
            for w in allNodes:
                phi_j_vw = vList.copy()
                phi_j_vw[j] = w
                jMultiset.append(C[tuple(phi_j_vw)])
            multisets.append(tuple(sorted(jMultiset)))

    elif isinstance(neighbours, dict): # Variant k-WL, neighbors is graph.adj
        adj = neighbours
        for j in range(k):
            jMultiset = []
            vList = list(kTuple)
            vjNeighbours = list(adj[kTuple[j]]) 

            for w in vjNeighbours:
                phi_j_vw = vList.copy()
                phi_j_vw[j] = w
                jMultiset.append(C[tuple(phi_j_vw)])
            multisets.append(tuple(sorted(jMultiset)))

    return tuple(multisets)

def run(G, k, L, variant=False):
    nodes = list(G.nodes())
    n = len(nodes)
    adj = G.adj

    allKTuples = list(itertools.product(nodes, repeat=k))

    c = {}
    for vTuple in allKTuples:
        c[vTuple] = hash(atomicType(G, vTuple))

    for _ in range(L):
        cNew = {}
        for vTuple in allKTuples:
            currColour = c[vTuple]
            if variant:
                neighbourMultisets = getColours(c, vTuple, adj)
            else:
                neighbourMultisets = getColours(c, vTuple, nodes)

            newColour = hash((currColour, neighbourMultisets))
            cNew[vTuple] = newColour

        if c == cNew:
            break
        c = cNew

    return c

def main():
    """
    Benchmarks the two k-WL variants on Erdős-Rényi graphs.
    """
    # Params
    n = 8       # Number of vertices
    k = 2       # k-WL
    L = 3       # Number of iterations

    pVals = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"Benchmarking k-WL vs. k-WL-Variant")
    print(f"Parameters: n={n}, k={k}, L={L}, {n**k} tuples per graph")

    res = []

    for p in pVals:
        G = nx.erdos_renyi_graph(n, p, seed=42)
        E = G.number_of_edges()
        # standard k-WL
        start_time = time.time()
        run(G, k, L, variant=False)
        timeStd = time.time() - start_time
        # variant k-WL
        start_time = time.time()
        run(G, k, L, variant=True)
        timeVar = time.time() - start_time

        res.append([p, E, f"{timeStd:.6f}", f"{timeVar:.6f}"])

    headers = ["p", "|E|", "Std. k-WL (s)", "Var. k-WL (s)"]
    print(tabulate(res, headers=headers, tablefmt="fancy_grid", stralign="right", numalign="right"))

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)

    stdT = [float(r[2]) for r in res]
    varT = [float(r[3]) for r in res]

    if all(t_std > varT[i] for i, t_std in enumerate(stdT)):
        print("Variant k-WL was faster for all p.")
    else:
        print("Standard k-WL was faster for some p.")

    if all(varT[i] <= varT[i+1] for i in range(len(varT)-1)):
        print("Variant k-WL computation time increased with p (more edges).")
    else:
         print("Variant k-WL computation time did not strictly increase with p.")

    stdAvg = sum(stdT) / len(stdT)
    stdDev = (sum((t - stdAvg) ** 2 for t in stdT) / len(stdT)) ** 0.5

    if stdDev / stdAvg < 0.2: # If standard deviation is < 20% of the mean
        print(f"Standard k-WL computation time was relatively constant (Avg: {stdAvg:.4f}s).")
    else:
        print(f"Standard k-WL computation time varied significantly.")

if __name__ == "__main__":
    main()
'''
