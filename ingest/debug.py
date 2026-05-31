import networkx as nx
import glob
import sys

files = sorted(glob.glob("./test_batch/*.graphml"))
if not files:
    print("No files found in ./dataset/test_batch")
    sys.exit()

filepath = files[0]
print(f"debugging file: {filepath} ... \n")

G = nx.read_graphml(filepath)

print("first 5 edges:")
# what key NetworkX used for the edge labels
for u, v, data in list(G.edges(data=True))[:5]:
    print(f"Node {u} -> Node {v} : {data}")

print("\ntracing cycle:")
try:
    # traces infinite loop and prints path
    cycle = nx.find_cycle(G, orientation="original")
    print("cycle path detected:")
    for u, v, direction in cycle:
        edge_data = G.edges[u, v]
        print(f"   {u} -> {v}  | attributes: {edge_data}")
except nx.NetworkXNoCycle:
    print("no cycle found in the raw graph :)")
except Exception as e:
    print(f"error finding cycle: {e}")