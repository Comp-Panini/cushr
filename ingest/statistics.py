import numpy as np
import os
import matplotlib.pyplot as plt

def compute_statistics(npz_path="cushr_data_full.npz", output_md="STATISTICS.md"):
    if not os.path.exists(npz_path):
        print(f"error: {npz_path} not found, run ingest.py first!")
        return

    print(f"loading {npz_path}...")
    data = np.load(npz_path)

    node_features = data['node_features']
    rowptr = data['rowptr']
    colidx = data['colidx']
    offsets = data['sentenceoffsets']
    goldmask = data['goldpathmask']

    total_nodes = len(node_features)
    total_edges = len(colidx)
    total_sentences = len(offsets)

    # statistics calculations

    # lattice node count distribution
    sentence_boundaries = np.append(offsets, total_nodes)
    nodes_per_sentence = np.diff(sentence_boundaries)

    mean_nodes = np.mean(nodes_per_sentence)
    median_nodes = np.median(nodes_per_sentence)
    p95_nodes = np.percentile(nodes_per_sentence, 95)
    p99_nodes = np.percentile(nodes_per_sentence, 99)
    max_nodes = np.max(nodes_per_sentence)

    # branching factor
    edges_per_node = np.diff(rowptr)
    avg_branching_factor = np.mean(edges_per_node)
    max_branching_factor = np.max(edges_per_node)

    # gold path length
    if np.sum(goldmask) > 0:
        gold_lengths = np.add.reduceat(goldmask, offsets)
        mean_gold_length = np.mean(gold_lengths)
        max_gold_length = np.max(gold_lengths)
    else:
        mean_gold_length = 0.0
        max_gold_length = 0
        print("\nnote: gold path mask is currently all zeros since we want the ground truth mapping")

    # matplotlib
    print("generating distribution histograms...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # graph 1: nodes per sentence
    ax1.hist(nodes_per_sentence, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Lattice Node Count Distribution')
    ax1.set_xlabel('Number of Nodes in Sentence')
    ax1.set_ylabel('Frequency (Total Count)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # graph 2: branching factor
    custom_bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                   20, 25, 30, 50, 70, 100, 150, 200, 250, 300, 400]
    
    clipped_edges = np.clip(edges_per_node, a_min=None, a_max=300)

    counts, edges = np.histogram(clipped_edges, bins=custom_bins)
    bin_widths = np.diff(edges)

    normalized_counts = counts / bin_widths

    bin_labels = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            bin_labels.append(f"{edges[i]}+")
        elif edges[i+1] - edges[i] == 1:
            bin_labels.append(f"{edges[i]}")
        else:
            bin_labels.append(f"{edges[i]}-{edges[i+1]-1}")

    x_positions = np.arange(len(bin_labels))

    ax2.bar(x_positions, normalized_counts, color='salmon', edgecolor='black', log=True, width=1.0)
    ax2.set_title('Branching Factor Distribution (Normalized Log Scale)')
    ax2.set_xlabel('Outgoing Edges (Degree Categories)')
    ax2.set_ylabel('Normalized Frequency (Avg Nodes per Degree)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')

    plt.tight_layout()
    plot_filename = "distribution_plots.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"saved plots to {plot_filename}")

    # generating the markdown
    md_content = f"""# cuSHR Dataset Statistics

**Total sentences:** {total_sentences:,}  
**Total nodes:** {total_nodes:,}  
**Total edges:** {total_edges:,}  

![Distribution Plots]({plot_filename})

## 1. Lattice node count distribution
*How many candidate words are in a single sentence graph?*
* **Mean:** {mean_nodes:.2f} nodes
* **Median:** {median_nodes:.2f} nodes
* **95th percentile:** {p95_nodes:.2f} nodes
* **99th percentile:** {p99_nodes:.2f} nodes
* **Max:** {max_nodes} nodes

## 2. Branching factor
*How many paths can you take from a given node?*
* **Average branching factor:** {avg_branching_factor:.4f} edges per node
* **Max branching factor:** {max_branching_factor} edges from a single node

## 3. Gold path length
*How many nodes make up the true, correct translation?*
* **Average gold path length:** {mean_gold_length:.2f} nodes
* **Max gold path length:** {max_gold_length} nodes
*(note: if these values are 0, ground truth labels have not yet been mapped).*
"""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"success! statistics calculated and written to {output_md}")
    print("-" * 40)

if __name__ == "__main__":
    compute_statistics()