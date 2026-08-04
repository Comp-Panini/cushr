import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import glob
import os

import ingest

def reconstruct_from_csr(data, sentence_idx):
    """Rebuilds a single sentence graph strictly from the flat CSR arrays."""
    rowptr = data['rowptr']
    colidx = data['colidx']
    offsets = data['sentenceoffsets']
    topo = data['topolevel']
    
    global_start = offsets[sentence_idx]
    if sentence_idx + 1 < len(offsets):
        global_end = offsets[sentence_idx + 1]
    else:
        global_end = len(rowptr) - 1 

    num_nodes = global_end - global_start
    G_recon = nx.DiGraph()

    for local_i in range(num_nodes):
        global_i = global_start + local_i
        G_recon.add_node(local_i, topo=topo[global_i])

    for local_u in range(num_nodes):
        global_u = global_start + local_u
        
        edge_start = rowptr[global_u]
        edge_end = rowptr[global_u + 1]
        
        for e in range(edge_start, edge_end):
            global_v = colidx[e]
            local_v = global_v - global_start 
            G_recon.add_edge(local_u, local_v)
            
    return G_recon

def load_and_filter_original(filepath, edge_order="id"):
    """Loads the original XML and applies the exact same filter as ingest.py.

    `edge_order` must match the setting the archive under test was built with,
    or every sentence will report a spurious edge mismatch.
    """
    return ingest.forward_edge_filter(nx.read_graphml(filepath), edge_order)

def verify_all_sentences(npz_path, graphml_dir, display_time=3.0, edge_order="id"):
    # Load the GPU Arrays once
    data = np.load(npz_path)
    
    # Get all original files in order
    filepaths = sorted(glob.glob(os.path.join(graphml_dir, "*.graphml")))
    total_sentences = len(filepaths)
    
    print(f"Starting batch verification for {total_sentences} sentences...\n")
    
    for sentence_idx in range(total_sentences):
        print(f"--- Verifying Sentence Index {sentence_idx} ---")
        original_filepath = filepaths[sentence_idx]
        print(f"Comparing against original file: {original_filepath}")
        
        # 1. Reconstruct and Filter
        G_recon = reconstruct_from_csr(data, sentence_idx)
        G_orig = load_and_filter_original(original_filepath, edge_order)
        
        # 2. Mathematical Isomorphism Check
        is_match = nx.is_isomorphic(G_recon, G_orig)
        
        if is_match:
            print("✅ SUCCESS: The CSR arrays perfectly match the original XML structure!\n")
        else:
            print("❌ FAILURE: The reconstructed graph shape differs from the original.")
            print(f"   Reconstructed: {G_recon.number_of_nodes()} nodes, {G_recon.number_of_edges()} edges")
            print(f"   Original:      {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges\n")

        # 3. Visualization
        plt.figure(figsize=(12, 6))
        plt.title(f"Reconstructed GPU Graph (Sentence {sentence_idx})\nFile: {os.path.basename(original_filepath)}")
        
        for node, n_data in G_recon.nodes(data=True):
            G_recon.nodes[node]['subset'] = n_data['topo']
            
        pos = nx.multipartite_layout(G_recon, subset_key="subset")
        
        nx.draw(G_recon, pos, with_labels=True, node_color='lightblue', 
                node_size=500, arrowsize=20, font_weight='bold')
        
        # The Automated Slideshow Logic
        plt.show(block=False)  # Draw the window without pausing the code
        plt.pause(display_time) # Force matplotlib to wait for exactly X seconds
        plt.close()            # Destroy the window and loop to the next sentence

if __name__ == "__main__":
    verify_all_sentences("cushr_data.npz", "./test_batch", display_time=2.0)