import matplotlib.pyplot as plt
import networkx as nx


def visualize_director_director_graph_filtered(director_graph, top_n=100):

    # Filter to most important nodes and edges for visualization
    # Keep directors with high degree (connected to many other directors)
    degree_dict = dict(director_graph.degree())
    sorted_directors = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)

    # Keep top directors by degree
    top_directors = [d[0] for d in sorted_directors[:top_n]]
    subgraph = director_graph.subgraph(top_directors).copy()

    # Further filter edges by weight (keep only significant transitions)
    edges_to_remove = []
    for u, v, data in subgraph.edges(data=True):
        if data['weight'] < 3:  # Only show transitions that happened 3+ times
            edges_to_remove.append((u, v))
    subgraph.remove_edges_from(edges_to_remove)

    # Remove isolated nodes after edge filtering
    isolated = list(nx.isolates(subgraph))
    subgraph.remove_nodes_from(isolated)

    print(f"Filtered graph for visualization:")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20))

    # Calculate layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Get edge weights for sizing
    edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1

    # Draw edges with width based on weight
    for u, v, data in subgraph.edges(data=True):
        weight = data['weight']
        width = 0.5 + (weight / max_weight) * 5  # Scale edge width
        alpha = 0.3 + (weight / max_weight) * 0.7  # Scale transparency
        nx.draw_networkx_edges(subgraph, pos, [(u, v)],
                               width=width, alpha=alpha,
                               edge_color='gray', arrows=True,
                               arrowsize=15, arrowstyle='->', ax=ax)

    # Draw nodes sized by degree
    node_sizes = [degree_dict[node] * 50 for node in subgraph.nodes()]
    node_colors = [degree_dict[node] for node in subgraph.nodes()]

    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                           node_color=node_colors, cmap='viridis',
                           alpha=0.8, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8,
                            font_weight='bold', ax=ax)

    plt.title("Director Transition Graph\n(Top directors, edges with weight ≥ 3)",
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('director_director_graph_figures/director_transition_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nGraph saved as 'director_transition_graph.png'")