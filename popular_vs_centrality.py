"""
Analysis script to find overlap between most popular actors and highest centrality nodes.
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_graph_and_data():
    """Load the actor-director network and celebrity data."""
    print("Loading data...")

    # Load datasets
    df_actors = pd.read_csv('Datasets/IMDB/actorfilms.csv', low_memory=False)
    df_movie_directors = pd.read_csv('Datasets/IMDB/title.crew.tsv', sep='\t', low_memory=False)
    df_name_to_id = pd.read_csv('Datasets/IMDB/name.basics.tsv', sep='\t', low_memory=False)
    df_celebrity = pd.read_csv('Datasets/Celebrity.csv', index_col=0)

    # Filter actors whose first film was after 1980
    actors_set = set(df_actors['Actor'].tolist())
    actors_after_1980 = set()
    for actor in actors_set:
        actor_films = df_actors[df_actors['Actor'] == actor]
        first_film_year = actor_films['Year'].min()
        if first_film_year >= 1980:
            actors_after_1980.add(actor)

    print(f"Found {len(actors_after_1980)} actors with first film >= 1980")

    # Build the graph
    from Artist_Director_Graph.actor_director_functions import making_director_actor_graph

    movie_directors_dict = df_movie_directors.set_index('tconst')['directors'].to_dict()
    name_lookup = df_name_to_id.set_index('nconst')['primaryName'].to_dict()
    actors_filtered = df_actors[df_actors['Actor'].isin(actors_after_1980)]
    actors_grouped = actors_filtered.groupby('Actor')

    print("Building actor-director graph...")
    actors_director_graph = making_director_actor_graph(
        actors_after_1980, actors_grouped, movie_directors_dict, name_lookup
    )

    # Filter graph (degree >= 10)
    from Artist_Director_Graph.actor_director_functions import filter_graph
    filtered_graph = filter_graph(degree_threshold=10, actors_director_graph=actors_director_graph)

    print(f"Graph nodes: {filtered_graph.number_of_nodes()}")
    print(f"Graph edges: {filtered_graph.number_of_edges()}")

    return filtered_graph, df_celebrity, actors_after_1980


def analyze_popular_vs_centrality(filtered_graph, df_celebrity, actors_after_1980,
                                   top_n_popular=100, top_k_centrality=100):
    """
    Analyze overlap between top n popular actors and top k centrality nodes.

    Args:
        filtered_graph: NetworkX graph of actor-director network
        df_celebrity: DataFrame with actor popularity data
        actors_after_1980: Set of actors to consider
        top_n_popular: Number of top popular actors to consider
        top_k_centrality: Number of top centrality nodes to consider

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Top {top_n_popular} Popular Actors vs Top {top_k_centrality} Centrality Nodes")
    print(f"{'='*80}\n")

    # Get top n popular actors
    df_actors_celebrity = df_celebrity[df_celebrity['known_for_department'] == 'Acting'].copy()
    df_actors_celebrity = df_actors_celebrity.sort_values('popularity', ascending=False)

    # Filter to only actors in our dataset
    df_actors_celebrity = df_actors_celebrity[df_actors_celebrity['name'].isin(actors_after_1980)]

    top_popular_actors = set(df_actors_celebrity.head(top_n_popular)['name'].tolist())
    print(f"Top {top_n_popular} popular actors identified")
    print(f"Examples: {', '.join(list(top_popular_actors)[:5])}")

    # Calculate different centrality measures
    print("\nCalculating centrality measures...")

    degree_centrality = nx.degree_centrality(filtered_graph)
    betweenness_centrality = nx.betweenness_centrality(filtered_graph)
    closeness_centrality = nx.closeness_centrality(filtered_graph)
    eigenvector_centrality = nx.eigenvector_centrality(filtered_graph, max_iter=1000)

    # Get top k nodes by each centrality measure
    centrality_measures = {
        'Degree': degree_centrality,
        'Betweenness': betweenness_centrality,
        'Closeness': closeness_centrality,
        'Eigenvector': eigenvector_centrality
    }

    results = {}

    for measure_name, centrality_dict in centrality_measures.items():
        # Get top k nodes
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        top_k_nodes = set([node for node, _ in sorted_nodes[:top_k_centrality]])

        # Find overlap with top popular actors
        overlap = top_popular_actors & top_k_nodes

        # Filter overlap to only actors (not directors)
        overlap_actors = [actor for actor in overlap if actor in actors_after_1980]

        # Calculate percentage
        overlap_percentage = (len(overlap_actors) / top_n_popular) * 100

        results[measure_name] = {
            'top_k_nodes': top_k_nodes,
            'overlap': overlap_actors,
            'overlap_count': len(overlap_actors),
            'overlap_percentage': overlap_percentage,
            'sorted_nodes': sorted_nodes[:top_k_centrality]
        }

        print(f"\n{measure_name} Centrality:")
        print(f"  Overlap: {len(overlap_actors)}/{top_n_popular} actors ({overlap_percentage:.1f}%)")
        print(f"  Top 5 nodes by {measure_name}: {', '.join([node for node, _ in sorted_nodes[:5]])}")

    return results, top_popular_actors, df_actors_celebrity


def print_detailed_results(results, top_popular_actors, df_actors_celebrity, measure='Degree'):
    """Print detailed results for a specific centrality measure."""
    print(f"\n{'='*80}")
    print(f"Detailed Results for {measure} Centrality")
    print(f"{'='*80}\n")

    result = results[measure]
    overlap_actors = result['overlap']

    # Get popularity scores for overlapping actors
    overlap_with_popularity = []
    for actor in overlap_actors:
        pop = df_actors_celebrity[df_actors_celebrity['name'] == actor]['popularity'].values
        if len(pop) > 0:
            overlap_with_popularity.append((actor, pop[0]))

    overlap_with_popularity.sort(key=lambda x: x[1], reverse=True)

    print(f"Top popular actors in top {measure} centrality nodes:")
    print("-" * 80)
    for i, (actor, popularity) in enumerate(overlap_with_popularity[:20], 1):
        print(f"{i:2d}. {actor:40s} (Popularity: {popularity:.2f})")

    # Also show which top popular actors are NOT in high centrality
    print(f"\n\nTop popular actors NOT in top {measure} centrality nodes:")
    print("-" * 80)
    non_overlap = top_popular_actors - set(overlap_actors)
    non_overlap_with_pop = []
    for actor in non_overlap:
        pop = df_actors_celebrity[df_actors_celebrity['name'] == actor]['popularity'].values
        if len(pop) > 0:
            non_overlap_with_pop.append((actor, pop[0]))

    non_overlap_with_pop.sort(key=lambda x: x[1], reverse=True)
    for i, (actor, popularity) in enumerate(non_overlap_with_pop[:20], 1):
        print(f"{i:2d}. {actor:40s} (Popularity: {popularity:.2f})")


def visualize_results(results, top_n_popular, top_k_centrality):
    """Create visualizations of the results."""
    # Create bar chart of overlap percentages
    measures = list(results.keys())
    overlap_percentages = [results[m]['overlap_percentage'] for m in measures]
    overlap_counts = [results[m]['overlap_count'] for m in measures]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart of overlap percentages
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(measures, overlap_percentages, color=colors)
    ax1.set_ylabel('Overlap Percentage (%)', fontsize=12)
    ax1.set_title(f'Top {top_n_popular} Popular Actors in Top {top_k_centrality} Centrality Nodes', fontsize=14)
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, count in zip(bars, overlap_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}/{top_n_popular}\n({height:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # Bar chart of overlap counts
    bars2 = ax2.bar(measures, overlap_counts, color=colors)
    ax2.set_ylabel('Number of Overlapping Actors', fontsize=12)
    ax2.set_title(f'Overlap Count by Centrality Measure', fontsize=14)
    ax2.set_ylim(0, top_n_popular)

    # Add horizontal line at top_n_popular for reference
    ax2.axhline(y=top_n_popular, color='gray', linestyle='--', alpha=0.5, label=f'Max possible ({top_n_popular})')
    ax2.legend()

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('popular_vs_centrality.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to 'popular_vs_centrality.png'")
    plt.show()


def run_analysis(top_n_popular=100, top_k_centrality=100):
    """Run the complete analysis."""
    # Load data
    filtered_graph, df_celebrity, actors_after_1980 = load_graph_and_data()

    # Analyze overlap
    results, top_popular_actors, df_actors_celebrity = analyze_popular_vs_centrality(
        filtered_graph, df_celebrity, actors_after_1980,
        top_n_popular=top_n_popular, top_k_centrality=top_k_centrality
    )

    # Print detailed results for degree centrality
    print_detailed_results(results, top_popular_actors, df_actors_celebrity, measure='Degree')

    # Visualize results
    visualize_results(results, top_n_popular, top_k_centrality)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Top {top_n_popular} popular actors analyzed")
    print(f"Top {top_k_centrality} centrality nodes considered")
    print()
    for measure_name, result in results.items():
        print(f"{measure_name} Centrality: {result['overlap_count']} actors overlap ({result['overlap_percentage']:.1f}%)")

    return results


if __name__ == "__main__":
    # You can adjust these parameters
    TOP_N_POPULAR = 100  # Number of top popular actors to consider
    TOP_K_CENTRALITY = 100  # Number of top centrality nodes to consider

    results = run_analysis(top_n_popular=TOP_N_POPULAR, top_k_centrality=TOP_K_CENTRALITY)