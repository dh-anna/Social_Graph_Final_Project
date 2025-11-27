import random
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def build_actor_director_dict(df_actors, movie_directors_dict, name_lookup):
    actor_directors_dict = {}
    unique_actors = df_actors['Actor'].unique()

    for i, actor in enumerate(unique_actors):
        actor_films = df_actors[df_actors['Actor'] == actor].sort_values('Year')

        directors_list = []

        for _, film in actor_films.iterrows():
            film_id = film['FilmID']

            director_row = movie_directors_dict.get(film_id)

            if director_row and director_row != '\\N':
                director_ids = director_row.split(',')

                for director_id in director_ids:
                    director_name = name_lookup.get(director_id)

                    if director_name:
                        directors_list.append({
                            'director': director_name,
                            'year': film['Year'],
                            'film': film['Film']
                        })

        actor_directors_dict[actor] = directors_list

    return actor_directors_dict

def make_director_graph(all_directors, actor_directors_dict ):
    director_graph = nx.DiGraph()

    # Add all directors as nodes
    director_graph.add_nodes_from(all_directors)

    # Iterate through each actor and their director sequence
    for actor, directors_list in actor_directors_dict.items():
        director_sequence = [d['director'] for d in directors_list]

        for i in range(len(director_sequence) - 1):
            director_from = director_sequence[i]
            director_to = director_sequence[i + 1]

            if director_graph.has_edge(director_from, director_to):
                director_graph[director_from][director_to]['weight'] += 1
            else:
                director_graph.add_edge(director_from, director_to, weight=1)

    return director_graph

def calculate_popular_directors_in_degree_centrality(degree_centrality, popular_directors):
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    node_to_rank = {node: rank + 1 for rank, (node, _) in enumerate(sorted_nodes)}

    # Find the rank of each popular director
    popular_director_ranks = []
    directors_not_in_graph = []

    for director in popular_directors:
        if director in node_to_rank:
            rank = node_to_rank[director]
            centrality_value = degree_centrality[director]
            popular_director_ranks.append((director, rank, centrality_value))
        else:
            directors_not_in_graph.append(director)

    # Sort by rank
    popular_director_ranks.sort(key=lambda x: x[1])

    # The minimum X is the maximum rank among all popular directors
    if popular_director_ranks:
        min_X = max(rank for _, rank, _ in popular_director_ranks)
        worst_ranked_director = [item for item in popular_director_ranks if item[1] == min_X][0]
    else:
        min_X = None
        worst_ranked_director = None

    results = {
        'min_X': min_X,
        'worst_ranked_director': worst_ranked_director,
        'popular_director_ranks': popular_director_ranks,
        'directors_not_in_graph': directors_not_in_graph,
        'sorted_nodes': sorted_nodes
    }
    return results



def correlate_with_director_popularity(G, director_popularity_df):
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality failed to converge, using numpy version")
        eigenvector_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

    pagerank_cent = nx.pagerank(G, weight='weight')
    in_degree_cent = dict(G.in_degree(weight='weight'))
    out_degree_cent = dict(G.out_degree(weight='weight'))

    # Create network metrics DataFrame
    network_df = pd.DataFrame({
        'director': list(eigenvector_cent.keys()),
        'eigenvector_centrality': list(eigenvector_cent.values()),
        'pagerank': list(pagerank_cent.values()),
        'in_degree': [in_degree_cent.get(d, 0) for d in eigenvector_cent.keys()],
        'out_degree': [out_degree_cent.get(d, 0) for d in eigenvector_cent.keys()]
    })

    director_popularity_df.rename(columns={'index': 'director_id'}, inplace=True)

    # Add popularity rank (1-based ranking, assuming df is already sorted by popularity)
    director_popularity_df['popularity_rank'] = range(1, len(director_popularity_df) + 1)

    # Merge with popularity data
    merged_df = pd.merge(
        network_df,
        director_popularity_df,
        on='director',
        how='inner'
    )

    # Calculate correlations for each network measure
    network_measures = ['eigenvector_centrality', 'pagerank', 'in_degree', 'out_degree']
    popularity_measures = ['total_popularity', "popularity_rank"]

    results = {}

    for net_measure in network_measures:
        results[net_measure] = {}

        for pop_measure in popularity_measures:
            # Remove any NaN values
            valid_data = merged_df[[net_measure, pop_measure]].dropna()

            # Calculate correlations
            pearson_r, pearson_p = stats.pearsonr(
                valid_data[net_measure],
                valid_data[pop_measure]
            )

            spearman_r, spearman_p = stats.spearmanr(
                valid_data[net_measure],
                valid_data[pop_measure]
            )

            results[net_measure][pop_measure] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(valid_data)
            }

    return results, merged_df


def visualize_director_correlations(merged_df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Eigenvector Centrality vs Total Popularity
    ax = axes[0, 0]
    valid_data = merged_df.dropna(subset=['eigenvector_centrality', 'total_popularity'])

    ax.scatter(valid_data['eigenvector_centrality'],
               valid_data['total_popularity'] / 1e6,  # Convert to millions
               alpha=0.6, s=50)

    # Add trend line
    z = np.polyfit(valid_data['eigenvector_centrality'],
                   valid_data['total_popularity'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['eigenvector_centrality'].min(),
                         valid_data['eigenvector_centrality'].max(), 100)
    ax.plot(x_line, p(x_line), "r-", alpha=0.8)

    # Calculate correlation for title
    r, p_val = stats.spearmanr(valid_data['eigenvector_centrality'],
                               valid_data['total_popularity'])

    ax.set_xlabel('Eigenvector Centrality')
    ax.set_ylabel('Total Popularity (millions)')
    ax.set_title(f'Network Centrality vs Popularity\n(Spearman r = {r:.3f}, p = {p_val:.3f})')

    # Label top directors
    top_directors = valid_data.nlargest(5, 'eigenvector_centrality')
    for _, row in top_directors.iterrows():
        ax.annotate(row['director'],
                    (row['eigenvector_centrality'], row['total_popularity'] / 1e6),
                    fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

    # Plot 2: Eigenvector Centrality vs Number of Movies
    ax = axes[0, 1]
    valid_data = merged_df.dropna(subset=['eigenvector_centrality', 'num_movies'])

    ax.scatter(valid_data['eigenvector_centrality'],
               valid_data['num_movies'],
               alpha=0.6, s=50)

    z = np.polyfit(valid_data['eigenvector_centrality'],
                   valid_data['num_movies'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['eigenvector_centrality'].min(),
                         valid_data['eigenvector_centrality'].max(), 100)
    ax.plot(x_line, p(x_line), "r-", alpha=0.8)

    r, p_val = stats.spearmanr(valid_data['eigenvector_centrality'],
                               valid_data['num_movies'])

    ax.set_xlabel('Eigenvector Centrality')
    ax.set_ylabel('Number of Movies')
    ax.set_title(f'Network Centrality vs Movie Count\n(Spearman r = {r:.3f}, p = {p_val:.3f})')

    # Plot 3: PageRank vs Total Popularity
    ax = axes[1, 0]
    valid_data = merged_df.dropna(subset=['pagerank', 'total_popularity'])

    ax.scatter(valid_data['pagerank'],
               valid_data['total_popularity'] / 1e6,
               alpha=0.6, s=50, c='green')

    z = np.polyfit(valid_data['pagerank'],
                   valid_data['total_popularity'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['pagerank'].min(),
                         valid_data['pagerank'].max(), 100)
    ax.plot(x_line, p(x_line), "g-", alpha=0.8)

    r, p_val = stats.spearmanr(valid_data['pagerank'],
                               valid_data['total_popularity'])

    ax.set_xlabel('PageRank')
    ax.set_ylabel('Total Popularity (millions)')
    ax.set_title(f'PageRank vs Popularity\n(Spearman r = {r:.3f}, p = {p_val:.3f})')

    # Plot 4: In-degree vs Total Popularity
    ax = axes[1, 1]
    valid_data = merged_df.dropna(subset=['in_degree', 'total_popularity'])

    ax.scatter(valid_data['in_degree'],
               valid_data['total_popularity'] / 1e6,
               alpha=0.6, s=50, c='orange')

    z = np.polyfit(valid_data['in_degree'],
                   valid_data['total_popularity'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['in_degree'].min(),
                         valid_data['in_degree'].max(), 100)
    ax.plot(x_line, p(x_line), "orange", alpha=0.8)

    r, p_val = stats.spearmanr(valid_data['in_degree'],
                               valid_data['total_popularity'])

    ax.set_xlabel('In-degree (weighted)')
    ax.set_ylabel('Total Popularity (millions)')
    ax.set_title(f'In-degree vs Popularity\n(Spearman r = {r:.3f}, p = {p_val:.3f})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def analyze_director_rankings(G, director_popularity_df):
    # Calculate network centrality with error handling
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality failed to converge, using numpy version")
        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')

    # Create network ranking
    network_ranking = pd.DataFrame({
        'director': list(centrality.keys()),
        'centrality': list(centrality.values())
    }).sort_values('centrality', ascending=False).reset_index(drop=True)
    network_ranking['network_rank'] = range(1, len(network_ranking) + 1)

    # Create popularity ranking
    pop_ranking = director_popularity_df.copy()
    pop_ranking = pop_ranking.sort_values('total_popularity', ascending=False).reset_index(drop=True)
    pop_ranking['popularity_rank'] = range(1, len(pop_ranking) + 1)

    # Merge rankings
    comparison = pd.merge(
        network_ranking,
        pop_ranking,
        on='director',
        how='inner'
    )

    # Calculate rank correlation
    if len(comparison) > 0:
        rank_corr, rank_p = stats.spearmanr(
            comparison['network_rank'],
            comparison['popularity_rank']
        )
    else:
        rank_corr, rank_p = 0, 1

    # Top-k overlap analysis
    top_k_overlaps = {}
    for k in [5, 10, 20]:
        if len(comparison) >= k:
            top_k_network = set(comparison.nsmallest(k, 'network_rank')['director'])
            top_k_popularity = set(comparison.nsmallest(k, 'popularity_rank')['director'])
            overlap = len(top_k_network & top_k_popularity) / k
            top_k_overlaps[k] = overlap

    # Print results
    print("Director ranking comparison: Network Centrality vs Popularity")
    print(f"\nRank correlation (Spearman): {rank_corr:.3f} (p={rank_p:.4f})")

    for k, overlap in top_k_overlaps.items():
        print(f"Top-{k} overlap: {overlap:.1%}")

    # Show top directors comparison
    print("Top 10 directors comparison")
    print(f"{'Director':<30} {'Network Rank':<15} {'Popularity Rank':<20} {'Total Popularity':<20}")
    print("-" * 85)

    for _, row in comparison.nsmallest(10, 'network_rank').iterrows():
        print(f"{row['director']:<30} {int(row['network_rank']):<15} "
              f"{int(row['popularity_rank']):<20} {row['total_popularity']:>20,.0f}")

    return comparison, rank_corr, top_k_overlaps


def print_correlation_summary(results):
    print("Correlation summary: Network Metrics vs Director Popularity")

    for network_measure, popularity_results in results.items():
        print(f"\n{network_measure.upper().replace('_', ' ')}:")

        for pop_measure, corr_stats in popularity_results.items():
            print(f"  vs {pop_measure.replace('_', ' ')}:")
            print(f"    Pearson r  = {corr_stats['pearson_r']:>7.3f} (p = {corr_stats['pearson_p']:.4f})")
            print(f"    Spearman r = {corr_stats['spearman_r']:>7.3f} (p = {corr_stats['spearman_p']:.4f})")

            # Interpretation
            r = abs(corr_stats['spearman_r'])
            if r > 0.7:
                strength = "Strong"
            elif r > 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"

            if corr_stats['spearman_p'] < 0.001:
                sig = "***"
            elif corr_stats['spearman_p'] < 0.01:
                sig = "**"
            elif corr_stats['spearman_p'] < 0.05:
                sig = "*"
            else:
                sig = "ns"

            print(f"    → {strength} correlation {sig}")



def analyze_director_prestige(G, director_popularity_df):
    results, merged_df = correlate_with_director_popularity(G, director_popularity_df)
    print_correlation_summary(results)
    fig = visualize_director_correlations(merged_df)
    comparison_df, rank_corr, overlaps = analyze_director_rankings(G, director_popularity_df)

    return results, merged_df, comparison_df



def calculate_network_prestige(G, method='eigenvector'):
    if method == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        except:
            centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    elif method == 'pagerank':
        # PageRank (Google's algorithm, similar concept)
        centrality = nx.pagerank(G, weight='weight')

    return centrality

def correlate_with_prestige_measures(G, external_prestige_data):

    # Calculate network centralities
    eigenvector_cent = calculate_network_prestige(G)
    pagerank_cent = calculate_network_prestige(G, 'pagerank')

    # Convert to DataFrame
    network_df = pd.DataFrame({
        'director': list(eigenvector_cent.keys()),
        'eigenvector_centrality': list(eigenvector_cent.values()),
        'pagerank': list(pagerank_cent.values())
    })

    # Merge with external data
    merged_df = pd.merge(network_df, external_prestige_data,
                         on='director', how='inner')

    # Calculate correlations
    results = {}

    # Get all prestige columns (exclude director name and network measures)
    prestige_columns = [col for col in external_prestige_data.columns
                       if col != 'director']

    for prestige_measure in prestige_columns:
        # Skip if column has NaN values
        valid_data = merged_df.dropna(subset=['eigenvector_centrality', prestige_measure])

        if len(valid_data) < 3:
            continue

        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(
            valid_data['eigenvector_centrality'],
            valid_data[prestige_measure]
        )

        # Spearman correlation (monotonic relationship, robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(
            valid_data['eigenvector_centrality'],
            valid_data[prestige_measure]
        )

        # Kendall's tau (rank correlation)
        kendall_tau, kendall_p = stats.kendalltau(
            valid_data['eigenvector_centrality'],
            valid_data[prestige_measure]
        )

        results[prestige_measure] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'n_samples': len(valid_data)
        }

    return results, merged_df

def visualize_correlations(merged_df, prestige_columns, save_path=None):

    n_measures = len(prestige_columns)
    fig, axes = plt.subplots(1, min(n_measures, 4), figsize=(5*min(n_measures, 4), 5))

    if n_measures == 1:
        axes = [axes]

    for idx, prestige_col in enumerate(prestige_columns[:4]):
        ax = axes[idx]

        # Create scatter plot
        valid_data = merged_df.dropna(subset=['eigenvector_centrality', prestige_col])

        ax.scatter(valid_data[prestige_col],
                  valid_data['eigenvector_centrality'],
                  alpha=0.6)

        # Add trend line
        z = np.polyfit(valid_data[prestige_col],
                      valid_data['eigenvector_centrality'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data[prestige_col].min(),
                            valid_data[prestige_col].max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, label='Trend')

        # Calculate correlation for title
        r, p_val = stats.spearmanr(valid_data[prestige_col],
                                   valid_data['eigenvector_centrality'])

        ax.set_xlabel(prestige_col)
        ax.set_ylabel('Eigenvector Centrality')
        ax.set_title(f'r = {r:.3f}, p = {p_val:.3f}')

        # Add director labels for top directors
        top_directors = valid_data.nlargest(3, 'eigenvector_centrality')
        for _, row in top_directors.iterrows():
            ax.annotate(row['director'],
                       (row[prestige_col], row['eigenvector_centrality']),
                       fontsize=8, alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return fig

def create_correlation_matrix(merged_df):

    # Select numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns

    # Calculate correlation matrix
    corr_matrix = merged_df[numeric_cols].corr(method='spearman')

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1)
    plt.title('Correlation Matrix: Network Centrality vs External Prestige Measures')
    plt.tight_layout()
    plt.show()

    return corr_matrix

def validate_ranking_consistency(G, external_rankings):

    # Get network centrality
    centrality = calculate_network_prestige(G, 'eigenvector')

    # Create rankings
    network_ranking = pd.DataFrame({
        'director': list(centrality.keys()),
        'centrality': list(centrality.values())
    }).sort_values('centrality', ascending=False).reset_index(drop=True)
    network_ranking['network_rank'] = range(1, len(network_ranking) + 1)

    # Merge with external rankings
    comparison = pd.merge(network_ranking, external_rankings, on='director')

    # 1. Spearman correlation between rankings
    if 'external_rank' in comparison.columns and len(comparison) > 0:
        rank_corr, rank_p = stats.spearmanr(comparison['network_rank'],
                                            comparison['external_rank'])
    else:
        rank_corr, rank_p = 0, 1

    # 2. Top-k overlap (e.g., top 10, 20, 50)
    top_k_overlaps = {}
    for k in [10, 20, 50]:
        if len(comparison) >= k:
            top_k_network = set(comparison.nsmallest(k, 'network_rank')['director'])
            if 'external_rank' in comparison.columns:
                top_k_external = set(comparison.nsmallest(k, 'external_rank')['director'])
                overlap = len(top_k_network & top_k_external) / k
                top_k_overlaps[k] = overlap

    print("Validation results")
    print(f"\nRank correlation (Spearman): {rank_corr:.3f} (p={rank_p:.4f})")

    for k, overlap in top_k_overlaps.items():
        print(f"Top-{k} overlap: {overlap:.1%}")

    # Show top directors comparison
    print("Top 10 directors comparison")
    print(f"{'Director':<30} {'Network Rank':<15}")
    print("-"*45)

    for _, row in comparison.nsmallest(10, 'network_rank').iterrows():
        print(f"{row['director']:<30} {row['network_rank']:<15}")

    return rank_corr, top_k_overlaps


# Main analysis
def analyze_prestige_correlation(G, external_prestige_df=None):
    # 1. Calculate correlations
    print("Correlation analysis")

    correlations, merged_df = correlate_with_prestige_measures(G, external_prestige_df)

    # Display results
    for measure, corr_stats in correlations.items():
        print(f"\n{measure}:")
        print(f"  Pearson r = {corr_stats['pearson_r']:.3f} (p = {corr_stats['pearson_p']:.4f})")
        print(f"  Spearman r = {corr_stats['spearman_r']:.3f} (p = {corr_stats['spearman_p']:.4f})")
        print(f"  Kendall tau = {corr_stats['kendall_tau']:.3f} (p = {corr_stats['kendall_p']:.4f})")

        # Interpretation
        if abs(corr_stats['spearman_r']) > 0.7:
            strength = "strong"
        elif abs(corr_stats['spearman_r']) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        if corr_stats['spearman_p'] < 0.05:
            print(f"  → {strength} and significant correlation")
        else:
            print(f"  → {strength} but not significant")

    # 2. Visualize correlations
    prestige_cols = [col for col in external_prestige_df.columns
                    if col != 'director']
    visualize_correlations(merged_df, prestige_cols[:4])

    # 3. Create correlation matrix
    corr_matrix = create_correlation_matrix(merged_df)

    # 4. Validate ranking consistency
    if 'external_rank' in external_prestige_df.columns:
        rank_corr, top_k = validate_ranking_consistency(G, external_prestige_df)

    return correlations, merged_df


def compute_transition_probabilities(G):
    out_degree_weights = dict(G.out_degree(weight='weight'))
    transition_probs = {}

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        out_weight = out_degree_weights[u]

        if out_weight > 0:
            prob = weight / out_weight
            transition_probs[(u, v)] = prob

    return transition_probs, out_degree_weights

def calculate_group_transition_probability(director_popularity, director_graph, transition_probs, top_n=10):
    # Analyze transition probabilities TO top 10 directors
    top_10_directors = director_popularity.nlargest(top_n, 'total_popularity')['director'].tolist()
    top_10_in_graph = [d for d in top_10_directors if director_graph.has_node(d)]

    # Create popularity groups
    director_popularity_sorted = director_popularity.sort_values('total_popularity', ascending=False)
    n_directors = len(director_popularity_sorted)
    director_popularity_sorted['group'] = pd.cut(
        range(n_directors),
        bins=[0, 100, 500, 1000, 5000, n_directors],
        labels=['Top 100', 'Top 500', 'Top 1000', 'Top 5000', 'Rest']
    )

    # Calculate avg probability of transitioning to top 10 for each group
    group_results = []

    for group in ['Top 100', 'Top 500', 'Top 1000', 'Top 5000', 'Rest']:
        group_directors = director_popularity_sorted[director_popularity_sorted['group'] == group]['director'].tolist()
        group_directors_in_graph = [d for d in group_directors if
                                    director_graph.has_node(d) and d not in top_10_in_graph]

        probs_to_top10 = []
        for d in group_directors_in_graph:
            # Sum of transition probabilities to any top 10 director
            total_prob = sum(transition_probs.get((d, top), 0) for top in top_10_in_graph)
            probs_to_top10.append(total_prob)

        if probs_to_top10:
            avg_prob = np.mean(probs_to_top10)
            group_results.append({'group': group, 'n_directors': len(probs_to_top10), 'avg_prob_to_top10': avg_prob})
    return group_results


def analyze_network_accessibility(G, director_popularity_df, top_n=10):
    top_directors = director_popularity_df.nlargest(top_n, 'total_popularity')['director'].tolist()
    top_in_graph = [d for d in top_directors if G.has_node(d)]

    reachable_from_top = set()
    for top_dir in top_in_graph:
        successors = set(G.successors(top_dir))
        reachable_from_top.update(successors)
    reachable_from_top = reachable_from_top - set(top_in_graph)

    print(f"\n1. Directors who can receive actors from top {top_n}:")
    print(
        f"   {len(reachable_from_top):,} directors ({len(reachable_from_top) / G.number_of_nodes() * 100:.1f}% of network)")

    can_reach_top = set()
    for top_dir in top_in_graph:
        predecessors = set(G.predecessors(top_dir))
        can_reach_top.update(predecessors)

    can_reach_top = can_reach_top - set(top_in_graph)

    print(f"\n2. Directors who can send actors to top {top_n}:")
    print(f"   {len(can_reach_top):,} directors ({len(can_reach_top) / G.number_of_nodes() * 100:.1f}% of network)")


    bidirectional = reachable_from_top & can_reach_top
    print(f"\n3. Directors with bidirectional access to top {top_n}:")
    print(f"   {len(bidirectional):,} directors ({len(bidirectional) / G.number_of_nodes() * 100:.1f}% of network)")

    all_connected = reachable_from_top | can_reach_top | set(top_in_graph)
    no_direct_access = set(G.nodes()) - all_connected

    print(f"\n4. Directors with no direct connection to top {top_n}:")
    print(
        f"   {len(no_direct_access):,} directors ({len(no_direct_access) / G.number_of_nodes() * 100:.1f}% of network)")



    print("Popularity comparison of Accessible vs Non-Accessible Directors")
    accessible_dirs = list(reachable_from_top | can_reach_top)
    non_accessible_dirs = list(no_direct_access)

    # Get popularity for each group
    acc_pop = director_popularity_df[director_popularity_df['director'].isin(accessible_dirs)]['total_popularity']
    non_acc_pop = director_popularity_df[director_popularity_df['director'].isin(non_accessible_dirs)]['total_popularity']

    print(f"\n{'Metric':<30} {'Connected to Top 10':<25} {'Not Connected':<25}")
    print(f"{'Count':<30} {len(acc_pop):<25,} {len(non_acc_pop):<25,}")
    print(f"{'Mean Popularity':<30} {acc_pop.mean():<25,.0f} {non_acc_pop.mean():<25,.0f}")
    print(f"{'Median Popularity':<30} {acc_pop.median():<25,.0f} {non_acc_pop.median():<25,.0f}")
    print(f"{'Max Popularity':<30} {acc_pop.max():<25,.0f} {non_acc_pop.max():<25,.0f}")

    # Statistical test
    from scipy import stats
    if len(acc_pop) > 0 and len(non_acc_pop) > 0:
        u_stat, p_value = stats.mannwhitneyu(acc_pop, non_acc_pop, alternative='greater')
        print(f"\nMann-Whitney U test (accessible > non-accessible):")
        print(f"  U-statistic: {u_stat:,.0f}")
        print(f"  P-value: {p_value:.4e}")

        if p_value < 0.001:
            print("  => Highly significant: Directors connected to top 10 are more popular")
        else:
            print("=> Directors connected to top 10 are not more popular")




def random_walk_career(G, start_director, n_steps=10, transition_probs=None):
    if not G.has_node(start_director):
        return [start_director]

    path = [start_director]
    current = start_director

    for _ in range(n_steps):
        # Get neighbors (possible next directors)
        neighbors = list(G.successors(current))

        if not neighbors:
            break  # Dead end - no outgoing transitions

        # Get transition probabilities
        probs = []
        for neighbor in neighbors:
            if transition_probs:
                prob = transition_probs.get((current, neighbor), 0)
            else:
                # Compute on the fly
                weight = G[current][neighbor].get('weight', 1)
                total_weight = sum(G[current][n].get('weight', 1) for n in neighbors)
                prob = weight / total_weight
            probs.append(prob)

        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1 / len(neighbors)] * len(neighbors)

        # Choose next director
        current = np.random.choice(neighbors, p=probs)
        path.append(current)

    return path

def get_transition_probabilities(director_graph):
    out_degree_weights = dict(director_graph.out_degree(weight='weight'))
    transition_probs = {}
    for u, v, data in director_graph.edges(data=True):
        weight = data.get('weight', 1)
        out_weight = out_degree_weights[u]
        if out_weight > 0:
            transition_probs[(u, v)] = weight / out_weight

    return transition_probs


def simulate_many_careers(G, n_simulations=10000, n_steps=10, transition_probs=None):
    # Get directors with outgoing edges as possible starting points
    starting_directors = [d for d in G.nodes() if G.out_degree(d) > 0]

    # Track visits to each director
    visit_counts = Counter()
    final_positions = Counter()

    for _ in range(n_simulations):
        # Random starting director (weighted by in-degree to simulate entry points)
        start = random.choice(starting_directors)

        # Simulate career
        path = random_walk_career(G, start, n_steps, transition_probs)

        # Count visits
        for director in path:
            visit_counts[director] += 1

        # Track final position
        final_positions[path[-1]] += 1

    return visit_counts, final_positions


def top_20_most_visited_directors_and_total_popularity(visit_counts, director_popularity ):

    print("\nTop 20 most visited directors (random walk simulation):")
    print(f"{'Rank':<6} {'Director':<35} {'Visits':<10} {'Actual Popularity':<20}")

    for rank, (director, visits) in enumerate(visit_counts.most_common(20), 1):
        pop_data = director_popularity[director_popularity['director'] == director]
        actual_pop = pop_data['total_popularity'].values[0] if not pop_data.empty else 0
        print(f"{rank:<6} {director[:33]:<35} {visits:<10} {actual_pop:>18,.0f}")

    # Calculate correlation between visit frequency and actual popularity
    visit_df = pd.DataFrame(list(visit_counts.items()), columns=['director', 'visit_count'])
    merged = pd.merge(visit_df, director_popularity[['director', 'total_popularity']], on='director')

    spearman_r, spearman_p = stats.spearmanr(merged['visit_count'], merged['total_popularity'])
    pearson_r, pearson_p = stats.pearsonr(merged['visit_count'], merged['total_popularity'])

    print("Random Walk Visits vs Actual Popularity")
    print(f"Sample size: {len(merged):,} directors")
    print(f"\nSpearman r = {spearman_r:.4f} (p = {spearman_p:.4e})")
    print(f"Pearson r  = {pearson_r:.4f} (p = {pearson_p:.4e})")

    if spearman_r > 0.5:
        print("\n=> STRONG POSITIVE correlation: Random walk model predicts actual popularity well.")
    elif spearman_r > 0.3:
        print("\n=> MODERATE POSITIVE correlation: Random walk partially explains popularity")
    else:
        print("\n=> WEAK correlation: Other factors beyond network structure determine popularity")

    return merged, spearman_r, pearson_r

def visualize_top_20_most_visited_directors_and_total_popularity(merged, spearman_r, visit_counts, director_popularity):
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(merged['visit_count'], merged['total_popularity'] / 1e6, alpha=0.3, s=20)
    ax1.set_xlabel('Random Walk Visit Count', fontsize=12)
    ax1.set_ylabel('Actual Popularity (millions)', fontsize=12)
    ax1.set_title(f'Random Walk Visits vs Popularity\n(Spearman r = {spearman_r:.3f})', fontsize=12)
    ax1.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(merged['visit_count'], merged['total_popularity'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['visit_count'].min(), merged['visit_count'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2)

    # Plot 2: Compare top 20 directors
    ax2 = axes[1]
    top_20_rw = visit_counts.most_common(20)
    top_20_pop = director_popularity.nlargest(20, 'total_popularity')['director'].tolist()

    # How many top-20 from random walk are in actual top-20?
    overlap = len(set([d for d, _ in top_20_rw]) & set(top_20_pop))

    rw_ranks = {d: i + 1 for i, (d, _) in enumerate(visit_counts.most_common())}
    pop_ranks = {d: i + 1 for i, d in
                 enumerate(director_popularity.sort_values('total_popularity', ascending=False)['director'])}

    comparison_data = []
    for d in top_20_pop[:10]:
        if d in rw_ranks:
            comparison_data.append({'director': d, 'rw_rank': rw_ranks[d], 'pop_rank': pop_ranks[d]})

    comp_df = pd.DataFrame(comparison_data)
    x = range(len(comp_df))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], comp_df['pop_rank'], width, label='Popularity Rank', color='blue', alpha=0.7)
    ax2.bar([i + width / 2 for i in x], comp_df['rw_rank'], width, label='Random Walk Rank', color='orange', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d[:15] for d in comp_df['director']], rotation=45, ha='right')
    ax2.set_ylabel('Rank (lower = better)')
    ax2.set_title(f'Top 10 Popular Directors: Popularity vs Random Walk Rank')
    ax2.legend()
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('random_walk_vs_popularity.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTop-20 overlap: {overlap}/20 directors appear in both random walk and popularity top-20")


def normalize_prestige(pi, prestige_min, prestige_max):
    return (pi - prestige_min) / (prestige_max - prestige_min + 1e-10)


def compute_average_reputation(career_history, director_prestige, prestige_min,prestige_max, n_tau=None ):
    if not career_history:
        return 0.5  # Default for new actors

    # Get last n_tau directors (or all if n_tau is None)
    if n_tau is not None and n_tau < len(career_history):
        recent_directors = career_history[-n_tau:]
    else:
        recent_directors = career_history

    # Compute average prestige
    prestiges = [normalize_prestige(director_prestige.get(d, prestige_min), prestige_min,prestige_max ) for d in recent_directors]
    m_tau = np.mean(prestiges)

    return m_tau

def get_bin(value, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 2  # Last bin


def mu_memory_function(pi_next, m_tau, P_pi, P_pi_given_m, prestige_bins):
    m_bin = get_bin(m_tau, prestige_bins)
    pi_bin = get_bin(pi_next, prestige_bins)

    # P[π_(i_(t+1))]
    p_marginal = P_pi[pi_bin]

    # P[π_(i_(t+1)) | m_τ]
    p_conditional = P_pi_given_m[m_bin, pi_bin]

    # Avoid division by zero
    if p_marginal < 1e-10:
        return 1.0

    mu = p_conditional / p_marginal
    return mu


def analyze_distance_to_top_directors(G, director_popularity_df, top_n=10):
    top_directors = director_popularity_df.nlargest(top_n, 'total_popularity')['director'].tolist()
    print(f"Top {top_n} most popular directors:")
    for i, d in enumerate(top_directors, 1):
        pop = director_popularity_df[director_popularity_df['director'] == d]['total_popularity'].values[0]
        print(f"  {i:2d}. {d} (popularity: {pop:,.0f})")

    top_directors_in_graph = [d for d in top_directors if G.has_node(d)]
    print(f"\nTop directors found in graph: {len(top_directors_in_graph)}/{top_n}")

    if len(top_directors_in_graph) == 0:
        print("ERROR: No top directors found in the graph!")
        return None

    print("\nCalculating shortest path distances...")
    print("(Measures: how many career steps to reach a top director)")

    distances = {}

    for node in G.nodes():
        if node in top_directors_in_graph:
            distances[node] = 0
        else:
            min_distance = float('inf')
            for top_dir in top_directors_in_graph:
                try:
                    dist = nx.shortest_path_length(G, node, top_dir)
                    min_distance = min(min_distance, dist)
                except nx.NetworkXNoPath:
                    continue
            distances[node] = min_distance if min_distance != float('inf') else None
    reachable = sum(1 for d in distances.values() if d is not None and d > 0)
    unreachable = sum(1 for d in distances.values() if d is None)
    print(f"\nReachability from directors to top 10:")
    print(f"  Can reach top 10: {reachable:,} directors")
    print(f"  Cannot reach top 10: {unreachable:,} directors (no directed path exists)")

    results_list = []
    for director, distance in distances.items():
        if distance is not None:
            pop_data = director_popularity_df[director_popularity_df['director'] == director]
            if not pop_data.empty:
                results_list.append({
                    'director': director,
                    'distance_to_top10': distance,
                    'total_popularity': pop_data['total_popularity'].values[0],
                    'num_movies': pop_data['num_movies'].values[0],
                    'is_top_director': director in top_directors_in_graph
                })

    results_df = pd.DataFrame(results_list)
    # Calculate correlations
    print("Distance to Top 10 vs Popularity")

    analysis_df = results_df[results_df['distance_to_top10'] > 0]

    spearman_r, spearman_p = stats.spearmanr(
        analysis_df['distance_to_top10'],
        analysis_df['total_popularity']
    )

    pearson_r, pearson_p = stats.pearsonr(
        analysis_df['distance_to_top10'],
        analysis_df['total_popularity']
    )

    print(f"\nSample size: {len(analysis_df)} directors (excluding top 10)")
    print(f"\nDistance to nearest top 10 director vs Total Popularity:")
    print(f"  Pearson r  = {pearson_r:>7.4f} (p = {pearson_p:.4e})")
    print(f"  Spearman r = {spearman_r:>7.4f} (p = {spearman_p:.4e})")

    if spearman_p < 0.001:
        sig_level = "highly significant (p < 0.001)"
    elif spearman_p < 0.01:
        sig_level = "significant (p < 0.01)"
    elif spearman_p < 0.05:
        sig_level = "significant (p < 0.05)"
    else:
        sig_level = "not significant"

    if spearman_r < -0.3:
        print(f"\n=> Strong negative correlation ({sig_level})")
        print("   Directors with shorter path TO top 10 tend to be MORE popular")
    elif spearman_r < -0.1:
        print(f"\n=> Moderate negative correlation ({sig_level})")
        print("   Directors closer to top 10 tend to be somewhat more popular")
    elif spearman_r < 0.1:
        print(f"\n=> Weak/No correlation ({sig_level})")
        print("   Directed distance to top 10 is NOT closely linked to popularity")
    else:
        print(f"\n=> Positive correlation ({sig_level})")
        print("   Directors farther from top 10 tend to be more popular")

    # Group analysis by distance
    print(f"\n{'=' * 70}")
    print("Popularity by directed distance")
    print(f"{'=' * 70}")

    print(f"\n{'Distance':<12} {'Count':<10} {'Mean Popularity':<20} {'Median Popularity':<20}")
    print("-" * 62)

    for dist in sorted(analysis_df['distance_to_top10'].unique()):
        group = analysis_df[analysis_df['distance_to_top10'] == dist]
        count = len(group)
        mean_pop = group['total_popularity'].mean()
        median_pop = group['total_popularity'].median()
        print(f"{int(dist):<12} {count:<10} {mean_pop:<20,.0f} {median_pop:<20,.0f}")

    return results_df, spearman_r, spearman_p

def calculate_transitions_with_memory_context(actor_directors_dict, director_prestige, prestige_min, prestige_max):
    transitions_data = []  # (m_tau, pi_next)
    all_prestiges = []  # For marginal P[π]

    for actor, directors_list in actor_directors_dict.items():
        if len(directors_list) < 2:
            continue

        director_sequence = [d['director'] for d in directors_list]

        for t in range(1, len(director_sequence)):
            # Career history up to time t
            history = director_sequence[:t]
            next_director = director_sequence[t]

            if next_director in director_prestige:
                # Compute m_tau from history
                m_tau = compute_average_reputation(history, director_prestige, prestige_min, prestige_max, n_tau=5)
                pi_next = normalize_prestige(director_prestige.get(next_director, prestige_min), prestige_min,
                                             prestige_max)

                transitions_data.append((m_tau, pi_next))
                all_prestiges.append(pi_next)
    return transitions_data, all_prestiges

def estimate_Pn(n_bins, all_prestiges, prestige_bins):
    prestige_counts = np.zeros(n_bins)
    for pi in all_prestiges:
        bin_idx = get_bin(pi, prestige_bins)
        prestige_counts[bin_idx] += 1
    P_pi = prestige_counts / prestige_counts.sum()
    return P_pi

def estimate_pim(n_bins, transitions_data, prestige_bins):
    joint_counts = np.zeros((n_bins, n_bins))
    m_tau_counts = np.zeros(n_bins)

    for m_tau, pi_next in transitions_data:
        m_bin = get_bin(m_tau, prestige_bins)
        pi_bin = get_bin(pi_next, prestige_bins)
        joint_counts[m_bin, pi_bin] += 1
        m_tau_counts[m_bin] += 1
    return joint_counts, m_tau_counts

def get_P_pi_given_m(n_bins, m_tau_counts, joint_counts):
    P_pi_given_m = np.zeros((n_bins, n_bins))
    for m_bin in range(n_bins):
        if m_tau_counts[m_bin] > 0:
            P_pi_given_m[m_bin, :] = joint_counts[m_bin, :] / m_tau_counts[m_bin]

    return P_pi_given_m

def visualize_memory_effect(prestige_bins, P_pi, P_pi_given_m):
    # Visualize the memory effect: P[π|m] vs P[π]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Marginal distribution P[π]
    ax1 = axes[0]
    bin_centers = (prestige_bins[:-1] + prestige_bins[1:]) / 2
    ax1.bar(bin_centers, P_pi, width=0.04, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Normalized Prestige π', fontsize=12)
    ax1.set_ylabel('P[π]', fontsize=12)
    ax1.set_title('Marginal Distribution of Director Prestige', fontsize=12)
    ax1.grid(alpha=0.3)

    # Plot 2: Conditional distribution P[π|m] for different memory levels
    ax2 = axes[1]
    memory_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0, 1, len(memory_levels)))

    for m_tau, color in zip(memory_levels, colors):
        m_bin = get_bin(m_tau, prestige_bins)
        ax2.plot(bin_centers, P_pi_given_m[m_bin, :],
                 label=f'm_τ = {m_tau:.1f}', color=color, linewidth=2)

    ax2.set_xlabel('Next Director Prestige π', fontsize=12)
    ax2.set_ylabel('P[π | m_τ]', fontsize=12)
    ax2.set_title('Conditional Distribution by Memory Level', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Memory function μ = P[π|m] / P[π]
    ax3 = axes[2]
    for m_tau, color in zip(memory_levels, colors):
        mu_values = []
        for pi in bin_centers:
            mu = mu_memory_function(pi, m_tau, P_pi, P_pi_given_m, prestige_bins)
            mu_values.append(mu)
        ax3.plot(bin_centers, mu_values, label=f'm_τ = {m_tau:.1f}', color=color, linewidth=2)

    ax3.axhline(y=1, color='black', linestyle='--', linewidth=1, label='μ = 1 (no effect)')
    ax3.set_xlabel('Next Director Prestige π', fontsize=12)
    ax3.set_ylabel('μ[π; m_τ]', fontsize=12)
    ax3.set_title('Memory Function μ = P[π|m_τ] / P[π]', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('memory_function_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nInterpretation:")
    print("- μ > 1: Actor with memory m_τ is MORE likely to work with director of prestige π")
    print("- μ < 1: Actor with memory m_τ is LESS likely to work with director of prestige π")
    print("- μ = 1: No memory effect (same as marginal probability)")


def memory_random_walk_eq1(G, start_director, prestige_min, prestige_max, n_steps=10, n_tau=5, transition_probs=None, director_prestige=None,
                           P_pi=None, P_pi_given_m=None, prestige_bins=None):
    if not G.has_node(start_director):
        return [start_director]

    path = [start_director]
    current = start_director

    for step in range(n_steps):
        neighbors = list(G.successors(current))
        if not neighbors:
            break

        # Compute actor's average reputation m_τ from past n_tau films
        m_tau = compute_average_reputation(path, director_prestige, prestige_min, prestige_max, n_tau=n_tau)

        # Compute modified probabilities using Equation (1)
        modified_probs = []
        for neighbor in neighbors:
            # P[i_(t+1) | i_t] - base transition probability
            base_prob = transition_probs.get((current, neighbor), 1e-10)

            # π_(i_(t+1)) - prestige of potential next director
            pi_next = normalize_prestige(director_prestige.get(neighbor, prestige_min), prestige_min, prestige_max)

            # μ[π_(i_(t+1)); m_τ] = P[π | m_τ] / P[π]
            mu = mu_memory_function(pi_next, m_tau, P_pi, P_pi_given_m, prestige_bins)

            # Combined probability (before normalization by K)
            modified_probs.append(mu * base_prob)

        # Normalize by K to ensure probabilities sum to 1
        total = sum(modified_probs)
        if total > 0:
            modified_probs = [p / total for p in modified_probs]
        else:
            modified_probs = [1 / len(neighbors)] * len(neighbors)

        # Sample next director
        current = np.random.choice(neighbors, p=modified_probs)
        path.append(current)

    return path

def simulating_careers(director_graph, prestige_min, prestige_max,prestige_bins, P_pi_given_m, P_pi,director_prestige=None,  n_steps=10, n_tau=5, transition_probs=None, ):
    starting_directors = [d for d in director_graph.nodes() if director_graph.out_degree(d) > 0]
    memory_visits = Counter()
    memory_careers = []

    for _ in range(5000):
        start = random.choice(starting_directors)
        path = memory_random_walk_eq1(director_graph, start, prestige_min, prestige_max, n_steps=n_steps, n_tau=n_tau,
                                      transition_probs=transition_probs, director_prestige=director_prestige, P_pi=P_pi,
                                      P_pi_given_m=P_pi_given_m, prestige_bins=prestige_bins, )
        for d in path:
            memory_visits[d] += 1
        memory_careers.append(path)

    return memory_visits, memory_careers

def get_prestige_trajectory(career, director_prestige, prestige_min, prestige_max):
    return [normalize_prestige(director_prestige.get(d, prestige_min), prestige_min, prestige_max) for d in career]

def pad_trajectory(traj, max_len):
    if len(traj) < max_len:
        return traj + [traj[-1]] * (max_len - len(traj))
    return traj[:max_len]

def visualize_prestige_evolution(max_len,mean_prestige, std_prestige, start_prestiges, memory_trajectories ):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    steps = range(max_len)
    ax1.plot(steps, mean_prestige, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(steps, mean_prestige - std_prestige, mean_prestige + std_prestige,
                     alpha=0.3, color='blue', label='±1 std')
    ax1.set_xlabel('Career Step', fontsize=12)
    ax1.set_ylabel('Normalized Prestige', fontsize=12)
    ax1.set_title('Prestige Evolution (Memory Model)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Start vs End prestige scatter
    ax2 = axes[1]
    ax2.scatter(start_prestiges[:1000], [t[-1] for t in memory_trajectories[:1000] if len(t) > 1][:1000],
                alpha=0.3, s=20)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No change')
    ax2.set_xlabel('Starting Prestige', fontsize=12)
    ax2.set_ylabel('Ending Prestige', fontsize=12)
    ax2.set_title('Career Start vs End (Memory Effect)', fontsize=12)
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('memory_career_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_actors_to_popular_directors(actor_directors_dict, popular_directors):
    actors_with_popular = set()
    actors_without_popular = set()
    for actor, directors_list in actor_directors_dict.items():
        worked_with_popular = any(d['director'] in popular_directors for d in directors_list)

        if worked_with_popular:
            actors_with_popular.add(actor)
        else:
            actors_without_popular.add(actor)
    return actors_with_popular, actors_without_popular

def visualize_actor_popularity(popularity_without_popular, popularity_with_popular ):
    # Visualize the comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Box plot comparison
    ax1 = axes[0]
    data_to_plot = [popularity_without_popular, popularity_with_popular]
    bp = ax1.boxplot(data_to_plot, labels=['Never worked with\npopular directors', 'Worked with\npopular directors'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax1.set_ylabel('Actor Popularity Score', fontsize=12)
    ax1.set_title('Actor Popularity Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add mean markers
    means = [np.mean(popularity_without_popular), np.mean(popularity_with_popular)]
    ax1.plot([1, 2], means, 'D', color='red', markersize=8, label='Mean', zorder=3)
    ax1.legend()

    # Plot 2: Histogram comparison
    ax2 = axes[1]
    bins = np.linspace(0, max(max(popularity_with_popular), max(popularity_without_popular)), 30)
    ax2.hist(popularity_without_popular, bins=bins, alpha=0.5, label='Never worked with popular directors',
             color='coral', edgecolor='black')
    ax2.hist(popularity_with_popular, bins=bins, alpha=0.5, label='Worked with popular directors',
             color='green', edgecolor='black')

    ax2.axvline(np.mean(popularity_without_popular), color='red', linestyle='--', linewidth=2,
                label=f'Mean (no popular): {np.mean(popularity_without_popular):.1f}')
    ax2.axvline(np.mean(popularity_with_popular), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean (with popular): {np.mean(popularity_with_popular):.1f}')

    ax2.set_xlabel('Actor Popularity Score', fontsize=12)
    ax2.set_ylabel('Number of Actors', fontsize=12)
    ax2.set_title('Actor Popularity Distribution (Histogram)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('actor_quality_popular_directors.png', dpi=300, bbox_inches='tight')
    plt.show()


