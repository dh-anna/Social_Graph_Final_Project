import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


ANALYSIS_TYPE = 'genre'


def get_analysis_config(analysis_type):
    configs = {
        'genre': {
            'column_name': 'genre',
            'dataframe_name': 'df_imdb',
            'title': 'Genre',
            'singular_label': 'genre',
            'plural_label': 'genres',
            'heatmap_color': 'YlOrRd',
            'viz_filename': 'cluster_genre_distribution.png',
            'heatmap_filename': 'genre_heatmap.png'
        },
        'production_company': {
            'column_name': 'production_company',
            'dataframe_name': 'df_imdb',
            'title': 'Production Company',
            'singular_label': 'production company',
            'plural_label': 'production companies',
            'heatmap_color': 'YlGnBu',
            'viz_filename': 'cluster_production_company_distribution.png',
            'heatmap_filename': 'production_company_heatmap.png'
        }
    }
    return configs.get(analysis_type, configs['genre'])


def map_movies_to_attributes(cluster_movies, df, analysis_type=None):
    if analysis_type is None:
        analysis_type = ANALYSIS_TYPE

    config = get_analysis_config(analysis_type)
    column_name = config['column_name']

    # Create a mapping from imdb_id to attributes
    imdb_to_attributes = {}
    for imdb_id, row in df.iterrows():
        if pd.notna(imdb_id) and pd.notna(row[column_name]):
            imdb_to_attributes[imdb_id] = row[column_name]

    cluster_attributes = {}

    for cluster_id, movies in cluster_movies.items():
        attributes_list = []

        for movie in movies:
            film_id = movie['film_id']

            # Check if this film_id is in our mapping
            if film_id in imdb_to_attributes:
                attributes_str = imdb_to_attributes[film_id]
                # Split attributes by comma and strip whitespace
                attributes = [a.strip() for a in attributes_str.split(',')]
                attributes_list.extend(attributes)

        cluster_attributes[cluster_id] = attributes_list

    return cluster_attributes


def analyze_attribute_distribution(cluster_attributes, cluster_to_nodes, analysis_type=None, top_n=10):
    if analysis_type is None:
        analysis_type = ANALYSIS_TYPE

    config = get_analysis_config(analysis_type)
    plural_label = config['plural_label']
    title = config['title']

    cluster_stats = {}

    print(f"{title} analysis by cluster")

    for cluster_id in sorted(cluster_attributes.keys()):
        attributes = cluster_attributes[cluster_id]

        if not attributes:
            print(f"\nCluster {cluster_id} ({len(cluster_to_nodes[cluster_id])} members): No {plural_label} data available")
            continue

        # Count attribute occurrences
        attribute_counts = Counter(attributes)
        total_attribute_mentions = len(attributes)

        # Get top attributes
        top_attributes = attribute_counts.most_common(top_n)

        # Calculate statistics
        cluster_stats[cluster_id] = {
            'total_mentions': total_attribute_mentions,
            'unique_count': len(attribute_counts),
            'top_items': top_attributes,
            'distribution': attribute_counts,
            'cluster_size': len(cluster_to_nodes[cluster_id])
        }

        print(f"\nCluster {cluster_id} ({len(cluster_to_nodes[cluster_id])} movies)")
        print(f"  Total movies which have {plural_label} information: {total_attribute_mentions}")
        print(f"  From this, there was  {len(attribute_counts)} number of {plural_label}:")
        print(f"  Top {min(top_n, len(top_attributes))} {plural_label}:")

        for i, (attribute, count) in enumerate(top_attributes, 1):
            percentage = (count / total_attribute_mentions) * 100
            width = 35
            print(f"    {i:2d}. {attribute:{width}s} {count:5d} ({percentage:5.1f}%)")

    return cluster_stats


def get_specialization_score(cluster_attributes):
    specialization_scores = {}

    for cluster_id, attributes in cluster_attributes.items():
        if not attributes:
            continue

        attribute_counts = Counter(attributes)
        total = len(attributes)

        # Calculate probabilities
        probs = np.array([count / total for count in attribute_counts.values()])

        # Calculate Shannon entropy (diversity index)
        shannon_entropy = entropy(probs, base=2)

        # Normalize by max possible entropy (log2 of number of unique attributes)
        max_entropy = np.log2(len(attribute_counts)) if len(attribute_counts) > 1 else 1
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0

        specialization_scores[cluster_id] = {
            'entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'unique_count': len(attribute_counts),
            'total_mentions': total
        }

    return specialization_scores


def compare_clusters_by_attribute(cluster_attributes ):

    # Get all unique attributes across all clusters
    all_attributes = set()
    for attributes in cluster_attributes.values():
        all_attributes.update(attributes)

    all_attributes = sorted(all_attributes)

    # Create a matrix: clusters x attributes
    cluster_ids = sorted(cluster_attributes.keys())
    matrix = np.zeros((len(cluster_ids), len(all_attributes)))

    for i, cluster_id in enumerate(cluster_ids):
        attributes = cluster_attributes[cluster_id]
        if attributes:
            attribute_counts = Counter(attributes)
            total = len(attributes)

            for j, attribute in enumerate(all_attributes):
                # Store as percentage
                matrix[i, j] = (attribute_counts.get(attribute, 0) / total * 100) if total > 0 else 0

    # Create DataFrame for easier viewing
    df_matrix = pd.DataFrame(
        matrix,
        index=[f"Cluster {cid}" for cid in cluster_ids],
        columns=all_attributes
    )

    return df_matrix


def visualize_top_attributes_by_cluster(cluster_stats, analysis_type=None, top_clusters=10, top_items=5):
    if analysis_type is None:
        analysis_type = ANALYSIS_TYPE

    config = get_analysis_config(analysis_type)
    viz_filename = config['viz_filename']

    # Sort clusters by size
    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda x: x[1]['cluster_size'],
        reverse=True
    )[:top_clusters]

    # Prepare data for plotting
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()

    for idx, (cluster_id, stats) in enumerate(sorted_clusters):
        if idx >= len(axes):
            break

        ax = axes[idx]

        top_data = stats['top_items'][:top_items]
        items = [item[0] for item in top_data]
        counts = [item[1] for item in top_data]

        # Create bar plot
        bars = ax.barh(items, counts)
        ax.set_xlabel('Count')
        ax.set_title(f'Cluster {cluster_id} ({stats["cluster_size"]} members)')
        ax.invert_yaxis()

        # Add percentage labels
        total = stats['total_mentions']
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            ax.text(count, i, f' {percentage:.1f}%', va='center')

    # Hide unused subplots
    for idx in range(len(sorted_clusters), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{viz_filename}'")

    return fig


def print_specialization_ranking(specialization_scores, cluster_attributes=None, analysis_type=None, show_top_n=3):
    if analysis_type is None:
        analysis_type = ANALYSIS_TYPE

    config = get_analysis_config(analysis_type)
    title = config['title']
    plural_label = config['plural_label']

    print(f"{title} Specialization ranking")
    print(f"\nMost Specialized (focused on fewer {plural_label}):")

    sorted_by_specialization = sorted(
        specialization_scores.items(),
        key=lambda x: x[1]['normalized_entropy']
    )

    for rank, (cluster_id, stats) in enumerate(sorted_by_specialization[:10], 1):
        print(f"{rank:2d}. Cluster {cluster_id:2d}: "
              f"Diversity={stats['normalized_entropy']:.3f}, "
              f"Unique={stats['unique_count']}, "
              f"Total mentions={stats['total_mentions']}")

        # Show top attributes for this cluster if data is provided
        if cluster_attributes is not None and cluster_id in cluster_attributes:
            attributes = cluster_attributes[cluster_id]
            if attributes:
                attribute_counts = Counter(attributes)
                top_attrs = attribute_counts.most_common(show_top_n)
                total = len(attributes)
                top_list = ', '.join([f"{attr} ({count/total*100:.1f}%)" for attr, count in top_attrs])
                print(f"      → Top {plural_label}: {top_list}")

    print(f"\nMost Diverse (spread across many {plural_label}):")
    for rank, (cluster_id, stats) in enumerate(reversed(sorted_by_specialization[-10:]), 1):
        print(f"{rank:2d}. Cluster {cluster_id:2d}: "
              f"Diversity={stats['normalized_entropy']:.3f}, "
              f"Unique={stats['unique_count']}, "
              f"Total mentions={stats['total_mentions']}")

        # Show top attributes for diverse clusters too
        if cluster_attributes is not None and cluster_id in cluster_attributes:
            attributes = cluster_attributes[cluster_id]
            if attributes:
                attribute_counts = Counter(attributes)
                top_attrs = attribute_counts.most_common(show_top_n)
                total = len(attributes)
                top_list = ', '.join([f"{attr} ({count/total*100:.1f}%)" for attr, count in top_attrs])
                print(f"      → Top {plural_label}: {top_list}")


def create_attribute_heatmap(df_matrix, analysis_type=None, top_n_clusters=15, top_n_items=15):
    if analysis_type is None:
        analysis_type = ANALYSIS_TYPE

    config = get_analysis_config(analysis_type)
    title = config['title']
    singular_label = config['singular_label'].capitalize()
    heatmap_color = config['heatmap_color']
    heatmap_filename = config['heatmap_filename']

    # Select top N clusters by total mentions
    cluster_totals = df_matrix.sum(axis=1).sort_values(ascending=False)
    top_clusters = cluster_totals.head(top_n_clusters).index

    # Select top N items by total occurrences
    item_totals = df_matrix.sum(axis=0).sort_values(ascending=False)
    top_items = item_totals.head(top_n_items).index

    # Filter matrix
    df_filtered = df_matrix.loc[top_clusters, top_items]

    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_filtered, annot=True, fmt='.1f', cmap=heatmap_color, cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'{title.capitalize()} Distribution: Top {top_n_clusters} Clusters × Top {top_n_items} {singular_label}s')
    plt.xlabel(singular_label)
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as '{heatmap_filename}'")

    return df_filtered
