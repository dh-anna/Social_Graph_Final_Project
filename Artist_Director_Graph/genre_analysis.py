import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy



def map_movies_to_genres(cluster_movies, df_tmdb):
    # Create a mapping from imdb_id to genres
    imdb_to_genres = {}
    for imdb_id, row in df_tmdb.iterrows():
        if pd.notna(imdb_id) and pd.notna(row['genre']):
            imdb_to_genres[imdb_id] = row['genre']

    cluster_genres = {}

    for cluster_id, movies in cluster_movies.items():
        genres_list = []

        for movie in movies:
            film_id = movie['film_id']

            # Check if this film_id is in our TMDB mapping
            if film_id in imdb_to_genres:
                genres_str = imdb_to_genres[film_id]
                # Split genres by comma and strip whitespace
                genres = [g.strip() for g in genres_str.split(',')]
                genres_list.extend(genres)

        cluster_genres[cluster_id] = genres_list

    return cluster_genres


def analyze_genre_distribution(cluster_genres, cluster_to_nodes, top_n=10):
    cluster_genre_stats = {}

    print("=" * 80)
    print("GENRE ANALYSIS BY CLUSTER")
    print("=" * 80)

    for cluster_id in sorted(cluster_genres.keys()):
        genres = cluster_genres[cluster_id]

        if not genres:
            print(f"\nCluster {cluster_id} ({len(cluster_to_nodes[cluster_id])} members): No genre data available")
            continue

        # Count genre occurrences
        genre_counts = Counter(genres)
        total_genre_mentions = len(genres)

        # Get top genres
        top_genres = genre_counts.most_common(top_n)

        # Calculate statistics
        cluster_genre_stats[cluster_id] = {
            'total_genre_mentions': total_genre_mentions,
            'unique_genres': len(genre_counts),
            'top_genres': top_genres,
            'genre_distribution': genre_counts,
            'cluster_size': len(cluster_to_nodes[cluster_id])
        }

        print(f"\nCluster {cluster_id} ({len(cluster_to_nodes[cluster_id])} members)")
        print(f"  Total genre mentions: {total_genre_mentions}")
        print(f"  Unique genres: {len(genre_counts)}")
        print(f"  Top {min(top_n, len(top_genres))} genres:")

        for i, (genre, count) in enumerate(top_genres, 1):
            percentage = (count / total_genre_mentions) * 100
            print(f"    {i:2d}. {genre:25s} {count:5d} ({percentage:5.1f}%)")

    return cluster_genre_stats


def get_genre_specialization_score(cluster_genres):

    specialization_scores = {}

    for cluster_id, genres in cluster_genres.items():
        if not genres:
            continue

        genre_counts = Counter(genres)
        total = len(genres)

        # Calculate probabilities
        probs = np.array([count / total for count in genre_counts.values()])

        # Calculate Shannon entropy (diversity index)
        shannon_entropy = entropy(probs, base=2)

        # Normalize by max possible entropy (log2 of number of unique genres)
        max_entropy = np.log2(len(genre_counts)) if len(genre_counts) > 1 else 1
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0

        specialization_scores[cluster_id] = {
            'entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'unique_genres': len(genre_counts),
            'total_mentions': total
        }

    return specialization_scores


def compare_clusters_by_genre(cluster_genres, cluster_to_nodes):
    # Get all unique genres across all clusters
    all_genres = set()
    for genres in cluster_genres.values():
        all_genres.update(genres)

    all_genres = sorted(all_genres)

    # Create a matrix: clusters x genres
    cluster_ids = sorted(cluster_genres.keys())
    matrix = np.zeros((len(cluster_ids), len(all_genres)))

    for i, cluster_id in enumerate(cluster_ids):
        genres = cluster_genres[cluster_id]
        if genres:
            genre_counts = Counter(genres)
            total = len(genres)

            for j, genre in enumerate(all_genres):
                # Store as percentage
                matrix[i, j] = (genre_counts.get(genre, 0) / total * 100) if total > 0 else 0

    # Create DataFrame for easier viewing
    df_matrix = pd.DataFrame(
        matrix,
        index=[f"Cluster {cid}" for cid in cluster_ids],
        columns=all_genres
    )

    return df_matrix


def visualize_top_genres_by_cluster(cluster_genre_stats, top_clusters=10, top_genres=5):
    # Sort clusters by size
    sorted_clusters = sorted(
        cluster_genre_stats.items(),
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

        top_genre_data = stats['top_genres'][:top_genres]
        genres = [g[0] for g in top_genre_data]
        counts = [g[1] for g in top_genre_data]

        # Create bar plot
        bars = ax.barh(genres, counts)
        ax.set_xlabel('Count')
        ax.set_title(f'Cluster {cluster_id} ({stats["cluster_size"]} members)')
        ax.invert_yaxis()

        # Add percentage labels
        total = stats['total_genre_mentions']
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            ax.text(count, i, f' {percentage:.1f}%', va='center')

    # Hide unused subplots
    for idx in range(len(sorted_clusters), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('cluster_genre_distribution.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'cluster_genre_distribution.png'")

    return fig


def print_genre_specialization_ranking(specialization_scores):
    print("\n" + "=" * 80)
    print("GENRE SPECIALIZATION RANKING")
    print("=" * 80)
    print("\nMost Specialized (focused on fewer genres):")

    sorted_by_specialization = sorted(
        specialization_scores.items(),
        key=lambda x: x[1]['normalized_entropy']
    )

    for rank, (cluster_id, stats) in enumerate(sorted_by_specialization[:10], 1):
        print(f"{rank:2d}. Cluster {cluster_id:2d}: "
              f"Diversity={stats['normalized_entropy']:.3f}, "
              f"Unique genres={stats['unique_genres']}, "
              f"Total mentions={stats['total_mentions']}")

    print("\nMost Diverse (spread across many genres):")
    for rank, (cluster_id, stats) in enumerate(reversed(sorted_by_specialization[-10:]), 1):
        print(f"{rank:2d}. Cluster {cluster_id:2d}: "
              f"Diversity={stats['normalized_entropy']:.3f}, "
              f"Unique genres={stats['unique_genres']}, "
              f"Total mentions={stats['total_mentions']}")


def create_genre_heatmap(df_matrix, top_n_clusters=15, top_n_genres=15):
    # Select top N clusters by total genre mentions
    cluster_totals = df_matrix.sum(axis=1).sort_values(ascending=False)
    top_clusters = cluster_totals.head(top_n_clusters).index

    # Select top N genres by total occurrences
    genre_totals = df_matrix.sum(axis=0).sort_values(ascending=False)
    top_genres = genre_totals.head(top_n_genres).index

    # Filter matrix
    df_filtered = df_matrix.loc[top_clusters, top_genres]

    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_filtered, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Genre Distribution: Top {top_n_clusters} Clusters × Top {top_n_genres} Genres')
    plt.xlabel('Genre')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('genre_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as 'genre_heatmap.png'")

    return df_filtered