import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def analyze_cluster_time_periods(cluster_unique_movies, cluster_to_nodes):
    time_stats = []

    for cluster_id in sorted(cluster_unique_movies.keys()):
        movies = cluster_unique_movies[cluster_id]

        if not movies:
            continue

        years = [movie['year'] for movie in movies if movie['year'] is not None]

        if not years:
            continue

        time_stats.append({
            'cluster_id': cluster_id,
            'num_movies': len(movies),
            'num_members': len(cluster_to_nodes[cluster_id]),
            'min_year': min(years),
            'max_year': max(years),
            'mean_year': np.mean(years),
            'median_year': np.median(years),
            'std_year': np.std(years),
            'year_range': max(years) - min(years),
            'q1_year': np.percentile(years, 25),
            'q3_year': np.percentile(years, 75)
        })

    df_stats = pd.DataFrame(time_stats)
    return df_stats


def categorize_by_era(cluster_unique_movies, cluster_to_nodes):
    era_stats = []

    for cluster_id in sorted(cluster_unique_movies.keys()):
        movies = cluster_unique_movies[cluster_id]

        if not movies:
            continue

        classic = 0
        golden = 0
        modern = 0
        contemporary = 0

        for movie in movies:
            year = movie['year']
            if year is None:
                continue

            if year < 2000:
                classic += 1
            elif 2000 <= year < 2008:
                golden += 1
            elif 2008 <= year < 2016:
                modern += 1
            else:
                contemporary += 1

        total = classic + golden + modern + contemporary

        if total == 0:
            continue

        era_stats.append({
            'cluster_id': cluster_id,
            'num_members': len(cluster_to_nodes[cluster_id]),
            'total_movies': total,
            'classic_count': classic,
            'classic_pct': (classic / total) * 100,
            'golden_count': golden,
            'golden_pct': (golden / total) * 100,
            'modern_count': modern,
            'modern_pct': (modern / total) * 100,
            'contemporary_count': contemporary,
            'contemporary_pct': (contemporary / total) * 100,
            'dominant_era': max(
                [('Classic', classic), ('Golden Age', golden),
                 ('Modern', modern), ('Contemporary', contemporary)],
                key=lambda x: x[1]
            )[0]
        })

    df_era = pd.DataFrame(era_stats)
    return df_era


def print_time_period_summary(df_stats, df_era, top_n=15):
    print("TIme period analysis by cluster:")

    # Sort by number of movies
    df_sorted = df_stats.sort_values('num_movies', ascending=False).head(top_n)

    print(f"\n{'Cluster':<8} {'Movies':<8} {'Members':<8} {'Year Range':<15} {'Mean Year':<12} {'Median':<8} {'Dominant Era':<15}")
    print("-" * 100)

    for _, row in df_sorted.iterrows():
        cluster_id = int(row['cluster_id'])
        era_row = df_era[df_era['cluster_id'] == cluster_id].iloc[0]

        year_range_str = f"{int(row['min_year'])}-{int(row['max_year'])}"

        print(f"{cluster_id:<8} {row['num_movies']:<8.0f} {row['num_members']:<8.0f} "
              f"{year_range_str:<15} {row['mean_year']:<12.1f} {row['median_year']:<8.0f} "
              f"{era_row['dominant_era']:<15}")

    print("\n" + "=" * 80)
    print("Era distribution by cluster")
    print("=" * 80)
    print("\nEra Definitions:")
    print("  - Classic: Before 1970")
    print("  - Golden Age: 1970-1989")
    print("  - Modern: 1990-2009")
    print("  - Contemporary: 2010-2021")
    print()

    # Sort by total movies
    df_era_sorted = df_era.sort_values('total_movies', ascending=False).head(top_n)

    print(f"\n{'Cluster':<8} {'Movies':<8} {'Classic':<12} {'Golden':<12} {'Modern':<12} {'Contemporary':<12}")
    print("-" * 80)

    for _, row in df_era_sorted.iterrows():
        cluster_id = int(row['cluster_id'])
        print(f"{cluster_id:<8} {row['total_movies']:<8.0f} "
              f"{row['classic_count']:>4.0f} ({row['classic_pct']:>4.1f}%)  "
              f"{row['golden_count']:>4.0f} ({row['golden_pct']:>4.1f}%)  "
              f"{row['modern_count']:>4.0f} ({row['modern_pct']:>4.1f}%)  "
              f"{row['contemporary_count']:>4.0f} ({row['contemporary_pct']:>4.1f}%)")


def find_era_specialized_clusters(df_era, threshold=60):
    print("\n" + "=" * 80)
    print(f"Clusters specialized in specific eras(>{threshold}% in one era)")
    print("=" * 80)

    specialized = []

    for _, row in df_era.iterrows():
        cluster_id = int(row['cluster_id'])

        if row['classic_pct'] > threshold:
            specialized.append({
                'cluster_id': cluster_id,
                'era': 'Classic (pre-2000)',
                'percentage': row['classic_pct'],
                'count': row['classic_count'],
                'total': row['total_movies']
            })
        elif row['golden_pct'] > threshold:
            specialized.append({
                'cluster_id': cluster_id,
                'era': 'Golden Age (2000-2008)',
                'percentage': row['golden_pct'],
                'count': row['golden_count'],
                'total': row['total_movies']
            })
        elif row['modern_pct'] > threshold:
            specialized.append({
                'cluster_id': cluster_id,
                'era': 'Modern (2008-2016)',
                'percentage': row['modern_pct'],
                'count': row['modern_count'],
                'total': row['total_movies']
            })
        elif row['contemporary_pct'] > threshold:
            specialized.append({
                'cluster_id': cluster_id,
                'era': 'Contemporary (2016-2021)',
                'percentage': row['contemporary_pct'],
                'count': row['contemporary_count'],
                'total': row['total_movies']
            })

    if specialized:
        df_specialized = pd.DataFrame(specialized)
        df_specialized = df_specialized.sort_values('percentage', ascending=False)

        print(f"\n{'Cluster':<10} {'Era':<30} {'Movies':<15} {'Percentage':<12}")
        print("-" * 70)

        for _, row in df_specialized.iterrows():
            print(f"{row['cluster_id']:<10} {row['era']:<30} "
                  f"{int(row['count'])}/{int(row['total']):<10} {row['percentage']:<12.1f}%")
    else:
        print(f"\nNo clusters found with >{threshold}% concentration in a single era.")

    return specialized


def visualize_era_distribution(df_era, output_file='cluster_era_distribution.png', top_n=20):
    # Sort by total movies and take top N
    df_plot = df_era.sort_values('total_movies', ascending=False).head(top_n)

    # Prepare data for stacked bar chart
    clusters = df_plot['cluster_id'].astype(int).astype(str)
    classic = df_plot['classic_pct']
    golden = df_plot['golden_pct']
    modern = df_plot['modern_pct']
    contemporary = df_plot['contemporary_pct']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bars
    ax.barh(clusters, classic, label='Classic (<2000)', color='#8B4513')
    ax.barh(clusters, golden, left=classic, label='Golden Age (2000-2008)', color='#DAA520')
    ax.barh(clusters, modern, left=classic+golden, label='Modern (2008-2016)', color='#4682B4')
    ax.barh(clusters, contemporary, left=classic+golden+modern,
            label='Contemporary (2016-2021)', color='#32CD32')

    ax.set_xlabel('Percentage of Movies (%)', fontsize=12)
    ax.set_ylabel('Cluster ID', fontsize=12)
    ax.set_title(f'Movie Era Distribution by Cluster (Top {top_n} Clusters by Movie Count)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{output_file}'")

    return fig


def create_timeline_heatmap(cluster_unique_movies, output_file='cluster_timeline_heatmap.png', top_n=20):
    # Count movies by cluster and decade
    cluster_decade_counts = defaultdict(lambda: defaultdict(int))
    cluster_totals = defaultdict(int)

    for cluster_id, movies in cluster_unique_movies.items():
        for movie in movies:
            year = movie['year']
            if year is None:
                continue

            decade = (year // 10) * 10
            cluster_decade_counts[cluster_id][decade] += 1
            cluster_totals[cluster_id] += 1

    # Get top N clusters by total movies
    top_clusters = sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_cluster_ids = [c[0] for c in top_clusters]

    # Get all decades
    all_decades = set()
    for cluster_id in top_cluster_ids:
        all_decades.update(cluster_decade_counts[cluster_id].keys())

    all_decades = sorted(all_decades)

    # Create matrix (normalized by cluster total)
    matrix = []
    for cluster_id in top_cluster_ids:
        row = []
        total = cluster_totals[cluster_id]
        for decade in all_decades:
            count = cluster_decade_counts[cluster_id][decade]
            percentage = (count / total * 100) if total > 0 else 0
            row.append(percentage)
        matrix.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(
        matrix,
        xticklabels=[f"{int(d)}s" for d in all_decades],
        yticklabels=[f"Cluster {c}" for c in top_cluster_ids],
        cmap='YlOrRd',
        annot=True,
        fmt='.1f',
        cbar_kws={'label': 'Percentage of Cluster Movies (%)'},
        ax=ax
    )

    ax.set_title(f'Movie Timeline Distribution by Cluster (Top {top_n} Clusters)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timeline heatmap saved as '{output_file}'")

    return fig