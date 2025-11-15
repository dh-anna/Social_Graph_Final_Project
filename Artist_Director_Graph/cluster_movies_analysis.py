import pandas as pd
import numpy as np
from collections import defaultdict



def get_director_id_mapping(name_lookup):
    director_name_to_id = {name: nconst for nconst, name in name_lookup.items()}
    return director_name_to_id


def collect_cluster_movies(cluster_to_nodes, df_actors, movie_directors_dict, name_lookup):
    # Create reverse mapping
    director_name_to_id = get_director_id_mapping(name_lookup)

    # Pre-process: Create mapping of director_id -> list of movies
    print("Building director-to-movies mapping...")
    director_movies = defaultdict(list)

    # Iterate through unique films
    for film_id in df_actors['FilmID'].unique():
        if film_id in movie_directors_dict:
            director_ids = movie_directors_dict[film_id]
            if pd.notna(director_ids) and director_ids != '\\N':
                # Handle multiple directors
                for dir_id in director_ids.split(','):
                    director_movies[dir_id].append(film_id)

    print(f"Found movies for {len(director_movies)} directors")

    # Collect movies for each cluster
    cluster_movies = {}
    print("\nCollecting movies for each cluster...")

    for cluster_id in sorted(cluster_to_nodes.keys()):
        movies_in_cluster = []
        nodes_in_cluster = cluster_to_nodes[cluster_id]

        # For each director in the cluster
        for name in nodes_in_cluster:
            # Get director ID
            director_id = director_name_to_id.get(name)

            if director_id and director_id in director_movies:
                # Get all films by this director
                film_ids = director_movies[director_id]

                # Get film details from df_actors
                films = df_actors[df_actors['FilmID'].isin(film_ids)]

                # Add to cluster movies
                for _, row in films.iterrows():
                    movies_in_cluster.append({
                        'director': name,
                        'director_id': director_id,
                        'film': row['Film'],
                        'film_id': row['FilmID'],
                        'year': row['Year'],
                        'actor': row['Actor'],
                        'rating': row['Rating'],
                        'votes': row['Votes']
                    })

        cluster_movies[cluster_id] = movies_in_cluster

        if len(movies_in_cluster) > 0:
            # Get unique movies
            unique_films = len(set(m['film_id'] for m in movies_in_cluster))
            unique_directors = len(set(m['director'] for m in movies_in_cluster))
            print(f"  Cluster {cluster_id:2d}: {len(movies_in_cluster):6d} actor-movie entries, "
                  f"{unique_films:5d} unique films, {unique_directors:4d} directors")

    return cluster_movies


def get_unique_movies_per_cluster(cluster_movies):
    cluster_unique_movies = {}

    for cluster_id, movies in cluster_movies.items():
        if not movies:
            cluster_unique_movies[cluster_id] = []
            continue

        # Group by film_id to get unique movies
        unique_movies = {}
        for movie in movies:
            film_id = movie['film_id']
            if film_id not in unique_movies:
                # Keep first entry, we'll aggregate info
                unique_movies[film_id] = {
                    'film': movie['film'],
                    'film_id': film_id,
                    'year': movie['year'],
                    'rating': movie['rating'],
                    'votes': movie['votes'],
                    'directors': set(),
                    'actors': set()
                }

            unique_movies[film_id]['directors'].add(movie['director'])
            unique_movies[film_id]['actors'].add(movie['actor'])

        # Convert sets to lists and create final list
        movies_list = []
        for film_id, info in unique_movies.items():
            movies_list.append({
                'film': info['film'],
                'film_id': film_id,
                'year': info['year'],
                'rating': info['rating'],
                'votes': info['votes'],
                'directors': list(info['directors']),
                'num_directors': len(info['directors']),
                'actors': list(info['actors']),
                'num_actors': len(info['actors'])
            })

        # Sort by year
        movies_list.sort(key=lambda x: x['year'])
        cluster_unique_movies[cluster_id] = movies_list

    return cluster_unique_movies


def print_cluster_movies_summary(cluster_unique_movies, cluster_to_nodes, top_clusters=10):
    """Print summary statistics for each cluster's movies."""

    print(f"\n{'='*80}")
    print("MOVIES BY CLUSTER - SUMMARY")
    print(f"{'='*80}\n")

    # Create summary data
    summary = []
    for cluster_id, movies in cluster_unique_movies.items():
        if movies:
            summary.append({
                'cluster_id': cluster_id,
                'num_movies': len(movies),
                'num_members': len(cluster_to_nodes[cluster_id]),
                'avg_rating': np.mean([m['rating'] for m in movies]),
                'year_range': f"{min(m['year'] for m in movies)}-{max(m['year'] for m in movies)}"
            })

    # Sort by number of movies
    summary.sort(key=lambda x: x['num_movies'], reverse=True)

    print(f"{'Cluster':<8} {'Movies':<8} {'Members':<8} {'Avg Rating':<12} {'Year Range'}")
    print(f"{'-'*60}")

    for s in summary[:top_clusters]:
        print(f"{s['cluster_id']:<8} {s['num_movies']:<8} {s['num_members']:<8} "
              f"{s['avg_rating']:<12.2f} {s['year_range']}")

    return summary


def print_cluster_movies_detail(cluster_id, cluster_unique_movies, top_n=20):

    movies = cluster_unique_movies.get(cluster_id, [])

    print(f"Cluster {cluster_id} - top {min(top_n, len(movies))} movies")


    if not movies:
        print("No movies found for this cluster.")
        return

    # Sort by rating * log(votes) to get popular + high-rated
    movies_sorted = sorted(movies,
                          key=lambda x: x['rating'] * np.log10(max(x['votes'], 1)),
                          reverse=True)

    for i, movie in enumerate(movies_sorted[:top_n], 1):
        directors_str = ", ".join(movie['directors'][:3])
        if movie['num_directors'] > 3:
            directors_str += f" (+ {movie['num_directors'] - 3} more)"

        print(f"{i:2d}. {movie['film']}")
        print(f"    Year: {movie['year']} | Rating: {movie['rating']:.1f} | Votes: {movie['votes']:,}")
        print(f"    Director(s): {directors_str}")
        print(f"    Actors in dataset: {movie['num_actors']}")
        print()

