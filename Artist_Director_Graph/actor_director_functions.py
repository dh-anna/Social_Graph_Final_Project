import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from networkx import Graph
from networkx.algorithms.community.quality import modularity
from sklearn.manifold import TSNE

import plotly.graph_objects as go
import networkx as nx
import plotly.io as pio
from typing import Dict, List, Set, Tuple, Any, Hashable


def cut_off_actors_whose_first_movie_was_before_1980(actors_set:Set, df_actors:pd.DataFrame)->Set:
    actors_after_1980 = set()
    for actor in actors_set:
        actor_films = df_actors[df_actors['Actor'] == actor]
        first_film_year = actor_films['Year'].min()
        if first_film_year >= 1980:
            actors_after_1980.add(actor)
    return actors_after_1980

def making_director_actor_graph(actors_after_1980:Set, actors_grouped, movie_directors_dict:Dict, name_lookup:Dict)->nx.DiGraph:
    actors_director_graph = nx.DiGraph()
    actors_director_graph.add_nodes_from(actors_after_1980)
    for actor, actor_movies in actors_grouped:
        # Sort once per actor, not in the initial filter
        actor_movies = actor_movies.sort_values('Year')

        # Count collaborations with directors
        director_counts = {}


        for movie_id in actor_movies['FilmID']:
            # Use dictionary lookup instead of DataFrame search
            directors_str = movie_directors_dict.get(movie_id)
            if directors_str:
                director_ids = directors_str.split(',')
                for director_id in director_ids:
                    director_name = name_lookup.get(director_id)
                    if director_name:
                        director_counts[director_name] = director_counts.get(director_name, 0) + 1

        for director_name, weight in director_counts.items():
            if director_name not in actors_director_graph:
                actors_director_graph.add_node(director_name)
            actors_director_graph.add_edge(actor, director_name, weight=weight)
    return actors_director_graph

def clusters_to_node(nodes:List[str], cluster_labels:np.ndarray)->Dict[int, List[str]]:
    cluster_to_nodes = {}
    for node, cluster in zip(nodes, cluster_labels):
        if cluster not in cluster_to_nodes:
            cluster_to_nodes[cluster] = []
        cluster_to_nodes[cluster].append(node)
    return cluster_to_nodes

def calculate_edge_trace(subgraph, pos ):
    edge_traces = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=min(weight * 0.5, 5), color='rgba(125,125,125,0.5)'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
    return edge_traces

def filter_graph(degree_threshold:int, actors_director_graph:nx.DiGraph)-> nx.Graph:
    degree_threshold = degree_threshold
    filtered_nodes = [n for n, d in actors_director_graph.degree() if d >= degree_threshold]
    filtered_graph = actors_director_graph.subgraph(filtered_nodes)
    return filtered_graph

def calculate_partition(communities):
    partition = {}
    for cluster_id, community in enumerate(communities):
        for node in community:
            partition[node] = cluster_id
    return partition

def find_person_cluster(person_name:str, nodes:List[str], cluster_labels:Dict, cluster_to_nodes:Dict):
    if person_name in nodes:
        idx = nodes.index(person_name)
        cluster = cluster_labels[idx]
        cluster_members = cluster_to_nodes[cluster]
        print(f"{person_name} is in Cluster {cluster}")
        print(f"Other members include: {', '.join(cluster_members[:5])}...")
    else:
        print(f"{person_name} not found (might have been filtered out)")

def calculate_popular_directors_for_each_cluster(cluster_to_nodes:Dict[int, List[str]],popular_directors:List[str])->Tuple[Dict[int, int], Dict[int, List[str]]]:
    cluster_popular_counts = {}
    cluster_popular_directors = {}
    for cluster_id in sorted(cluster_to_nodes.keys()):
        members = cluster_to_nodes[cluster_id]
        # Count how many members are popular directors
        popular_in_cluster = [m for m in members if m in popular_directors]
        cluster_popular_directors[cluster_id] = popular_in_cluster
        cluster_popular_counts[cluster_id] = len(popular_in_cluster)


    return cluster_popular_counts, cluster_popular_directors

def print_popular_directors_in_each_cluster(cluster_popular_counts:Dict[int, int], cluster_popular_directors:Dict[int, List[str]], cluster_to_nodes:Dict[int, List[str]]):
    for cluster_id in sorted(cluster_popular_counts.keys()):
        print(
            f"Cluster {cluster_id}: {len(cluster_popular_directors[cluster_id])} popular directors out of {len(cluster_to_nodes[cluster_id])} total members ({len(cluster_popular_directors[cluster_id]) / len(cluster_to_nodes[cluster_id]) * 100:.1f}%)")
        if cluster_popular_directors[cluster_id]:
            print(f"  Examples: {', '.join(cluster_popular_directors[cluster_id][:5])}")
        print()

def check_how_many_popular_directors_in_the_graph(director_popularity:pd.DataFrame, nodes:List[str]):
    popular_directors_list = list(director_popularity['director'])
    directors_in_graph = [d for d in popular_directors_list if d in nodes]
    directors_not_in_graph = [d for d in popular_directors_list if d not in nodes]

    print(f"  Found in graph: {len(directors_in_graph)}")
    print(f"  NOT found in graph: {len(directors_not_in_graph)}\n")

    if directors_in_graph:
        print(f"Popular directors IN the graph:")
        for d in directors_in_graph[:10]:
            print(f"  - {d}")
        print()

    if directors_not_in_graph:
        print(f"Popular directors NOT in the graph (name mismatch or filtered out):")
        for d in directors_not_in_graph[:10]:
            print(f"  - {d}")
        print()

def list_actors_who_worked_with_popular_directors_in_cluster(cluster_to_nodes:Dict[int, List[str]], cluster_popular_directors:Dict[int, List[str]], filtered_graph:nx.DiGraph):
    for cluster_id in sorted(cluster_to_nodes.keys()):
        popular_directors_in_cluster = cluster_popular_directors[cluster_id]

        if popular_directors_in_cluster:
            print(f"\n{'=' * 80}")
            print(f"CLUSTER {cluster_id} - {len(popular_directors_in_cluster)} popular director(s)")
            print(f"{'=' * 80}")

            for director in popular_directors_in_cluster:
                # Find actors who worked with this director (predecessors in the directed graph)
                # The graph is built as: actor -> director edges
                actors_worked_with = list(filtered_graph.predecessors(director))

                # Get the edge weights (number of collaborations)
                collaborations = [(actor, filtered_graph[actor][director]['weight'])
                                  for actor in actors_worked_with]
                # Sort by number of collaborations
                collaborations.sort(key=lambda x: x[1], reverse=True)

                print(f"\n{director}:")
                print(f"  Worked with {len(actors_worked_with)} actors in this cluster")
                print(f"  Top collaborators:")
                for actor, weight in collaborations[:10]:
                    print(f"    - {actor} ({weight} film{'s' if weight > 1 else ''})")
        else:
            print(f"\nCluster {cluster_id}: No popular directors")


def list_actors_in_cluster(cluster_to_nodes, actor_popularity_map, actors_after_1980, cluster_id, show_top_n=20):
    # Get all members of the cluster
    members = cluster_to_nodes[cluster_id]

    # Filter to only actors (those in the actors_after_1980 set or in Celebrity dataset)
    actors_in_cluster = []
    for member in members:
        if member in actor_popularity_map:
            actors_in_cluster.append({
                'name': member,
                'popularity': actor_popularity_map[member]
            })
        elif member in actors_after_1980:
            actors_in_cluster.append({
                'name': member,
                'popularity': 0
            })

    actors_in_cluster.sort(key=lambda x: x['popularity'], reverse=True)

    # Display results
    print(f"Actors in cluster {cluster_id}")
    print(f"Total members: {len(members)}")
    print(f"Actors identified: {len(actors_in_cluster)}")
    print(f"Directors: {len(members) - len(actors_in_cluster)}")
    print(f"\nTop {min(show_top_n, len(actors_in_cluster))} Actors by Popularity:")

    for i, actor in enumerate(actors_in_cluster[:show_top_n], 1):
        pop_str = f"{actor['popularity']:.2f}" if actor['popularity'] > 0 else "N/A"
        print(f"{i:2d}. {actor['name']:40s} (Popularity: {pop_str})")

    return actors_in_cluster

def calculate_cluster_avg_popularity(cluster_to_nodes, actor_popularity_map):
    cluster_avg_popularity = {}
    for cluster_id in sorted(cluster_to_nodes.keys()):
        members = cluster_to_nodes[cluster_id]

        # Find actors in this cluster who are in the Celebrity dataset
        actors_with_popularity = []
        for member in members:
            if member in actor_popularity_map:
                actors_with_popularity.append(actor_popularity_map[member])

        if actors_with_popularity:
            avg_pop = np.mean(actors_with_popularity)
            #ENI TODO: popular actors score / all actors
            cluster_avg_popularity[cluster_id] = avg_pop
        else:
            cluster_avg_popularity[cluster_id] = 0

    return cluster_avg_popularity

def map_nodes_to_cluster(cluster_to_nodes:Dict[int, List[str]])->Dict[str, int]:
    node_to_cluster = {}
    for cluster_id, members in cluster_to_nodes.items():
        for member in members:
            node_to_cluster[member] = cluster_id

    return node_to_cluster

def calculate_edges_between_clusters(cluster_to_nodes:Dict[int, List[str]], filtered_graph:nx.DiGraph, node_to_cluster:Dict[str, int])->Tuple[int, np.ndarray]:
    n_clusters = len(cluster_to_nodes)
    inter_cluster_edges = np.zeros((n_clusters, n_clusters), dtype=int)

    # Count edges between different clusters
    for edge in filtered_graph.edges():
        source, target = edge
        source_cluster = node_to_cluster.get(source)
        target_cluster = node_to_cluster.get(target)

        if source_cluster is not None and target_cluster is not None:
            inter_cluster_edges[source_cluster][target_cluster] += 1

    return n_clusters, inter_cluster_edges

def visualize_louvain_communities(filtered_graph:nx.DiGraph, nodes:np.ndarray, cluster_labels:np.ndarray)->List[str]:
    adj_matrix = nx.adjacency_matrix(filtered_graph, nodelist=nodes)

    # Use TSNE for dimensionality reduction (for easier visualization)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(adj_matrix.toarray())

    plt.figure(figsize=(15, 15))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                          c=cluster_labels, cmap='tab20',
                          s=10, alpha=0.7)
    plt.colorbar(scatter, label='Community')
    plt.title(f'Actor-Director Network - Louvain Communities (TSNE visualization, {len(nodes)} nodes)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('actor_director_louvain.png', dpi=300, bbox_inches='tight')
    plt.show()
    return embeddings

def calculate_member_centralities(cluster_to_nodes:Dict, degree_centrality:Dict)->Dict:
    cluster_centralities = {}

    for cluster_id, members in cluster_to_nodes.items():
        # Calculate average and max centrality for this cluster
        centralities = [degree_centrality[member] for member in members]
        avg_centrality = np.mean(centralities)
        max_centrality = max(centralities)

        cluster_centralities[cluster_id] = {
            'avg_centrality': avg_centrality,
            'max_centrality': max_centrality,
            'members': members,
            'size': len(members)
        }
    return cluster_centralities

def get_popular_directors(df_movies_IMDB:pd.DataFrame)->pd.DataFrame:
    df_expanded = df_movies_IMDB.copy()
    df_expanded['director'] = df_expanded['director'].str.split(', ')
    df_expanded = df_expanded.explode('director')

    # We remove whitespaces from director names
    df_expanded['director'] = df_expanded['director'].str.strip()

    # Create popularity field for each movie by multiplying avg_vote by votes (cause just avg vote was not enough)
    df_expanded['popularity'] = df_expanded['avg_vote'] * df_expanded['votes']

    # Group by director and calculate
    director_stats = df_expanded.groupby('director').agg({
        'popularity': ['sum'],
        'title': 'count'
    }).round(2)

    # Flatten column names
    director_stats.columns = ['total_popularity',
                              'num_movies']
    director_stats = director_stats.reset_index("director")

    # Sort by average popularity
    director_stats_sorted = director_stats.sort_values('total_popularity', ascending=False)
    return director_stats_sorted

def visualize_actor_director_graph_500_nodes(actors_director_graph:nx.DiGraph):

    pio.renderers.default = 'notebook'

    # Filter to most important nodes/edges
    degree_dict = dict(actors_director_graph.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:500]
    subgraph = actors_director_graph.subgraph(top_nodes)

    # Calculate layout (this may take a minute)
    pos = nx.spring_layout(subgraph, k=1, iterations=50)

    # Create edge traces
    edge_traces = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=min(weight * 0.5, 5), color='rgba(125,125,125,0.5)'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = [pos[node][0] for node in subgraph.nodes()]
    node_y = [pos[node][1] for node in subgraph.nodes()]
    node_text = [f"{node}<br>Connections: {degree_dict[node]}" for node in subgraph.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=[min(degree_dict[node] / 10, 50) for node in subgraph.nodes()],
            color=[degree_dict[node] for node in subgraph.nodes()],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Degree")
        )
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='Actor-Director Network (Top 500 nodes)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.show()

def visualize_actor_director_graph(embeddings:List[str],cluster_labels:np.ndarray, nodes:np.ndarray ):
    # Create interactive plot with hover labels
    fig = go.Figure(data=[go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        text=nodes,  # This adds hover labels
        hovertemplate='%{text}<br>Cluster: %{marker.color}',
        marker=dict(
            size=8,
            color=cluster_labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster")
        )
    )])

    fig.update_layout(
        title=f'Actor-Director Network Clusters ({len(nodes)} nodes)',
        width=1000,
        height=800,
        hovermode='closest'
    )
    fig.show()

def make_louvain_communities(filtered_graph:nx.DiGraph, actors_director_graph:nx.DiGraph)->Tuple[np.ndarray, List[str], np.ndarray]:
    # Convert to undirected graph for Louvain (Louvain works on undirected graphs)
    undirected_graph = filtered_graph.to_undirected().copy()

    # Apply Louvain community detection
    communities = nx.community.louvain_communities(undirected_graph, seed=42)
    # Convert communities (list of sets) to node to cluster mapping
    partition = calculate_partition(communities)

    # Extract cluster labels
    nodes = list(filtered_graph.nodes())
    cluster_labels = np.array([partition[node] for node in nodes])
    n_clusters = len(communities)

    print(f"Louvain detected {n_clusters} communities")

    # Calculate modularity (quality metric for community detection)
    mod = modularity(undirected_graph, communities)
    print(f"The modularity: {mod:.4f}")

    embeddings = visualize_louvain_communities(filtered_graph, nodes, cluster_labels)

    # Show statistics
    print(f"Number of communities detected: {n_clusters}")
    print(f"\nCommunity sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for comm_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Community {comm_id}: {count} members")

    return cluster_labels, nodes, embeddings

def print_members_for_each_clusters(cluster_to_nodes):
    for cluster_id in sorted(cluster_to_nodes.keys()):
        members = cluster_to_nodes[cluster_id]
        print(f"\nCluster {cluster_id} ({len(members)} members):")
        print(", ".join(members[:10]))
        if len(members) > 10:
            print(f"... and {len(members) - 10} more")

def find_top_n_members_of_each_cluster(n:int, cluster_to_nodes:Dict,degree_centrality:Dict, filtered_graph:nx.DiGraph ):
    # Find top members by centrality in each cluster
    for cluster_id in sorted(cluster_to_nodes.keys()):
        members = cluster_to_nodes[cluster_id]
        sorted_members = sorted(members,key=lambda x: degree_centrality[x], reverse=True)
        print(f"\nCluster {cluster_id} - Top {n} most connected:")
        for member in sorted_members[:n]:
            degree = filtered_graph.degree(member)
            print(f"  {member} (degree: {degree})")

def find_cluster_with_highest_avg_centrality(sorted_by_average_centrality, degree_centrality, filtered_graph):
    top_cluster_id = sorted_by_average_centrality[0][0]
    top_cluster_data = sorted_by_average_centrality[0][1]
    members_with_centrality = [(member, degree_centrality[member], filtered_graph.degree(member))for member in top_cluster_data['members']]
    members_with_centrality.sort(key=lambda x: x[1], reverse=True)

    print(f"\nCluster {top_cluster_id} has the highest average centrality of {top_cluster_data['avg_centrality']:.6f}")
    print(f"{'Rank':<6} {'Director or Actor':<40} {'Centrality':<15} {'Degree':<10}")
    for rank, (director, centrality, degree) in enumerate(members_with_centrality[:20], 1):
        print(f"{rank:<6} {director:<40} {centrality:<15.6f} {degree:<10}")

    return members_with_centrality

def clusters_by_average_actors_popularity(df_celebrity:pd.DataFrame, cluster_to_nodes:Dict[int, List[str]])->Dict[str, float]:
    # Filter for actors only
    df_actors_celebrity = df_celebrity[df_celebrity['known_for_department'] == 'Acting']

    # Create a mapping of actor name to popularity
    actor_popularity_map = dict(zip(df_actors_celebrity['name'], df_actors_celebrity['popularity']))

    # Calculate average popularity for actors in each cluster
    cluster_avg_popularity = calculate_cluster_avg_popularity(cluster_to_nodes, actor_popularity_map)

    # Show clusters ranked by average actor popularity
    sorted_clusters = sorted(cluster_avg_popularity.items(), key=lambda x: x[1], reverse=True)
    print("Clusters ranked by average actor popularity:")
    for cluster_id, avg_pop in sorted_clusters:
        if avg_pop > 0:
            print(f"Cluster {cluster_id}: {avg_pop:.2f}")
    return actor_popularity_map

def calculate_between_cluster_connections(n_clusters:int, inter_cluster_edges:np.ndarray, cluster_to_nodes:Dict[int, List[str]]):
    print("Inter-cluster connections (edges between clusters):\n")

    # Create a summary
    cluster_connections = []
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j and inter_cluster_edges[i][j] > 0:
                cluster_connections.append((i, j, inter_cluster_edges[i][j]))

    # Sort by number of edges
    cluster_connections.sort(key=lambda x: x[2], reverse=True)

    # Show top connections
    print("Top 15 inter-cluster connections:")
    for i, (source, target, count) in enumerate(cluster_connections[:15], 1):
        print(f"{i}. Cluster {source} -> Cluster {target}: {count} edges")

    print("Summary by cluster:")
    for cluster_id in sorted(cluster_to_nodes.keys()):
        # Count outgoing edges to other clusters
        outgoing = sum(inter_cluster_edges[cluster_id][j] for j in range(n_clusters) if j != cluster_id)
        # Count incoming edges from other clusters
        incoming = sum(inter_cluster_edges[i][cluster_id] for i in range(n_clusters) if i != cluster_id)
        # Internal edges
        internal = inter_cluster_edges[cluster_id][cluster_id]

        print(f"Cluster {cluster_id}: Internal={internal}, Outgoing={outgoing}, Incoming={incoming}")

def clusters_by_incoming_edges(cluster_to_nodes:Dict[int, List[str]], inter_cluster_edges:np.ndarray, n_clusters:int):
    cluster_incoming = []
    for cluster_id in sorted(cluster_to_nodes.keys()):
        incoming = sum(inter_cluster_edges[i][cluster_id] for i in range(n_clusters) if i != cluster_id)
        cluster_incoming.append((cluster_id, incoming))

    # Sort by incoming edges
    cluster_incoming.sort(key=lambda x: x[1], reverse=True)

    print("Clusters ranked by INCOMING edges (actors from other clusters working with directors in this cluster):")
    for rank, (cluster_id, incoming) in enumerate(cluster_incoming, 1):
        num_members = len(cluster_to_nodes[cluster_id])
        print(f"{rank}. Cluster {cluster_id}: {incoming} incoming edges ({num_members} members)")