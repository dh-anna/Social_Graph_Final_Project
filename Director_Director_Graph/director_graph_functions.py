import networkx as nx


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