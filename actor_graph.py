import networkx as nx


def remove_characters_from_crew_list(crew):
    crew_list = crew.split(",")
    actor_list = crew_list[::2]
    return actor_list


class ActorGraph:
    def __init__(self):
        self.graph = nx.Graph()


    def add_actor_nodes(self, actors_name):
        for actor_name in actors_name:
            self.graph.add_node(actor_name)


    # this method takes a list of movies, where each movie is represented as a list of actor names
    def get_collaborators(self, movies):
        for movie in movies:
            for i in range(len(movie)):
                for j in range(i + 1, len(movie)):
                    self.graph.add_edge(movie[i], movie[j])

    def make_graph(self, movies):
        self.add_actor_nodes(set(actor for movie in movies for actor in movie))
        self.get_collaborators(movies)

    def __str__(self):
        return str(self.graph)