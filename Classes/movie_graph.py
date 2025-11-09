from Classes.graph_variables import GraphVariables


class MovieGraph(GraphVariables):
    def __init__(self):
        super().__init__()

    def movies_with_common_actors(self, movies_crew_lists_dictionary):
        for movie1 in movies_crew_lists_dictionary:
            for movie2 in movies_crew_lists_dictionary:
                if movie1 != movie2:
                    common_actors = set(movies_crew_lists_dictionary[movie1]) & set(movies_crew_lists_dictionary[movie2])
                    if common_actors:
                        self.graph.add_edge(movie1, movie2, weight=len(common_actors))



    # Here the movies_crew_lists is a list of strings consisting of actors
    def make_graph(self, movies_crew_lists_dictionary):
        self.add_nodes_list(set(movies_crew_lists_dictionary.keys()))
        self.movies_with_common_actors(movies_crew_lists_dictionary)