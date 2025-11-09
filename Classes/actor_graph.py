from Classes.graph_variables import GraphVariables




class ActorGraph(GraphVariables):
    def __init__(self):
        super().__init__()

    # this method takes a list of movies, where each movie is represented as a list of actor names
    def get_collaborators(self, movies):
        for movie in movies:
            for i in range(len(movie)):
                for j in range(i + 1, len(movie)):
                    if self.graph.has_edge(movie[i], movie[j]):
                        self.graph[movie[i]][movie[j]]['weight'] += 1
                    else:
                        self.graph.add_edge(movie[i], movie[j], weight=1)

    ##Here the movies_crew_lists is a list of strings consisting of actors
    def make_graph(self, movies_crew_lists):
        self.add_nodes_list(set(actor for movie in movies_crew_lists for actor in movie))
        self.get_collaborators(movies_crew_lists)

    def __str__(self):
        return str(self.graph)