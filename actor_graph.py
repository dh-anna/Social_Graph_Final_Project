import networkx as nx
from tqdm import tqdm

from graph_variables import GraphVariables


def remove_characters_from_crew_list(crew):
    crew_list = crew.split(",")
    actor_list = crew_list[::2]
    return actor_list


class ActorGraph(GraphVariables):
    def __init__(self):
        super().__init__()

    def making_actor_lists(self, movies_crew):
        movies_actors_list = []  # the list our movies and each list is an actor list for that movie
        for i in range(len(movies_crew)):
            if isinstance(movies_crew[i], str):
                movies_actors_list.append(remove_characters_from_crew_list(movies_crew[i]))
        return movies_actors_list


    def add_actor_nodes(self, actors_name):
        for actor_name in actors_name:
            self.graph.add_node(actor_name)


    # this method takes a list of movies, where each movie is represented as a list of actor names
    def get_collaborators(self, movies):
        for movie in movies:
            for i in range(len(movie)):
                for j in range(i + 1, len(movie)):
                    self.graph.add_edge(movie[i], movie[j])

    #Here the movies_crew_lists is a list of strings consisting of actors and their characters in the movie separated by commas
    def make_graph(self, movies_crew_lists):
        movies_actors_list = self.making_actor_lists(movies_crew_lists)
        self.add_actor_nodes(set(actor for movie in movies_actors_list for actor in movie))
        self.get_collaborators(movies_actors_list)

    def __str__(self):
        return str(self.graph)