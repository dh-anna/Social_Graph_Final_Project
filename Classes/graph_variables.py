import networkx as nx
from matplotlib import pyplot as plt
from fa2 import ForceAtlas2

def remove_characters_from_crew_list(crew):
    crew_list = crew.split(",")
    actor_list = crew_list[::2]
    return actor_list


def making_actor_lists( movies_crew):
    movies_actors_list = []  # the list our movies and each list is an actor list for that movie
    for i in range(len(movies_crew)):
        if isinstance(movies_crew[i], str):
            movies_actors_list.append(remove_characters_from_crew_list(movies_crew[i]))
    return movies_actors_list


class GraphVariables:
    def __init__(self):
        self.graph = nx.Graph()

    def add_nodes_list(self, node_list):
        for node in node_list:
            self.graph.add_node(node)


    def plot_graph(self):
        forceatlas2 = ForceAtlas2()
        pos = forceatlas2.forceatlas2_networkx_layout(self.graph, pos=None, iterations=50)
        plt.figure(1, figsize=(10, 10))
        nx.draw_networkx(self.graph, pos=pos, with_labels=False, node_size=10, alpha=0.3)

    def __str__(self):
        return str(self.graph)

