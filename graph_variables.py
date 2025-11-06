import networkx as nx
from matplotlib import pyplot as plt


class GraphVariables:
    def __init__(self):
        self.graph = nx.Graph()



    def plot_graph(self):
        pos = nx.forceatlas2_layout(self.graph, gravity=10)
        plt.figure(1, figsize=(10, 10))
        nx.draw_networkx(self.graph, pos=pos, with_labels=False, node_size=10, alpha=0.3)

    def __str__(self):
        return str(self.graph)