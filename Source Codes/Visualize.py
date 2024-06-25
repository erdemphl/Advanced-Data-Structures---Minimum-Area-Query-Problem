import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from graphviz import Digraph


def draw_cartesian_tree(root):
    dot = Digraph(comment='Cartesian Tree')

    def add_edges(node, parent_data):
        if node.leftNode:
            child_data = f"{node.leftNode.data}__{id(node.leftNode)}_left"
            dot.edge(parent_data, child_data)
            add_edges(node.leftNode, child_data)
        if node.rightNode:
            child_data = f"{node.rightNode.data}__{id(node.rightNode)}_right"
            dot.edge(parent_data, child_data)
            add_edges(node.rightNode, child_data)

    add_edges(root, str(root.data))

    dot.attr(rankdir='TB')  # Set rank direction (Top to Bottom)
    dot.attr(splines='false')  # Avoid curved edges
    dot.attr(directed='true')  # Use directed edges
    dot.format = 'png'  # You can change the output format if needed
    dot.render('cartesian_tree', view=True)  # Renders and saves the tree visualization as 'cartesian_tree.png'
