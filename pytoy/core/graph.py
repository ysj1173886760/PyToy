# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Time    :   2021/11/29 19:03:55
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Compute Graph
'''

import numpy as np

class Graph:

    def __init__(self) -> None:
        self.nodes = []
        self.training = True
        self.low_precise = False

    def add_node(self, node):
        self.nodes.append(node)

    # clear all the graident which computed in this pass
    def clear_graident(self):
        for node in self.nodes:
            node.clear_graident()
    
    def reset_value(self):
        for node in self.nodes:
            node.reset_value(False)
        
    def node_count(self):
        return len(self.nodes)
    
    def train(self):
        """[Train Mode]
        """
        self.training = True
    
    def evaluate(self):
        """[Evaluate Mode]
        """
        self.training = False
    
    def low_precise(self):
        self.low_precise = True
    
    def reset_mark(self):
        for node in self.nodes:
            node.mark = False
    
    def reset_pece_succ(self):
        for node in self.nodes:
            node.reset_pece_succ()

    def calc_degree(self):
        for node in self.nodes:
            node.calc_degree()
    
    def set_mark(self, node):
        node.set_mark()
    
    def param_count(self):
        total = 0
        trainable = 0
        for node in self.nodes:
            if hasattr(node, 'dims'):
                total += np.product(list(node.dims))
                if hasattr(node, 'trainable') and node.trainable:
                    trainable += np.product(list(node.dims))
        print('total params: {} trainable params: {}'.format(total, trainable))

    def draw(self, ax=None):
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np
        except:
            raise Exception("Need Module networkx")

        G = nx.DiGraph()

        already = []
        labels = {}
        for node in self.nodes:
            G.add_node(node)
            # labels[node] = node.__class__.__name__ + ("({:s})".format(str(node.dims)) if hasattr(node, "dims") else "") \
            #     + ("\n[{:.3f}]".format(np.linalg.norm(node.graident))
            #        if node.graident is not None else "") + ("\n{}".format(node.name))
            labels[node] = node.name + "\n{:s}".format(str(node.dims) if hasattr(node, "dims") else "")
            for c in node.get_children():
                if {node, c} not in already:
                    G.add_edge(node, c)
                    already.append({node, c})

        if ax is None:
            fig = plt.figure(figsize=(50, 100))
            ax = fig.add_subplot(111)

        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        node_size = 2000

        # 有雅克比的变量节点
        cm = plt.cm.Reds
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.graident is not None]
        colorlist = [np.linalg.norm(n.graident) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=node_size, alpha=1.0, ax=ax)

        # 无雅克比的变量节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.graident is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=node_size, alpha=1.0, ax=ax)

        # 有雅克比的计算节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variable" and n.graident is not None]
        colorlist = [np.linalg.norm(n.graident) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=node_size, alpha=1.0, ax=ax)

        # 无雅克比的中间
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variable" and n.graident is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=node_size, alpha=1.0, ax=ax)

        # 边
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#014b66", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", font_color="#6c6c6c", font_size=8,
                                ax=ax)

        # 保存图像
        plt.savefig("computing_graph.png")  # save as png

    
    
default_graph = Graph()
