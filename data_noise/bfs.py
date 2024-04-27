import rustworkx as rx
from rustworkx.visit import DFSVisitor

class TreeEdgesRecorder(DFSVisitor):

    def __init__(self):
        self.edges = []

    def tree_edge(self, edge):
        self.edges.append(edge)




if __name__ == '__main__':
    graph = rx.PyGraph()
    graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2), (4,5)])
    vis = TreeEdgesRecorder()
    rx.dfs_search(graph, [0], vis)
    print('Tree edges:', vis.edges)