import numpy as np


def edge2mat(link, num_node) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A) -> np.ndarray:
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward) -> np.ndarray:
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, labeling_mode="spatial", graph="wlasl"):
        if graph in ["wlasl", "vsl"]:
            num_node = 27
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [
                (5, 6),
                (5, 7),
                (6, 8),
                (8, 10),
                (7, 9),
                (9, 11),
                (12, 13),
                (12, 14),
                (12, 16),
                (12, 18),
                (12, 20),
                (14, 15),
                (16, 17),
                (18, 19),
                (20, 21),
                (22, 23),
                (22, 24),
                (22, 26),
                (22, 28),
                (22, 30),
                (24, 25),
                (26, 27),
                (28, 29),
                (30, 31),
                (10, 12),
                (11, 22),
            ]

            inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == "kinetics":
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14),
            ]
            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == "ntu":
            num_node = 25
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode: str = None) -> np.ndarray:
        if labeling_mode == "spatial":
            return get_spatial_graph(
                self.num_node, self.self_link, self.inward, self.outward
            )
        raise ValueError()
