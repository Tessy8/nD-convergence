import networkx as nx


def bottleneck_3(D):
    h = D // 2

    def block(nodes):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        m = len(nodes)
        for i in range(m):
            G.add_edge(nodes[i], nodes[(i + 1) % m])
        for i in range(m // 2):
            G.add_edge(nodes[i], nodes[i + m // 2])
        return G

    G = nx.Graph()
    G.add_nodes_from(range(D))
    G.add_edges_from(block(list(range(h))).edges())
    G.add_edges_from(block(list(range(h, D))).edges())

    G.remove_edge(0, 1)
    G.remove_edge(h, h + 1)
    G.add_edge(0, h)
    G.add_edge(1, h + 1)
    return G


def prism_3(D):
    return nx.circular_ladder_graph(D // 2)


def expander_3(D):
    seed = 12345
    G = nx.random_regular_graph(3, D, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.random_regular_graph(3, D, seed=seed)
    return G


def bottleneck_4(D):
    h = D // 2

    def block(nodes):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        m = len(nodes)
        for i in range(m):
            G.add_edge(nodes[i], nodes[(i + 1) % m])
            G.add_edge(nodes[i], nodes[(i + 2) % m])
        return G

    G = nx.Graph()
    G.add_nodes_from(range(D))
    G.add_edges_from(block(list(range(h))).edges())
    G.add_edges_from(block(list(range(h, D))).edges())

    for (a, b), (c, d) in [((0, 1), (h, h + 1)), ((2, 3), (h + 2, h + 3))]:
        G.remove_edge(a, b)
        G.remove_edge(c, d)
        G.add_edge(a, c)
        G.add_edge(b, d)
    return G


def circulant_4(D):
    G = nx.Graph()
    G.add_nodes_from(range(D))
    for i in range(D):
        G.add_edge(i, (i + 1) % D)
        G.add_edge(i, (i + 2) % D)
    return G


def expander_4(D):
    seed = 54321
    G = nx.random_regular_graph(4, D, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.random_regular_graph(4, D, seed=seed)
    return G


def families(D):
    return {
        "d3_bottleneck": bottleneck_3(D),
        "d3_prism": prism_3(D),
        "d3_expander": expander_3(D),
        "d4_bottleneck": bottleneck_4(D),
        "d4_circulant": circulant_4(D),
        "d4_expander": expander_4(D),
    }
