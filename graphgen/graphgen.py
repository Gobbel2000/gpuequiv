#!/usr/bin/env python3

import json
import sys
from typing import Any
import random

import networkx as nx

DEFAULT_NODES = 100
UPDATE_LEN = 6
ENERGY_CONF = {"elements": UPDATE_LEN, "max": 3}

def random_update():
    update = [0] * UPDATE_LEN
    # Select one element that should be non-zero
    pos = random.randrange(UPDATE_LEN)
    if random.random() >= 0.5:
        # Decrement
        val = -1
    else:
        # Min-update with element at index `val`
        val = random.randrange(1, UPDATE_LEN)
        if val >= pos:
            val += 1
    update[pos] = val
    return update

def format_graph(G: nx.DiGraph) -> dict[str, Any]:
    n_vertices = G.number_of_nodes()
    adj = [list(a) for a in G.adj.values()]
    weights = []
    for node_adj in adj:
        weights.append([random_update() for _ in range(len(node_adj))])
    attacker_pos = random.choices([False, True], k=n_vertices)
    return {
        "conf": ENERGY_CONF,
        "adj": adj,
        "weights": weights,
        "attacker_pos": attacker_pos,
    }

if len(sys.argv) >= 2:
    nodes = int(sys.argv[1])
else:
    nodes = DEFAULT_NODES

G = nx.gnr_graph(nodes, 0).reverse()
data = format_graph(G)
serialized = json.dumps(data)
print(serialized)
