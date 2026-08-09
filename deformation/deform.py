import numpy as np
from collections import defaultdict, deque
import igraph as ig
from sympy.abc import x, y
from qldpc import codes, circuits
from qldpc.objects import Pauli
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from deformation.graph_helper_functions import *

"""

This code is for generating deformed stabilizers through a measurement graph attached to a code. It is used for logical measurement and is the input to ``deform_logical_to_tri_lattice(...)``. 

"""

def deform_code_for_logical(H, basis, logical):

    """
    Args:
    - H : parity check matrix of a CSS code. It should have the form [Hx, 0; 0, Hz]
    - basis : the basis of the logical
    - logical : the logical vector we want to measure

    Returns:
    - res : a result dictionary that contains relevant information for the deformation.
    """

    logical = np.asarray(logical, dtype=np.uint8) % 2
    logical_qubits_index = np.where(logical == 1)[0]
    n_data = logical.shape[0]
    qubit_to_vertex = {int(q): i for i, q in enumerate(logical_qubits_index)}
    vertex_to_qubit = {i: int(q) for i, q in enumerate(logical_qubits_index)}
    n_vertices = len(logical_qubits_index)
    g = ig.Graph(n_vertices, edges=[])

    if basis == Pauli.Z:
        H_basis_all = H[:, n_data:]
        H_opposite_basis_all = H[:, :n_data]
    else:
        H_basis_all = H[:, :n_data]
        H_opposite_basis_all = H[:, n_data:]

    edge_set = set()
    for check in H_opposite_basis_all:
        overlap = np.where((logical == 1) & (check == 1))[0]
        if len(overlap) > 0 and len(overlap) % 2 == 0:
            n_pairs = len(overlap) // 2
            for i in range(n_pairs):
                v1 = qubit_to_vertex[int(overlap[2 * i])]
                v2 = qubit_to_vertex[int(overlap[2 * i + 1])]
                edge_set.add(normalize_edge(v1, v2))

    if edge_set:
        g.add_edges(list(edge_set))

    comps = g.components()
    if len(comps) > 1:
        reps = [comp[0] for comp in comps]
        for i in range(len(reps) - 1):
            e = normalize_edge(reps[i], reps[i + 1])
            if not g.are_adjacent(*e):
                g.add_edge(*e)

    row_weights = np.sum(np.array(H_basis_all), axis=1)
    max_cycle_weight = int(np.max(row_weights)) if H_basis_all.shape[0] else 4
    max_cycle_weight = max(max_cycle_weight, 3)

    cycles_edges = find_short_cycle_basis(
        [tuple(map(int, e)) for e in g.get_edgelist()],
        return_edges=True,
    )

    cycles_edges = split_heavy_cycles(
        cycles_edges=cycles_edges,
        max_cycle_weight=max_cycle_weight,
        g=g,
    )

    final_edgelist = [normalize_edge(*e) for e in g.get_edgelist()]
    edge_to_eid = {e: eid for eid, e in enumerate(final_edgelist)}

    n_edges = len(final_edgelist)
    n_qubits = n_data + n_edges

    H_opposite_basis_new = np.zeros((len(cycles_edges), n_qubits), dtype=np.uint8)
    for i, cyc in enumerate(cycles_edges):
        for e in cyc:
            eid = edge_to_eid[normalize_edge(*e)]
            H_opposite_basis_new[i, n_data + eid] ^= 1

    H_basis_new = np.zeros((n_vertices, n_qubits), dtype=np.uint8)
    for v in range(n_vertices):
        H_basis_new[v, vertex_to_qubit[v]] = 1
        for eid in g.incident(v):
            H_basis_new[v, n_data + eid] ^= 1

    H_basis_old = H_basis_all[np.any(H_basis_all, axis=1)]
    H_basis_old_padded = np.pad(H_basis_old, ((0, 0), (0, n_edges)), mode="constant").astype(np.uint8)

    H_opposite_basis_old_padded = deform_old_opposite_basis_checks_with_graph_edges(
        g=g,
        H_opposite_basis_all=H_opposite_basis_all,
        logical_qubits=logical,
        logical_qubits_index=logical_qubits_index,
        n_data=n_data,
    )

    H_opposite_basis_def = np.vstack([H_opposite_basis_old_padded, H_opposite_basis_new]) % 2
    H_basis_def = np.vstack([H_basis_old_padded, H_basis_new]) % 2

    res = {
        "n_qubits" : n_qubits,
        "n_original_qubits" : n_data,
        "n_edges" : n_edges,
        "H_basis_def" : H_basis_def,
        "H_opposite_basis_def" : H_opposite_basis_def,
        "H_basis_new" : H_basis_new,
        "H_basis_old" : H_basis_old_padded,
        "H_opposite_basis_new" : H_opposite_basis_new,
        "H_opposite_basis_old_padded" : H_opposite_basis_old_padded,
        "g" : g,
        "logical" : logical,
        "qubit_to_vertex": qubit_to_vertex,
    }

    return res
