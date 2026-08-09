"""Triangular-lattice compilation for logical-measurement gadgets.

This module is the standalone triangular-gadget compiler:

1. ``tri_layout`` planarizes the abstract auxiliary graph, places its vertices
   on a triangular lattice with a Tutte-style drawing, and routes abstract
   graph edges along lattice edges.
2. ``deform_logical_to_tri_lattice`` takes that placed layout and compiles it
   into the fixed Stim gadget indexing used by the sweep scripts.

"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from deformation.graph_helper_functions import (
    find_short_cycle_basis,
    normalize_edge,
    split_heavy_cycles,
)


# ---------------------------------------------------------------------------
# Stage 1: planarize, place, and route the abstract auxiliary graph.
# ---------------------------------------------------------------------------

def layout_ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def layout_segments_cross(p1, p2, p3, p4):
    if any(np.allclose(a, b) for a, b in [(p1, p3), (p1, p4), (p2, p3), (p2, p4)]):
        return False
    return layout_ccw(p1, p3, p4) != layout_ccw(p2, p3, p4) and layout_ccw(p1, p2, p3) != layout_ccw(p1, p2, p4)


def layout_crossing_edges(G, pos):
    edges = [normalize_edge(*e) for e in G.edges()]
    crossings = []
    for i, e1 in enumerate(edges):
        for e2 in edges[i + 1:]:
            if set(e1) & set(e2):
                continue
            if layout_segments_cross(pos[e1[0]], pos[e1[1]], pos[e2[0]], pos[e2[1]]):
                crossings.append((e1, e2))
    return crossings


def layout_gf2_row_combination_for_target(rows, target):
    """Return coeffs such that coeffs @ rows == target over GF(2)."""
    rows = np.asarray(rows, dtype=np.uint8) % 2
    target = np.asarray(target, dtype=np.uint8) % 2
    m, n = rows.shape
    if target.shape[0] != n:
        raise ValueError(f"target length {target.shape[0]} does not match row width {n}")

    # Solve rows.T @ coeffs = target.
    A = np.concatenate([rows.T.copy(), target.reshape(n, 1)], axis=1)
    rank = 0
    pivots = []
    for col in range(m):
        pivot = None
        for r in range(rank, n):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(n):
            if r != rank and A[r, col]:
                A[r] ^= A[rank]
        pivots.append(col)
        rank += 1

    for r in range(rank, n):
        if A[r, -1]:
            raise ValueError("target logical is not in the row span")

    coeffs = np.zeros(m, dtype=np.uint8)
    for r, col in enumerate(pivots):
        coeffs[col] = A[r, -1]
    return coeffs


# -----------------------------------------------------------------------------
# Stage 1: planarize the auxiliary graph 
# -----------------------------------------------------------------------------


def layout_lift_old_graph_matrix(deformation_result, g_planar, M_old):
    """Rewrite old abstract-edge columns after planarization splits edges.

    ``M_old`` may contain data columns followed by columns for the old abstract
    graph edges.  If an old edge was split into a path in ``g_planar``, this
    replaces the old edge column by the XOR of the new path-edge columns.
    """
    n_data = int(deformation_result["n_original_qubits"])
    n_edges = g_planar.ecount()
    n_total = n_data + n_edges
    old_edges = [normalize_edge(*e) for e in deformation_result["g"].get_edgelist()]
    new_edge_to_eid = {normalize_edge(*e): eid for eid, e in enumerate(g_planar.get_edgelist())}
    G_planar = nx.Graph(g_planar.get_edgelist())

    M_old = np.asarray(M_old, dtype=np.uint8) % 2
    out = np.zeros((M_old.shape[0], n_total), dtype=np.uint8)
    data_width = min(n_data, M_old.shape[1])
    out[:, :data_width] = M_old[:, :data_width]

    for old_eid, edge in enumerate(old_edges):
        old_col = n_data + old_eid
        if old_col >= M_old.shape[1]:
            continue
        rows = np.where(M_old[:, old_col] == 1)[0]
        if len(rows) == 0:
            continue
        path = nx.shortest_path(G_planar, edge[0], edge[1])
        for a, b in zip(path, path[1:]):
            new_col = n_data + new_edge_to_eid[normalize_edge(a, b)]
            out[rows, new_col] ^= 1
    return out


def layout_rebuild_deformation_checks(deformation_result, g_planar):
    """Build abstract stabilizer matrices after planarization.

    New basis checks are vertex checks.  New opposite-basis checks are cycle
    checks.  Old checks are lifted through any edge splits introduced during
    planarization.
    """
    n_data = int(deformation_result["n_original_qubits"])
    vertex_to_qubit = {int(v): int(q) for q, v in deformation_result["qubit_to_vertex"].items()}

    cycles_edges = find_short_cycle_basis(g_planar.get_edgelist(), return_edges=True)
    H_basis_old_raw = np.asarray(deformation_result["H_basis_old"], dtype=np.uint8) % 2
    basis_weights = np.sum(H_basis_old_raw, axis=1) if H_basis_old_raw.size else np.array([4])
    max_cycle_weight = max(int(np.max(basis_weights)) if len(basis_weights) else 4, 3)
    cycles_edges = split_heavy_cycles(cycles_edges, max_cycle_weight=max_cycle_weight, g=g_planar)

    # split_heavy_cycles may add chord edges, so all sizes are read afterward.
    n_edges = g_planar.ecount()
    n_total = n_data + n_edges
    final_edges = [normalize_edge(*e) for e in g_planar.get_edgelist()]
    edge_to_eid = {e: eid for eid, e in enumerate(final_edges)}

    H_basis_new = np.zeros((g_planar.vcount(), n_total), dtype=np.uint8)
    for v in range(g_planar.vcount()):
        if v in vertex_to_qubit:
            H_basis_new[v, vertex_to_qubit[v]] = 1
        for eid in g_planar.incident(v):
            H_basis_new[v, n_data + eid] ^= 1

    H_opposite_basis_new = np.zeros((len(cycles_edges), n_total), dtype=np.uint8)
    for row, cycle in enumerate(cycles_edges):
        for e in cycle:
            H_opposite_basis_new[row, n_data + edge_to_eid[normalize_edge(*e)]] ^= 1

    H_basis_old = layout_lift_old_graph_matrix(deformation_result, g_planar, deformation_result["H_basis_old"])
    H_opposite_basis_old_padded = layout_lift_old_graph_matrix(
        deformation_result,
        g_planar,
        deformation_result["H_opposite_basis_old_padded"],
    )
    return H_basis_new, H_opposite_basis_new, H_basis_old, H_opposite_basis_old_padded


def layout_planarize_deformation(deformation_result, layout_seed=7):
    """Planarize the abstract auxiliary graph and rebuild abstract checks.

    This is a simple geometric planarization pass:
    1. draw the graph using a planar layout if possible, otherwise spring layout;
    2. find crossing edge pairs in that drawing;
    3. replace each crossing by a new vertex and split the two crossed edges;
    4. rebuild vertex, cycle, and old-check matrices for the split graph.

    The largest cycle of the original graph is saved as ``tutte_fixed_cycle``
    and later used as the Tutte outer boundary.
    """
    g_old = deformation_result["g"]
    G = nx.Graph(g_old.get_edgelist())
    original_cycles = nx.cycle_basis(G)
    tutte_fixed_cycle = max(original_cycles, key=len) if original_cycles else None

    if G.number_of_nodes() == 0:
        pos = {}
    elif nx.is_planar(G):
        pos = nx.planar_layout(G)
    else:
        pos = nx.spring_layout(G, seed=layout_seed)

    g_planar = g_old.copy()
    decomposed_edges = {}
    new_qubit_indices = []
    next_new_vertex = g_planar.vcount()

    while True:
        G_current = nx.Graph(g_planar.get_edgelist())
        missing = [v for v in G_current.nodes if v not in pos]
        if missing:
            fallback = nx.spring_layout(G_current, seed=layout_seed)
            for v in missing:
                pos[v] = np.asarray(fallback.get(v, (0.0, 0.0)))

        crossings = layout_crossing_edges(G_current, pos)
        if not crossings:
            break

        crossing = next_new_vertex
        next_new_vertex += 1
        g_planar.add_vertex()
        pos[crossing] = (
            np.asarray(pos[crossings[0][0][0]])
            + np.asarray(pos[crossings[0][0][1]])
            + np.asarray(pos[crossings[0][1][0]])
            + np.asarray(pos[crossings[0][1][1]])
        ) / 4

        for edge in crossings[0]:
            edge = normalize_edge(*edge)
            eid = g_planar.get_eid(*edge, directed=False, error=False)
            if eid == -1:
                continue
            g_planar.delete_edges([eid])
            pieces = [normalize_edge(edge[0], crossing), normalize_edge(crossing, edge[1])]
            g_planar.add_edges(pieces)
            decomposed_edges[edge] = tuple(pieces)
            new_qubit_indices.append(deformation_result["n_original_qubits"] + g_planar.ecount() - 1)

    H_basis_new, H_opposite_basis_new, H_basis_old, H_opposite_basis_old_padded = layout_rebuild_deformation_checks(
        deformation_result,
        g_planar,
    )
    H_basis_def = np.vstack([H_basis_old, H_basis_new]) % 2
    H_opposite_basis_def = np.vstack([H_opposite_basis_old_padded, H_opposite_basis_new]) % 2

    return {
        **deformation_result,
        "n_qubits": int(deformation_result["n_original_qubits"] + g_planar.ecount()),
        "n_edges": int(g_planar.ecount()),
        "H_basis_def": H_basis_def,
        "H_opposite_basis_def": H_opposite_basis_def,
        "H_basis_new": H_basis_new,
        "H_opposite_basis_new": H_opposite_basis_new,
        "H_basis_old": H_basis_old,
        "H_opposite_basis_old_padded": H_opposite_basis_old_padded,
        "new_qubit_indices": new_qubit_indices,
        "decomposed_edges": decomposed_edges,
        "original_g": g_old,
        "tutte_fixed_cycle": tutte_fixed_cycle,
        "g": g_planar,
    }


# -----------------------------------------------------------------------------
# Stage 2: Tutte placement and routing on a triangular lattice
# -----------------------------------------------------------------------------


def layout_as_networkx_graph(g):
    if isinstance(g, nx.Graph):
        return nx.Graph(g)
    if hasattr(g, "get_edgelist"):
        G = nx.Graph()
        G.add_nodes_from(range(g.vcount()))
        G.add_edges_from([tuple(map(int, e)) for e in g.get_edgelist()])
        return G
    return nx.Graph(g)


def layout_triangular_lattice(rows, cols, spacing=1.0):
    L = nx.Graph()
    pos = {}
    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            pos[node] = np.array([spacing * (c + 0.5 * r), spacing * math.sqrt(3) * r / 2])
            L.add_node(node)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in [(0, 1), (1, 0), (1, -1)]:
                nb = (r + dr, c + dc)
                if nb in pos:
                    L.add_edge((r, c), nb)
    return L, pos


def layout_largest_cycle_nodes(G):
    cycles = nx.cycle_basis(G)
    return max(cycles, key=len) if cycles else None


def layout_tutte_positions(G, fixed_cycle=None):
    """Return continuous Tutte-style positions for graph vertices.

    The outer cycle is fixed on a circle.  Every interior vertex is placed at
    the average of its neighbors by solving the barycentric linear system.
    If no usable cycle exists, this falls back to a spring layout.
    """
    H = nx.convert_node_labels_to_integers(G, label_attribute="original_label")
    original = nx.get_node_attributes(H, "original_label")
    inverse = {v: k for k, v in original.items()}

    if fixed_cycle is None:
        outer = layout_largest_cycle_nodes(G)
    else:
        outer = [v for v in fixed_cycle if v in inverse]

    if outer is None or len(outer) < 3:
        pos = nx.spring_layout(H, seed=7)
        return {original[n]: np.asarray(p) for n, p in pos.items()}, []

    outer_h = [inverse[v] for v in outer]
    fixed = {}
    for i, v in enumerate(outer_h):
        theta = 2 * math.pi * i / len(outer_h)
        fixed[v] = np.array([math.cos(theta), math.sin(theta)])

    interior = [v for v in H.nodes if v not in fixed]
    pos = dict(fixed)
    if interior:
        idx = {v: i for i, v in enumerate(interior)}
        A = np.zeros((len(interior), len(interior)))
        bx = np.zeros(len(interior))
        by = np.zeros(len(interior))
        for v in interior:
            row = idx[v]
            nbrs = list(H.neighbors(v))
            A[row, row] = 1.0
            w = 1.0 / len(nbrs)
            for u in nbrs:
                if u in fixed:
                    bx[row] += w * fixed[u][0]
                    by[row] += w * fixed[u][1]
                else:
                    A[row, idx[u]] -= w
        xs = np.linalg.solve(A, bx)
        ys = np.linalg.solve(A, by)
        for v in interior:
            pos[v] = np.array([xs[idx[v]], ys[idx[v]]])
    return {original[n]: pos[n] for n in H.nodes}, [original[n] for n in outer_h]


def layout_scale_to_lattice(tutte_pos, lattice_pos, margin=1):
    """Affine-scale continuous Tutte positions into the lattice bounding box."""

    P = np.array(list(tutte_pos.values()))
    Q = np.array(list(lattice_pos.values()))

    # bounding box size of the measurement graph with Tutte positions
    p_min, p_max = P.min(axis=0), P.max(axis=0)
    
    # bounding box size of the measurement graph with Tutte positions
    q_min, q_max = Q.min(axis=0) + margin, Q.max(axis=0) - margin
    denom = np.maximum(p_max - p_min, 1e-9)
    return {v: q_min + (p - p_min) / denom * (q_max - q_min) for v, p in tutte_pos.items()}


def layout_nearest_free_sites(target_pos, lattice_pos, rng=None, jitter=0.0):
    """Snap each graph vertex to a distinct nearest lattice site."""
    rng = np.random.default_rng(0) if rng is None else rng
    nodes = sorted(target_pos, key=lambda v: (target_pos[v][1], target_pos[v][0], v))
    if jitter:
        nodes = sorted(nodes, key=lambda v: (target_pos[v][1] + jitter * rng.normal(), target_pos[v][0] + jitter * rng.normal()))
    assignment = {}
    used = set()
    for v in nodes:
        target = target_pos[v]
        site = min(
            (s for s in lattice_pos if s not in used),
            key=lambda s: float(np.linalg.norm(lattice_pos[s] - target)) + jitter * float(rng.random()),
        )
        assignment[v] = site
        used.add(site)
    return assignment


def layout_placement_energy(G, assignment, lattice_graph, lattice_pos, target_pos, anchor_weight=0.03, bbox_weight=0.35):
    """Score a snapped placement: short routes, compact box, small drift."""
    lengths = [nx.shortest_path_length(lattice_graph, assignment[u], assignment[v]) for u, v in G.edges()]
    edge_cost = sum(lengths)
    max_cost = max(lengths) if lengths else 0
    anchor = sum(float(np.linalg.norm(lattice_pos[s] - target_pos[v]) ** 2) for v, s in assignment.items())
    pts = np.array([lattice_pos[s] for s in assignment.values()])
    span = pts.max(axis=0) - pts.min(axis=0) if len(pts) else np.zeros(2)
    return edge_cost + 1.5 * max_cost + anchor_weight * anchor + bbox_weight * float(span[0] * span[1])


def layout_improve_vertex_sites(G, assignment, lattice_graph, lattice_pos, target_pos, rng=None, rounds=5):
    """Greedily move vertices to nearby free sites if the placement score drops."""
    rng = np.random.default_rng(0) if rng is None else rng
    assignment = dict(assignment)
    best = layout_placement_energy(G, assignment, lattice_graph, lattice_pos, target_pos)
    all_sites = list(lattice_pos)
    nodes = list(G.nodes())
    for _ in range(rounds):
        improved = False
        rng.shuffle(nodes)
        used = set(assignment.values())
        for u in nodes:
            old = assignment[u]
            candidates = [s for s in all_sites if s not in used or s == old]
            candidates.sort(key=lambda s: (np.linalg.norm(lattice_pos[s] - lattice_pos[old]), rng.random()))
            for site in candidates[:16]:
                if site == old:
                    continue
                assignment[u] = site
                score = layout_placement_energy(G, assignment, lattice_graph, lattice_pos, target_pos)
                if score + 1e-9 < best:
                    used.remove(old)
                    used.add(site)
                    old = site
                    best = score
                    improved = True
                else:
                    assignment[u] = old
        if not improved:
            break
    return assignment, best


def layout_path_edges(path):
    return [tuple(sorted((a, b))) for a, b in zip(path, path[1:])]


def layout_route_abstract_edges(G, lattice_graph, vertex_sites, rng=None, route_randomness=0.03):
    """Route each abstract graph edge as a short lattice path.

    Edges are processed from shortest to longest.  Previously used lattice
    edges get an added weight, so routes prefer short paths but avoid severe
    congestion when an alternative is available.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    used = defaultdict(int)
    routes = {}
    edges = list(G.edges())
    rng.shuffle(edges)
    edges.sort(key=lambda e: nx.shortest_path_length(lattice_graph, vertex_sites[e[0]], vertex_sites[e[1]]))
    bias = {tuple(sorted(e)): route_randomness * rng.random() for e in lattice_graph.edges()}
    for u, v in edges:
        src, dst = vertex_sites[u], vertex_sites[v]
        def weight(a, b, data):
            e = tuple(sorted((a, b)))
            return 1.0 + bias[e] + 30.0 * used[e]
        path = nx.shortest_path(lattice_graph, src, dst, weight=weight)
        routes[normalize_edge(u, v)] = path
        for e in layout_path_edges(path):
            used[e] += 1
    return routes


# -----------------------------------------------------------------------------
# Stage 3: define stabilizers on physical lattice-edge qubits
# -----------------------------------------------------------------------------


def layout_graph_edge_order(deformation_result):
    return [normalize_edge(*e) for e in deformation_result["g"].get_edgelist()]


def layout_all_pairings(items):
    items = tuple(items)
    if not items:
        yield []
        return
    first = items[0]
    for i in range(1, len(items)):
        second = items[i]
        rest = items[1:i] + items[i + 1:]
        for tail in layout_all_pairings(rest):
            yield [(first, second)] + tail


def layout_bfs_shortest_path(lattice_graph, a, b, participation=None, max_participation=None):
    """Shortest unweighted physical-lattice path, with a congestion fallback."""
    if participation is not None and max_participation is not None:
        uncongested = nx.Graph()
        uncongested.add_nodes_from(lattice_graph.nodes())
        for u, v in lattice_graph.edges():
            e = tuple(sorted((u, v)))
            if participation[e] < max_participation:
                uncongested.add_edge(u, v)
        if a in uncongested and b in uncongested:
            try:
                return nx.shortest_path(uncongested, a, b)
            except nx.NetworkXNoPath:
                pass
    return nx.shortest_path(lattice_graph, a, b)


def layout_path_congestion_score(path, participation):
    edges = layout_path_edges(path)
    if not edges:
        return (0, 0)
    counts = [participation[e] for e in edges]
    return (max(counts), sum(counts))


def layout_choose_old_bb_support(lattice_graph, terminals, participation, max_participation=7):
    """Pair logical-overlap vertices and route each pair by BFS on the lattice.

    The even support is matched into pairs. Each pair contributes one connected
    shortest physical path. The union of different pair paths is allowed to be
    disconnected; forcing all pairs into one component is not part of the adapter
    construction.
    """
    terminals = list(terminals)
    if len(terminals) % 2:
        raise ValueError("Old BB check has odd number of terminals")
    if not terminals:
        return set(), []

    best = None
    for pairs in layout_all_pairings(terminals):
        paths = []
        for a, b in pairs:
            paths.append(layout_bfs_shortest_path(
                lattice_graph,
                a,
                b,
                participation=participation,
                max_participation=max_participation,
            ))
        support = layout_xor_path_edges(paths)
        path_lengths = [len(path) - 1 for path in paths]
        congestion = [layout_path_congestion_score(path, participation) for path in paths]
        over_cap_edges = sum(participation[e] >= max_participation for e in support)
        candidate = (
            sum(path_lengths),
            max(path_lengths) if path_lengths else 0,
            over_cap_edges,
            max((c[0] for c in congestion), default=0),
            sum(c[1] for c in congestion),
            pairs,
            support,
            paths,
        )
        if best is None or candidate[:5] < best[:5]:
            best = candidate
    return best[6], best[7]


def layout_xor_path_edges(paths):
    support = set()
    for path in paths:
        for e in layout_path_edges(path):
            if e in support:
                support.remove(e)
            else:
                support.add(e)
    return support


def layout_ordered_cycle(edge_support):
    edge_support = {tuple(sorted(e)) for e in edge_support}
    adj = defaultdict(list)
    for a, b in edge_support:
        adj[a].append(b)
        adj[b].append(a)
    if not edge_support or any(len(n) != 2 for n in adj.values()):
        return None
    start = min(adj)
    vertices = [start]
    edges = []
    prev = None
    cur = start
    for _ in range(len(edge_support)):
        n0, n1 = adj[cur]
        nxt = n0 if n0 != prev else n1
        edges.append(tuple(sorted((cur, nxt))))
        if nxt == start:
            break
        vertices.append(nxt)
        prev, cur = cur, nxt
    return (vertices, edges) if set(edges) == edge_support else None


def layout_decompose_even_support(edge_support):
    G = nx.Graph()
    G.add_edges_from(edge_support)
    if any(d % 2 for _, d in G.degree()):
        return [set(edge_support)]
    pieces = []
    H = G.copy()
    while H.number_of_edges():
        cyc = nx.find_cycle(H)
        edges = {tuple(sorted((a, b))) for a, b, *_ in cyc}
        pieces.append(edges)
        H.remove_edges_from(edges)
        H.remove_nodes_from([v for v, d in list(H.degree()) if d == 0])
    return pieces


def layout_split_large_cycle_supports(cycle_supports, lattice_graph, max_weight=6):
    rows = []
    split_count = 0
    for support, meta in cycle_supports:
        comps = layout_decompose_even_support(support)
        if len(comps) > 1:
            split_count += len(comps) - 1
        for i, comp in enumerate(comps):
            m = dict(meta)
            if len(comps) > 1:
                m["decomposed_component"] = i
            rows.append((comp, m))
    out = []
    while rows:
        support, meta = rows.pop(0)
        if len(support) <= max_weight:
            out.append((support, meta))
            continue
        cyc = layout_ordered_cycle(support)
        if cyc is None:
            out.append((support, meta))
            continue
        vertices, edges = cyc
        k = len(edges)
        best = None
        for i in range(k):
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue
                arc1 = set(edges[i:j])
                arc2 = set(edges[j:] + edges[:i])
                H = lattice_graph.copy()
                H.remove_edges_from(support)
                try:
                    chord_path = nx.shortest_path(H, vertices[i], vertices[j])
                except nx.NetworkXNoPath:
                    chord_path = nx.shortest_path(lattice_graph, vertices[i], vertices[j])
                chord = set(layout_path_edges(chord_path))
                c1, c2 = arc1 ^ chord, arc2 ^ chord
                cand = (max(len(c1), len(c2)), len(c1) + len(c2), c1, c2)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None or best[0] >= len(support):
            out.append((support, meta))
            continue
        split_count += 1
        for piece, child in enumerate([best[2], best[3]]):
            m = dict(meta)
            m["split_from_local_row"] = meta.get("local_row")
            m["split_piece"] = piece
            rows.append((child, m))
    return out, split_count


def layout_build_lattice_stabilizers(
    deformation_result,
    lattice_graph,
    vertex_sites,
    abstract_routes,
    max_old_participation=7,
    max_cycle_weight=6,
):
    """Materialize lattice-edge qubits and lift every stabilizer matrix.

    Each used triangular-lattice edge becomes one new
    auxiliary qubit.  Abstract auxiliary edges are replaced by their routed
    lattice-edge paths.

    Returned matrices:
      - ``H_basis_old``: unchanged old basis checks, expanded to lattice qubits;
      - ``H_basis_new``: vertex checks on lattice-edge qubits and port data;
      - ``H_opposite_basis_old_padded``: old opposite-basis checks with routed
        auxiliary support;
      - ``H_opposite_basis_new``: cycle checks on lattice-edge qubits.
    """
    n_data = int(deformation_result["n_original_qubits"])
    graph_edges = layout_graph_edge_order(deformation_result)
    graph_edge_to_col = {e: i for i, e in enumerate(graph_edges)}
    logical = np.asarray(deformation_result["logical"], dtype=np.uint8)
    qubit_to_vertex = {int(q): int(v) for q, v in deformation_result["qubit_to_vertex"].items()}

    physical_edges = set()
    graph_edge_to_lattice_edges = {}
    for e, path in abstract_routes.items():
        es = set(layout_path_edges(path))
        graph_edge_to_lattice_edges[e] = es
        physical_edges |= es

    participation = defaultdict(int)
    for e in physical_edges:
        participation[e] += 2  # two endpoint vertex checks

    M_old_opp = np.asarray(deformation_result["H_opposite_basis_old_padded"], dtype=np.uint8) % 2
    old_supports = []
    old_paths = []
    for row in M_old_opp:
        overlap = [int(q) for q in np.where((row[:n_data] == 1) & (logical == 1))[0]]
        terminals = [vertex_sites[qubit_to_vertex[q]] for q in overlap]
        support, paths = layout_choose_old_bb_support(
            lattice_graph,
            terminals,
            participation,
            max_participation=max_old_participation,
        )
        old_supports.append(support)
        old_paths.append(paths)
        physical_edges |= support
        for e in support:
            participation[e] += 1

    M_cycle = np.asarray(deformation_result["H_opposite_basis_new"], dtype=np.uint8) % 2
    raw_cycles = []
    for row_id, row in enumerate(M_cycle):
        support = set()
        for e, col in graph_edge_to_col.items():
            old_col = n_data + col
            if old_col < row.shape[0] and row[old_col]:
                support ^= graph_edge_to_lattice_edges.get(e, set())
        raw_cycles.append((support, {"type": "cycle check", "local_row": row_id}))
    cycle_supports, split_count = layout_split_large_cycle_supports(raw_cycles, lattice_graph, max_weight=max_cycle_weight)
    for support, _ in cycle_supports:
        physical_edges |= support

    physical_edges = sorted(physical_edges)
    lattice_edge_to_qubit = {e: n_data + i for i, e in enumerate(physical_edges)}
    n_total = n_data + len(physical_edges)

    def expand_abstract(M):
        M = np.asarray(M, dtype=np.uint8) % 2
        out = np.zeros((M.shape[0], n_total), dtype=np.uint8)
        out[:, :n_data] = M[:, :n_data]
        for e, col in graph_edge_to_col.items():
            old_col = n_data + col
            if old_col >= M.shape[1]:
                continue
            rows = np.where(M[:, old_col] == 1)[0]
            for le in graph_edge_to_lattice_edges.get(e, set()):
                out[rows, lattice_edge_to_qubit[le]] ^= 1
        return out

    H_basis_old = expand_abstract(deformation_result["H_basis_old"])
    H_old_opp = np.zeros((M_old_opp.shape[0], n_total), dtype=np.uint8)
    H_old_opp[:, :n_data] = M_old_opp[:, :n_data]
    for r, support in enumerate(old_supports):
        for e in support:
            H_old_opp[r, lattice_edge_to_qubit[e]] ^= 1

    H_cycle = np.zeros((len(cycle_supports), n_total), dtype=np.uint8)
    cycle_sources = []
    for r, (support, meta) in enumerate(cycle_supports):
        for e in support:
            H_cycle[r, lattice_edge_to_qubit[e]] ^= 1
        cycle_sources.append(meta)

    site_to_row = {}
    vertex_rows = []
    vertex_to_qubit = {int(v): int(q) for q, v in deformation_result["qubit_to_vertex"].items()}
    def row_for_site(site):
        if site not in site_to_row:
            site_to_row[site] = len(vertex_rows)
            vertex_rows.append(np.zeros(n_total, dtype=np.uint8))
        return vertex_rows[site_to_row[site]]
    for v, site in vertex_sites.items():
        row = row_for_site(site)
        if int(v) in vertex_to_qubit:
            row[vertex_to_qubit[int(v)]] = 1
    for e in physical_edges:
        q = lattice_edge_to_qubit[e]
        a, b = e
        row_for_site(a)[q] ^= 1
        row_for_site(b)[q] ^= 1
    H_vertex = np.vstack(vertex_rows) if vertex_rows else np.zeros((0, n_total), dtype=np.uint8)

    original_vertex_to_vertex_check_row = {
        int(v): int(site_to_row[site]) for v, site in vertex_sites.items() if site in site_to_row
    }
    port_qubit_to_vertex_check_row = {
        int(q): original_vertex_to_vertex_check_row[int(v)]
        for q, v in deformation_result["qubit_to_vertex"].items()
        if int(v) in original_vertex_to_vertex_check_row
    }
    port_vertex_rows = sorted(set(port_qubit_to_vertex_check_row.values()))
    basis_new_row_offset = H_basis_old.shape[0]
    port_vertex_rows_global = [basis_new_row_offset + r for r in port_vertex_rows]
    port_qubit_to_basis_row = {
        q: basis_new_row_offset + r for q, r in port_qubit_to_vertex_check_row.items()
    }

    H_basis_def = np.vstack([H_basis_old, H_vertex]) % 2
    H_opp_def = np.vstack([H_old_opp, H_cycle]) % 2
    materialized_graph = nx.Graph()
    materialized_graph.add_edges_from(physical_edges)
    original_vertex_to_site = dict(vertex_sites)
    site_to_original_vertex = {site: v for v, site in vertex_sites.items()}
    opposite_sources = []
    for i in range(H_old_opp.shape[0]):
        support_graph = nx.Graph(list(old_supports[i]))
        component_count = nx.number_connected_components(support_graph) if support_graph.number_of_edges() else 0
        opposite_sources.append({
            "type": "updated BB stabilizer with auxiliary support",
            "local_row": i,
            "paths": old_paths[i],
            "path_count": len(old_paths[i]),
            "path_lengths": [len(path) - 1 for path in old_paths[i]],
            "union_component_count": component_count,
        })
    opposite_sources += cycle_sources
    return {
        "n_data_qubits": n_data,
        "n_lattice_edge_qubits": len(physical_edges),
        "n_total_qubits": n_total,
        "lattice_edge_to_qubit": lattice_edge_to_qubit,
        "graph_edge_to_lattice_qubits": {e: [lattice_edge_to_qubit[x] for x in sorted(s)] for e, s in graph_edge_to_lattice_edges.items()},
        "materialized_graph": materialized_graph,
        "materialized_edges": physical_edges,
        "original_vertex_to_site": original_vertex_to_site,
        "site_to_original_vertex": site_to_original_vertex,
        "site_to_vertex_check_row": site_to_row,
        "original_vertex_to_vertex_check_row": original_vertex_to_vertex_check_row,
        "port_qubit_to_vertex_check_row": port_qubit_to_vertex_check_row,
        "port_vertex_rows": port_vertex_rows,
        "port_vertex_rows_global": port_vertex_rows_global,
        "port_qubit_to_basis_row": port_qubit_to_basis_row,
        "basis_new_row_offset": basis_new_row_offset,
        "H_basis_old": H_basis_old,
        "H_basis_new": H_vertex,
        "H_basis_def": H_basis_def,
        "H_opposite_basis_old_padded": H_old_opp,
        "H_opposite_basis_new": H_cycle,
        "H_opposite_basis_def": H_opp_def,
        "opposite_row_sources": opposite_sources,
        "cycle_split_count": split_count,
        "max_lattice_cycle_weight": max_cycle_weight,
    }












# -----------------------------------------------------------------------------
# Stage 2: embed and route the lattice layout
# -----------------------------------------------------------------------------


def layout_candidate_grid_sizes(n, rows, cols, grid_extra=4):
    if rows is not None and cols is not None:
        return [(rows, cols)]
    base = max(4, int(math.ceil(math.sqrt(n))) + 2)
    out = []
    for extra in range(grid_extra + 1):
        for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            pair = (base + extra + dr, base + extra + dc)
            if pair[0] * pair[1] >= n and pair not in out:
                out.append(pair)
    return out


def layout_route_stats(routes):
    lengths = [len(p) - 1 for p in routes.values()]
    return {"total_route_length": int(sum(lengths)), "max_route_length": int(max(lengths)) if lengths else 0, "mean_route_length": float(np.mean(lengths)) if lengths else 0.0}


def layout_embed_tutte_style(
    g,
    deformation_result=None,
    rows=None,
    cols=None,
    spacing=1.0,
    figsize=(8, 8),
    random_restarts=2,
    rng_seed=0,
    placement_jitter=0.08,
    route_randomness=0.03,
    max_old_participation=7,
    max_lattice_cycle_weight=6,
):
    """Place an auxiliary graph on a triangular lattice.

    The placement stage uses a continuous Tutte-style drawing, scales it onto
    candidate triangular lattices, snaps vertices to distinct sites, improves
    the placement by local swaps, and routes each abstract edge on the lattice.
    """
    del figsize  # Kept only so older notebook calls do not break.

    G = layout_as_networkx_graph(g)
    n = max(G.number_of_nodes(), 1)
    fixed_cycle = None
    if deformation_result is not None:
        fixed_cycle = deformation_result.get("tutte_fixed_cycle")
        if fixed_cycle is None and "original_g" in deformation_result:
            fixed_cycle = layout_largest_cycle_nodes(layout_as_networkx_graph(deformation_result["original_g"]))

    tutte_pos, tutte_fixed_cycle = layout_tutte_positions(G, fixed_cycle=fixed_cycle)
    root_rng = np.random.default_rng(rng_seed)
    best = None

    for rr, cc in layout_candidate_grid_sizes(n, rows, cols):
        lattice_graph, lattice_pos = layout_triangular_lattice(rr, cc, spacing=spacing)
        margin = 1 if min(rr, cc) <= 6 else 2
        target_pos = layout_scale_to_lattice(tutte_pos, lattice_pos, margin=margin)
        for restart in range(max(1, random_restarts)):
            rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
            jitter = placement_jitter if restart else 0.0
            init = layout_nearest_free_sites(target_pos, lattice_pos, rng=rng, jitter=jitter)
            vertex_sites, placement_score = layout_improve_vertex_sites(
                G,
                init,
                lattice_graph,
                lattice_pos,
                target_pos,
                rng=rng,
            )
            routes = layout_route_abstract_edges(
                G,
                lattice_graph,
                vertex_sites,
                rng=rng,
                route_randomness=route_randomness if restart else 0.0,
            )
            stats = layout_route_stats(routes)
            score = 10 * rr * cc + stats["total_route_length"] + 2 * stats["max_route_length"] + placement_score
            candidate = (
                score,
                rr * cc,
                stats["total_route_length"],
                stats["max_route_length"],
                rr,
                cc,
                restart,
                lattice_graph,
                lattice_pos,
                target_pos,
                vertex_sites,
                routes,
                placement_score,
                stats,
            )
            if best is None or candidate[:6] < best[:6]:
                best = candidate

    (
        _,
        _,
        _,
        _,
        rows,
        cols,
        best_restart,
        lattice_graph,
        lattice_pos,
        target_pos,
        vertex_sites,
        routes,
        placement_score,
        stats,
    ) = best

    state = {
        "G": G,
        "lattice_graph": lattice_graph,
        "lattice_pos": lattice_pos,
        "tutte_pos": tutte_pos,
        "tutte_fixed_cycle": tutte_fixed_cycle,
        "target_pos": target_pos,
        "vertex_sites": vertex_sites,
        "routes": routes,
        "rows": rows,
        "cols": cols,
        "best_restart": best_restart,
        "rng_seed": rng_seed,
        "placement_energy": placement_score,
        **stats,
    }

    if deformation_result is not None:
        state["lattice_stabilizers"] = layout_build_lattice_stabilizers(
            deformation_result,
            lattice_graph,
            vertex_sites,
            routes,
            max_old_participation=max_old_participation,
            max_cycle_weight=max_lattice_cycle_weight,
        )
        state["materialized_graph"] = state["lattice_stabilizers"]["materialized_graph"]
        state["materialized_edges"] = state["lattice_stabilizers"]["materialized_edges"]
        state["original_vertex_to_site"] = state["lattice_stabilizers"]["original_vertex_to_site"]
        state["site_to_original_vertex"] = state["lattice_stabilizers"]["site_to_original_vertex"]

    return state

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------


def tri_layout(
    deformation_result,
    rows=None,
    cols=None,
    spacing=1.0,
    figsize=(8, 8),
    random_restarts=2,
    rng_seed=0,
    placement_jitter=0.08,
    route_randomness=0.03,
    max_old_participation=7,
    max_lattice_cycle_weight=6,
):
    """Planarize, place, route, and lift one logical deformation."""
    planarized = layout_planarize_deformation(deformation_result)
    state = layout_embed_tutte_style(
        planarized["g"],
        deformation_result=planarized,
        rows=rows,
        cols=cols,
        spacing=spacing,
        figsize=figsize,
        random_restarts=random_restarts,
        rng_seed=rng_seed,
        placement_jitter=placement_jitter,
        route_randomness=route_randomness,
        max_old_participation=max_old_participation,
        max_lattice_cycle_weight=max_lattice_cycle_weight,
    )
    stabilizers = state["lattice_stabilizers"]

    logical_full = np.zeros(stabilizers["n_total_qubits"], dtype=np.uint8)
    logical = np.asarray(planarized["logical"], dtype=np.uint8)
    logical_full[:len(logical)] = logical

    logical_observable_row_coeffs = layout_gf2_row_combination_for_target(
        stabilizers["H_basis_def"],
        logical_full,
    )
    logical_observable_rows = np.flatnonzero(logical_observable_row_coeffs).astype(int).tolist()
    extra_logical_observable_rows = sorted(
        set(logical_observable_rows) - set(stabilizers["port_vertex_rows_global"])
    )

    return {
        "state": state,
        "deformation_result": planarized,
        "g": planarized["g"],
        "auxiliary_graph": stabilizers["materialized_graph"],
        "lattice_graph": state["lattice_graph"],
        "lattice_pos": state["lattice_pos"],
        "vertex_sites": state["vertex_sites"],
        "routes": state["routes"],
        "stabilizers": stabilizers,
        "H_basis": stabilizers["H_basis_def"],
        "H_opposite_basis": stabilizers["H_opposite_basis_def"],
        "H_basis_def": stabilizers["H_basis_def"],
        "H_opposite_basis_def": stabilizers["H_opposite_basis_def"],
        "H_basis_old": stabilizers["H_basis_old"],
        "H_basis_new": stabilizers["H_basis_new"],
        "H_opposite_basis_old_padded": stabilizers["H_opposite_basis_old_padded"],
        "H_opposite_basis_new": stabilizers["H_opposite_basis_new"],
        "logical_full": logical_full,
        "logical_observable_rows": logical_observable_rows,
        "logical_observable_row_coeffs": logical_observable_row_coeffs,
        "extra_logical_observable_rows": extra_logical_observable_rows,
        "n_data_qubits": stabilizers["n_data_qubits"],
        "n_lattice_edge_qubits": stabilizers["n_lattice_edge_qubits"],
        "n_total_qubits": stabilizers["n_total_qubits"],
        "lattice_edge_to_qubit": stabilizers["lattice_edge_to_qubit"],
        "graph_edge_to_lattice_qubits": stabilizers["graph_edge_to_lattice_qubits"],
        "site_to_vertex_check_row": stabilizers["site_to_vertex_check_row"],
        "original_vertex_to_vertex_check_row": stabilizers["original_vertex_to_vertex_check_row"],
        "port_qubit_to_vertex_check_row": stabilizers["port_qubit_to_vertex_check_row"],
        "port_vertex_rows": stabilizers["port_vertex_rows"],
        "port_vertex_rows_global": stabilizers["port_vertex_rows_global"],
        "port_qubit_to_basis_row": stabilizers["port_qubit_to_basis_row"],
        "basis_new_row_offset": stabilizers["basis_new_row_offset"],
        "tutte_fixed_cycle": state.get("tutte_fixed_cycle"),
    }



# ---------------------------------------------------------------------------
# Stage 2: compile the placed layout into the fixed Stim gadget indexing.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedTriangularTemplate:
    coords: Dict[int, Tuple[float, float]]
    data_qubits: List[int]
    basis_check_qubits: List[int]
    opposite_check_qubits: List[int]
    vertex_check_qubits: List[int]
    edge_qubits: List[int]
    shuttling_edge_qubits: List[int]
    cycle_check_qubits: List[int]
    shuttling_cycle_check_qubits: List[int]
    port_qubits: List[int]
    row_qubits: List[List[int]]
    edge_qubit_to_vertices: Dict[int, Tuple[int, int]]
    vertex_pair_to_edge_qubit: Dict[Tuple[int, int], int]
    vertex_site_to_qubit: Dict[Tuple[int, int], int]
    vertex_qubit_to_site: Dict[int, Tuple[int, int]]
    cycle_check_to_vertices: Dict[int, Tuple[int, int, int]]
    cycle_check_to_edge_qubits: Dict[int, Tuple[int, int, int]]
    code_patch_rect: Tuple[float, float, float, float]
    code_patch_label: str
    n_data: int
    n_basis_checks: int
    n_opposite_checks: int
    n_total_qubits: int
    n_h_qubits: int


def compile_code_patch_coords(n_data: int, n_basis_checks: int, n_opposite_checks: int) -> tuple[Dict[int, Tuple[float, float]], Tuple[float, float, float, float], str]:
    """Place generic code qubits/ancillas inside a compact white-rectangle region."""
    coords: Dict[int, Tuple[float, float]] = {}
    if n_data:
        cols = int(np.ceil(np.sqrt(n_data)))
        for q in range(n_data):
            r, c = divmod(q, cols)
            coords[q] = (float(c), float(r))
    else:
        cols = 1

    width = max(cols, 6)
    height = int(np.ceil(max(n_data, 1) / cols))
    rect = (-1.0, -0.8, float(width + 1), float(height + 1.4))
    label = f"code patch\n{n_data} data qubits\n{n_basis_checks} basis checks\n{n_opposite_checks} opposite checks"
    return coords, rect, label


def compile_sort_qubits_by_coord(qubits: Iterable[int], coords: Dict[int, Tuple[float, float]]) -> List[int]:
    return sorted(qubits, key=lambda q: (coords[q][0], coords[q][1], q))


def compile_alternating_row_lengths(rows: int, cols: int) -> List[int]:
    """Row lengths for an alternating triangular grid, starting with the wide row."""
    return [int(cols) if r % 2 == 0 else int(cols) - 1 for r in range(int(rows))]


def fixed_triangular_template(
    rows: int = 8,
    cols: int = 9,
    spacing: float = 2.0,
    ports: int = 20,
    n_data: int = 72,
    n_basis_checks: int = 36,
    n_opposite_checks: int = 36,
    shuttling_edges: int = 0,
    shuttling_cycle_checks: int | None = None,
) -> FixedTriangularTemplate:
    """Return a dynamic alternating triangular gadget template.
    Indices are assigned as data, gadget edges, old basis-check ancillas, old opposite-check ancillas,
    floor cycle ancillas, gadget vertex ancillas, port qubits, then shuttling
    cycle ancillas.
    """
    if rows < 2 or cols < 2:
        raise ValueError("triangular gadget needs at least 2 rows and 2 columns")
    if shuttling_cycle_checks is None:
        shuttling_cycle_checks = int(shuttling_edges)

    coords, code_patch_rect, code_patch_label = compile_code_patch_coords(n_data, n_basis_checks, n_opposite_checks)
    data_qubits = list(range(n_data))

    row_lengths = compile_alternating_row_lengths(rows, cols)
    sites = [(r, c) for r, count in enumerate(row_lengths) for c in range(count)]
    vertex_edges = []
    for r, count in enumerate(row_lengths):
        for c in range(count - 1):
            vertex_edges.append(((r, c), (r, c + 1)))
    for r in range(rows - 1):
        upper_count = row_lengths[r]
        lower_count = row_lengths[r + 1]
        if upper_count == lower_count + 1:
            for c in range(lower_count):
                vertex_edges.append(((r, c), (r + 1, c)))
                vertex_edges.append(((r, c + 1), (r + 1, c)))
        elif lower_count == upper_count + 1:
            for c in range(upper_count):
                vertex_edges.append(((r, c), (r + 1, c)))
                vertex_edges.append(((r, c), (r + 1, c + 1)))
        else:
            raise ValueError("adjacent triangular rows must differ by exactly one site")

    vertex_edges = sorted({tuple(sorted(e)) for e in vertex_edges})

    triangles = []
    for r in range(rows - 1):
        upper_count = row_lengths[r]
        lower_count = row_lengths[r + 1]
        if upper_count == lower_count + 1:
            for c in range(lower_count):
                triangles.append(((r, c), (r, c + 1), (r + 1, c)))
            for c in range(lower_count - 1):
                triangles.append(((r + 1, c), (r + 1, c + 1), (r, c + 1)))
        else:
            for c in range(upper_count):
                triangles.append(((r + 1, c), (r + 1, c + 1), (r, c)))
            for c in range(upper_count - 1):
                triangles.append(((r, c), (r, c + 1), (r + 1, c + 1)))

    edge_start = n_data
    shuttling_start = edge_start + len(vertex_edges)
    basis_start = shuttling_start + int(shuttling_edges)
    opposite_start = basis_start + n_basis_checks
    cycle_start = opposite_start + n_opposite_checks
    vertex_start = cycle_start + len(triangles)
    port_start = vertex_start + len(sites)
    shuttling_cycle_start = port_start + ports
    n_total = shuttling_cycle_start + int(shuttling_cycle_checks)
    n_h = basis_start

    shuttling_edge_qubits = list(range(shuttling_start, basis_start))
    basis_check_qubits = list(range(basis_start, opposite_start))
    opposite_check_qubits = list(range(opposite_start, cycle_start))
    cycle_check_qubits = list(range(cycle_start, vertex_start))
    vertex_check_qubits = list(range(vertex_start, port_start))
    port_qubits = list(range(port_start, shuttling_cycle_start))
    shuttling_cycle_check_qubits = list(range(shuttling_cycle_start, n_total))

    row_qubits = []
    site_to_vertex = {}
    for r, count in enumerate(row_lengths):
        row = []
        for c in range(count):
            q = vertex_check_qubits[len(site_to_vertex)]
            site_to_vertex[(r, c)] = q
            x0 = 0.0 if r % 2 == 0 else 0.5
            coords[q] = (float(spacing * (x0 + c)), float(spacing * r + 8.0))
            row.append(q)
        row_qubits.append(row)

    vertex_qubit_to_site = {q: site for site, q in site_to_vertex.items()}

    def midpoint(edge):
        a_site, b_site = edge
        a, b = site_to_vertex[a_site], site_to_vertex[b_site]
        xa, ya = coords[a]
        xb, yb = coords[b]
        return ((xa + xb) / 2, (ya + yb) / 2)

    vertex_edges = sorted(vertex_edges, key=lambda e: (midpoint(e)[0], midpoint(e)[1], e))
    edge_qubits = list(range(edge_start, shuttling_start))
    edge_qubit_to_vertices = {}
    vertex_pair_to_edge_qubit = {}
    for q, edge in zip(edge_qubits, vertex_edges):
        a_site, b_site = edge
        a, b = site_to_vertex[a_site], site_to_vertex[b_site]
        pair = tuple(sorted((a, b)))
        coords[q] = midpoint(edge)
        edge_qubit_to_vertices[q] = pair
        vertex_pair_to_edge_qubit[pair] = q

    def centroid(tri):
        qs = [site_to_vertex[s] for s in tri]
        return (sum(coords[q][0] for q in qs) / 3, sum(coords[q][1] for q in qs) / 3)

    triangles = sorted(triangles, key=lambda t: (centroid(t)[0], centroid(t)[1], t))
    cycle_check_to_vertices = {}
    cycle_check_to_edge_qubits = {}
    for q, tri in zip(cycle_check_qubits, triangles):
        verts = tuple(site_to_vertex[s] for s in tri)
        coords[q] = centroid(tri)
        cycle_check_to_vertices[q] = verts
        a, b, c = verts
        cycle_check_to_edge_qubits[q] = tuple(
            vertex_pair_to_edge_qubit[tuple(sorted(e))]
            for e in [(a, b), (b, c), (a, c)]
        )

    # Put old code-check ancillas as two simple rows in the code rectangle.
    basis_y = code_patch_rect[1] + code_patch_rect[3] + 0.45
    opposite_y = basis_y + 0.55
    for i, q in enumerate(basis_check_qubits):
        coords[q] = (float(i % max(1, int(np.ceil(np.sqrt(max(n_basis_checks, 1)))))), float(basis_y + i // max(1, int(np.ceil(np.sqrt(max(n_basis_checks, 1)))))))
    for i, q in enumerate(opposite_check_qubits):
        coords[q] = (float(i % max(1, int(np.ceil(np.sqrt(max(n_opposite_checks, 1)))))), float(opposite_y + i // max(1, int(np.ceil(np.sqrt(max(n_opposite_checks, 1)))))))

    floor_width = spacing * (max(row_lengths) - 1)
    port_spacing = spacing if ports <= 1 else min(spacing, max(0.55, floor_width / max(ports - 1, 1)))
    port_x0 = (floor_width - port_spacing * max(ports - 1, 0)) / 2
    port_y = 8.0 + spacing * rows + 1.0
    for i, q in enumerate(port_qubits):
        coords[q] = (float(port_x0 + port_spacing * i), float(port_y))
    for i, q in enumerate(shuttling_cycle_check_qubits):
        coords[q] = (float(floor_width + 3.25), float(8.0 + 0.48 * i))

    vertex_check_qubits = compile_sort_qubits_by_coord(vertex_check_qubits, coords)
    edge_qubits = compile_sort_qubits_by_coord(edge_qubits, coords)
    cycle_check_qubits = compile_sort_qubits_by_coord(cycle_check_qubits, coords)
    compile_validate_fixed_ordering(coords, vertex_check_qubits, edge_qubits, cycle_check_qubits)
    return FixedTriangularTemplate(
        coords=coords,
        data_qubits=data_qubits,
        basis_check_qubits=basis_check_qubits,
        opposite_check_qubits=opposite_check_qubits,
        vertex_check_qubits=vertex_check_qubits,
        edge_qubits=edge_qubits,
        shuttling_edge_qubits=shuttling_edge_qubits,
        cycle_check_qubits=cycle_check_qubits,
        shuttling_cycle_check_qubits=shuttling_cycle_check_qubits,
        port_qubits=port_qubits,
        row_qubits=row_qubits,
        edge_qubit_to_vertices=edge_qubit_to_vertices,
        vertex_pair_to_edge_qubit=vertex_pair_to_edge_qubit,
        vertex_site_to_qubit=site_to_vertex,
        vertex_qubit_to_site=vertex_qubit_to_site,
        cycle_check_to_vertices=cycle_check_to_vertices,
        cycle_check_to_edge_qubits=cycle_check_to_edge_qubits,
        code_patch_rect=code_patch_rect,
        code_patch_label=code_patch_label,
        n_data=n_data,
        n_basis_checks=n_basis_checks,
        n_opposite_checks=n_opposite_checks,
        n_total_qubits=n_total,
        n_h_qubits=n_h,
    )

def compile_validate_fixed_ordering(coords, vertex_qubits, edge_qubits, cycle_qubits) -> None:
    for name, qubits in [
        ("vertex checks", vertex_qubits),
        ("edge qubits", edge_qubits),
        ("cycle checks", cycle_qubits),
    ]:
        ordered = compile_sort_qubits_by_coord(qubits, coords)
        if list(qubits) != ordered:
            raise AssertionError(f"{name} are not ordered left-to-right, top-to-bottom")


def compile_copy_columns_to_fixed(M, old_to_fixed, width=None):
    M = np.asarray(M, dtype=np.uint8) % 2
    if width is None:
        fixed_cols = []
        for cols in old_to_fixed.values():
            if isinstance(cols, int):
                fixed_cols.append(cols)
            else:
                fixed_cols.extend(cols)
        width = max([M.shape[1], *[int(q) + 1 for q in fixed_cols]] or [M.shape[1]])
    out = np.zeros((M.shape[0], int(width)), dtype=np.uint8)
    for old_col, fixed_cols in old_to_fixed.items():
        if isinstance(fixed_cols, int):
            fixed_cols = [fixed_cols]
        if old_col >= M.shape[1]:
            continue
        for fixed_col in fixed_cols:
            if fixed_col < width:
                out[:, fixed_col] ^= M[:, old_col]
    return out


def compile_cycle_vertices_to_edge_qubits(cycle_vertices, pair_to_qubit):
    edge_qubits = []
    vertices = list(cycle_vertices)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        pair = tuple(sorted((int(a), int(b))))
        q = pair_to_qubit.get(pair)
        if q is None:
            return None
        edge_qubits.append(int(q))
    return edge_qubits


def compile_recompute_cycle_basis_from_fixed_edges(fixed_edge_qubits, template, width=None):
    """Compute simple cycle checks from the placed fixed-gadget edge graph.

    The old routed cycle rows can contain interior edges.  This recomputes the
    gadget cycle checks after placement as a graph cycle basis over the fixed
    triangular-lattice edge graph.  Each row is therefore only the boundary of a
    simple cycle in the placed graph.
    """
    if width is None:
        width = int(template.n_h_qubits)
    G = nx.Graph()
    pair_to_qubit = {}
    for q in sorted(set(int(q) for q in fixed_edge_qubits)):
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = template.edge_qubit_to_vertices[q]
        G.add_edge(int(a), int(b), qubit=int(q))
        pair_to_qubit[tuple(sorted((int(a), int(b))))] = int(q)

    cycles = nx.minimum_cycle_basis(G)
    rows = []
    metadata = []
    seen = set()
    for cycle in cycles:
        edge_qubits = compile_cycle_vertices_to_edge_qubits(cycle, pair_to_qubit)
        if not edge_qubits:
            continue
        key = tuple(sorted(edge_qubits))
        if key in seen:
            continue
        seen.add(key)
        row = np.zeros(width, dtype=np.uint8)
        row[list(key)] = 1
        rows.append(row)
        xs = [template.coords[int(v)][0] for v in cycle]
        ys = [template.coords[int(v)][1] for v in cycle]
        metadata.append({
            "type": "fixed-gadget simple cycle basis row",
            "cycle_vertices": [int(v) for v in cycle],
            "edge_qubits": list(key),
            "weight": len(key),
            "centroid": (float(np.mean(xs)), float(np.mean(ys))),
        })

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            metadata[i]["centroid"][0],
            metadata[i]["centroid"][1],
            metadata[i]["weight"],
            metadata[i]["edge_qubits"],
        ),
    )
    rows = [rows[i] for i in order]
    metadata = [metadata[i] for i in order]
    if rows:
        H = np.vstack(rows).astype(np.uint8)
    else:
        H = np.zeros((0, width), dtype=np.uint8)
    return H, metadata


def compile_ordered_cycle_from_edge_support(edge_qubits, template):
    """Return ordered vertices and edge qubits for one simple cycle support."""
    edge_qubits = [int(q) for q in edge_qubits]
    adjacency = {}
    pair_to_qubit = {}
    for q in edge_qubits:
        if q not in template.edge_qubit_to_vertices:
            return None
        a, b = map(int, template.edge_qubit_to_vertices[q])
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        pair_to_qubit[tuple(sorted((a, b)))] = q
    if not edge_qubits or any(len(nbrs) != 2 for nbrs in adjacency.values()):
        return None

    start = min(adjacency)
    vertices = [start]
    edges = []
    prev = None
    cur = start
    for _ in range(len(edge_qubits)):
        nbrs = adjacency[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        edge = tuple(sorted((cur, nxt)))
        edges.append(pair_to_qubit[edge])
        if nxt == start:
            break
        vertices.append(nxt)
        prev, cur = cur, nxt
    if set(edges) != set(edge_qubits):
        return None
    return vertices, edges


def compile_row_from_edge_support(edge_qubits, width):
    row = np.zeros(int(width), dtype=np.uint8)
    row[sorted(set(map(int, edge_qubits)))] = 1
    return row


def compile_split_floor_cycle_supports(
    H_basis_new,
    cycle_sources,
    floor_edge_qubits_used,
    template,
    vertex_row_to_check_qubit,
    width,
    *,
    max_cycle_weight=6,
    max_vertex_degree=6,
):
    """Split heavy fixed-floor cycles using degree-capped chord paths."""
    max_cycle_weight = int(max_cycle_weight)
    floor_edges = sorted(set(map(int, floor_edge_qubits_used)))
    allowed_vertices = set(map(int, template.vertex_check_qubits))
    queue = [(list(map(int, src["edge_qubits"])), dict(src)) for src in cycle_sources]
    rows = []
    out_sources = []

    while queue:
        support, source = queue.pop(0)
        support = sorted(set(map(int, support)))
        if len(support) <= max_cycle_weight:
            rows.append(compile_row_from_edge_support(support, width))
            out_sources.append({**source, "edge_qubits": support, "weight": len(support)})
            continue

        ordered = compile_ordered_cycle_from_edge_support(support, template)
        if ordered is None:
            rows.append(compile_row_from_edge_support(support, width))
            out_sources.append({**source, "edge_qubits": support, "weight": len(support)})
            continue

        vertices, ordered_edges = ordered
        k = len(ordered_edges)
        vertex_weights = compile_vertex_weights_from_rows(H_basis_new, vertex_row_to_check_qubit)
        best = None
        best_relaxed = None
        support_set = set(support)
        for i in range(k):
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue
                path = compile_shortest_floor_edge_path(
                    template,
                    vertices[i],
                    vertices[j],
                    allowed_vertices,
                    forbidden_edges=support_set,
                )
                if path is None:
                    continue
                arc1 = set(ordered_edges[i:j])
                arc2 = set(ordered_edges[j:] + ordered_edges[:i])
                chord = set(map(int, path))
                c1 = sorted(arc1 | chord)
                c2 = sorted(arc2 | chord)
                if not c1 or not c2:
                    continue
                max_vertex_after = compile_max_vertex_weight_after_adding_path(
                    chord,
                    floor_edges,
                    vertex_weights,
                    template,
                )
                candidate = (
                    max(len(c1), len(c2)),
                    len(c1) + len(c2),
                    len(chord - set(floor_edges)),
                    len(chord),
                    max_vertex_after,
                    c1,
                    c2,
                    sorted(chord),
                    (int(vertices[i]), int(vertices[j])),
                )
                cap_ok = compile_path_fits_vertex_weight_cap(
                    chord,
                    floor_edges,
                    vertex_weights,
                    template,
                    max_vertex_degree,
                )
                if cap_ok and (best is None or candidate[:4] < best[:4]):
                    best = candidate
                relaxed_key = (
                    candidate[0],
                    candidate[4],
                    candidate[1],
                    candidate[2],
                    candidate[3],
                )
                if (
                    candidate[0] < len(support)
                    and candidate[4] < len(support)
                    and (best_relaxed is None or relaxed_key < best_relaxed[0])
                ):
                    best_relaxed = (relaxed_key, candidate)

        relaxed_vertex_cap = False
        if best is None:
            best = None if best_relaxed is None else best_relaxed[1]
            relaxed_vertex_cap = best is not None

        if best is None or best[0] >= len(support):
            rows.append(compile_row_from_edge_support(support, width))
            out_sources.append({**source, "edge_qubits": support, "weight": len(support)})
            continue

        _, _, _, _, max_vertex_after, c1, c2, chord, endpoints = best
        H_basis_new, floor_edges, added_edges = compile_add_floor_edges_to_vertex_checks(
            H_basis_new,
            chord,
            floor_edges,
            template,
            vertex_row_to_check_qubit,
            allow_new_vertices=True,
        )
        for piece, child in enumerate([c1, c2]):
            queue.append((
                child,
                {
                    "type": "fixed-gadget split cycle basis row",
                    "split_from_type": source.get("type"),
                    "split_parent_weight": len(support),
                    "split_piece": int(piece),
                    "chord_edge_qubits": [int(q) for q in chord],
                    "chord_endpoint_vertices": endpoints,
                    "added_floor_edges": [int(q) for q in added_edges],
                    "relaxed_vertex_degree_cap": bool(relaxed_vertex_cap),
                    "max_vertex_weight_after_chord": int(max_vertex_after),
                    "edge_qubits": [int(q) for q in child],
                    "weight": len(child),
                },
            ))

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            out_sources[i].get("weight", int(np.sum(rows[i]))),
            out_sources[i].get("edge_qubits", []),
        ),
    )
    rows = [rows[i] for i in order]
    out_sources = [out_sources[i] for i in order]
    H = np.vstack(rows).astype(np.uint8) if rows else np.zeros((0, width), dtype=np.uint8)
    return H_basis_new, H, out_sources, floor_edges


def compile_partition_cycle_rows_by_shuttles(H_cycles, cycle_sources, shuttling_edge_qubits):
    H_cycles = np.asarray(H_cycles, dtype=np.uint8)
    shuttles = set(map(int, shuttling_edge_qubits))
    floor_rows = []
    floor_sources = []
    shuttle_rows = []
    shuttle_sources = []
    for row, source in zip(H_cycles, cycle_sources):
        support = set(map(int, np.flatnonzero(row)))
        if support & shuttles:
            shuttle_rows.append(row)
            shuttle_sources.append({**source, "type": source.get("type", "combined cycle basis row with shuttle")})
        else:
            floor_rows.append(row)
            floor_sources.append(source)
    width = H_cycles.shape[1] if H_cycles.ndim == 2 else 0
    H_floor = np.vstack(floor_rows).astype(np.uint8) if floor_rows else np.zeros((0, width), dtype=np.uint8)
    H_shuttle = np.vstack(shuttle_rows).astype(np.uint8) if shuttle_rows else np.zeros((0, width), dtype=np.uint8)
    return H_floor, floor_sources, H_shuttle, shuttle_sources


def compile_cycle_rows_are_simple_boundaries(H_cycle_fixed, template):
    checks = []
    for row in np.asarray(H_cycle_fixed, dtype=np.uint8):
        support = [int(q) for q in np.flatnonzero(row) if int(q) in template.edge_qubit_to_vertices]
        degree = {}
        for q in support:
            a, b = template.edge_qubit_to_vertices[q]
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        checks.append(bool(support) and all(d == 2 for d in degree.values()))
    return checks




def compile_fixed_auxiliary_graph_data(edge_qubits, template):
    """Return incidence data and an igraph for the used fixed triangular graph.

    Rows of the incidence matrix are fixed vertex-check qubits; columns are used
    fixed edge qubits. Each column has 1s at the two endpoint vertices.
    """
    edge_qubits = sorted(int(q) for q in edge_qubits)
    vertex_qubits = sorted(
        {
            int(v)
            for q in edge_qubits
            for v in template.edge_qubit_to_vertices[int(q)]
        },
        key=lambda q: (*template.coords[q], q),
    )
    vertex_to_row = {q: i for i, q in enumerate(vertex_qubits)}

    incidence = np.zeros((len(vertex_qubits), len(edge_qubits)), dtype=np.uint8)
    ig_edges = []
    for col, q in enumerate(edge_qubits):
        a, b = template.edge_qubit_to_vertices[int(q)]
        a = int(a)
        b = int(b)
        incidence[vertex_to_row[a], col] = 1
        incidence[vertex_to_row[b], col] = 1
        ig_edges.append((vertex_to_row[a], vertex_to_row[b]))

    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError(
            "python-igraph is required to return auxiliary_graph_igraph"
        ) from exc

    graph = ig.Graph(n=len(vertex_qubits), edges=ig_edges, directed=False)
    graph.vs["qubit"] = vertex_qubits
    graph.vs["coord"] = [tuple(template.coords[q]) for q in vertex_qubits]
    graph.vs["site"] = [template.vertex_qubit_to_site.get(q) for q in vertex_qubits]
    graph.es["qubit"] = edge_qubits
    graph.es["endpoint_qubits"] = [
        tuple(map(int, template.edge_qubit_to_vertices[q]))
        for q in edge_qubits
    ]

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(len(vertex_qubits)))
    for vertex_id, qubit in enumerate(vertex_qubits):
        nx_graph.nodes[vertex_id]["qubit"] = int(qubit)
        nx_graph.nodes[vertex_id]["coord"] = tuple(template.coords[int(qubit)])
        nx_graph.nodes[vertex_id]["site"] = template.vertex_qubit_to_site.get(int(qubit))
    for col, q in enumerate(edge_qubits):
        a, b = template.edge_qubit_to_vertices[int(q)]
        u, v = vertex_to_row[int(a)], vertex_to_row[int(b)]
        nx_graph.add_edge(u, v, qubit=int(q), column=int(col), endpoint_qubits=(int(a), int(b)))

    edge_qubit_to_vertex_ids = {
        int(q): tuple(vertex_to_row[int(v)] for v in template.edge_qubit_to_vertices[int(q)])
        for q in edge_qubits
    }
    edge_qubit_to_vertex_qubits = {
        int(q): tuple(map(int, template.edge_qubit_to_vertices[int(q)]))
        for q in edge_qubits
    }
    vertex_qubit_to_vertex_id = {int(q): int(i) for i, q in enumerate(vertex_qubits)}
    vertex_id_to_qubit = {int(i): int(q) for i, q in enumerate(vertex_qubits)}

    return {
        "auxiliary_graph_incidence_matrix": incidence,
        "auxiliary_graph_incidence_vertices": vertex_qubits,
        "auxiliary_graph_incidence_edge_qubits": edge_qubits,
        "auxiliary_graph_igraph": graph,
        "measurement_graph": nx_graph,
        "measurement_graph_networkx": nx_graph,
        "measurement_graph_igraph": graph,
        "measurement_graph_incidence_matrix": incidence,
        "measurement_graph_vertex_qubits": vertex_qubits,
        "measurement_graph_edge_qubits": edge_qubits,
        "edge_qubit_to_vertex_ids": edge_qubit_to_vertex_ids,
        "edge_qubit_to_endpoint_vertex_ids": edge_qubit_to_vertex_ids,
        "qubit_to_endpoint_vertex_ids": edge_qubit_to_vertex_ids,
        "edge_qubit_to_vertex_qubits": edge_qubit_to_vertex_qubits,
        "vertex_qubit_to_vertex_id": vertex_qubit_to_vertex_id,
        "vertex_id_to_qubit": vertex_id_to_qubit,
    }

def compile_used_layout_sites(layout):
    sites = set()
    for a, b in layout["lattice_edge_to_qubit"]:
        sites.add(a)
        sites.add(b)
    for site in layout["site_to_vertex_check_row"]:
        sites.add(site)
    return sites


def compile_fit_layout_sites_to_template(layout, template):
    """Map tri_layout lattice sites exactly into the fixed triangular template.

    ``tri_layout`` and the reusable gadget use the same triangular-lattice
    combinatorics.  Therefore the only allowed freedom is an integer row/column
    translation.  If the placed graph does not fit as an exact subgraph of the
    fixed gadget lattice, this raises instead of silently changing the graph.
    """
    used_sites = sorted(compile_used_layout_sites(layout), key=lambda s: (s[0], s[1]))
    if len(used_sites) > len(template.vertex_check_qubits):
        raise ValueError(
            f"layout uses {len(used_sites)} lattice sites, fixed template has "
            f"{len(template.vertex_check_qubits)} vertex sites"
        )

    used_rows = [s[0] for s in used_sites]
    template_sites = set(template.vertex_site_to_qubit)
    template_rows = [s[0] for s in template_sites]
    template_cols = [s[1] for s in template_sites]

    row_span = max(used_rows) - min(used_rows)
    if row_span > max(template_rows) - min(template_rows):
        raise ValueError("tri_layout uses more lattice rows than the fixed gadget has")

    best = None
    for dr in range(min(template_rows) - min(used_rows), max(template_rows) - max(used_rows) + 1):
        skew_cols = [compile_skew_to_alternating_site(site, dr, 0)[1] for site in used_sites]
        for dc in range(min(template_cols) - min(skew_cols), max(template_cols) - max(skew_cols) + 1):
            shifted = [compile_skew_to_alternating_site(site, dr, dc) for site in used_sites]
            if all(site in template_sites for site in shifted):
                # Prefer centered placements, then stable small translations.
                shifted_rows = [r for r, _ in shifted]
                shifted_cols = [c for _, c in shifted]
                row_center_error = abs((min(shifted_rows) + max(shifted_rows)) - (min(template_rows) + max(template_rows)))
                col_center_error = abs((min(shifted_cols) + max(shifted_cols)) - (min(template_cols) + max(template_cols)))
                candidate = (row_center_error + col_center_error, abs(dr) + abs(dc), dr, dc)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise ValueError("tri_layout lattice sites do not fit exactly into the fixed triangular gadget")

    _, _, dr, dc = best
    return {
        site: template.vertex_site_to_qubit[compile_skew_to_alternating_site(site, dr, dc)]
        for site in used_sites
    }, (dr, dc)


def compile_skew_to_alternating_site(site, row_shift, col_shift):
    """Convert tri_layout's skew lattice coordinates to the 8/7 gadget rows.

    tri_layout positions site ``(r, c)`` at x = c + r/2.  The fixed Stim gadget
    draws rows with only parity offset: x = c' + (r' mod 2)/2.  This conversion
    preserves the physical lattice point exactly, up to integer row/column
    translation.
    """
    r, c = site
    if row_shift % 2 == 0:
        c_shift_from_skew = r // 2
    else:
        c_shift_from_skew = (r + 1) // 2
    return (r + row_shift, c + c_shift_from_skew + col_shift)


def compile_map_layout_edges_to_fixed(layout, template, site_to_vertex):
    """Map each tri_layout lattice-edge qubit to exactly one fixed edge qubit."""
    old_edge_to_fixed_edges = {}
    missing_edges = []

    for layout_edge, old_q in layout["lattice_edge_to_qubit"].items():
        a, b = layout_edge
        va = site_to_vertex[a]
        vb = site_to_vertex[b]
        pair = tuple(sorted((va, vb)))
        if pair not in template.vertex_pair_to_edge_qubit:
            missing_edges.append((tuple(layout_edge), pair, int(old_q)))
            continue
        old_edge_to_fixed_edges[int(old_q)] = [template.vertex_pair_to_edge_qubit[pair]]

    if missing_edges:
        details = ", ".join(
            f"layout edge {edge} -> fixed vertices {pair} for old qubit {q}"
            for edge, pair, q in missing_edges[:5]
        )
        raise ValueError(f"tri_layout edge is not an exact fixed-gadget edge: {details}")

    return old_edge_to_fixed_edges


def layout_graph_edge_order(deformation_result):
    return [
        tuple(sorted(map(int, e)))
        for e in deformation_result["g"].get_edgelist()
    ]


def compile_fixed_vertex_graph(template, allowed_vertices=None):
    allowed = None if allowed_vertices is None else set(map(int, allowed_vertices))
    G = nx.Graph()
    vertices = template.vertex_check_qubits if allowed is None else allowed
    G.add_nodes_from(map(int, vertices))
    for q in template.edge_qubits:
        a, b = template.edge_qubit_to_vertices[int(q)]
        a = int(a)
        b = int(b)
        if allowed is not None and (a not in allowed or b not in allowed):
            continue
        G.add_edge(a, b, qubit=int(q))
    return G


def compile_route_edge_qubits_on_fixed_graph(G, start, stop, used=None, randomness=0.0, rng=None):
    used = {} if used is None else used
    rng = np.random.default_rng(0) if rng is None else rng

    def weight(a, b, data):
        q = int(data["qubit"])
        return 1.0 + 30.0 * used.get(q, 0) + randomness * float(rng.random())

    vertices = nx.shortest_path(G, int(start), int(stop), weight=weight)
    out = []
    for a, b in zip(vertices, vertices[1:]):
        out.append(int(G.edges[int(a), int(b)]["qubit"]))
    return out


def compile_fixed_route_length(template, a, b):
    G = compile_fixed_vertex_graph(template)
    return nx.shortest_path_length(G, int(a), int(b))


def compile_placement_score_for_floor_edges(template, placement, floor_edges, target_fixed, anchor_weight=0.03, bbox_weight=0.35):
    G = compile_fixed_vertex_graph(template)
    lengths = [
        nx.shortest_path_length(G, placement[u], placement[v])
        for u, v in floor_edges
    ]
    edge_cost = sum(lengths)
    max_cost = max(lengths) if lengths else 0
    anchor = 0.0
    for v, q in placement.items():
        x, y = template.coords[int(q)]
        tx, ty = template.coords[int(target_fixed[v])]
        anchor += (x - tx) ** 2 + (y - ty) ** 2
    pts = np.array([template.coords[int(q)] for q in placement.values()])
    span = pts.max(axis=0) - pts.min(axis=0) if len(pts) else np.zeros(2)
    return edge_cost + 1.5 * max_cost + anchor_weight * anchor + bbox_weight * float(span[0] * span[1])


def compile_improve_fixed_vertex_placement(template, initial, floor_edges, rounds=5):
    """Second local placement pass after long abstract routes become shuttles."""
    if not floor_edges:
        return dict(initial)
    rng = np.random.default_rng(0)
    placement = {int(v): int(q) for v, q in initial.items()}
    target_fixed = dict(placement)
    all_sites = list(map(int, template.vertex_check_qubits))
    best = compile_placement_score_for_floor_edges(template, placement, floor_edges, target_fixed)
    nodes = list(placement)

    for _ in range(int(rounds)):
        improved = False
        rng.shuffle(nodes)
        used = set(placement.values())
        for v in nodes:
            old = int(placement[v])
            candidates = [q for q in all_sites if q not in used or q == old]
            candidates.sort(
                key=lambda q: (
                    np.linalg.norm(np.asarray(template.coords[q]) - np.asarray(template.coords[old])),
                    rng.random(),
                )
            )
            for q in candidates[:16]:
                q = int(q)
                if q == old:
                    continue
                placement[v] = q
                score = compile_placement_score_for_floor_edges(template, placement, floor_edges, target_fixed)
                if score + 1e-9 < best:
                    used.remove(old)
                    used.add(q)
                    old = q
                    best = score
                    improved = True
                else:
                    placement[v] = old
        if not improved:
            break
    return placement


def compile_reroute_floor_after_abstract_shuttling(layout, template, site_to_fixed_vertex, shuttling_threshold):
    """Classify abstract graph edges as floor/shuttle, then compactly reroute floor edges."""
    deformation_result = layout["deformation_result"]
    initial = {
        int(v): int(site_to_fixed_vertex[site])
        for v, site in layout["vertex_sites"].items()
    }
    graph_edges = layout_graph_edge_order(deformation_result)
    threshold = None if shuttling_threshold is None else int(shuttling_threshold)
    shuttled_edges = set()
    floor_graph_edges = []
    for edge in graph_edges:
        a, b = edge
        length = compile_fixed_route_length(template, initial[int(a)], initial[int(b)])
        if threshold is not None and length > threshold:
            shuttled_edges.add(edge)
        else:
            floor_graph_edges.append(edge)

    placement = compile_improve_fixed_vertex_placement(template, initial, floor_graph_edges)
    G = compile_fixed_vertex_graph(template)
    used = {}
    floor_routes = {}
    floor_edges = set()
    for edge in sorted(
        floor_graph_edges,
        key=lambda e: nx.shortest_path_length(G, placement[int(e[0])], placement[int(e[1])]),
    ):
        path = compile_route_edge_qubits_on_fixed_graph(
            G,
            placement[int(edge[0])],
            placement[int(edge[1])],
            used=used,
        )
        floor_routes[edge] = path
        floor_edges.update(path)
        for q in path:
            used[int(q)] = used.get(int(q), 0) + 1

    return placement, floor_routes, shuttled_edges, sorted(floor_edges)


def compile_increment_vertex_weights_for_edges(vertex_weights, edge_qubits, template):
    """Update vertex weights after materializing edge qubits."""
    for q in map(int, edge_qubits):
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = template.edge_qubit_to_vertices[q]
        vertex_weights[int(a)] = vertex_weights.get(int(a), 0) + 1
        vertex_weights[int(b)] = vertex_weights.get(int(b), 0) + 1


def compile_choose_shuttling_endpoint(start, template, vertex_weights, max_vertex_degree, max_distance=None):
    """Choose a shuttle endpoint without exceeding the vertex-degree cap when possible."""
    start = int(start)
    if max_vertex_degree is None or vertex_weights.get(start, 0) < int(max_vertex_degree):
        return start, []
    max_distance = max(2, int(max_vertex_degree)) if max_distance is None else int(max_distance)
    allowed_vertices = set(map(int, template.vertex_check_qubits))
    endpoint, path_edges = compile_choose_nearby_low_weight_vertex(
        start,
        allowed_vertices,
        template,
        vertex_weights,
        max_vertex_degree,
        max_distance,
    )
    if endpoint is not None:
        return int(endpoint), [int(q) for q in path_edges]

    # Fallback: keep the construction alive, but choose the least loaded nearby
    # vertex instead of blindly piling onto the preferred endpoint.
    G = compile_fixed_floor_graph(template, allowed_vertices=allowed_vertices)
    if start not in G:
        return start, []
    lengths = nx.single_source_shortest_path_length(G, start, cutoff=max_distance)
    candidates = []
    for vertex, distance in lengths.items():
        path = compile_shortest_floor_edge_path(template, start, vertex, allowed_vertices)
        if path is None:
            continue
        candidates.append((vertex_weights.get(int(vertex), 0), int(distance), int(vertex), path))
    if not candidates:
        return start, []
    _, _, endpoint, path_edges = min(candidates)
    return int(endpoint), [int(q) for q in path_edges]


def compile_choose_shuttling_endpoints(preferred, template, vertex_weights, max_vertex_degree):
    del template, vertex_weights, max_vertex_degree
    return tuple(sorted(map(int, preferred))), []


def compile_initial_vertex_weights_for_shuttling(deformation_result, template, placement, floor_edge_qubits):
    """Weights before shuttles: data support plus already routed floor edges."""
    weights = {}
    n_data = int(deformation_result["n_original_qubits"])
    for q, v in deformation_result["qubit_to_vertex"].items():
        q = int(q)
        v = int(v)
        if q < n_data and v in placement:
            vertex = int(placement[v])
            weights[vertex] = weights.get(vertex, 0) + 1
    compile_increment_vertex_weights_for_edges(weights, floor_edge_qubits, template)
    return weights


def compile_allocate_abstract_route_shuttles(
    deformation_result,
    template,
    placement,
    shuttled_edges,
    used_shuttles=None,
    vertex_weights=None,
    max_vertex_degree=6,
):
    used_shuttles = set() if used_shuttles is None else set(map(int, used_shuttles))
    vertex_weights = {} if vertex_weights is None else vertex_weights
    shuttle_iter = iter([q for q in template.shuttling_edge_qubits if int(q) not in used_shuttles])
    records = []
    for edge in sorted(shuttled_edges):
        try:
            q = int(next(shuttle_iter))
        except StopIteration as exc:
            raise ValueError("not enough reserved shuttling edge qubits") from exc
        preferred = (int(placement[int(edge[0])]), int(placement[int(edge[1])]))
        endpoint_vertices, adjustment_edges = compile_choose_shuttling_endpoints(
            preferred,
            template,
            vertex_weights,
            max_vertex_degree,
        )
        template.edge_qubit_to_vertices[q] = endpoint_vertices
        compile_increment_vertex_weights_for_edges(vertex_weights, [q], template)
        records.append({
            "qubit": q,
            "type": "abstract_route_shuttle",
            "abstract_edge": tuple(map(int, edge)),
            "path_length": int(compile_fixed_route_length(template, *preferred)),
            "preferred_endpoint_vertices": tuple(sorted(preferred)),
            "endpoint_vertices": endpoint_vertices,
            "endpoint_adjustment_floor_edges": [int(x) for x in adjustment_edges],
            "endpoint_sites": (
                tuple(template.vertex_qubit_to_site[endpoint_vertices[0]]),
                tuple(template.vertex_qubit_to_site[endpoint_vertices[1]]),
            ),
            "replaced_floor_edge_qubits": [],
        })
    compile_place_shuttling_edge_coords(template, records)
    return [int(r["qubit"]) for r in records], records


def compile_seed_vertex_rows_from_placement(deformation_result, template, placement, width):
    n_data = int(deformation_result["n_original_qubits"])
    vertices = sorted(set(map(int, placement.values())), key=lambda q: (*template.coords[q], q))
    vertex_row_to_check_qubit = {row: int(q) for row, q in enumerate(vertices)}
    vertex_to_row = {int(q): int(row) for row, q in vertex_row_to_check_qubit.items()}
    H = np.zeros((len(vertices), width), dtype=np.uint8)
    for q, v in deformation_result["qubit_to_vertex"].items():
        q = int(q)
        v = int(v)
        if q >= n_data or int(v) not in placement:
            continue
        row = vertex_to_row[int(placement[int(v)])]
        H[row, q] = 1
    return H, vertex_row_to_check_qubit


def compile_expand_old_matrix_on_fixed_routes(
    deformation_result,
    matrix_key,
    floor_routes,
    abstract_shuttle_records,
    width,
):
    M = np.asarray(deformation_result[matrix_key], dtype=np.uint8) % 2
    n_data = int(deformation_result["n_original_qubits"])
    out = np.zeros((M.shape[0], width), dtype=np.uint8)
    out[:, : min(n_data, M.shape[1])] = M[:, : min(n_data, M.shape[1])]
    graph_edges = layout_graph_edge_order(deformation_result)
    shuttle_by_edge = {
        tuple(record["abstract_edge"]): [
            int(record["qubit"]),
            *map(int, record.get("endpoint_adjustment_floor_edges", [])),
        ]
        for record in abstract_shuttle_records
    }
    for col, edge in enumerate(graph_edges):
        old_col = n_data + col
        if old_col >= M.shape[1]:
            continue
        rows = np.flatnonzero(M[:, old_col])
        if len(rows) == 0:
            continue
        if edge in shuttle_by_edge:
            for q in shuttle_by_edge[edge]:
                out[rows, int(q)] ^= 1
        else:
            for q in floor_routes.get(edge, []):
                out[rows, int(q)] ^= 1
    return out


def compile_expand_old_basis_on_fixed_routes(deformation_result, floor_routes, abstract_shuttle_records, width):
    return compile_expand_old_matrix_on_fixed_routes(
        deformation_result,
        "H_basis_old",
        floor_routes,
        abstract_shuttle_records,
        width,
    )


def compile_fixed_support_for_planar_edge(edge, floor_routes, abstract_shuttle_records):
    shuttle_by_edge = {
        tuple(record["abstract_edge"]): [
            int(record["qubit"]),
            *map(int, record.get("endpoint_adjustment_floor_edges", [])),
        ]
        for record in abstract_shuttle_records
    }
    edge = tuple(sorted(map(int, edge)))
    return list(map(int, floor_routes.get(edge, shuttle_by_edge.get(edge, []))))


def compile_compress_old_opposite_components_with_shuttles(
    deformation_result,
    template,
    placement,
    H_opp_old,
    floor_routes,
    abstract_shuttle_records,
    floor_edge_qubits_used,
    shuttling_edge_qubits_used,
    vertex_weights,
    *,
    shuttling_threshold,
    max_vertex_degree=6,
):
    """Replace long post-planar old-opposite path components by shuttle edges."""
    if shuttling_threshold is None:
        return H_opp_old, floor_edge_qubits_used, shuttling_edge_qubits_used, []

    threshold = int(shuttling_threshold)
    H_opp_old = np.asarray(H_opp_old, dtype=np.uint8).copy()
    graph_edges = layout_graph_edge_order(deformation_result)
    used_shuttles = set(map(int, shuttling_edge_qubits_used))
    shuttle_iter = iter([
        int(q) for q in template.shuttling_edge_qubits if int(q) not in used_shuttles
    ])
    floor_edges = set(map(int, floor_edge_qubits_used))
    shuttle_edges = set(map(int, shuttling_edge_qubits_used))
    records = []

    n_rows = np.asarray(deformation_result["H_opposite_basis_old_padded"]).shape[0]
    for row in range(n_rows):
        for component in compile_old_opposite_edge_components(deformation_result, row):
            odd = component["odd_vertices"]
            if len(odd) != 2:
                continue
            fixed_support = []
            for edge_col in component["edge_cols"]:
                if int(edge_col) >= len(graph_edges):
                    continue
                fixed_support.extend(compile_fixed_support_for_planar_edge(
                    graph_edges[int(edge_col)],
                    floor_routes,
                    abstract_shuttle_records,
                ))
            fixed_support = sorted(set(map(int, fixed_support)))
            if len(fixed_support) <= threshold:
                continue
            try:
                q = int(next(shuttle_iter))
            except StopIteration as exc:
                raise ValueError("not enough reserved shuttling edge qubits") from exc

            preferred = (int(placement[int(odd[0])]), int(placement[int(odd[1])]))
            endpoints, adjustment_edges = compile_choose_shuttling_endpoints(
                preferred,
                template,
                vertex_weights,
                max_vertex_degree,
            )
            template.edge_qubit_to_vertices[q] = endpoints
            compile_increment_vertex_weights_for_edges(vertex_weights, [q], template)

            for old_q in fixed_support:
                H_opp_old[row, int(old_q)] ^= 1
            H_opp_old[row, q] ^= 1
            for adj_q in adjustment_edges:
                H_opp_old[row, int(adj_q)] ^= 1
                floor_edges.add(int(adj_q))

            shuttle_edges.add(q)
            records.append({
                "qubit": q,
                "type": "old_opposite_component_shuttle",
                "opposite_row": int(row),
                "planar_edge_cols": [int(c) for c in component["edge_cols"]],
                "path_length": int(len(fixed_support)),
                "preferred_endpoint_vertices": tuple(sorted(preferred)),
                "endpoint_vertices": endpoints,
                "endpoint_adjustment_floor_edges": [int(x) for x in adjustment_edges],
                "replaced_floor_edge_qubits": [int(x) for x in fixed_support],
                "force_replaced_path_closure": True,
            })

    compile_place_shuttling_edge_coords(template, abstract_shuttle_records + records)
    return H_opp_old, sorted(floor_edges), sorted(shuttle_edges), records


def compile_map_vertex_rows_to_fixed(layout, site_to_vertex):
    offset = int(layout["basis_new_row_offset"])
    return {
        offset + int(local_row): int(site_to_vertex[site])
        for site, local_row in layout["site_to_vertex_check_row"].items()
    }


def compile_ordered_cycle_vertices(edge_vertex_pairs):
    """Return ordered vertex loops for a support made from simple cycle(s)."""
    adjacency = {}
    for a, b in edge_vertex_pairs:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    unused = {tuple(sorted(edge)) for edge in edge_vertex_pairs}
    loops = []
    while unused:
        start_edge = min(unused)
        start, cur = start_edge
        prev = None
        loop = [start]
        for _ in range(len(unused) + len(edge_vertex_pairs) + 1):
            loop.append(cur)
            edge = tuple(sorted((prev, cur))) if prev is not None else tuple(sorted((start, cur)))
            unused.discard(edge)
            if cur == start:
                loops.append(loop[:-1])
                break
            nbrs = adjacency.get(cur, [])
            if len(nbrs) != 2:
                return []
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
            prev, cur = cur, nxt
        else:
            return []
    return loops


def compile_point_on_segment(p, a, b, eps=1e-9):
    px, py = p
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps


def compile_point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test, including boundary."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if compile_point_on_segment(point, a, b):
            return True
        xi, yi = a
        xj, yj = b
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
    return inside


def compile_cycle_checks_inside_edge_support(support, template):
    support_edges = [
        template.edge_qubit_to_vertices[q]
        for q in sorted(support)
        if q in template.edge_qubit_to_vertices
    ]
    loops = compile_ordered_cycle_vertices(support_edges)
    if not loops:
        return []

    polygons = [
        [template.coords[v] for v in loop]
        for loop in loops
        if len(loop) >= 3
    ]
    checks = []
    for q in template.cycle_check_qubits:
        point = template.coords[q]
        if any(compile_point_in_polygon(point, polygon) for polygon in polygons):
            checks.append(int(q))
    return checks


def compile_nearest_cycle_check_for_edge_support(support, template):
    points = []
    for q in sorted(map(int, support)):
        if q in template.edge_qubit_to_vertices:
            for vertex in template.edge_qubit_to_vertices[q]:
                points.append(template.coords[int(vertex)])
        elif q in template.coords:
            points.append(template.coords[int(q)])
    if not points or not template.cycle_check_qubits:
        return None
    center = np.mean(np.asarray(points, dtype=float), axis=0)
    return int(min(
        template.cycle_check_qubits,
        key=lambda q: float(np.linalg.norm(np.asarray(template.coords[int(q)]) - center)),
    ))


def compile_map_cycle_rows_to_triangle_checks(H_cycle_fixed, template):
    row_to_checks = {}
    row_to_first_check = {}
    triangle_edges = {
        q: set(edges)
        for q, edges in template.cycle_check_to_edge_qubits.items()
    }
    for row, check in enumerate(np.asarray(H_cycle_fixed, dtype=np.uint8)):
        support = set(np.flatnonzero(check).astype(int))
        checks = compile_cycle_checks_inside_edge_support(support, template)
        if not checks:
            checks = [
                int(q)
                for q, edges in triangle_edges.items()
                if len(support & edges) >= 2
            ]
        if not checks:
            nearest = compile_nearest_cycle_check_for_edge_support(support, template)
            checks = [] if nearest is None else [nearest]
        row_to_checks[int(row)] = checks
        row_to_first_check[int(row)] = checks[0] if checks else None
    return row_to_checks, row_to_first_check


def compile_map_opposite_rows_to_cycle_ancillas(H_floor_cycles, H_shuttling_cycles, template):
    """Map floor cycles to floor ancillas and shuttling cycles to external ancillas."""
    floor_map, floor_first = compile_map_cycle_rows_to_triangle_checks(H_floor_cycles, template)
    row_to_checks = {}
    row_to_first_check = {}

    for row, checks in floor_map.items():
        row_to_checks[int(row)] = list(map(int, checks))
        row_to_first_check[int(row)] = floor_first.get(int(row))

    offset = int(np.asarray(H_floor_cycles).shape[0])
    n_shuttle_rows = int(np.asarray(H_shuttling_cycles).shape[0])
    if n_shuttle_rows > len(template.shuttling_cycle_check_qubits):
        raise ValueError(
            f"layout needs {n_shuttle_rows} shuttling cycle ancillas, "
            f"template has {len(template.shuttling_cycle_check_qubits)}"
        )
    for local_row in range(n_shuttle_rows):
        q = int(template.shuttling_cycle_check_qubits[local_row])
        row = offset + local_row
        row_to_checks[row] = [q]
        row_to_first_check[row] = q

    return row_to_checks, row_to_first_check



def compile_estimate_relative_expansion(edge_qubits, port_vertex_qubits, template, samples=50_000, seed=0, t=None):
    """Randomly estimate beta_t(G, U) for the current fixed auxiliary graph."""
    graph_data = compile_fixed_auxiliary_graph_data(edge_qubits, template)
    G = graph_data["auxiliary_graph_incidence_matrix"]
    vertices = graph_data["auxiliary_graph_incidence_vertices"]
    vertex_to_row = {q: i for i, q in enumerate(vertices)}
    port_rows = np.asarray([vertex_to_row[q] for q in port_vertex_qubits if q in vertex_to_row], dtype=int)
    if len(port_rows) < 2:
        return {"beta": np.inf, "bad_subset": [], "bad_port_vertices": [], "graph_data": graph_data}

    rng = np.random.default_rng(seed)
    t = len(port_rows) if t is None else int(t)
    best = {"beta": np.inf, "bad_subset": [], "bad_port_vertices": [], "graph_data": graph_data}
    for _ in range(int(samples)):
        subset = np.flatnonzero(rng.random(G.shape[0]) < rng.uniform(0.05, 0.5))
        u_rows = np.intersect1d(subset, port_rows, assume_unique=False)
        complement_u_rows = np.setdiff1d(port_rows, u_rows, assume_unique=False)
        denom = min(t, len(u_rows), len(complement_u_rows))
        if denom == 0:
            continue
        boundary = int(np.sum(np.bitwise_xor.reduce(G[subset], axis=0)))
        beta = boundary / denom
        if beta < best["beta"]:
            # Add edges from the smaller bad port side, because that is the side
            # appearing in the relative-expansion denominator.
            bad_rows = u_rows if len(u_rows) <= len(complement_u_rows) else complement_u_rows
            best = {
                "beta": float(beta),
                "bad_subset": [vertices[i] for i in subset],
                "bad_port_vertices": [vertices[i] for i in bad_rows],
                "graph_data": graph_data,
            }
    return best


def compile_unused_incident_edges(
    vertex_qubit,
    used_edge_qubits,
    allowed_vertices,
    template,
    *,
    vertex_check_weights=None,
    max_vertex_weight=None,
):
    used = set(used_edge_qubits)
    allowed = set(allowed_vertices)
    out = []
    for q, (a, b) in template.edge_qubit_to_vertices.items():
        if q in used or vertex_qubit not in (a, b):
            continue
        other = b if a == vertex_qubit else a
        if vertex_check_weights is not None and max_vertex_weight is not None:
            if vertex_check_weights.get(int(vertex_qubit), 0) >= max_vertex_weight:
                continue
            if vertex_check_weights.get(int(other), 0) >= max_vertex_weight:
                continue
        if other in allowed:
            out.append(int(q))
    return out


def compile_repair_relative_expansion(
    edge_qubits,
    port_vertex_qubits,
    allowed_vertices,
    template,
    *,
    threshold=1.0,
    samples=50_000,
    seed=0,
    max_rounds=5,
    vertex_check_weights=None,
    max_vertex_weight=6,
):
    """Add unused fixed-template edges adjacent to bad port vertices until beta >= threshold."""
    edge_qubits = sorted(set(map(int, edge_qubits)))
    vertex_check_weights = None if vertex_check_weights is None else {
        int(k): int(v) for k, v in vertex_check_weights.items()
    }
    added = []
    history = []
    rng = np.random.default_rng(seed)

    for round_id in range(int(max_rounds) + 1):
        estimate = compile_estimate_relative_expansion(
            edge_qubits,
            port_vertex_qubits,
            template,
            samples=samples,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        history.append({
            "round": round_id,
            "beta": estimate["beta"],
            "bad_port_vertices": estimate["bad_port_vertices"],
        })
        if estimate["beta"] >= threshold or round_id == max_rounds:
            return edge_qubits, added, history, estimate

        new_edges = []
        for v in estimate["bad_port_vertices"]:
            candidates = compile_unused_incident_edges(
                v,
                edge_qubits + new_edges,
                allowed_vertices,
                template,
                vertex_check_weights=vertex_check_weights,
                max_vertex_weight=max_vertex_weight,
            )
            if candidates:
                q = int(rng.choice(candidates))
                new_edges.append(q)
                if vertex_check_weights is not None:
                    a, b = template.edge_qubit_to_vertices[q]
                    vertex_check_weights[int(a)] = vertex_check_weights.get(int(a), 0) + 1
                    vertex_check_weights[int(b)] = vertex_check_weights.get(int(b), 0) + 1
        if not new_edges:
            return edge_qubits, added, history, estimate

        added.extend(new_edges)
        edge_qubits = sorted(set(edge_qubits + new_edges))

    return edge_qubits, added, history, estimate


def compile_add_edge_columns_to_vertex_checks(
    H_vertex,
    edge_qubits,
    template,
    vertex_row_to_check_qubit,
    *,
    allow_new_vertices=False,
):
    """Update vertex-check rows when repair edges are added to the fixed graph."""
    H_vertex = np.asarray(H_vertex, dtype=np.uint8).copy()
    qubit_to_row = {int(q): int(row) for row, q in vertex_row_to_check_qubit.items()}
    for q in edge_qubits:
        a, b = template.edge_qubit_to_vertices[int(q)]
        for vertex in (int(a), int(b)):
            if vertex in qubit_to_row:
                continue
            if not allow_new_vertices:
                raise ValueError(f"repair edge {q} touches a vertex outside the placed graph")
            row = H_vertex.shape[0]
            H_vertex = np.vstack([H_vertex, np.zeros(H_vertex.shape[1], dtype=np.uint8)])
            vertex_row_to_check_qubit[int(row)] = vertex
            qubit_to_row[vertex] = int(row)
        H_vertex[qubit_to_row[int(a)], int(q)] ^= 1
        H_vertex[qubit_to_row[int(b)], int(q)] ^= 1
    return H_vertex

def compile_row_to_ancilla_map(qubits, n_rows):
    if n_rows > len(qubits):
        raise ValueError(f"need {n_rows} ancillas, but only have {len(qubits)}")
    return {row: int(qubits[row]) for row in range(n_rows)}


def compile_gf2_row_combination_for_target(rows, target):
    """Return coeffs such that coeffs @ rows == target over GF(2)."""
    rows = np.asarray(rows, dtype=np.uint8) % 2
    target = np.asarray(target, dtype=np.uint8) % 2
    m, n = rows.shape
    if target.shape[0] != n:
        raise ValueError(f"target length {target.shape[0]} does not match row width {n}")

    # Solve rows.T @ coeffs = target.
    A = np.concatenate([rows.T.copy(), target.reshape(n, 1)], axis=1)
    rank = 0
    pivots = []
    for col in range(m):
        pivot = None
        for r in range(rank, n):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(n):
            if r != rank and A[r, col]:
                A[r] ^= A[rank]
        pivots.append(col)
        rank += 1

    for r in range(rank, n):
        if A[r, -1]:
            raise ValueError("target logical is not in the basis-check row span")

    coeffs = np.zeros(m, dtype=np.uint8)
    for r, col in enumerate(pivots):
        coeffs[col] = A[r, -1]
    return coeffs


def compile_basis_is_x(basis):
    if basis is None:
        return False
    name = getattr(basis, "name", None)
    if name is not None:
        return str(name).upper() == "X"
    return str(basis).upper().endswith("X") or str(basis).upper() == "X"


def compile_skiptree_port_order(g, port_vertices, root=None):
    """SkipTree ordering restricted to port vertices.

    The BFS/tree traversal is over the full auxiliary graph, so paths may pass
    through internal non-port vertices.  Only vertices in ``port_vertices`` are
    assigned port labels.
    """
    port_vertices = [int(v) for v in port_vertices]
    port_set = set(port_vertices)
    if not port_vertices:
        return {
            "port_tree_rows": [],
            "port_vertex_to_label": {},
            "label_to_port_vertex": [],
            "tree_edge_ids": [],
        }
    if g.is_directed():
        raise ValueError("skip-tree port ordering requires an undirected graph")

    root = int(port_vertices[0] if root is None else root)
    n = g.vcount()
    if not (0 <= root < n):
        raise ValueError("skip-tree root is not a valid graph vertex")

    adj = [[] for _ in range(n)]
    for eid, edge in enumerate(g.es):
        u, v = edge.tuple
        adj[int(u)].append((int(v), int(eid)))
        adj[int(v)].append((int(u), int(eid)))

    parent = [-1] * n
    parent_edge = [-1] * n
    children = [[] for _ in range(n)]
    seen = [False] * n
    q = deque([root])
    seen[root] = True
    while q:
        v = q.popleft()
        for w, eid in adj[v]:
            if not seen[w]:
                seen[w] = True
                parent[w] = v
                parent_edge[w] = eid
                children[v].append(w)
                q.append(w)

    missing = sorted(port_set - {v for v, ok in enumerate(seen) if ok})
    if missing:
        raise ValueError(f"port vertices are not connected to skip-tree root: {missing}")

    label_to_port_vertex = []
    port_vertex_to_label = {}

    def label_if_port(v):
        if v not in port_set or v in port_vertex_to_label:
            return
        port_vertex_to_label[v] = len(label_to_port_vertex)
        label_to_port_vertex.append(v)

    def label_first(v, skip=False):
        label_if_port(v)
        for idx, child in enumerate(children[v]):
            is_youngest = idx == len(children[v]) - 1
            if is_youngest and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v):
        for child in children[v]:
            label_first(child, skip=True)
        label_if_port(v)

    label_first(root, skip=False)
    if set(label_to_port_vertex) != port_set:
        missing = sorted(port_set - set(label_to_port_vertex))
        raise ValueError(f"skip-tree did not label all port vertices: {missing}")

    def tree_path_edges(a, b):
        path_a = []
        path_b = []
        ancestors_a = {}
        x = int(a)
        while x != -1:
            ancestors_a[x] = len(path_a)
            if parent[x] != -1:
                path_a.append(parent_edge[x])
            x = parent[x]

        y = int(b)
        while y not in ancestors_a:
            path_b.append(parent_edge[y])
            y = parent[y]
        return path_a[:ancestors_a[y]] + path_b

    port_tree_rows = [
        set(tree_path_edges(label_to_port_vertex[i], label_to_port_vertex[i + 1]))
        for i in range(len(label_to_port_vertex) - 1)
    ]
    tree_edge_ids = [int(eid) for eid in parent_edge if eid != -1]
    return {
        "port_tree_rows": port_tree_rows,
        "port_vertex_to_label": port_vertex_to_label,
        "label_to_port_vertex": label_to_port_vertex,
        "tree_edge_ids": tree_edge_ids,
    }


def compile_port_order_from_deformation(deformation_result):
    logical = np.asarray(deformation_result["logical"], dtype=np.uint8)
    q_to_v = {
        int(q): int(v)
        for q, v in deformation_result["qubit_to_vertex"].items()
        if int(q) < len(logical) and logical[int(q)]
    }
    v_to_q = {v: q for q, v in q_to_v.items()}
    skip = compile_skiptree_port_order(deformation_result["g"], list(q_to_v.values()))
    return {
        **skip,
        "port_vertex_to_logical_qubit": v_to_q,
        "logical_qubit_to_port_vertex": q_to_v,
    }

def compile_template_shape_for_layout(layout, min_rows=2, min_cols=2):
    """Smallest alternating-row triangular template that contains the layout sites."""
    used_sites = sorted(compile_used_layout_sites(layout), key=lambda s: (s[0], s[1]))
    if not used_sites:
        return max(2, min_rows), max(2, min_cols)

    used_rows = [s[0] for s in used_sites]
    row_span = max(used_rows) - min(used_rows)
    rows = max(int(min_rows), row_span + 1)

    best_cols = None
    for cols in range(max(2, int(min_cols)), max(2, int(min_cols)) + 64):
        template = fixed_triangular_template(
            rows=rows,
            cols=cols,
            ports=1,
            n_data=int(layout["n_data_qubits"]),
            n_basis_checks=int(np.asarray(layout["H_basis_old"]).shape[0]),
            n_opposite_checks=int(np.asarray(layout["H_opposite_basis_old_padded"]).shape[0]),
        )
        try:
            compile_fit_layout_sites_to_template(layout, template)
        except ValueError:
            continue
        best_cols = cols
        break
    if best_cols is None:
        raise ValueError("could not find a triangular gadget size that fits the layout")
    return rows, best_cols


def compile_make_triangular_layout(deformation_result, **tri_layout_kwargs):
    """Run the abstract deformation through the triangular-layout stage."""
    tri_layout_kwargs.pop("plot", None)
    return tri_layout(deformation_result, **tri_layout_kwargs)





def compile_long_abstract_route_count(layout, shuttling_threshold):
    if shuttling_threshold is None:
        return 0
    threshold = int(shuttling_threshold)
    return sum(
        1
        for path in layout.get("routes", {}).values()
        if len(path) >= 2 and len(path) - 1 > threshold
    )


def compile_old_opposite_edge_components(deformation_result, row):
    M = np.asarray(deformation_result["H_opposite_basis_old_padded"], dtype=np.uint8) % 2
    n_data = int(deformation_result["n_original_qubits"])
    edge_cols = np.flatnonzero(M[int(row), n_data:]).astype(int).tolist()
    graph_edges = layout_graph_edge_order(deformation_result)
    G = nx.Graph()
    for col in edge_cols:
        if col >= len(graph_edges):
            continue
        a, b = graph_edges[col]
        G.add_edge(int(a), int(b), edge_col=int(col))
    components = []
    for vertices in nx.connected_components(G):
        H = G.subgraph(vertices)
        cols = [int(data["edge_col"]) for _, _, data in H.edges(data=True)]
        odd = [int(v) for v, degree in H.degree() if degree % 2]
        components.append({"edge_cols": sorted(cols), "odd_vertices": sorted(odd)})
    return components




def compile_old_opposite_two_endpoint_component_count(layout):
    """Conservative shuttle-capacity bound for old opposite components.

    Whether an old opposite component is long can change after abstract routes
    are replaced by shuttles.  Reserving one possible shuttle per two-endpoint
    component avoids running out of side qubits during that later compression
    pass.  Unused reserved qubits remain outside the returned H supports.
    """
    M = np.asarray(layout["H_opposite_basis_old_padded"], dtype=np.uint8)
    count = 0
    for row in range(M.shape[0]):
        for component in compile_old_opposite_edge_components(layout["deformation_result"], row):
            if len(component["odd_vertices"]) == 2:
                count += 1
    return count


def compile_place_shuttling_edge_coords(template, records):
    if not records:
        return
    floor_vertices = list(template.vertex_check_qubits)
    max_x = max(template.coords[q][0] for q in floor_vertices)
    min_y = min(template.coords[q][1] for q in floor_vertices)
    max_y = max(template.coords[q][1] for q in floor_vertices)
    spacing = 0.48
    column_gap = 0.75
    max_per_column = max(8, int((max_y - min_y) / spacing) + 1)
    n_stack = max(len(records), len(template.shuttling_cycle_check_qubits))
    for i in range(n_stack):
        column, row = divmod(i, max_per_column)
        base_x = float(max_x + 2.5 + column_gap * column)
        base_y = float(min_y + spacing * row)
        if i < len(template.shuttling_cycle_check_qubits):
            template.coords[int(template.shuttling_cycle_check_qubits[i])] = (base_x + 0.32, base_y)
    for i, record in enumerate(records):
        q = int(record["qubit"])
        column, row = divmod(i, max_per_column)
        template.coords[q] = (
            float(max_x + 2.5 + column_gap * column),
            float(min_y + spacing * row),
        )


def compile_fixed_floor_graph(template, allowed_vertices=None):
    allowed = None if allowed_vertices is None else set(map(int, allowed_vertices))
    G = nx.Graph()
    for q, pair in template.edge_qubit_to_vertices.items():
        q = int(q)
        if q not in template.edge_qubits:
            continue
        a, b = map(int, pair)
        if allowed is not None and (a not in allowed or b not in allowed):
            continue
        G.add_edge(a, b, qubit=q)
    return G


def compile_shortest_floor_edge_path(template, start, stop, allowed_vertices, forbidden_edges=None):
    start = int(start)
    stop = int(stop)
    if start == stop:
        return []
    forbidden_edges = set() if forbidden_edges is None else set(map(int, forbidden_edges))
    G = compile_fixed_floor_graph(template, allowed_vertices=allowed_vertices)
    for q in forbidden_edges:
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = map(int, template.edge_qubit_to_vertices[q])
        if G.has_edge(a, b):
            G.remove_edge(a, b)
    if start not in G or stop not in G:
        return None
    try:
        vertices = nx.shortest_path(G, start, stop)
    except nx.NetworkXNoPath:
        return None
    edges = []
    for a, b in zip(vertices, vertices[1:]):
        edges.append(int(G.edges[int(a), int(b)]["qubit"]))
    return edges


def compile_vertex_weights_from_rows(H_vertex, vertex_row_to_check_qubit):
    H_vertex = np.asarray(H_vertex, dtype=np.uint8)
    return {
        int(qubit): int(np.sum(H_vertex[int(row)]))
        for row, qubit in vertex_row_to_check_qubit.items()
    }


def compile_path_fits_vertex_weight_cap(path_edges, materialized_edges, vertex_weights, template, max_vertex_degree):
    if max_vertex_degree is None:
        return True
    materialized = set(map(int, materialized_edges))
    increments = {}
    for q in map(int, path_edges):
        if q in materialized:
            continue
        a, b = template.edge_qubit_to_vertices[q]
        increments[int(a)] = increments.get(int(a), 0) + 1
        increments[int(b)] = increments.get(int(b), 0) + 1
    for vertex, inc in increments.items():
        if vertex_weights.get(int(vertex), 0) + inc > int(max_vertex_degree):
            return False
    return True


def compile_max_vertex_weight_after_adding_path(path_edges, materialized_edges, vertex_weights, template):
    materialized = set(map(int, materialized_edges))
    increments = {}
    for q in map(int, path_edges):
        if q in materialized:
            continue
        a, b = template.edge_qubit_to_vertices[int(q)]
        increments[int(a)] = increments.get(int(a), 0) + 1
        increments[int(b)] = increments.get(int(b), 0) + 1
    best = max(vertex_weights.values(), default=0)
    for vertex, inc in increments.items():
        best = max(best, vertex_weights.get(int(vertex), 0) + inc)
    return int(best)


def compile_add_floor_edges_to_vertex_checks(
    H_vertex,
    edge_qubits,
    materialized_edges,
    template,
    vertex_row_to_check_qubit,
    *,
    allow_new_vertices=False,
):
    new_edges = sorted(set(map(int, edge_qubits)) - set(map(int, materialized_edges)))
    if not new_edges:
        return H_vertex, sorted(set(map(int, materialized_edges))), []
    H_vertex = compile_add_edge_columns_to_vertex_checks(
        H_vertex,
        new_edges,
        template,
        vertex_row_to_check_qubit,
        allow_new_vertices=allow_new_vertices,
    )
    return H_vertex, sorted(set(map(int, materialized_edges)) | set(new_edges)), new_edges


def compile_rebuild_vertex_checks_from_edges(
    H_vertex,
    floor_edge_qubits,
    shuttling_edge_qubits,
    template,
    vertex_row_to_check_qubit,
    n_data,
):
    """Rebuild gadget vertex rows from the materialized edge graph.
    This removes stale vertex-check rows left behind by floor routes that were
    replaced by shuttling.  Data/port support on original vertices is preserved.
    """
    H_vertex = np.asarray(H_vertex, dtype=np.uint8)
    width = H_vertex.shape[1]
    n_data = int(n_data)

    vertex_to_data = {}
    for row, vertex in vertex_row_to_check_qubit.items():
        row = int(row)
        vertex = int(vertex)
        data = np.flatnonzero(H_vertex[row, :n_data]).astype(int).tolist()
        if data:
            vertex_to_data.setdefault(vertex, set()).update(data)

    materialized_edges = sorted(set(map(int, floor_edge_qubits)) | set(map(int, shuttling_edge_qubits)))
    used_vertices = set(vertex_to_data)
    for q in materialized_edges:
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = template.edge_qubit_to_vertices[int(q)]
        used_vertices.add(int(a))
        used_vertices.add(int(b))

    ordered_vertices = sorted(
        used_vertices,
        key=lambda q: (*template.coords[int(q)], int(q)),
    )
    new_row_to_vertex = {row: int(vertex) for row, vertex in enumerate(ordered_vertices)}
    vertex_to_row = {int(vertex): int(row) for row, vertex in new_row_to_vertex.items()}

    out = np.zeros((len(ordered_vertices), width), dtype=np.uint8)
    for vertex, data in vertex_to_data.items():
        row = vertex_to_row[int(vertex)]
        out[row, sorted(data)] = 1
    for q in materialized_edges:
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = template.edge_qubit_to_vertices[int(q)]
        if int(a) in vertex_to_row:
            out[vertex_to_row[int(a)], int(q)] ^= 1
        if int(b) in vertex_to_row:
            out[vertex_to_row[int(b)], int(q)] ^= 1

    return out, new_row_to_vertex


def compile_choose_nearby_low_weight_vertex(
    start,
    allowed_vertices,
    template,
    vertex_weights,
    max_vertex_degree,
    max_distance,
    forbidden_edges=None,
):
    start = int(start)
    forbidden_edges = set() if forbidden_edges is None else set(map(int, forbidden_edges))
    G = compile_fixed_floor_graph(template, allowed_vertices=allowed_vertices)
    for q in forbidden_edges:
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = map(int, template.edge_qubit_to_vertices[q])
        if G.has_edge(a, b):
            G.remove_edge(a, b)
    if start not in G:
        return None, None
    lengths = nx.single_source_shortest_path_length(G, start, cutoff=max(1, int(max_distance)))
    candidates = []
    for vertex, distance in lengths.items():
        vertex = int(vertex)
        if vertex == start:
            continue
        if vertex_weights.get(vertex, 0) >= int(max_vertex_degree):
            continue
        path_edges = compile_shortest_floor_edge_path(
            template,
            start,
            vertex,
            allowed_vertices,
            forbidden_edges=forbidden_edges,
        )
        if path_edges is None:
            continue
        candidates.append((int(distance), vertex_weights.get(vertex, 0), vertex, path_edges))
    if not candidates:
        return None, None
    _, _, vertex, path_edges = min(candidates)
    return int(vertex), path_edges


def compile_build_shuttling_cycle_rows(
    template,
    H_basis_new,
    floor_edge_qubits_used,
    shuttling_edge_qubits_used,
    shuttling_records,
    vertex_row_to_check_qubit,
    *,
    max_vertex_degree=6,
    max_floor_path_length=None,
    forbidden_floor_edges=None,
):
    """Add explicit opposite-basis cycle rows that contain shuttling edges."""
    width = int(H_basis_new.shape[1])
    floor_edges = sorted(set(map(int, floor_edge_qubits_used)))
    shuttle_edges = list(map(int, shuttling_edge_qubits_used))
    spare_shuttles = [
        int(q) for q in template.shuttling_edge_qubits if int(q) not in set(shuttle_edges)
    ]
    spare_iter = iter(spare_shuttles)
    allowed_vertices = set(vertex_row_to_check_qubit.values())
    forbidden_floor_edges = set() if forbidden_floor_edges is None else set(map(int, forbidden_floor_edges))
    max_floor_path_length = (
        max(2, int(max_vertex_degree))
        if max_floor_path_length is None
        else int(max_floor_path_length)
    )

    rows = []
    metadata = []
    extra_shuttles = []
    extra_records = []

    def add_cycle_row(edge_support, meta):
        support = sorted(set(map(int, edge_support)))
        if not support:
            return
        row = np.zeros(width, dtype=np.uint8)
        row[support] = 1
        rows.append(row)
        metadata.append({**meta, "edge_qubits": support, "weight": len(support)})

    def existing_shuttle_closure(q1, v, w, vertex_weights):
        best = None
        for q2 in shuttle_edges:
            q2 = int(q2)
            if q2 == int(q1) or q2 not in template.edge_qubit_to_vertices:
                continue
            a, b = map(int, template.edge_qubit_to_vertices[q2])
            for pairing in [((v, a), (w, b)), ((v, b), (w, a))]:
                paths = []
                ok = True
                for start, stop in pairing:
                    path = compile_shortest_floor_edge_path(
                        template,
                        start,
                        stop,
                        allowed_vertices,
                        forbidden_edges=forbidden_floor_edges,
                    )
                    if path is None or len(path) > max_floor_path_length:
                        ok = False
                        break
                    paths.extend(path)
                if not ok:
                    continue
                if not compile_path_fits_vertex_weight_cap(
                    paths,
                    floor_edges,
                    vertex_weights,
                    template,
                    max_vertex_degree,
                ):
                    continue
                cand = (len(set(map(int, paths))), q2, sorted(set(map(int, paths))), pairing)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        return best

    for record in shuttling_records:
        q1 = int(record["qubit"])
        v, w = map(int, record["endpoint_vertices"])
        vertex_weights = compile_vertex_weights_from_rows(H_basis_new, vertex_row_to_check_qubit)

        if record.get("force_replaced_path_closure"):
            closure_support = sorted(
                set(map(int, record.get("replaced_floor_edge_qubits", [])))
                | set(map(int, record.get("endpoint_adjustment_floor_edges", [])))
            )
            closure_floor_edges = [q for q in closure_support if int(q) in set(map(int, template.edge_qubits))]
            H_basis_new, floor_edges, added_edges = compile_add_floor_edges_to_vertex_checks(
                H_basis_new,
                closure_floor_edges,
                floor_edges,
                template,
                vertex_row_to_check_qubit,
                allow_new_vertices=True,
            )
            add_cycle_row(
                [q1] + closure_support,
                {
                    "type": "shuttling cycle closed by replaced planar component",
                    "shuttling_qubit": q1,
                    "endpoint_vertices": (v, w),
                    "floor_path_edges": [int(q) for q in closure_support],
                    "added_floor_edges": added_edges,
                },
            )
            continue

        direct_path = compile_shortest_floor_edge_path(
            template,
            v,
            w,
            allowed_vertices,
            forbidden_edges=forbidden_floor_edges,
        )
        direct_ok = (
            direct_path is not None
            and len(direct_path) <= max_floor_path_length
            and compile_path_fits_vertex_weight_cap(
                direct_path,
                floor_edges,
                vertex_weights,
                template,
                max_vertex_degree,
            )
        )
        if direct_ok:
            H_basis_new, floor_edges, added_edges = compile_add_floor_edges_to_vertex_checks(
                H_basis_new,
                direct_path,
                floor_edges,
                template,
                vertex_row_to_check_qubit,
                allow_new_vertices=True,
            )
            add_cycle_row(
                [q1] + direct_path,
                {
                    "type": "shuttling cycle closed by floor path",
                    "shuttling_qubit": q1,
                    "endpoint_vertices": (v, w),
                    "floor_path_edges": [int(q) for q in direct_path],
                    "added_floor_edges": added_edges,
                },
            )
            continue

        reusable = existing_shuttle_closure(q1, v, w, vertex_weights)
        if reusable is not None:
            _, q2, reusable_paths, pairing = reusable
            H_basis_new, floor_edges, added_edges = compile_add_floor_edges_to_vertex_checks(
                H_basis_new,
                reusable_paths,
                floor_edges,
                template,
                vertex_row_to_check_qubit,
                allow_new_vertices=True,
            )
            add_cycle_row(
                [q1, q2] + reusable_paths,
                {
                    "type": "shuttling cycle closed by existing shuttle",
                    "shuttling_qubit": q1,
                    "second_shuttling_qubit": q2,
                    "endpoint_vertices": (v, w),
                    "second_endpoint_vertices": tuple(map(int, template.edge_qubit_to_vertices[q2])),
                    "floor_path_edges": [int(q) for q in reusable_paths],
                    "floor_path_pairing": [
                        (int(a), int(b)) for a, b in pairing
                    ],
                    "added_floor_edges": added_edges,
                },
            )
            continue

        try:
            q2 = int(next(spare_iter))
        except StopIteration:
            q2 = None

        made_two_shuttle_cycle = False
        if q2 is not None:
            high, low = (v, w)
            if vertex_weights.get(w, 0) > vertex_weights.get(v, 0):
                high, low = (w, v)

            u, path_high = compile_choose_nearby_low_weight_vertex(
                high,
                allowed_vertices,
                template,
                vertex_weights,
                max_vertex_degree,
                max_floor_path_length,
                forbidden_edges=forbidden_floor_edges,
            )
            if vertex_weights.get(low, 0) < int(max_vertex_degree):
                p = low
                path_low = []
            else:
                p, path_low = compile_choose_nearby_low_weight_vertex(
                    low,
                    allowed_vertices,
                    template,
                    vertex_weights,
                    max_vertex_degree,
                    max_floor_path_length,
                    forbidden_edges=forbidden_floor_edges,
                )

            if u is not None and p is not None and u != p:
                candidate_paths = list(path_high or []) + list(path_low or [])
                endpoints_fit = (
                    vertex_weights.get(int(u), 0) + 1 <= int(max_vertex_degree)
                    and vertex_weights.get(int(p), 0) + 1 <= int(max_vertex_degree)
                )
                paths_fit = compile_path_fits_vertex_weight_cap(
                    candidate_paths,
                    floor_edges,
                    vertex_weights,
                    template,
                    max_vertex_degree,
                )
                if endpoints_fit and paths_fit:
                    H_basis_new, floor_edges, added_edges = compile_add_floor_edges_to_vertex_checks(
                        H_basis_new,
                        candidate_paths,
                        floor_edges,
                        template,
                        vertex_row_to_check_qubit,
                        allow_new_vertices=True,
                    )
                    pair = tuple(sorted((int(u), int(p))))
                    template.edge_qubit_to_vertices[q2] = pair
                    for endpoint in pair:
                        row = {int(q): int(r) for r, q in vertex_row_to_check_qubit.items()}[endpoint]
                        H_basis_new[row, q2] ^= 1
                    extra_shuttles.append(q2)
                    shuttle_edges.append(q2)
                    record2 = {
                        "qubit": q2,
                        "type": "cycle_closing_shuttle",
                        "paired_with": q1,
                        "endpoint_vertices": pair,
                    }
                    extra_records.append(record2)
                    add_cycle_row(
                        [q1, q2] + candidate_paths,
                        {
                            "type": "shuttling cycle closed by second shuttle",
                            "shuttling_qubit": q1,
                            "second_shuttling_qubit": q2,
                            "endpoint_vertices": (v, w),
                            "second_endpoint_vertices": pair,
                            "floor_path_edges": [int(q) for q in candidate_paths],
                            "added_floor_edges": added_edges,
                        },
                    )
                    made_two_shuttle_cycle = True

        if made_two_shuttle_cycle:
            continue

        # Last-resort closure: keep a shortest floor path so the shuttle is part
        # of a deterministic cycle row even when it violates the preferred caps.
        fallback = direct_path if direct_path is not None else list(record.get("replaced_floor_edge_qubits", []))
        H_basis_new, floor_edges, added_edges = compile_add_floor_edges_to_vertex_checks(
            H_basis_new,
            fallback,
            floor_edges,
            template,
            vertex_row_to_check_qubit,
            allow_new_vertices=True,
        )
        add_cycle_row(
            [q1] + fallback,
            {
                "type": "shuttling cycle fallback floor closure",
                "shuttling_qubit": q1,
                "endpoint_vertices": (v, w),
                "floor_path_edges": [int(q) for q in fallback],
                "added_floor_edges": added_edges,
            },
        )

    compile_place_shuttling_edge_coords(template, shuttling_records + extra_records)
    H = np.vstack(rows).astype(np.uint8) if rows else np.zeros((0, width), dtype=np.uint8)
    return H_basis_new, H, metadata, floor_edges, extra_shuttles, extra_records


def compile_basis_ancilla_maps(template, n_basis_rows, n_opposite_rows):
    return {
        "BB_basis_to_ancilla": compile_row_to_ancilla_map(template.basis_check_qubits, n_basis_rows),
        "BB_opposite_to_ancilla": compile_row_to_ancilla_map(template.opposite_check_qubits, n_opposite_rows),
    }


def compile_logical_observable_data(layout, H_basis_fixed, H_basis_new_fixed, n_basis_old_rows, vertex_row_to_check_qubit, width):
    n_data = int(layout["n_data_qubits"])
    logical_full = np.zeros(width, dtype=np.uint8)
    logical = np.asarray(layout["logical_full"][:n_data], dtype=np.uint8)
    logical_full[:len(logical)] = logical

    vertex_coeffs = compile_gf2_row_combination_for_target(H_basis_new_fixed, logical_full)
    vertex_rows = np.flatnonzero(vertex_coeffs).astype(int)
    basis_rows = (n_basis_old_rows + vertex_rows).astype(int)

    row_coeffs = np.zeros(H_basis_fixed.shape[0], dtype=np.uint8)
    row_coeffs[basis_rows] = 1
    vertex_ancillas = [vertex_row_to_check_qubit[int(row)] for row in vertex_rows]

    return {
        "logical_full": logical_full,
        "logical_observable_row_coeffs": row_coeffs,
        "logical_observable_vertex_row_coeffs": vertex_coeffs,
        "logical_observable_basis_rows": basis_rows,
        "logical_observable_vertex_rows": vertex_rows,
        "logical_observable_vertex_ancillas": vertex_ancillas,
        "logical_observable_ancillas": vertex_ancillas,
    }



def compile_compile_layout_to_fixed_gadget(
    layout,
    template,
    *,
    port_order=None,
    expansion_threshold=1.0,
    expansion_samples=50_000,
    expansion_max_rounds=5,
    shuttling_threshold=None,
    max_vertex_degree=6,
    shuttling_cycle_max_floor_path_length=None,
    floor_cycle_max_weight=None,
):
    """Map a triangular layout onto the fixed 8-row Stim gadget indexing."""
    n_data = int(layout["n_data_qubits"])
    width = int(template.n_h_qubits)

    site_to_fixed_vertex, lattice_translation = compile_fit_layout_sites_to_template(layout, template)
    (
        optimized_vertex_placement,
        floor_routes,
        shuttled_abstract_edges,
        floor_edge_qubits_used,
    ) = compile_reroute_floor_after_abstract_shuttling(
        layout,
        template,
        site_to_fixed_vertex,
        shuttling_threshold,
    )
    shuttling_vertex_weights = compile_initial_vertex_weights_for_shuttling(
        layout["deformation_result"],
        template,
        optimized_vertex_placement,
        floor_edge_qubits_used,
    )
    abstract_shuttling_edge_qubits, abstract_shuttling_records = compile_allocate_abstract_route_shuttles(
        layout["deformation_result"],
        template,
        optimized_vertex_placement,
        shuttled_abstract_edges,
        vertex_weights=shuttling_vertex_weights,
        max_vertex_degree=max_vertex_degree,
    )
    abstract_adjustment_floor_edges = sorted({
        int(q)
        for record in abstract_shuttling_records
        for q in record.get("endpoint_adjustment_floor_edges", [])
    })
    floor_edge_qubits_used = sorted(set(floor_edge_qubits_used) | set(abstract_adjustment_floor_edges))
    H_basis_old = compile_expand_old_basis_on_fixed_routes(
        layout["deformation_result"],
        floor_routes,
        abstract_shuttling_records,
        width,
    )
    H_opp_old = compile_expand_old_matrix_on_fixed_routes(
        layout["deformation_result"],
        "H_opposite_basis_old_padded",
        floor_routes,
        abstract_shuttling_records,
        width,
    )
    shuttling_edge_qubits_used = sorted(set(abstract_shuttling_edge_qubits))
    shuttling_records = list(abstract_shuttling_records)

    basis_new_row_offset = int(layout["basis_new_row_offset"])
    shuttling_removed_floor_edges = []
    H_basis_new, vertex_row_to_check_qubit = compile_seed_vertex_rows_from_placement(
        layout["deformation_result"],
        template,
        optimized_vertex_placement,
        width,
    )
    H_basis_new, vertex_row_to_check_qubit = compile_rebuild_vertex_checks_from_edges(
        H_basis_new,
        floor_edge_qubits_used,
        shuttling_edge_qubits_used,
        template,
        vertex_row_to_check_qubit,
        n_data,
    )
    component_vertex_weights = compile_vertex_weights_from_rows(H_basis_new, vertex_row_to_check_qubit)
    (
        H_opp_old,
        floor_edge_qubits_used,
        shuttling_edge_qubits_used,
        component_shuttling_records,
    ) = compile_compress_old_opposite_components_with_shuttles(
        layout["deformation_result"],
        template,
        optimized_vertex_placement,
        H_opp_old,
        floor_routes,
        abstract_shuttling_records,
        floor_edge_qubits_used,
        shuttling_edge_qubits_used,
        component_vertex_weights,
        shuttling_threshold=shuttling_threshold,
        max_vertex_degree=max_vertex_degree,
    )
    if component_shuttling_records:
        shuttling_records.extend(component_shuttling_records)
        H_basis_new, vertex_row_to_check_qubit = compile_rebuild_vertex_checks_from_edges(
            H_basis_new,
            floor_edge_qubits_used,
            shuttling_edge_qubits_used,
            template,
            vertex_row_to_check_qubit,
            n_data,
        )
    allowed_vertices = set(vertex_row_to_check_qubit.values())
    port_vertex_qubits = [
        int(qubit)
        for row, qubit in vertex_row_to_check_qubit.items()
        if np.any(H_basis_new[int(row), :n_data])
    ]
    vertex_check_weights = {
        int(qubit): int(np.sum(H_basis_new[int(row)]))
        for row, qubit in vertex_row_to_check_qubit.items()
    }
    edge_qubits_for_expansion = sorted(set(floor_edge_qubits_used) | set(shuttling_edge_qubits_used))
    fixed_edge_qubits_used, expansion_added_edges, expansion_history, expansion_estimate = compile_repair_relative_expansion(
        edge_qubits_for_expansion,
        port_vertex_qubits,
        allowed_vertices,
        template,
        threshold=expansion_threshold,
        samples=expansion_samples,
        seed=0,
        max_rounds=expansion_max_rounds,
        vertex_check_weights=vertex_check_weights,
        max_vertex_weight=max_vertex_degree,
    )
    floor_edge_qubits_used = sorted(set(floor_edge_qubits_used) | set(expansion_added_edges))
    H_basis_new = compile_add_edge_columns_to_vertex_checks(
        H_basis_new,
        expansion_added_edges,
        template,
        vertex_row_to_check_qubit,
    )
    fixed_edge_qubits_used = sorted(set(floor_edge_qubits_used) | set(shuttling_edge_qubits_used))
    H_cycles, cycle_sources = compile_recompute_cycle_basis_from_fixed_edges(fixed_edge_qubits_used, template, width=width)
    (
        H_basis_new,
        H_cycles,
        cycle_sources,
        floor_edge_qubits_used,
    ) = compile_split_floor_cycle_supports(
        H_basis_new,
        cycle_sources,
        floor_edge_qubits_used,
        template,
        vertex_row_to_check_qubit,
        width,
        max_cycle_weight=max_vertex_degree if floor_cycle_max_weight is None else floor_cycle_max_weight,
        max_vertex_degree=max_vertex_degree,
    )
    H_floor_cycles, floor_cycle_sources, H_shuttling_cycles, shuttling_cycle_sources = compile_partition_cycle_rows_by_shuttles(
        H_cycles,
        cycle_sources,
        shuttling_edge_qubits_used,
    )
    H_basis_new, vertex_row_to_check_qubit = compile_rebuild_vertex_checks_from_edges(
        H_basis_new,
        floor_edge_qubits_used,
        shuttling_edge_qubits_used,
        template,
        vertex_row_to_check_qubit,
        n_data,
    )
    global_vertex_row_to_check_qubit = {
        int(basis_new_row_offset + row): int(qubit)
        for row, qubit in vertex_row_to_check_qubit.items()
    }
    fixed_edge_qubits_used = sorted(set(floor_edge_qubits_used) | set(shuttling_edge_qubits_used))
    H_basis_def = np.vstack([H_basis_old, H_basis_new]) % 2
    if H_shuttling_cycles.shape[0]:
        H_opp_new = np.vstack([H_floor_cycles, H_shuttling_cycles]).astype(np.uint8)
        cycle_sources = floor_cycle_sources + shuttling_cycle_sources
    else:
        H_opp_new = H_floor_cycles
        cycle_sources = floor_cycle_sources
    if H_floor_cycles.shape[0] > len(template.cycle_check_qubits):
        raise ValueError(
            f"layout needs {H_floor_cycles.shape[0]} floor cycle checks, "
            f"fixed template has {len(template.cycle_check_qubits)}"
        )
    if H_shuttling_cycles.shape[0] > len(template.shuttling_cycle_check_qubits):
        raise ValueError(
            f"layout needs {H_shuttling_cycles.shape[0]} shuttling cycle checks, "
            f"fixed template has {len(template.shuttling_cycle_check_qubits)}"
        )
    H_opposite_def = np.vstack([H_opp_old, H_opp_new]) % 2

    cycle_row_to_check_qubits, cycle_row_to_check_qubit = compile_map_opposite_rows_to_cycle_ancillas(
        H_floor_cycles,
        H_shuttling_cycles,
        template,
    )
    graph_edges = layout_graph_edge_order(layout["deformation_result"])
    shuttle_by_edge = {
        tuple(record["abstract_edge"]): [int(record["qubit"])]
        for record in abstract_shuttling_records
    }
    old_to_fixed = {q: q for q in range(n_data)}
    for col, edge in enumerate(graph_edges):
        old_to_fixed[n_data + col] = list(floor_routes.get(edge, shuttle_by_edge.get(edge, [])))

    logical_support = np.flatnonzero(np.asarray(layout["logical_full"][:n_data], dtype=np.uint8))
    if port_order is None:
        label_to_port_vertex = []
        port_vertex_to_logical_qubit = {}
        ordered_port_qubits = template.port_qubits[: len(logical_support)]
        logical_support_to_port_qubit = {
            int(q): template.port_qubits[i]
            for i, q in enumerate(logical_support[: len(template.port_qubits)])
        }
    else:
        label_to_port_vertex = [int(v) for v in port_order["label_to_port_vertex"]]
        port_vertex_to_logical_qubit = {
            int(v): int(q)
            for v, q in port_order["port_vertex_to_logical_qubit"].items()
        }
        ordered_port_qubits = template.port_qubits[: len(label_to_port_vertex)]
        logical_support_to_port_qubit = {
            port_vertex_to_logical_qubit[int(v)]: int(ordered_port_qubits[label])
            for label, v in enumerate(label_to_port_vertex)
            if int(v) in port_vertex_to_logical_qubit
        }
    port_label_to_qubit = {
        int(label): int(q)
        for label, q in enumerate(ordered_port_qubits)
    }
    port_label_to_vertex = {
        int(label): int(v)
        for label, v in enumerate(label_to_port_vertex)
    }
    port_label_to_logical_support_qubit = {
        int(label): int(port_vertex_to_logical_qubit[int(v)])
        for label, v in enumerate(label_to_port_vertex)
        if int(v) in port_vertex_to_logical_qubit
    }
    fixed_vertex_to_local_row = {
        int(qubit): int(row)
        for row, qubit in vertex_row_to_check_qubit.items()
    }
    original_vertex_to_site = layout.get(
        "original_vertex_to_site",
        layout.get("state", {}).get("original_vertex_to_site", {}),
    )
    port_label_to_vertex_row = {}
    for label, v in enumerate(label_to_port_vertex):
        fixed_vertex = optimized_vertex_placement.get(int(v))
        if fixed_vertex is None:
            site = original_vertex_to_site.get(int(v))
            fixed_vertex = None if site is None else site_to_fixed_vertex.get(site)
        if fixed_vertex is None:
            continue
        row = fixed_vertex_to_local_row.get(int(fixed_vertex))
        if row is not None:
            port_label_to_vertex_row[int(label)] = int(row)

    unused_edge_qubits = sorted(set(template.edge_qubits) - set(floor_edge_qubits_used))
    auxiliary_graph_data = compile_fixed_auxiliary_graph_data(fixed_edge_qubits_used, template)
    vertex_qubit_to_vertex_id = auxiliary_graph_data["vertex_qubit_to_vertex_id"]
    abstract_vertex_to_fixed_vertex_qubit = {
        int(v): int(q)
        for v, q in optimized_vertex_placement.items()
        if int(q) in vertex_qubit_to_vertex_id
    }
    abstract_vertex_to_measurement_vertex_id = {
        int(v): int(vertex_qubit_to_vertex_id[int(q)])
        for v, q in abstract_vertex_to_fixed_vertex_qubit.items()
    }
    data_qubit_to_vertex_qubit = {}
    data_qubit_to_vertex_id = {}
    data_qubit_to_vertex_qubits = {}
    data_qubit_to_vertex_ids = {}
    vertex_to_qubit_items = layout["deformation_result"].get("vertex_to_qubit", {})
    if vertex_to_qubit_items:
        items = [(int(v), int(q)) for v, q in vertex_to_qubit_items.items()]
    else:
        items = [(int(v), int(q)) for q, v in layout["deformation_result"].get("qubit_to_vertex", {}).items()]
    for v, q in items:
        if q >= n_data or v not in abstract_vertex_to_fixed_vertex_qubit:
            continue
        fixed_vertex = abstract_vertex_to_fixed_vertex_qubit[v]
        vertex_id = int(vertex_qubit_to_vertex_id[fixed_vertex])
        data_qubit_to_vertex_qubit.setdefault(q, fixed_vertex)
        data_qubit_to_vertex_id.setdefault(q, vertex_id)
        data_qubit_to_vertex_qubits.setdefault(q, []).append(fixed_vertex)
        data_qubit_to_vertex_ids.setdefault(q, []).append(vertex_id)

    port_label_to_measurement_vertex_id = {
        int(label): int(abstract_vertex_to_measurement_vertex_id[int(v)])
        for label, v in port_label_to_vertex.items()
        if int(v) in abstract_vertex_to_measurement_vertex_id
    }
    logical_support_to_measurement_vertex_id = {
        int(q): int(data_qubit_to_vertex_id[int(q)])
        for q in logical_support
        if int(q) in data_qubit_to_vertex_id
    }
    return {
        "n_data": n_data,
        "n_edges": len(fixed_edge_qubits_used),
        "n_floor_edges": len(floor_edge_qubits_used),
        "n_shuttling_edges": len(shuttling_edge_qubits_used),
        "n_cycles": int(H_opp_new.shape[0]),
        "n_vertices": len(auxiliary_graph_data["auxiliary_graph_incidence_vertices"]),
        "n_edge_capacity": len(template.edge_qubits) + len(template.shuttling_edge_qubits),
        "n_floor_edge_capacity": len(template.edge_qubits),
        "n_shuttling_edge_capacity": len(template.shuttling_edge_qubits),
        "n_vertex_capacity": len(template.vertex_check_qubits),
        "n_cycle_capacity": len(template.cycle_check_qubits),
        "n_shuttling_cycle_capacity": len(template.shuttling_cycle_check_qubits),
        "n_h_qubits": width,
        "n_total_qubits": int(template.n_total_qubits),
        "BB_H_basis": H_basis_old,
        "BB_H_opposite": H_opp_old,
        "gadget_H_basis": H_basis_new,
        "gadget_H_opposite": H_opp_new,
        "H_basis": H_basis_def,
        "H_opposite_basis": H_opposite_def,
        "H_basis_def": H_basis_def,
        "H_opposite_basis_def": H_opposite_def,
        "H_basis_old": H_basis_old,
        "H_basis_new": H_basis_new,
        "H_opposite_basis_old_padded": H_opp_old,
        "H_opposite_basis_new": H_opp_new,
        "cycle_basis_source": "minimum_cycle_basis_after_triangular_lattice_placement",
        "cycle_rows_are_simple_boundaries": compile_cycle_rows_are_simple_boundaries(H_opp_new, template),
        "cycle_sources": cycle_sources,
        "relative_expansion_estimate": expansion_estimate["beta"],
        "relative_expansion_port_vertices": port_vertex_qubits,
        "relative_expansion_added_edges": expansion_added_edges,
        "relative_expansion_repair_max_vertex_weight": int(max_vertex_degree),
        "relative_expansion_history": expansion_history,
        "floor_edge_qubits_used": floor_edge_qubits_used,
        "shuttling_edge_qubits": shuttling_edge_qubits_used,
        "shuttling_records": shuttling_records,
        "shuttling_removed_floor_edge_qubits": shuttling_removed_floor_edges,
        "shuttling_threshold": None if shuttling_threshold is None else int(shuttling_threshold),
        "shuttling_cycle_sources": shuttling_cycle_sources,
        "shuttling_cycle_check_qubits": list(map(int, template.shuttling_cycle_check_qubits)),
        "shuttling_cycle_max_floor_path_length": (
            None
            if shuttling_cycle_max_floor_path_length is None
            else int(shuttling_cycle_max_floor_path_length)
        ),
        "floor_cycle_max_weight": int(max_vertex_degree if floor_cycle_max_weight is None else floor_cycle_max_weight),
        **auxiliary_graph_data,
        "abstract_vertex_to_fixed_vertex_qubit": abstract_vertex_to_fixed_vertex_qubit,
        "abstract_vertex_to_measurement_vertex_id": abstract_vertex_to_measurement_vertex_id,
        "data_qubit_to_vertex_qubit": data_qubit_to_vertex_qubit,
        "data_qubit_to_vertex_id": data_qubit_to_vertex_id,
        "data_qubit_to_vertex_qubits": data_qubit_to_vertex_qubits,
        "data_qubit_to_vertex_ids": data_qubit_to_vertex_ids,
        "qubit_to_vertex_id": data_qubit_to_vertex_id,
        "port_label_to_measurement_vertex_id": port_label_to_measurement_vertex_id,
        "logical_support_to_measurement_vertex_id": logical_support_to_measurement_vertex_id,
        "gadget_basis_to_ancilla": vertex_row_to_check_qubit,
        "gadget_opposite_to_ancillas": cycle_row_to_check_qubits,
        "port_qubits": list(map(int, ordered_port_qubits)),
        "port_label_to_qubit": port_label_to_qubit,
        "port_label_to_vertex": port_label_to_vertex,
        "port_label_to_logical_support_qubit": port_label_to_logical_support_qubit,
        "port_label_to_vertex_row": port_label_to_vertex_row,
        "port_tree_rows": [] if port_order is None else port_order["port_tree_rows"],
        "port_tree_edge_ids": [] if port_order is None else port_order["tree_edge_ids"],
        "logical_support_to_port_qubit": logical_support_to_port_qubit,
        "old_to_fixed_qubit": old_to_fixed,
        "vertex_sites": layout["vertex_sites"],
        "optimized_vertex_placement": optimized_vertex_placement,
        "site_to_fixed_vertex_check_qubit": site_to_fixed_vertex,
        "lattice_translation": lattice_translation,
        "fixed_edge_qubits_used": fixed_edge_qubits_used,
        "unused_fixed_edge_qubits": unused_edge_qubits,
        "geometric_rerouted_edges": {},
        "vertex_row_to_check_qubit": vertex_row_to_check_qubit,
        "global_vertex_row_to_check_qubit": global_vertex_row_to_check_qubit,
        "cycle_row_to_check_qubits": cycle_row_to_check_qubits,
        "cycle_row_to_check_qubit": cycle_row_to_check_qubit,
    }


def deform_logical_to_tri_lattice(
    deformation_result,
    basis=None,
    expansion_threshold=1.0,
    expansion_samples=50_000,
    expansion_max_rounds=5,
    shuttling_threshold=None,
    max_vertex_degree=6,
    shuttling_cycle_max_floor_path_length=None,
    floor_cycle_max_weight=None,
    **tri_layout_kwargs,
):
    """Run the full logical-deformation-to-fixed-triangular-gadget pipeline.

    The pipeline is:
        1. use ``tri_layout`` to planarize, place, and route the abstract
           auxiliary deformation on a triangular lattice;
        2. translate that placed lattice into the fixed 8-row Stim gadget;
        3. rebuild H matrices over BB data + gadget edge qubits only;
        4. return the measurement-ancilla maps needed by Stim circuits.

    The returned H matrices have columns ``0..n_data-1`` for code data qubits
    and ``n_data..n_h_qubits-1`` for gadget edge qubits. Check qubits and ports
    are returned as row-to-ancilla metadata.
    """
    tri_layout_kwargs.pop("plot", None)
    layout = compile_make_triangular_layout(deformation_result, **tri_layout_kwargs)
    n_data = int(layout["n_data_qubits"])
    n_basis_checks = int(np.asarray(layout["H_basis_old"]).shape[0])
    n_opposite_checks = int(np.asarray(layout["H_opposite_basis_old_padded"]).shape[0])
    logical_weight = int(np.sum(np.asarray(layout["logical_full"][:n_data], dtype=np.uint8)))
    port_order = compile_port_order_from_deformation(deformation_result)
    n_ports = max(1, len(port_order["label_to_port_vertex"]), logical_weight)
    rows, cols = compile_template_shape_for_layout(layout)
    n_abstract_shuttles = compile_long_abstract_route_count(layout, shuttling_threshold)
    n_old_opposite_shuttle_capacity = compile_old_opposite_two_endpoint_component_count(layout)
    n_long_paths = n_abstract_shuttles + n_old_opposite_shuttle_capacity
    n_shuttling_records_capacity = n_long_paths
    n_shuttling_edges = 2 * n_shuttling_records_capacity
    template = fixed_triangular_template(
        rows=rows,
        cols=cols,
        ports=n_ports,
        n_data=n_data,
        n_basis_checks=n_basis_checks,
        n_opposite_checks=n_opposite_checks,
        shuttling_edges=n_shuttling_edges,
        shuttling_cycle_checks=n_shuttling_records_capacity,
    )
    fixed = compile_compile_layout_to_fixed_gadget(
        layout,
        template,
        port_order=port_order,
        expansion_threshold=expansion_threshold,
        expansion_samples=expansion_samples,
        expansion_max_rounds=expansion_max_rounds,
        shuttling_threshold=shuttling_threshold,
        max_vertex_degree=max_vertex_degree,
        shuttling_cycle_max_floor_path_length=shuttling_cycle_max_floor_path_length,
        floor_cycle_max_weight=floor_cycle_max_weight,
    )

    fixed.update(compile_basis_ancilla_maps(
        template,
        fixed["BB_H_basis"].shape[0],
        fixed["BB_H_opposite"].shape[0],
    ))
    fixed.update(compile_logical_observable_data(
        layout,
        fixed["H_basis"],
        fixed["gadget_H_basis"],
        fixed["BB_H_basis"].shape[0],
        fixed["vertex_row_to_check_qubit"],
        fixed["n_h_qubits"],
    ))

    result = {
        "coords": template.coords,
        "fixed_template": template,
        "n_total_qubits": template.n_total_qubits,
        "n_h_qubits": template.n_h_qubits,
        "code_patch_rect": template.code_patch_rect,
        "code_patch_label": template.code_patch_label,
        "triangular_gadget_rows": rows,
        "triangular_gadget_cols": cols,
        "basis": "X" if compile_basis_is_x(basis) else "Z",
        "fixed_ordering_ok": True,
        **fixed,
    }
    return result


__all__ = [
    "FixedTriangularTemplate",
    "tri_layout",
    "fixed_triangular_template",
    "deform_logical_to_tri_lattice",
]
