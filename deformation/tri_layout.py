"""Triangular-lattice layout for logical-measurement auxiliary graphs.

``tri_layout`` is stage 1 of the triangular deformation.  It takes
the abstract auxiliary graph from ``deform_code_for_logical``, planarizes it,
places it on a triangular lattice using a Tutte-style embedding, routes graph
edges along lattice edges, and rebuilds the stabilizer matrices for that
routed lattice graph.
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
    """Pair logical-overlap vertices and route each pair on the lattice.

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
    """Updated the stablizers for the physical triangular lattice.

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

__all__ = ["tri_layout"]
