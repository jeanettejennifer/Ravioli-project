import numpy as np
from collections import defaultdict, deque
import igraph as ig
from sympy.abc import x, y
from qldpc import codes, circuits
from qldpc.objects import Pauli
import numpy as np

def deform_old_x_checks_with_graph_edges(
    g,
    Hx_all,
    logical_qubits,
    logical_qubits_index,
    n_data,
    n_edges,
    n_L):
    """
    Build the deformed old X-check matrix by adding ancilla-edge support.

    Assumes:
      - g has vertices [0..n_L-1] = logical-side vertices L_i
      - g has vertices [n_L..2*n_L-1] = check-side vertices C_i
      - ancilla qubit for graph edge eid lives at qubit index n_data + eid
      - an old X check that overlaps the chosen logical on exactly two BB qubits
        should gain X support on the ancilla qubit of the corresponding graph edge

    Returns:
      Hx_old_padded: filtered X-check rows, padded to n_data+n_edges qubits
                     and deformed by adding ancilla support
    """
    n_qubits = n_data + n_edges

    Hx_old = Hx_all[np.any(Hx_all, axis=1)]
    Hx_old_padded = np.pad(Hx_old, ((0, 0), (0, n_edges)), mode="constant")

    bb_to_logical_vertex = {int(q): i for i, q in enumerate(logical_qubits_index)}

    filtered_row = 0
    for full_row in range(Hx_all.shape[0]):
        row = Hx_all[full_row]
        if not np.any(row):
            continue

        overlap = np.where((logical_qubits == 1) & (row == 1))[0]

        if len(overlap) == 2:
            q1, q2 = map(int, overlap)
            l1 = bb_to_logical_vertex[q1]
            l2 = bb_to_logical_vertex[q2]

            u = n_L + l1
            v = n_L + l2

            eid = g.get_eid(u, v, directed=False, error=False)
            if eid == -1:
                raise ValueError(
                    f"No graph edge found between check vertices C{l1} and C{l2} "
                    f"for X-check row {full_row}."
                )
            Hx_old_padded[filtered_row, n_data + eid] = 1

        filtered_row += 1

    return Hx_old_padded

                                     
def find_short_cycle_basis(pairs, return_edges=True):
    """
    Greedy shortest cycle basis for an undirected simple graph.

    Args:
        pairs:
            Iterable of edges [(u,v), ...].
        return_edges:
            If False, return cycles as ordered vertex lists, like your old function.
            If True, return cycles as edge lists [(u,v), ...] in cyclic order.

    Returns:
        List of cycles, greedily chosen from shortest candidates first.
        The number of returned cycles is the cycle rank m - n + c.
    """
    adj = defaultdict(set)
    edges = set()
    for u, v in pairs:
        u, v = int(u), int(v)
        if u == v:
            continue
        e = tuple(sorted((u, v)))
        if e in edges:
            continue
        edges.add(e)
        a, b = e
        adj[a].add(b)
        adj[b].add(a)

    vertices = sorted(adj.keys())
    if not vertices:
        return []
    edge_list = sorted(edges)
    edge_index = {e: i for i, e in enumerate(edge_list)}

    def connected_components():
        seen = set()
        comps = 0
        for s in vertices:
            if s in seen:
                continue
            comps += 1
            q = [s]
            seen.add(s)
            while q:
                x = q.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
        return comps

    def shortest_path_avoiding_edge(src, dst, forbidden_edge):
        """
        BFS shortest path from src to dst avoiding one specific undirected edge.
        Returns a vertex path [src, ..., dst], or None if disconnected.
        """
        q = deque([src])
        parent = {src: None}

        while q:
            x = q.popleft()
            if x == dst:
                break
            for y in adj[x]:
                if tuple(sorted((x, y))) == forbidden_edge:
                    continue
                if y not in parent:
                    parent[y] = x
                    q.append(y)
        if dst not in parent:
            return None
        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def path_to_cycle_edges(path, closing_edge):
        """
        path = [v0, v1, ..., vk], closing_edge closes vk -> v0
        Returns ordered cycle edges.
        """
        cyc_edges = []
        for i in range(len(path) - 1):
            cyc_edges.append(tuple(sorted((path[i], path[i + 1]))))
        cyc_edges.append(tuple(sorted(closing_edge)))
        return cyc_edges

    def canonical_cycle_vertices(path):
        """
        Convert path [u,...,v] + closing edge (v,u) into cyclic vertex list
        without repeating the start vertex.
        """
        cyc = path[:]  
        return normalize_cycle_vertices(cyc)

    def normalize_cycle_vertices(cycle):
        """
        Canonicalize a cyclic vertex list up to rotation and reversal.
        Example: [5,7,9,6] equivalent to [7,9,6,5] and reversed versions.
        """
        c = list(cycle)
        n = len(c)
        rots = [tuple(c[i:] + c[:i]) for i in range(n)]
        rc = list(reversed(c))
        rots_rev = [tuple(rc[i:] + rc[:i]) for i in range(n)]
        return list(min(rots + rots_rev))

    def cycle_bitset(cycle_edges):
        """
        Represent cycle as integer bitset over the graph's edges.
        """
        bits = 0
        for e in cycle_edges:
            bits ^= (1 << edge_index[e])
        return bits
    basis = {}

    def is_independent_and_add(vec):
        x = vec
        while x:
            pivot = x.bit_length() - 1
            if pivot in basis:
                x ^= basis[pivot]
            else:
                basis[pivot] = x
                return True
        return False

    candidates = []
    seen_cycles = set()

    for e in edge_list:
        u, v = e
        path = shortest_path_avoiding_edge(u, v, e)
        if path is None:
            continue
        if len(path) < 3:
            continue

        cyc_edges = path_to_cycle_edges(path, e)
        cyc_vertices = canonical_cycle_vertices(path)
        cyc_key = tuple(cyc_vertices)

        if cyc_key in seen_cycles:
            continue
        seen_cycles.add(cyc_key)

        candidates.append({
            "vertices": cyc_vertices,
            "edges": cyc_edges,
            "bitset": cycle_bitset(cyc_edges),
            "length": len(cyc_edges),
        })
    candidates.sort(key=lambda c: (c["length"], c["vertices"]))

    n = len(vertices)
    m = len(edge_list)
    c = connected_components()
    target_rank = m - n + c

    chosen = []
    for cand in candidates:
        if is_independent_and_add(cand["bitset"]):
            chosen.append(cand)
            if len(chosen) == target_rank:
                break
    if return_edges:
        return [cyc["edges"] for cyc in chosen]
    return [cyc["vertices"] for cyc in chosen]

def gf2_solve_left(H, v):
    """
    Solve a @ H = v over GF(2).
    Returns one solution a if it exists, else None.
    H: (m, n)
    v: (n,)
    """
    H = H.copy().astype(np.uint8) % 2
    v = v.copy().astype(np.uint8) % 2

    m, n = H.shape
    A = np.concatenate([H.T, v[:, None]], axis=1)
    rows, cols = A.shape 
    pivot_cols = []
    r = 0
    for c in range(m):
        pivot = None
        for i in range(r, rows):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if A[i, -1]:
            return None
    x = np.zeros(m, dtype=np.uint8)
    for i, c in enumerate(pivot_cols):
        x[c] = A[i, -1]
    return x

def to_gf2(A):
    return np.asarray(A, dtype=np.uint8) % 2


def normalize_edge(u, v):
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def order_cycle_edges(cycle):
    if len(cycle) == 0:
        return []

    edges = [normalize_edge(*e) for e in cycle]
    nbrs = defaultdict(list)
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)

    bad = {v: ns for v, ns in nbrs.items() if len(ns) != 2}
    if bad:
        raise ValueError(f"Not a simple cycle; cycle-vertex degrees are {bad}")

    start = min(nbrs.keys())
    prev = None
    curr = start
    ordered_vertices = [start]

    while True:
        a, b = nbrs[curr]
        nxt = a if a != prev else b

        if nxt == start:
            break

        ordered_vertices.append(nxt)
        prev, curr = curr, nxt

        if len(ordered_vertices) > len(edges):
            raise ValueError(f"Failed to order cycle consistently: {cycle}")

    return [
        (ordered_vertices[i], ordered_vertices[(i + 1) % len(ordered_vertices)])
        for i in range(len(ordered_vertices))
    ]


def cycle_vertices(cycle):
    if len(cycle) == 0:
        return []

    verts = [cycle[0][0]]
    for u, v in cycle:
        if u != verts[-1]:
            raise ValueError(f"Cycle is not ordered consistently: {cycle}")
        verts.append(v)

    if verts[-1] != verts[0]:
        raise ValueError(f"Cycle does not close: {cycle}")

    return verts[:-1]


def path_edges(path_vertices):
    return [
        normalize_edge(path_vertices[i], path_vertices[i + 1])
        for i in range(len(path_vertices) - 1)
    ]


def split_cycle_in_center(cycle):
    ordered_cycle = order_cycle_edges(cycle)
    verts = cycle_vertices(ordered_cycle)
    n = len(verts)

    if n < 4:
        raise ValueError(f"Cannot split a cycle with fewer than 4 vertices: {cycle}")

    i = 0
    j = n // 2
    e = normalize_edge(verts[i], verts[j])

    path1_verts = verts[i:j + 1]
    path2_verts = verts[j:] + [verts[0]]

    cycle1 = path_edges(path1_verts) + [e]
    cycle2 = path_edges(path2_verts) + [e]

    return e, [cycle1, cycle2]


def split_heavy_cycles(cycles_edges, max_cycle_weight, g):
    pending = [[normalize_edge(*e) for e in cyc] for cyc in cycles_edges]
    final_cycles = []

    while pending:
        cyc = pending.pop(0)

        if len(cyc) > max_cycle_weight:
            e, (c1, c2) = split_cycle_in_center(cyc)
            if not g.are_adjacent(*e):
                g.add_edge(*e)
            pending.append(c1)
            pending.append(c2)
        else:
            final_cycles.append([normalize_edge(*e) for e in cyc])

    return final_cycles


def shortest_path_edges(g, src, dst):
    vpath = g.get_shortest_paths(src, to=dst, mode="ALL", output="vpath")[0]
    if len(vpath) < 2:
        if src == dst:
            return []
        raise ValueError(f"No path between {src} and {dst}")
    return [normalize_edge(vpath[i], vpath[i + 1]) for i in range(len(vpath) - 1)]


def greedy_pairing_path_union(g, vertices):
    """
    Path-pairing for an even set of vertices.
    Returns the GF(2) union of shortest paths.
    """
    verts = sorted(map(int, vertices))
    if len(verts) % 2 != 0:
        raise ValueError(f"Expected even overlap, got {verts}")

    remaining = set(verts)
    chosen_edges = set()

    while remaining:
        u = min(remaining)

        best_v = None
        best_dist = None
        for v in remaining:
            if v == u:
                continue
            d = g.distances(u, v)[0][0]
            if d == float("inf"):
                raise ValueError(f"No path between {u} and {v}")
            if best_dist is None or d < best_dist or (d == best_dist and v < best_v):
                best_dist = d
                best_v = v

        for e in shortest_path_edges(g, u, best_v):
            if e in chosen_edges:
                chosen_edges.remove(e)
            else:
                chosen_edges.add(e)

        remaining.remove(u)
        remaining.remove(best_v)

    return sorted(chosen_edges)


def deform_old_opposite_basis_checks_with_graph_edges(
    g,
    H_opposite_basis_all,
    logical_qubits,
    logical_qubits_index,
    n_data):
    """
    For a check overlapping the logical on an even set L_s,
    add X-support on edge qubits along path pairings μ(L_s). 
    """
    H_opposite_basis_all = to_gf2(H_opposite_basis_all)
    logical_qubits = to_gf2(logical_qubits)
    logical_qubits_index = np.asarray(logical_qubits_index, dtype=int)

    H_old = H_opposite_basis_all[np.any(H_opposite_basis_all, axis=1)]
    n_edges = g.ecount()
    H_old_padded = np.pad(H_old, ((0, 0), (0, n_edges)), mode="constant").astype(np.uint8)

    qubit_to_vertex = {int(q): i for i, q in enumerate(logical_qubits_index)}
    edge_to_eid = {normalize_edge(*e): eid for eid, e in enumerate(g.get_edgelist())}

    for row_idx, check in enumerate(H_old):
        overlap = np.where((logical_qubits == 1) & (check == 1))[0]

        if len(overlap) == 0:
            continue

        if len(overlap) % 2 != 0:
            raise ValueError(
                f"Row {row_idx} overlaps logical support on odd number of qubits: {len(overlap)}"
            )

        overlap_vertices = [qubit_to_vertex[int(q)] for q in overlap]
        matching_edges = greedy_pairing_path_union(g, overlap_vertices)

        for e in matching_edges:
            eid = edge_to_eid[e]
            H_old_padded[row_idx, n_data + eid] ^= 1

    return H_old_padded