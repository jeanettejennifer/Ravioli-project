import numpy as np
from collections import defaultdict, deque
import igraph as ig
from sympy.abc import x, y
from qldpc import codes, circuits
from qldpc.objects import Pauli
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

def to_gf2(A):
    return np.asarray(A, dtype=np.uint8) % 2

def normalize_edge(u, v):
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def deform_old_x_checks_with_graph_edges(
    g,
    Hx_all,
    logical_qubits,
    logical_qubits_index,
    n_data,
    n_edges,
    n_L):

    """
    Build the deformed X-check matrix by adding ancilla-edge support.

    Assumes:
      - g has vertices [0..n_L-1]
      - g has vertices [n_L..2*n_L-1] = check-side vertices C_i
      - ancilla qubit for graph edge eid lives at qubit index n_data + eid
      - an old X check that overlaps the chosen logical on exactly two BB qubits
        should gain X support on the ancilla qubit of the corresponding graph edge

    Returns:
      Hx_padded: X-check rows with added edge qubit support
    """

    Hx_old = Hx_all[np.any(Hx_all, axis=1)]
    Hx_padded = np.pad(Hx_old, ((0, 0), (0, n_edges)), mode="constant")

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
            Hx_padded[filtered_row, n_data + eid] = 1

        filtered_row += 1

    return Hx_padded


                                     
def find_short_cycle_basis(pairs, return_edges=True):
    """
    Greedy shortest cycle basis for the measurement graph.

    Args:
        pairs:
            List of edges [(u,v), ...].
        return_edges:
            If False, return cycles as ordered vertex lists
            If True, return cycles as edge lists [(u,v), ...] in cyclic order.

    Returns:
        List of cycles, greedily chosen from shortest candidates.
        The number of returned cycles is the cycle rank m (num edges) - n (num vertices) + c (num connected components, usually 1).
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
        Helper function for making short cycles.
        
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
        Standardized cycle representation to avoid adding the same cycle multiple times.
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


def split_heavy_cycles(
    cycles_edges,
    max_cycle_weight,
    g,
    target_max_vertex_degree=None,
    degree_slack=0,
):
    """
    Split cycles so each final cycle has <= max_cycle_weight edges.

    Priority:
      1. Use the fewest final pieces possible.
      2. Make child cycles as balanced as possible.
      3. Prefer splitting endpoints with lower post-split degree.
      4. Prefer existing edges for splitting when available.

    """

    if max_cycle_weight < 3:
        raise ValueError("max_cycle_weight must be at least 3")

    def cycle_vertices_from_edges(cyc):
        if not cyc:
            return []

        cyc = [normalize_edge(*e) for e in cyc]

        adj = defaultdict(list)
        for a, b in cyc:
            adj[a].append(b)
            adj[b].append(a)

        bad = [v for v, nbrs in adj.items() if len(nbrs) != 2]
        if bad:
            raise ValueError(
                f"Edge set is not a simple cycle. Bad vertices: {bad}, cycle: {cyc}"
            )

        start = cyc[0][0]
        ordered = [start]
        prev = None
        curr = start

        while True:
            nbrs = adj[curr]
            nxt = nbrs[0] if prev is None else (nbrs[0] if nbrs[1] == prev else nbrs[1])

            if nxt == start:
                break

            ordered.append(nxt)
            prev, curr = curr, nxt

            if len(ordered) > len(cyc):
                raise ValueError(f"Failed to reconstruct cycle ordering for cycle: {cyc}")

        return ordered

    def min_final_pieces(k):
        """
        Lower bound for decomposing a k-cycle into cycles of size <= W
        using noncrossing chords.

        If p final cycles are produced, total final cycle weight is:
            k + 2 * (p - 1)
        so we need:
            k + 2 * (p - 1) <= p * W
            p >= (k - 2) / (W - 2)
        """
        if k <= max_cycle_weight:
            return 1
        return int(math.ceil((k - 2) / (max_cycle_weight - 2)))

    def choose_balanced_chord(cyc):
        verts = cycle_vertices_from_edges(cyc)
        k = len(verts)

        if k < 4:
            raise ValueError("Cannot split a cycle of length < 4")

        best = None
        best_score = None

        for i in range(k):
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue

                a = verts[i]
                b = verts[j]
                chord = normalize_edge(a, b)

                arc1_len = j - i
                arc2_len = k - arc1_len

                c1_len = arc1_len + 1
                c2_len = arc2_len + 1

                if c1_len >= k or c2_len >= k:
                    continue

                p1 = min_final_pieces(c1_len)
                p2 = min_final_pieces(c2_len)
                total_pieces = p1 + p2

                # Balance expected final pieces, not only immediate child lengths.
                avg1 = c1_len / p1
                avg2 = c2_len / p2
                final_balance_cost = abs(avg1 - avg2)

                immediate_balance_cost = abs(c1_len - c2_len)

                adds_new_edge = not g.are_adjacent(*chord)
                deg_a_after = g.degree(a) + int(adds_new_edge)
                deg_b_after = g.degree(b) + int(adds_new_edge)

                max_endpoint_degree = max(deg_a_after, deg_b_after)
                sum_endpoint_degree = deg_a_after + deg_b_after

                if target_max_vertex_degree is None:
                    degree_overflow = 0
                else:
                    cap = target_max_vertex_degree + degree_slack
                    degree_overflow = max(0, deg_a_after - cap) + max(0, deg_b_after - cap)

                score = (
                    degree_overflow,          # first: respect degree cap if provided
                    total_pieces,             # then: fewest final cycles
                    final_balance_cost,       # then: balanced recursive decomposition
                    immediate_balance_cost,   # then: balanced immediate split
                    max_endpoint_degree,      # then: avoid high-degree endpoints
                    sum_endpoint_degree,
                    int(adds_new_edge),       # prefer existing chord
                    i,
                    j,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best = (i, j, chord)

        if best is None:
            raise ValueError(f"Could not find a valid chord split for cycle {cyc}")

        return best

    def split_once(cyc):
        verts = cycle_vertices_from_edges(cyc)
        i, j, chord = choose_balanced_chord(cyc)

        if not g.are_adjacent(*chord):
            g.add_edge(*chord)

        arc1 = verts[i:j + 1]
        arc2 = verts[j:] + verts[:i + 1]

        c1 = [normalize_edge(arc1[t], arc1[t + 1]) for t in range(len(arc1) - 1)]
        c1.append(chord)

        c2 = [normalize_edge(arc2[t], arc2[t + 1]) for t in range(len(arc2) - 1)]
        c2.append(chord)

        return c1, c2

    pending = [[normalize_edge(*e) for e in cyc] for cyc in cycles_edges]
    final_cycles = []

    max_splits = 50 * max(1, len(cycles_edges))
    split_count = 0

    while pending:
        cyc = pending.pop()

        if len(cyc) <= max_cycle_weight:
            final_cycles.append(cyc)
            continue

        if split_count >= max_splits:
            final_cycles.append(cyc)
            continue

        try:
            c1, c2 = split_once(cyc)
        except ValueError:
            final_cycles.append(cyc)
            continue

        if len(c1) < len(cyc) and len(c2) < len(cyc):
            pending.append(c1)
            pending.append(c2)
            split_count += 1
        else:
            final_cycles.append(cyc)

    return final_cycles


def shortest_path_edge_list(g, src, dst):
    """
    Return one shortest path from src to dst as a list of normalized edges.
    Returns [] if src == dst.
    Raises ValueError if no path exists.
    """
    if src == dst:
        return []

    vpath = g.get_shortest_paths(src, to=dst, output="vpath")[0]
    if vpath is None or len(vpath) == 0:
        raise ValueError(f"No path found between vertices {src} and {dst}")

    return [normalize_edge(vpath[i], vpath[i + 1]) for i in range(len(vpath) - 1)]


def greedy_pairing_path_union(g, overlap_vertices):
    """
    Given an even-size list of graph vertices, greedily pair them by shortest-path
    distance and return the union of edges on those pairing paths.

    Parameters
    ----------
    g : igraph.Graph
    overlap_vertices : list[int]
        Vertices to pair. Must have even length.

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of normalized edges appearing in the union of the chosen paths.
    """
    overlap_vertices = [int(v) for v in overlap_vertices]

    if len(overlap_vertices) % 2 != 0:
        raise ValueError(
            f"greedy_pairing_path_union requires an even number of vertices, got {len(overlap_vertices)}"
        )

    if len(overlap_vertices) == 0:
        return []

    # Precompute pairwise shortest-path lengths on just the relevant vertices
    # distances[i][j] corresponds to overlap_vertices[i] -> overlap_vertices[j]
    distances = g.distances(source=overlap_vertices, target=overlap_vertices)

    remaining = set(range(len(overlap_vertices)))
    used_edges = set()

    while remaining:
        i = min(remaining)

        # Choose nearest remaining partner j for i
        best_j = None
        best_score = None
        vi = overlap_vertices[i]

        for j in remaining:
            if j == i:
                continue

            vj = overlap_vertices[j]
            dij = distances[i][j]

            if dij is None or np.isinf(dij):
                continue

            # tie-break by vertex label for determinism
            score = (dij, min(vi, vj), max(vi, vj))
            if best_score is None or score < best_score:
                best_score = score
                best_j = j

        if best_j is None:
            raise ValueError(
                f"Could not find a path to pair vertex {vi} with any remaining vertex."
            )

        vj = overlap_vertices[best_j]
        path_edges = shortest_path_edge_list(g, vi, vj)
        used_edges.update(path_edges)

        remaining.remove(i)
        remaining.remove(best_j)

    return sorted(used_edges)

def deform_old_opposite_basis_checks_with_graph_edges(
    g,
    H_opposite_basis_all,
    logical_qubits,
    logical_qubits_index,
    n_data):
    """
    For a check overlapping the logical on an even set L_s,
    add support on edge qubits along path pairings mu(L_s).
    """
    H_opposite_basis_all = to_gf2(H_opposite_basis_all)
    logical_qubits = to_gf2(logical_qubits)
    logical_qubits_index = np.asarray(logical_qubits_index, dtype=int)

    H_old = H_opposite_basis_all[np.any(H_opposite_basis_all, axis=1)]
    n_edges = g.ecount()
    data_width = H_old.shape[1]
    full_data_width = max(int(n_data), data_width)
    H_old_padded = np.zeros((H_old.shape[0], full_data_width + n_edges), dtype=np.uint8)
    H_old_padded[:, :data_width] = H_old

    qubit_to_vertex = {int(q): i for i, q in enumerate(logical_qubits_index)}
    edge_to_eid = {normalize_edge(*e): eid for eid, e in enumerate(g.get_edgelist())}
    logical_support = logical_qubits[:data_width]

    for row_idx, check in enumerate(H_old):
        overlap = np.where((logical_support == 1) & (check == 1))[0]

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
            col = full_data_width + eid
            if col >= H_old_padded.shape[1]:
                raise IndexError(
                    f"edge column {col} is outside old-check matrix with shape "
                    f"{H_old_padded.shape}; n_data={n_data}, data_width={data_width}, "
                    f"n_edges={n_edges}, eid={eid}"
                )
            H_old_padded[row_idx, col] ^= 1

    return H_old_padded




