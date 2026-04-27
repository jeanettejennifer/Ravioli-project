import numpy as np
from collections import defaultdict, deque
import igraph as ig
from sympy.abc import x, y
from qldpc import codes, circuits
from qldpc.objects import Pauli
import numpy as np
import math
import matplotlib.pyplot as plt


def deform_code_for_logical(H, basis, logical):
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
        if len(overlap)>0 and len(overlap)%2==0:
            n_pairs = len(overlap) // 2
            for i in range(n_pairs):
                v1 = qubit_to_vertex[int(overlap[2*i])]
                v2 = qubit_to_vertex[int(overlap[2*i+1])]
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
    #print_degree_stats(g, "after adding edges")

    H_basis_nonzero = H_basis_all[np.any(H_basis_all, axis=1)]
    max_cycle_weight = int(np.max(np.sum(H_basis_nonzero, axis=1))) if len(H_basis_nonzero) else 4
    max_cycle_weight = max(max_cycle_weight, 3)

    cycles_edges = find_short_cycle_basis(
        [tuple(map(int, e)) for e in g.get_edgelist()],
        return_edges=True,
    )
    #print_degree_stats(g, "Before splitting cycles")
    cycles_edges = split_heavy_cycles(
        cycles_edges=cycles_edges,
        max_cycle_weight=max_cycle_weight,
        g=g,
    )
    #print_degree_stats(g, "After splitting cycles")
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
        "qubit_to_vertex": qubit_to_vertex
    }
    
    return res



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


def print_degree_stats(g, label):
    degs = g.degree()
    print(f"{label}: max degree = {max(degs) if degs else 0}")
    hist = defaultdict(int)
    for d in degs:
        hist[d] += 1
    print(f"{label}: degree histogram = {dict(sorted(hist.items()))}")

def split_heavy_cycles(cycles_edges, max_cycle_weight, g):
    """
    Split cycles until every cycle has <= max_cycle_weight edges.

    Strategy:
      - Ignore balance.
      - At each step, choose the chord whose endpoints have the smallest
        post-split vertex-degree cost.
      - Keep splitting recursively until all child cycles are short enough.

    """

    from collections import defaultdict

    def cycle_vertices_from_edges(cyc):
        """
        Recover an ordered vertex cycle from an unordered simple cycle edge list.
        Returns [v0, v1, ..., v_{k-1}] with wraparound implicit.
        """
        if not cyc:
            return []

        cyc = [normalize_edge(*e) for e in cyc]

        adj = defaultdict(list)
        for a, b in cyc:
            adj[a].append(b)
            adj[b].append(a)

        bad = [v for v, nbrs in adj.items() if len(nbrs) != 2]
        if bad:
            raise ValueError(f"Edge set is not a simple cycle. Bad vertices: {bad}, cycle: {cyc}")

        start = cyc[0][0]
        ordered = [start]
        prev = None
        curr = start

        while True:
            nbrs = adj[curr]
            if prev is None:
                nxt = nbrs[0]
            else:
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]

            if nxt == start:
                break

            ordered.append(nxt)
            prev, curr = curr, nxt

            if len(ordered) > len(cyc):
                raise ValueError(f"Failed to reconstruct cycle ordering for cycle: {cyc}")

        return ordered

    def choose_min_degree_chord(cyc, g):
        """
        Choose a chord that minimizes endpoint degree growth, subject to actually
        splitting the cycle into two strictly smaller cycles.
        """
        verts = cycle_vertices_from_edges(cyc)
        k = len(verts)

        if k < 4:
            raise ValueError("Cannot split a cycle of length < 4")

        best = None
        best_score = None

        for i in range(k):
            for j in range(i + 2, k):
                # skip wraparound adjacency
                if i == 0 and j == k - 1:
                    continue

                a = verts[i]
                b = verts[j]
                chord = normalize_edge(a, b)

                arc1_len = j - i
                arc2_len = k - arc1_len

                # resulting cycles include the chord
                c1_len = arc1_len + 1
                c2_len = arc2_len + 1

                # both children must be strictly smaller
                if c1_len >= k or c2_len >= k:
                    continue

                adds_new_edge = not g.are_adjacent(*chord)
                deg_a_after = g.degree(a) + (1 if adds_new_edge else 0)
                deg_b_after = g.degree(b) + (1 if adds_new_edge else 0)

                # purely degree-driven score
                score = (
                    max(deg_a_after, deg_b_after),   # minimize worst endpoint degree
                    deg_a_after + deg_b_after,       # then total endpoint degree
                    0 if adds_new_edge else -1,      # slight preference for existing chord
                    c1_len + c2_len,                 # tie-breaker
                    i, j,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best = (i, j, chord)

        if best is None:
            raise ValueError(f"Could not find a valid chord split for cycle {cyc}")

        return best

    def split_once(cyc, g):
        """
        Split one cycle using the minimum-degree chord.
        Returns two child cycles.
        """
        verts = cycle_vertices_from_edges(cyc)
        i, j, chord = choose_min_degree_chord(cyc, g)

        if not g.are_adjacent(*chord):
            g.add_edge(*chord)

        arc1 = verts[i:j + 1]
        arc2 = verts[j:] + verts[:i + 1]

        c1 = [normalize_edge(arc1[t], arc1[t + 1]) for t in range(len(arc1) - 1)] + [chord]
        c2 = [normalize_edge(arc2[t], arc2[t + 1]) for t in range(len(arc2) - 1)] + [chord]

        return c1, c2

    pending = [[normalize_edge(*e) for e in cyc] for cyc in cycles_edges]
    final_cycles = []

    # safety cap
    max_splits = 50 * max(1, len(cycles_edges))
    split_count = 0

    while pending:
        cyc = pending.pop()

        if len(cyc) <= max_cycle_weight:
            final_cycles.append(cyc)
            continue

        if split_count >= max_splits:
            # stop gracefully if something pathological happens
            final_cycles.append(cyc)
            continue

        try:
            c1, c2 = split_once(cyc, g)
        except ValueError:
            final_cycles.append(cyc)
            continue

        # Only accept the split if both children are strictly smaller
        if len(c1) < len(cyc) and len(c2) < len(cyc):
            pending.append(c1)
            pending.append(c2)
            split_count += 1
        else:
            final_cycles.append(cyc)

    return final_cycles


from collections import deque
import numpy as np


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

def normalize_edge(u, v):
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)

def plot_deformation_surgery_view(
    H,
    basis,
    logical,
    deform_fn=deform_code_for_logical,
    n_cols=None,
    aux_layout="kk",
    figsize=(16, 8),
    label_aux=True,
    ):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    out = deform_fn(H, basis, logical)
    n_data = out["n_original_qubits"]
    g = out["g"]

    logical_idx = np.where(logical == 1)[0]
    qubit_to_vertex = {int(q): i for i, q in enumerate(logical_idx)}

    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_data)))

    n_rows = int(np.ceil(n_data / n_cols))

    data_pos = {}
    for q in range(n_data):
        r, c = divmod(q, n_cols)
        data_pos[q] = (c, -r)

    aux_x0 = n_cols + 4.0
    aux_w = max(5.0, n_cols * 0.55)
    aux_h = max(5.0, n_rows * 0.85)

    layout = g.layout(aux_layout)
    xs = np.array([p[0] for p in layout])
    ys = np.array([p[1] for p in layout])

    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)

    aux_pos = {
        v: (
            aux_x0 + 0.6 + xs[v] * (aux_w - 1.2),
            -0.6 - ys[v] * (aux_h - 1.2),
        )
        for v in range(g.vcount())
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # data-qubit patch, aligned around all plotted qubit centers
    pad = 0.75
    patch = patches.Rectangle(
        (-pad, -(n_rows - 1) - pad),
        (n_cols - 1) + 2 * pad,
        (n_rows - 1) + 2 * pad,
        fill=False,
        linewidth=2,
        edgecolor="black",
    )
    ax.add_patch(patch)

    for q, (x, y) in data_pos.items():
        is_logical = logical[q] == 1
        ax.scatter(
            x,
            y,
            s=170 if is_logical else 95,
            facecolors="lightgray" if is_logical else "white",
            edgecolors="black" if is_logical else "0.6",
            linewidths=2 if is_logical else 1.4,
            zorder=3,
        )

    # auxiliary graph background
    aux_bg = patches.Rectangle(
        (aux_x0, -aux_h),
        aux_w,
        aux_h,
        facecolor="0.93",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(aux_bg)

    # auxiliary graph edges
    for u, v in g.get_edgelist():
        x1, y1 = aux_pos[u]
        x2, y2 = aux_pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.4, zorder=2)

    # dashed mapping lines from logical data qubits to graph vertices
    for q in logical_idx:
        v = qubit_to_vertex[int(q)]
        x1, y1 = data_pos[int(q)]
        x2, y2 = aux_pos[v]
        ax.plot([x1, x2], [y1, y2], "--", color="0.55", linewidth=1.2, alpha=0.75, zorder=1)

    # auxiliary graph vertices
    for v in range(g.vcount()):
        x, y = aux_pos[v]
        ax.scatter(
            x,
            y,
            s=210,
            marker="s",
            facecolors="white",
            edgecolors="black",
            linewidths=1.7,
            zorder=4,
        )
        if label_aux:
            ax.text(x, y, str(v), ha="center", va="center", fontsize=9, zorder=5)

    ax.set_title(f"Auxiliary-graph surgery view for logical {basis}", fontsize=18)
    ax.text(aux_x0 + aux_w / 2, 0.75, "auxiliary graph", ha="center", fontsize=14)
    ax.text((n_cols - 1) / 2, pad + 0.45, "data-qubit patch", ha="center", fontsize=14)

    ax.set_xlim(-1.2, aux_x0 + aux_w + 0.8)
    ax.set_ylim(-max(n_rows, aux_h) - 0.8, 1.4)

    plt.tight_layout()
    return fig, ax, out



def skiptree(g, root=0):
    """
    SkipTreeHR for an igraph.Graph.

    Args:
        g: igraph.Graph, undirected and connected
        root: root vertex index

    Returns:
        T_rows: list[set[int]]
            T row l is the set of original igraph edge IDs on the tree path
            from SkipTree label l to label l+1. Shape: (n-1) x m.
        P: list[int]
            P[v] = SkipTree label of original vertex v.
        label_to_vertex: list[int]
            label_to_vertex[l] = original vertex with SkipTree label l.
        tree_edge_ids: list[int]
            igraph edge IDs used in the spanning tree.
    """
    if g.is_directed():
        raise ValueError("g must be undirected.")
    if not g.is_connected():
        raise ValueError("g must be connected.")

    n = g.vcount()
    m = g.ecount()

    if not (0 <= root < n):
        raise ValueError("root must be a valid vertex index.")

    # Build adjacency: vertex -> [(neighbor, edge_id), ...]
    adj = [[] for _ in range(n)]
    for eid, edge in enumerate(g.es):
        u, v = edge.tuple
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    # BFS spanning tree
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

    label_to_vertex = []
    vertex_to_label = [-1] * n

    def label_vertex(v):
        label = len(label_to_vertex)
        label_to_vertex.append(v)
        vertex_to_label[v] = label

    def label_first(v, skip=False):
        label_vertex(v)

        for idx, child in enumerate(children[v]):
            is_youngest = idx == len(children[v]) - 1

            if is_youngest and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v):
        for child in children[v]:
            label_first(child, skip=True)

        label_vertex(v)

    label_first(root, skip=False)

    # P[v] = SkipTree label of original vertex v
    P = vertex_to_label

    def tree_path_edges(a, b):
        path_a = []
        path_b = []
        ancestors_a = {}

        x = a
        while x != -1:
            ancestors_a[x] = len(path_a)
            if parent[x] != -1:
                path_a.append(parent_edge[x])
            x = parent[x]

        y = b
        while y not in ancestors_a:
            path_b.append(parent_edge[y])
            y = parent[y]

        lca = y
        return path_a[:ancestors_a[lca]] + path_b

    T_rows = []
    for l in range(n - 1):
        a = label_to_vertex[l]
        b = label_to_vertex[l + 1]
        T_rows.append(set(tree_path_edges(a, b)))

    tree_edge_ids = [eid for eid in parent_edge if eid != -1]

    return T_rows, P, label_to_vertex, tree_edge_ids