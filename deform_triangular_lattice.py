"""Compile logical deformations onto a triangular-lattice Stim gadget.

Main entry point
----------------
    deform_logical_to_tri_lattice(deformation_result)

The pipeline is:

1. run the abstract deformation through the triangular layout stage;
2. make the smallest alternating triangular gadget that fits the placed graph;
3. rebuild compact H matrices over original data qubits + gadget edge qubits;
4. return check-row-to-ancilla maps and graph metadata for Stim.

The dynamic Stim index convention is:

    data qubits, gadget edge qubits, old basis-check ancillas,
    old opposite-check ancillas, gadget cycle ancillas, gadget vertex ancillas,
    port qubits.

Only data qubits and gadget edge qubits appear in stabilizer support columns.
Check/port qubits are measurement/helper qubits and are returned through
row-to-measurement-qubit metadata.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from tri_layout_deformation import tri_layout


BB_DATA_QUBITS = range(0, 72)
GADGET_EDGE_START = 72
H_MATRIX_QUBITS = 244
BB_Z_CHECK_QUBITS = range(244, 280)
BB_X_CHECK_QUBITS = range(280, 316)
GADGET_CYCLE_CHECK_START = 316
GADGET_VERTEX_CHECK_START = 421
GADGET_PORT_START = 489
N_FIXED_QUBITS = 509


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


def _code_patch_coords(n_data: int, n_basis_checks: int, n_opposite_checks: int) -> tuple[Dict[int, Tuple[float, float]], Tuple[float, float, float, float], str]:
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


def _sort_qubits_by_coord(qubits: Iterable[int], coords: Dict[int, Tuple[float, float]]) -> List[int]:
    return sorted(qubits, key=lambda q: (coords[q][0], coords[q][1], q))


def _alternating_row_lengths(rows: int, cols: int) -> List[int]:
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

    The historical name is kept for compatibility.  Indices are assigned as
    data, gadget edges, old basis-check ancillas, old opposite-check ancillas,
    floor cycle ancillas, gadget vertex ancillas, port qubits, then shuttling
    cycle ancillas.
    """
    if rows < 2 or cols < 2:
        raise ValueError("triangular gadget needs at least 2 rows and 2 columns")
    if shuttling_cycle_checks is None:
        shuttling_cycle_checks = int(shuttling_edges)

    coords, code_patch_rect, code_patch_label = _code_patch_coords(n_data, n_basis_checks, n_opposite_checks)
    data_qubits = list(range(n_data))

    row_lengths = _alternating_row_lengths(rows, cols)
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

    vertex_check_qubits = _sort_qubits_by_coord(vertex_check_qubits, coords)
    edge_qubits = _sort_qubits_by_coord(edge_qubits, coords)
    cycle_check_qubits = _sort_qubits_by_coord(cycle_check_qubits, coords)
    _validate_fixed_ordering(coords, vertex_check_qubits, edge_qubits, cycle_check_qubits)
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

def _validate_fixed_ordering(coords, vertex_qubits, edge_qubits, cycle_qubits) -> None:
    for name, qubits in [
        ("vertex checks", vertex_qubits),
        ("edge qubits", edge_qubits),
        ("cycle checks", cycle_qubits),
    ]:
        ordered = _sort_qubits_by_coord(qubits, coords)
        if list(qubits) != ordered:
            raise AssertionError(f"{name} are not ordered left-to-right, top-to-bottom")


def _copy_columns_to_fixed(M, old_to_fixed, width=H_MATRIX_QUBITS):
    M = np.asarray(M, dtype=np.uint8) % 2
    out = np.zeros((M.shape[0], width), dtype=np.uint8)
    for old_col, fixed_cols in old_to_fixed.items():
        if isinstance(fixed_cols, int):
            fixed_cols = [fixed_cols]
        if old_col >= M.shape[1]:
            continue
        for fixed_col in fixed_cols:
            if fixed_col < width:
                out[:, fixed_col] ^= M[:, old_col]
    return out


def _cycle_vertices_to_edge_qubits(cycle_vertices, pair_to_qubit):
    edge_qubits = []
    vertices = list(cycle_vertices)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        pair = tuple(sorted((int(a), int(b))))
        q = pair_to_qubit.get(pair)
        if q is None:
            return None
        edge_qubits.append(int(q))
    return edge_qubits


def _recompute_cycle_basis_from_fixed_edges(fixed_edge_qubits, template, width=H_MATRIX_QUBITS):
    """Compute simple cycle checks from the placed fixed-gadget edge graph.

    The old routed cycle rows can contain interior edges.  This recomputes the
    gadget cycle checks after placement as a graph cycle basis over the fixed
    triangular-lattice edge graph.  Each row is therefore only the boundary of a
    simple cycle in the placed graph.
    """
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
        edge_qubits = _cycle_vertices_to_edge_qubits(cycle, pair_to_qubit)
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


def _ordered_cycle_from_edge_support(edge_qubits, template):
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


def _row_from_edge_support(edge_qubits, width):
    row = np.zeros(int(width), dtype=np.uint8)
    row[sorted(set(map(int, edge_qubits)))] = 1
    return row


def _split_floor_cycle_supports(
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
            rows.append(_row_from_edge_support(support, width))
            out_sources.append({**source, "edge_qubits": support, "weight": len(support)})
            continue

        ordered = _ordered_cycle_from_edge_support(support, template)
        if ordered is None:
            rows.append(_row_from_edge_support(support, width))
            out_sources.append({**source, "edge_qubits": support, "weight": len(support)})
            continue

        vertices, ordered_edges = ordered
        k = len(ordered_edges)
        vertex_weights = _vertex_weights_from_rows(H_basis_new, vertex_row_to_check_qubit)
        best = None
        best_relaxed = None
        support_set = set(support)
        for i in range(k):
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue
                path = _shortest_floor_edge_path(
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
                max_vertex_after = _max_vertex_weight_after_adding_path(
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
                cap_ok = _path_fits_vertex_weight_cap(
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
            rows.append(_row_from_edge_support(support, width))
            out_sources.append({**source, "edge_qubits": support, "weight": len(support)})
            continue

        _, _, _, _, max_vertex_after, c1, c2, chord, endpoints = best
        H_basis_new, floor_edges, added_edges = _add_floor_edges_to_vertex_checks(
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


def _partition_cycle_rows_by_shuttles(H_cycles, cycle_sources, shuttling_edge_qubits):
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


def _cycle_rows_are_simple_boundaries(H_cycle_fixed, template):
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




def _fixed_auxiliary_graph_data(edge_qubits, template):
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

    return {
        "auxiliary_graph_incidence_matrix": incidence,
        "auxiliary_graph_incidence_vertices": vertex_qubits,
        "auxiliary_graph_incidence_edge_qubits": edge_qubits,
        "auxiliary_graph_igraph": graph,
    }

def _used_layout_sites(layout):
    sites = set()
    for a, b in layout["lattice_edge_to_qubit"]:
        sites.add(a)
        sites.add(b)
    for site in layout["site_to_vertex_check_row"]:
        sites.add(site)
    return sites


def _fit_layout_sites_to_template(layout, template):
    """Map tri_layout lattice sites exactly into the fixed triangular template.

    ``tri_layout`` and the reusable gadget use the same triangular-lattice
    combinatorics.  Therefore the only allowed freedom is an integer row/column
    translation.  If the placed graph does not fit as an exact subgraph of the
    fixed gadget lattice, this raises instead of silently changing the graph.
    """
    used_sites = sorted(_used_layout_sites(layout), key=lambda s: (s[0], s[1]))
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
        skew_cols = [_skew_to_alternating_site(site, dr, 0)[1] for site in used_sites]
        for dc in range(min(template_cols) - min(skew_cols), max(template_cols) - max(skew_cols) + 1):
            shifted = [_skew_to_alternating_site(site, dr, dc) for site in used_sites]
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
        site: template.vertex_site_to_qubit[_skew_to_alternating_site(site, dr, dc)]
        for site in used_sites
    }, (dr, dc)


def _skew_to_alternating_site(site, row_shift, col_shift):
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


def _map_layout_edges_to_fixed(layout, template, site_to_vertex):
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


def _layout_graph_edge_order(deformation_result):
    return [
        tuple(sorted(map(int, e)))
        for e in deformation_result["g"].get_edgelist()
    ]


def _fixed_vertex_graph(template, allowed_vertices=None):
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


def _route_edge_qubits_on_fixed_graph(G, start, stop, used=None, randomness=0.0, rng=None):
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


def _fixed_route_length(template, a, b):
    G = _fixed_vertex_graph(template)
    return nx.shortest_path_length(G, int(a), int(b))


def _placement_score_for_floor_edges(template, placement, floor_edges, target_fixed, anchor_weight=0.03, bbox_weight=0.35):
    G = _fixed_vertex_graph(template)
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


def _improve_fixed_vertex_placement(template, initial, floor_edges, rounds=5):
    """Second local placement pass after long abstract routes become shuttles."""
    if not floor_edges:
        return dict(initial)
    rng = np.random.default_rng(0)
    placement = {int(v): int(q) for v, q in initial.items()}
    target_fixed = dict(placement)
    all_sites = list(map(int, template.vertex_check_qubits))
    best = _placement_score_for_floor_edges(template, placement, floor_edges, target_fixed)
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
                score = _placement_score_for_floor_edges(template, placement, floor_edges, target_fixed)
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


def _reroute_floor_after_abstract_shuttling(layout, template, site_to_fixed_vertex, shuttling_threshold):
    """Classify abstract graph edges as floor/shuttle, then compactly reroute floor edges."""
    deformation_result = layout["deformation_result"]
    initial = {
        int(v): int(site_to_fixed_vertex[site])
        for v, site in layout["vertex_sites"].items()
    }
    graph_edges = _layout_graph_edge_order(deformation_result)
    threshold = None if shuttling_threshold is None else int(shuttling_threshold)
    shuttled_edges = set()
    floor_graph_edges = []
    for edge in graph_edges:
        a, b = edge
        length = _fixed_route_length(template, initial[int(a)], initial[int(b)])
        if threshold is not None and length > threshold:
            shuttled_edges.add(edge)
        else:
            floor_graph_edges.append(edge)

    placement = _improve_fixed_vertex_placement(template, initial, floor_graph_edges)
    G = _fixed_vertex_graph(template)
    used = {}
    floor_routes = {}
    floor_edges = set()
    for edge in sorted(
        floor_graph_edges,
        key=lambda e: nx.shortest_path_length(G, placement[int(e[0])], placement[int(e[1])]),
    ):
        path = _route_edge_qubits_on_fixed_graph(
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


def _increment_vertex_weights_for_edges(vertex_weights, edge_qubits, template):
    """Update vertex weights after materializing edge qubits."""
    for q in map(int, edge_qubits):
        if q not in template.edge_qubit_to_vertices:
            continue
        a, b = template.edge_qubit_to_vertices[q]
        vertex_weights[int(a)] = vertex_weights.get(int(a), 0) + 1
        vertex_weights[int(b)] = vertex_weights.get(int(b), 0) + 1


def _choose_shuttling_endpoint(start, template, vertex_weights, max_vertex_degree, max_distance=None):
    """Choose a shuttle endpoint without exceeding the vertex-degree cap when possible."""
    start = int(start)
    if max_vertex_degree is None or vertex_weights.get(start, 0) < int(max_vertex_degree):
        return start, []
    max_distance = max(2, int(max_vertex_degree)) if max_distance is None else int(max_distance)
    allowed_vertices = set(map(int, template.vertex_check_qubits))
    endpoint, path_edges = _choose_nearby_low_weight_vertex(
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
    G = _fixed_floor_graph(template, allowed_vertices=allowed_vertices)
    if start not in G:
        return start, []
    lengths = nx.single_source_shortest_path_length(G, start, cutoff=max_distance)
    candidates = []
    for vertex, distance in lengths.items():
        path = _shortest_floor_edge_path(template, start, vertex, allowed_vertices)
        if path is None:
            continue
        candidates.append((vertex_weights.get(int(vertex), 0), int(distance), int(vertex), path))
    if not candidates:
        return start, []
    _, _, endpoint, path_edges = min(candidates)
    return int(endpoint), [int(q) for q in path_edges]


def _choose_shuttling_endpoints(preferred, template, vertex_weights, max_vertex_degree):
    del template, vertex_weights, max_vertex_degree
    return tuple(sorted(map(int, preferred))), []


def _initial_vertex_weights_for_shuttling(deformation_result, template, placement, floor_edge_qubits):
    """Weights before shuttles: data support plus already routed floor edges."""
    weights = {}
    n_data = int(deformation_result["n_original_qubits"])
    for q, v in deformation_result["qubit_to_vertex"].items():
        q = int(q)
        v = int(v)
        if q < n_data and v in placement:
            vertex = int(placement[v])
            weights[vertex] = weights.get(vertex, 0) + 1
    _increment_vertex_weights_for_edges(weights, floor_edge_qubits, template)
    return weights


def _allocate_abstract_route_shuttles(
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
        endpoint_vertices, adjustment_edges = _choose_shuttling_endpoints(
            preferred,
            template,
            vertex_weights,
            max_vertex_degree,
        )
        template.edge_qubit_to_vertices[q] = endpoint_vertices
        _increment_vertex_weights_for_edges(vertex_weights, [q], template)
        records.append({
            "qubit": q,
            "type": "abstract_route_shuttle",
            "abstract_edge": tuple(map(int, edge)),
            "path_length": int(_fixed_route_length(template, *preferred)),
            "preferred_endpoint_vertices": tuple(sorted(preferred)),
            "endpoint_vertices": endpoint_vertices,
            "endpoint_adjustment_floor_edges": [int(x) for x in adjustment_edges],
            "endpoint_sites": (
                tuple(template.vertex_qubit_to_site[endpoint_vertices[0]]),
                tuple(template.vertex_qubit_to_site[endpoint_vertices[1]]),
            ),
            "replaced_floor_edge_qubits": [],
        })
    _place_shuttling_edge_coords(template, records)
    return [int(r["qubit"]) for r in records], records


def _seed_vertex_rows_from_placement(deformation_result, template, placement, width):
    n_data = int(deformation_result["n_original_qubits"])
    vertices = sorted(set(map(int, placement.values())), key=lambda q: (*template.coords[q], q))
    vertex_row_to_check_qubit = {row: int(q) for row, q in enumerate(vertices)}
    vertex_to_row = {int(q): int(row) for row, q in vertex_row_to_check_qubit.items()}
    H = np.zeros((len(vertices), width), dtype=np.uint8)
    for q, v in deformation_result["qubit_to_vertex"].items():
        q = int(q)
        if q >= n_data or int(v) not in placement:
            continue
        row = vertex_to_row[int(placement[int(v)])]
        H[row, q] = 1
    return H, vertex_row_to_check_qubit


def _expand_old_matrix_on_fixed_routes(
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
    graph_edges = _layout_graph_edge_order(deformation_result)
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


def _expand_old_basis_on_fixed_routes(deformation_result, floor_routes, abstract_shuttle_records, width):
    return _expand_old_matrix_on_fixed_routes(
        deformation_result,
        "H_basis_old",
        floor_routes,
        abstract_shuttle_records,
        width,
    )


def _fixed_support_for_planar_edge(edge, floor_routes, abstract_shuttle_records):
    shuttle_by_edge = {
        tuple(record["abstract_edge"]): [
            int(record["qubit"]),
            *map(int, record.get("endpoint_adjustment_floor_edges", [])),
        ]
        for record in abstract_shuttle_records
    }
    edge = tuple(sorted(map(int, edge)))
    return list(map(int, floor_routes.get(edge, shuttle_by_edge.get(edge, []))))


def _compress_old_opposite_components_with_shuttles(
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
    graph_edges = _layout_graph_edge_order(deformation_result)
    used_shuttles = set(map(int, shuttling_edge_qubits_used))
    shuttle_iter = iter([
        int(q) for q in template.shuttling_edge_qubits if int(q) not in used_shuttles
    ])
    floor_edges = set(map(int, floor_edge_qubits_used))
    shuttle_edges = set(map(int, shuttling_edge_qubits_used))
    records = []

    n_rows = np.asarray(deformation_result["H_opposite_basis_old_padded"]).shape[0]
    for row in range(n_rows):
        for component in _old_opposite_edge_components(deformation_result, row):
            odd = component["odd_vertices"]
            if len(odd) != 2:
                continue
            fixed_support = []
            for edge_col in component["edge_cols"]:
                if int(edge_col) >= len(graph_edges):
                    continue
                fixed_support.extend(_fixed_support_for_planar_edge(
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
            endpoints, adjustment_edges = _choose_shuttling_endpoints(
                preferred,
                template,
                vertex_weights,
                max_vertex_degree,
            )
            template.edge_qubit_to_vertices[q] = endpoints
            _increment_vertex_weights_for_edges(vertex_weights, [q], template)

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

    _place_shuttling_edge_coords(template, abstract_shuttle_records + records)
    return H_opp_old, sorted(floor_edges), sorted(shuttle_edges), records


def _map_vertex_rows_to_fixed(layout, site_to_vertex):
    offset = int(layout["basis_new_row_offset"])
    return {
        offset + int(local_row): int(site_to_vertex[site])
        for site, local_row in layout["site_to_vertex_check_row"].items()
    }


def _ordered_cycle_vertices(edge_vertex_pairs):
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


def _point_on_segment(p, a, b, eps=1e-9):
    px, py = p
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps


def _point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test, including boundary."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if _point_on_segment(point, a, b):
            return True
        xi, yi = a
        xj, yj = b
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
    return inside


def _cycle_checks_inside_edge_support(support, template):
    support_edges = [
        template.edge_qubit_to_vertices[q]
        for q in sorted(support)
        if q in template.edge_qubit_to_vertices
    ]
    loops = _ordered_cycle_vertices(support_edges)
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
        if any(_point_in_polygon(point, polygon) for polygon in polygons):
            checks.append(int(q))
    return checks


def _nearest_cycle_check_for_edge_support(support, template):
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


def _map_cycle_rows_to_triangle_checks(H_cycle_fixed, template):
    row_to_checks = {}
    row_to_first_check = {}
    triangle_edges = {
        q: set(edges)
        for q, edges in template.cycle_check_to_edge_qubits.items()
    }
    for row, check in enumerate(np.asarray(H_cycle_fixed, dtype=np.uint8)):
        support = set(np.flatnonzero(check).astype(int))
        checks = _cycle_checks_inside_edge_support(support, template)
        if not checks:
            checks = [
                int(q)
                for q, edges in triangle_edges.items()
                if len(support & edges) >= 2
            ]
        if not checks:
            nearest = _nearest_cycle_check_for_edge_support(support, template)
            checks = [] if nearest is None else [nearest]
        row_to_checks[int(row)] = checks
        row_to_first_check[int(row)] = checks[0] if checks else None
    return row_to_checks, row_to_first_check


def _map_opposite_rows_to_cycle_ancillas(H_floor_cycles, H_shuttling_cycles, template):
    """Map floor cycles to floor ancillas and shuttling cycles to external ancillas."""
    floor_map, floor_first = _map_cycle_rows_to_triangle_checks(H_floor_cycles, template)
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



def _estimate_relative_expansion(edge_qubits, port_vertex_qubits, template, samples=50_000, seed=0, t=None):
    """Randomly estimate beta_t(G, U) for the current fixed auxiliary graph."""
    graph_data = _fixed_auxiliary_graph_data(edge_qubits, template)
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


def _unused_incident_edges(
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


def _repair_relative_expansion(
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
        estimate = _estimate_relative_expansion(
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
            candidates = _unused_incident_edges(
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


def _add_edge_columns_to_vertex_checks(
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

def _row_to_ancilla_map(qubits, n_rows):
    if n_rows > len(qubits):
        raise ValueError(f"need {n_rows} ancillas, but only have {len(qubits)}")
    return {row: int(qubits[row]) for row in range(n_rows)}


def _gf2_row_combination_for_target(rows, target):
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


def _basis_is_x(basis):
    if basis is None:
        return False
    name = getattr(basis, "name", None)
    if name is not None:
        return str(name).upper() == "X"
    return str(basis).upper().endswith("X") or str(basis).upper() == "X"


def _skiptree_port_order(g, port_vertices, root=None):
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


def _port_order_from_deformation(deformation_result):
    logical = np.asarray(deformation_result["logical"], dtype=np.uint8)
    q_to_v = {
        int(q): int(v)
        for q, v in deformation_result["qubit_to_vertex"].items()
        if int(q) < len(logical) and logical[int(q)]
    }
    v_to_q = {v: q for q, v in q_to_v.items()}
    skip = _skiptree_port_order(deformation_result["g"], list(q_to_v.values()))
    return {
        **skip,
        "port_vertex_to_logical_qubit": v_to_q,
        "logical_qubit_to_port_vertex": q_to_v,
    }


def _template_shape_for_layout(layout, min_rows=2, min_cols=2):
    """Smallest alternating-row triangular template that contains the layout sites."""
    used_sites = sorted(_used_layout_sites(layout), key=lambda s: (s[0], s[1]))
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
            _fit_layout_sites_to_template(layout, template)
        except ValueError:
            continue
        best_cols = cols
        break
    if best_cols is None:
        raise ValueError("could not find a triangular gadget size that fits the layout")
    return rows, best_cols


def _make_triangular_layout(deformation_result, plot=False, **tri_layout_kwargs):
    """Run the abstract deformation through the triangular-layout stage."""
    layout = tri_layout(deformation_result, **tri_layout_kwargs)
    if not plot and layout.get("fig") is not None:
        import matplotlib.pyplot as plt

        plt.close(layout["fig"])
    return layout



def _long_opposite_path_count(layout, shuttling_threshold):
    if shuttling_threshold is None:
        return 0
    threshold = int(shuttling_threshold)
    sources = layout.get("stabilizers", {}).get("opposite_row_sources", [])
    count = 0
    for source in sources[: int(np.asarray(layout["H_opposite_basis_old_padded"]).shape[0])]:
        for path in source.get("paths", []):
            if len(path) >= 2 and len(path) - 1 > threshold:
                count += 1
    return count


def _long_abstract_route_count(layout, shuttling_threshold):
    if shuttling_threshold is None:
        return 0
    threshold = int(shuttling_threshold)
    return sum(
        1
        for path in layout.get("routes", {}).values()
        if len(path) >= 2 and len(path) - 1 > threshold
    )


def _old_opposite_edge_components(deformation_result, row):
    M = np.asarray(deformation_result["H_opposite_basis_old_padded"], dtype=np.uint8) % 2
    n_data = int(deformation_result["n_original_qubits"])
    edge_cols = np.flatnonzero(M[int(row), n_data:]).astype(int).tolist()
    graph_edges = _layout_graph_edge_order(deformation_result)
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


def _long_old_opposite_component_count(layout, shuttling_threshold):
    if shuttling_threshold is None:
        return 0
    threshold = int(shuttling_threshold)
    M = np.asarray(layout["H_opposite_basis_old_padded"], dtype=np.uint8)
    graph_edges = _layout_graph_edge_order(layout["deformation_result"])
    routes = layout.get("routes", {})
    count = 0
    for row in range(M.shape[0]):
        for component in _old_opposite_edge_components(layout["deformation_result"], row):
            if len(component["odd_vertices"]) != 2:
                continue
            routed_length = 0
            for edge_col in component["edge_cols"]:
                if int(edge_col) >= len(graph_edges):
                    continue
                path = routes.get(graph_edges[int(edge_col)], [])
                routed_length += max(1, len(path) - 1)
            if routed_length > threshold:
                count += 1
    return count


def _old_opposite_two_endpoint_component_count(layout):
    """Conservative shuttle-capacity bound for old opposite components.

    Whether an old opposite component is long can change after abstract routes
    are replaced by shuttles.  Reserving one possible shuttle per two-endpoint
    component avoids running out of side qubits during that later compression
    pass.  Unused reserved qubits remain outside the returned H supports.
    """
    M = np.asarray(layout["H_opposite_basis_old_padded"], dtype=np.uint8)
    count = 0
    for row in range(M.shape[0]):
        for component in _old_opposite_edge_components(layout["deformation_result"], row):
            if len(component["odd_vertices"]) == 2:
                count += 1
    return count


def _place_shuttling_edge_coords(template, records):
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


def _fixed_floor_graph(template, allowed_vertices=None):
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


def _shortest_floor_edge_path(template, start, stop, allowed_vertices, forbidden_edges=None):
    start = int(start)
    stop = int(stop)
    if start == stop:
        return []
    forbidden_edges = set() if forbidden_edges is None else set(map(int, forbidden_edges))
    G = _fixed_floor_graph(template, allowed_vertices=allowed_vertices)
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


def _vertex_weights_from_rows(H_vertex, vertex_row_to_check_qubit):
    H_vertex = np.asarray(H_vertex, dtype=np.uint8)
    return {
        int(qubit): int(np.sum(H_vertex[int(row)]))
        for row, qubit in vertex_row_to_check_qubit.items()
    }


def _path_fits_vertex_weight_cap(path_edges, materialized_edges, vertex_weights, template, max_vertex_degree):
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


def _max_vertex_weight_after_adding_path(path_edges, materialized_edges, vertex_weights, template):
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


def _add_floor_edges_to_vertex_checks(
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
    H_vertex = _add_edge_columns_to_vertex_checks(
        H_vertex,
        new_edges,
        template,
        vertex_row_to_check_qubit,
        allow_new_vertices=allow_new_vertices,
    )
    return H_vertex, sorted(set(map(int, materialized_edges)) | set(new_edges)), new_edges


def _rebuild_vertex_checks_from_edges(
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


def _choose_nearby_low_weight_vertex(
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
    G = _fixed_floor_graph(template, allowed_vertices=allowed_vertices)
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
        path_edges = _shortest_floor_edge_path(
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


def _build_shuttling_cycle_rows(
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
                    path = _shortest_floor_edge_path(
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
                if not _path_fits_vertex_weight_cap(
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
        vertex_weights = _vertex_weights_from_rows(H_basis_new, vertex_row_to_check_qubit)

        if record.get("force_replaced_path_closure"):
            closure_support = sorted(
                set(map(int, record.get("replaced_floor_edge_qubits", [])))
                | set(map(int, record.get("endpoint_adjustment_floor_edges", [])))
            )
            closure_floor_edges = [q for q in closure_support if int(q) in set(map(int, template.edge_qubits))]
            H_basis_new, floor_edges, added_edges = _add_floor_edges_to_vertex_checks(
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

        direct_path = _shortest_floor_edge_path(
            template,
            v,
            w,
            allowed_vertices,
            forbidden_edges=forbidden_floor_edges,
        )
        direct_ok = (
            direct_path is not None
            and len(direct_path) <= max_floor_path_length
            and _path_fits_vertex_weight_cap(
                direct_path,
                floor_edges,
                vertex_weights,
                template,
                max_vertex_degree,
            )
        )
        if direct_ok:
            H_basis_new, floor_edges, added_edges = _add_floor_edges_to_vertex_checks(
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
            H_basis_new, floor_edges, added_edges = _add_floor_edges_to_vertex_checks(
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

            u, path_high = _choose_nearby_low_weight_vertex(
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
                p, path_low = _choose_nearby_low_weight_vertex(
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
                paths_fit = _path_fits_vertex_weight_cap(
                    candidate_paths,
                    floor_edges,
                    vertex_weights,
                    template,
                    max_vertex_degree,
                )
                if endpoints_fit and paths_fit:
                    H_basis_new, floor_edges, added_edges = _add_floor_edges_to_vertex_checks(
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
        H_basis_new, floor_edges, added_edges = _add_floor_edges_to_vertex_checks(
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

    _place_shuttling_edge_coords(template, shuttling_records + extra_records)
    H = np.vstack(rows).astype(np.uint8) if rows else np.zeros((0, width), dtype=np.uint8)
    return H_basis_new, H, metadata, floor_edges, extra_shuttles, extra_records


def _basis_ancilla_maps(template, n_basis_rows, n_opposite_rows):
    return {
        "BB_basis_to_ancilla": _row_to_ancilla_map(template.basis_check_qubits, n_basis_rows),
        "BB_opposite_to_ancilla": _row_to_ancilla_map(template.opposite_check_qubits, n_opposite_rows),
    }


def _logical_observable_data(layout, H_basis_fixed, H_basis_new_fixed, n_basis_old_rows, vertex_row_to_check_qubit, width):
    n_data = int(layout["n_data_qubits"])
    logical_full = np.zeros(width, dtype=np.uint8)
    logical = np.asarray(layout["logical_full"][:n_data], dtype=np.uint8)
    logical_full[:len(logical)] = logical

    vertex_coeffs = _gf2_row_combination_for_target(H_basis_new_fixed, logical_full)
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


def plot_compiled_tri_lattice_gadget(tri_def, ax=None, show_labels=True, legend=True):
    """Plot the compiled code rectangle, triangular gadget floor, and ports."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    template = tri_def["fixed_template"]
    coords = template.coords
    used_edges = set(map(int, tri_def.get("fixed_edge_qubits_used", [])))
    shuttling_edges = set(map(int, tri_def.get("shuttling_edge_qubits", [])))
    floor_used_edges = used_edges - shuttling_edges
    used_vertices = set(map(int, tri_def.get("auxiliary_graph_incidence_vertices", [])))
    H_cycle = np.asarray(tri_def.get("gadget_H_opposite", []), dtype=np.uint8)
    cycle_sources = tri_def.get("cycle_sources", [])
    row_to_cycle_ancillas = tri_def.get("cycle_row_to_check_qubits", {})
    floor_cycle_ancillas = set()
    shuttling_cycle_ancillas = set()
    for row, qs in row_to_cycle_ancillas.items():
        row = int(row)
        ancillas = set(map(int, qs))
        support = set(map(int, np.flatnonzero(H_cycle[row]))) if row < H_cycle.shape[0] else set()
        source_type = ""
        if row < len(cycle_sources):
            source_type = str(cycle_sources[row].get("type", "")).lower()
        if support & shuttling_edges or "shuttling" in source_type:
            shuttling_cycle_ancillas |= ancillas
        else:
            floor_cycle_ancillas |= ancillas
    shared_cycle_ancillas = floor_cycle_ancillas & shuttling_cycle_ancillas
    floor_only_cycle_ancillas = floor_cycle_ancillas - shared_cycle_ancillas
    shuttling_only_cycle_ancillas = shuttling_cycle_ancillas - shared_cycle_ancillas
    logical_ports = set(map(int, tri_def.get("port_qubits", tri_def.get("logical_support_to_port_qubit", {}).values())))

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 8))
    else:
        fig = ax.figure

    x0, y0, w, h = template.code_patch_rect
    ax.add_patch(Rectangle(
        (x0, y0),
        w,
        h,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=1,
    ))
    ax.text(
        x0 + w / 2,
        y0 + h / 2,
        template.code_patch_label,
        ha="center",
        va="center",
        fontsize=10,
        zorder=2,
    )

    for q, (a, b) in template.edge_qubit_to_vertices.items():
        if int(q) in shuttling_edges:
            continue
        xa, ya = coords[a]
        xb, yb = coords[b]
        color = "#e8e8e8"
        lw = 0.6
        alpha = 0.55
        z = 1
        if q in floor_used_edges:
            color = "#d62728"
            lw = 2.3
            alpha = 0.95
            z = 4
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=z)

    def scatter(qubits, *, color, marker="o", size=20, label=None, zorder=3, edgecolor="none", alpha=1.0):
        qubits = [int(q) for q in qubits if int(q) in coords]
        if not qubits:
            return
        ax.scatter(
            [coords[q][0] for q in qubits],
            [coords[q][1] for q in qubits],
            s=size,
            c=color,
            marker=marker,
            label=label,
            zorder=zorder,
            edgecolors=edgecolor,
            alpha=alpha,
        )

    scatter(template.edge_qubits, color="#93c5fd", size=8, label="all floor edge qubits", zorder=2, alpha=0.32)
    scatter(floor_used_edges, color="#1d4ed8", size=14, label="used floor edge qubits", zorder=5)
    if shuttling_edges:
        shuttle_sorted = sorted(shuttling_edges)
        for x in sorted({round(coords[q][0], 8) for q in shuttle_sorted}):
            column = [q for q in shuttle_sorted if round(coords[q][0], 8) == x]
            if len(column) > 1:
                ax.plot(
                    [coords[q][0] for q in column],
                    [coords[q][1] for q in column],
                    color="#7c3aed",
                    linewidth=0.7,
                    alpha=0.35,
                    zorder=4,
                )
        scatter(shuttle_sorted, color="#7c3aed", marker="D", size=22, label="shuttling edge qubits", zorder=8, edgecolor="black")
    scatter(template.cycle_check_qubits, color="#fde68a", size=5, label="all cycle ancillas", zorder=3, alpha=0.22)
    scatter(template.shuttling_cycle_check_qubits, color="#fef3c7", size=7, label="all shuttling-cycle ancillas", zorder=3, alpha=0.45)
    scatter(floor_only_cycle_ancillas, color="#f59e0b", size=18, label="floor-cycle ancillas", zorder=6, edgecolor="black")
    scatter(shuttling_only_cycle_ancillas, color="#facc15", marker="o", size=20, label="shuttling-cycle ancillas", zorder=7, edgecolor="black")
    scatter(shared_cycle_ancillas, color="#f59e0b", marker="P", size=24, label="floor+shuttling cycle ancillas", zorder=7, edgecolor="#7c3aed")
    scatter(template.vertex_check_qubits, color="#9ca3af", size=9, label="all vertex ancillas", zorder=4, alpha=0.30)
    scatter(used_vertices, color="#111827", size=24, label="used vertex ancillas", zorder=7)
    scatter(template.port_qubits, color="#fca5a5", marker="s", size=18, label="all ports", zorder=5)
    scatter(logical_ports, color="#991b1b", marker="s", size=34, label="used logical ports", zorder=8)

    if show_labels and len(logical_ports) <= 24:
        for q in sorted(logical_ports):
            x, y = coords[q]
            ax.text(x, y + 0.28, str(q), ha="center", va="bottom", fontsize=6, color="#991b1b")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    visible_qubits = (
        list(template.edge_qubits)
        + list(template.vertex_check_qubits)
        + list(template.cycle_check_qubits)
        + list(template.shuttling_cycle_check_qubits)
        + list(template.port_qubits)
        + list(shuttling_edges)
    )
    visible_x = [coords[q][0] for q in visible_qubits if q in coords]
    visible_y = [coords[q][1] for q in visible_qubits if q in coords]
    rect_x = [x0, x0 + w]
    rect_y = [y0, y0 + h]
    ax.set_xlim(min(visible_x + rect_x) - 1.5, max(visible_x + rect_x) + 1.5)
    ax.set_ylim(min(visible_y + rect_y) - 1.0, max(visible_y + rect_y) + 1.0)
    ax.grid(True, alpha=0.12)
    ax.set_title("Compiled logical-measurement gadget")
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    return fig, ax


def _compile_layout_to_fixed_gadget(
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

    site_to_fixed_vertex, lattice_translation = _fit_layout_sites_to_template(layout, template)
    (
        optimized_vertex_placement,
        floor_routes,
        shuttled_abstract_edges,
        floor_edge_qubits_used,
    ) = _reroute_floor_after_abstract_shuttling(
        layout,
        template,
        site_to_fixed_vertex,
        shuttling_threshold,
    )
    shuttling_vertex_weights = _initial_vertex_weights_for_shuttling(
        layout["deformation_result"],
        template,
        optimized_vertex_placement,
        floor_edge_qubits_used,
    )
    abstract_shuttling_edge_qubits, abstract_shuttling_records = _allocate_abstract_route_shuttles(
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
    H_basis_old = _expand_old_basis_on_fixed_routes(
        layout["deformation_result"],
        floor_routes,
        abstract_shuttling_records,
        width,
    )
    H_opp_old = _expand_old_matrix_on_fixed_routes(
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
    H_basis_new, vertex_row_to_check_qubit = _seed_vertex_rows_from_placement(
        layout["deformation_result"],
        template,
        optimized_vertex_placement,
        width,
    )
    H_basis_new, vertex_row_to_check_qubit = _rebuild_vertex_checks_from_edges(
        H_basis_new,
        floor_edge_qubits_used,
        shuttling_edge_qubits_used,
        template,
        vertex_row_to_check_qubit,
        n_data,
    )
    component_vertex_weights = _vertex_weights_from_rows(H_basis_new, vertex_row_to_check_qubit)
    (
        H_opp_old,
        floor_edge_qubits_used,
        shuttling_edge_qubits_used,
        component_shuttling_records,
    ) = _compress_old_opposite_components_with_shuttles(
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
        H_basis_new, vertex_row_to_check_qubit = _rebuild_vertex_checks_from_edges(
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
    fixed_edge_qubits_used, expansion_added_edges, expansion_history, expansion_estimate = _repair_relative_expansion(
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
    H_basis_new = _add_edge_columns_to_vertex_checks(
        H_basis_new,
        expansion_added_edges,
        template,
        vertex_row_to_check_qubit,
    )
    fixed_edge_qubits_used = sorted(set(floor_edge_qubits_used) | set(shuttling_edge_qubits_used))
    H_cycles, cycle_sources = _recompute_cycle_basis_from_fixed_edges(fixed_edge_qubits_used, template, width=width)
    (
        H_basis_new,
        H_cycles,
        cycle_sources,
        floor_edge_qubits_used,
    ) = _split_floor_cycle_supports(
        H_basis_new,
        cycle_sources,
        floor_edge_qubits_used,
        template,
        vertex_row_to_check_qubit,
        width,
        max_cycle_weight=max_vertex_degree if floor_cycle_max_weight is None else floor_cycle_max_weight,
        max_vertex_degree=max_vertex_degree,
    )
    H_floor_cycles, floor_cycle_sources, H_shuttling_cycles, shuttling_cycle_sources = _partition_cycle_rows_by_shuttles(
        H_cycles,
        cycle_sources,
        shuttling_edge_qubits_used,
    )
    H_basis_new, vertex_row_to_check_qubit = _rebuild_vertex_checks_from_edges(
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

    cycle_row_to_check_qubits, cycle_row_to_check_qubit = _map_opposite_rows_to_cycle_ancillas(
        H_floor_cycles,
        H_shuttling_cycles,
        template,
    )
    graph_edges = _layout_graph_edge_order(layout["deformation_result"])
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
    auxiliary_graph_data = _fixed_auxiliary_graph_data(fixed_edge_qubits_used, template)
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
        "cycle_rows_are_simple_boundaries": _cycle_rows_are_simple_boundaries(H_opp_new, template),
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
        "debug": {
            "layout": layout,
            "fixed_template": template,
            "site_to_fixed_vertex_check_qubit": site_to_fixed_vertex,
            "lattice_translation": lattice_translation,
            "fixed_edge_qubits_used": fixed_edge_qubits_used,
            "unused_fixed_edge_qubits": unused_edge_qubits,
            "relative_expansion_history": expansion_history,
            "shuttling_records": shuttling_records,
            "shuttling_removed_floor_edge_qubits": shuttling_removed_floor_edges,
            "shuttling_cycle_sources": shuttling_cycle_sources,
        },
    }


def deform_logical_to_tri_lattice(
    deformation_result,
    basis=None,
    plot=False,
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
        3. rebuild compact H matrices over BB data + gadget edge qubits only;
        4. return the measurement-ancilla maps needed by Stim circuits.

    The returned H matrices have columns ``0..71`` for BB data qubits and
    ``72..243`` for gadget edge qubits. BB/gadget check qubits and ports are
    returned as row-to-ancilla metadata, not as H columns.
    """
    layout = _make_triangular_layout(deformation_result, plot=plot, **tri_layout_kwargs)
    n_data = int(layout["n_data_qubits"])
    n_basis_checks = int(np.asarray(layout["H_basis_old"]).shape[0])
    n_opposite_checks = int(np.asarray(layout["H_opposite_basis_old_padded"]).shape[0])
    logical_weight = int(np.sum(np.asarray(layout["logical_full"][:n_data], dtype=np.uint8)))
    port_order = _port_order_from_deformation(deformation_result)
    n_ports = max(1, len(port_order["label_to_port_vertex"]), logical_weight)
    rows, cols = _template_shape_for_layout(layout)
    n_abstract_shuttles = _long_abstract_route_count(layout, shuttling_threshold)
    n_old_opposite_shuttle_capacity = _old_opposite_two_endpoint_component_count(layout)
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
    fixed = _compile_layout_to_fixed_gadget(
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

    fixed.update(_basis_ancilla_maps(
        template,
        fixed["BB_H_basis"].shape[0],
        fixed["BB_H_opposite"].shape[0],
    ))
    fixed.update(_logical_observable_data(
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
        "basis": "X" if _basis_is_x(basis) else "Z",
        "fixed_ordering_ok": True,
        **fixed,
    }
    if plot:
        fig, ax = plot_compiled_tri_lattice_gadget(result)
        result["fig"] = fig
        result["ax"] = ax
    else:
        result["fig"] = None
        result["ax"] = None

    return result


def plot_highest_weight_stabilizer(tri_def, kind="both", ax=None, show_labels=True):
    """Plot the highest-weight stabilizer support on the fixed triangular gadget.

    Parameters
    ----------
    tri_def:
        Output of ``deform_logical_to_tri_lattice``.
    kind:
        ``"basis"``, ``"opposite"``, or ``"both"``.  ``"both"`` searches both
        ``H_basis_def`` and ``H_opposite_basis_def`` and plots the heaviest row.
    ax:
        Optional Matplotlib axis.
    show_labels:
        If true, label highlighted qubits/checks.
    """
    import matplotlib.pyplot as plt

    template = tri_def.get("fixed_template") or fixed_triangular_template()
    matrices = []
    if kind in ("basis", "both"):
        matrices.append(("basis", np.asarray(tri_def["H_basis_def"], dtype=np.uint8)))
    if kind in ("opposite", "both"):
        matrices.append(("opposite", np.asarray(tri_def["H_opposite_basis_def"], dtype=np.uint8)))
    if not matrices:
        raise ValueError("kind must be 'basis', 'opposite', or 'both'")

    best = None
    for name, H in matrices:
        weights = np.sum(H, axis=1).astype(int)
        if len(weights) == 0:
            continue
        row = int(np.argmax(weights))
        candidate = (int(weights[row]), name, row, H[row])
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("no stabilizer rows to plot")

    weight, matrix_name, row, support_row = best
    support = set(map(int, np.flatnonzero(support_row)))
    edge_support = sorted(q for q in support if q in template.edge_qubit_to_vertices)
    data_support = sorted(q for q in support if q < int(tri_def.get("n_data", BB_DATA_QUBITS.stop)))

    check_qubits = []
    if matrix_name == "basis":
        n_old = tri_def["BB_H_basis"].shape[0]
        if row < n_old:
            check_qubits = [tri_def["BB_basis_to_ancilla"].get(row)]
        else:
            check_qubits = [tri_def["gadget_basis_to_ancilla"].get(row - n_old)]
    else:
        n_old = tri_def["BB_H_opposite"].shape[0]
        if row < n_old:
            check_qubits = [tri_def["BB_opposite_to_ancilla"].get(row)]
        else:
            check_qubits = tri_def["cycle_row_to_check_qubits"].get(row - n_old, [])
    check_qubits = [int(q) for q in check_qubits if q is not None]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Full fixed triangular lattice, faint.
    shuttling_edges = set(map(int, tri_def.get("shuttling_edge_qubits", [])))
    for q, (a, b) in template.edge_qubit_to_vertices.items():
        if int(q) in shuttling_edges:
            continue
        xa, ya = template.coords[a]
        xb, yb = template.coords[b]
        ax.plot([xa, xb], [ya, yb], color="0.88", linewidth=0.7, zorder=0)

    # Used auxiliary graph edges.
    for q in tri_def.get("fixed_edge_qubits_used", []):
        if int(q) in shuttling_edges:
            continue
        a, b = template.edge_qubit_to_vertices[int(q)]
        xa, ya = template.coords[a]
        xb, yb = template.coords[b]
        ax.plot([xa, xb], [ya, yb], color="#8fb9dd", linewidth=1.3, alpha=0.8, zorder=1)
    for q in sorted(shuttling_edges):
        if q in template.coords:
            xq, yq = template.coords[q]
            ax.scatter([xq], [yq], s=38, marker="D", color="#7c3aed", edgecolor="black", linewidth=0.5, zorder=2)

    # Highlight stabilizer edge support.
    for q in edge_support:
        if q in shuttling_edges:
            xq, yq = template.coords[q]
            ax.scatter([xq], [yq], s=70, marker="D", color="#d62728", edgecolor="black", linewidth=0.6, zorder=5)
            if show_labels:
                ax.text(xq, yq, str(q), fontsize=7, ha="center", va="center", color="white", zorder=6)
            continue
        a, b = template.edge_qubit_to_vertices[q]
        xa, ya = template.coords[a]
        xb, yb = template.coords[b]
        ax.plot([xa, xb], [ya, yb], color="#d62728", linewidth=4.0, solid_capstyle="round", zorder=4)
        xq, yq = template.coords[q]
        ax.scatter([xq], [yq], s=32, color="#d62728", edgecolor="black", linewidth=0.4, zorder=5)
        if show_labels:
            ax.text(xq, yq, str(q), fontsize=7, ha="center", va="center", color="white", zorder=6)

    # Draw all used vertices and highlight vertices touched by edge support.
    used_vertices = set(tri_def.get("auxiliary_graph_incidence_vertices", []))
    touched_vertices = {v for q in edge_support for v in template.edge_qubit_to_vertices[q]}
    for q in sorted(used_vertices):
        xq, yq = template.coords[int(q)]
        color = "#f28e2b" if q in touched_vertices else "#6baed6"
        ax.scatter([xq], [yq], s=60, color=color, edgecolor="black", linewidth=0.5, zorder=3)

    # Highlight measurement ancilla(s) for this row.
    for q in check_qubits:
        if q in template.coords:
            xq, yq = template.coords[q]
            ax.scatter([xq], [yq], s=120, marker="*", color="#9467bd", edgecolor="black", linewidth=0.7, zorder=7)
            if show_labels:
                ax.text(xq, yq + 0.28, f"check {q}", fontsize=8, ha="center", va="bottom", zorder=8)

    # BB data support is off the triangular gadget but useful to report.
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"Highest-weight {matrix_name} stabilizer row {row}, weight {weight}\n"
        f"edge support={len(edge_support)}, BB data support={data_support}"
    )
    fig.tight_layout()
    return fig, ax, {
        "matrix": matrix_name,
        "row": row,
        "weight": weight,
        "support": sorted(support),
        "edge_support": edge_support,
        "data_support": data_support,
        "check_qubits": check_qubits,
    }


__all__ = [
    "fixed_triangular_template",
    "deform_logical_to_tri_lattice",
    "plot_highest_weight_stabilizer",
    "BB_DATA_QUBITS",
    "BB_X_CHECK_QUBITS",
    "BB_Z_CHECK_QUBITS",
    "GADGET_VERTEX_CHECK_START",
    "GADGET_EDGE_START",
    "H_MATRIX_QUBITS",
    "GADGET_CYCLE_CHECK_START",
    "GADGET_PORT_START",
    "N_FIXED_QUBITS",
]
