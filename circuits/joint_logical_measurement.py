""" joint-logical merge between two triangular deformation gadgets.

Main entry point
----------------
    merge_code_stabilizers(...)

The merge connects matching qubits in ports from two compiled
``deform_logical_to_tri_lattice`` results.  Each port-pair gets one adapter edge
qubit. Stabilizers are updates with the new adapter qubit included.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qldpc.objects import Pauli

from deformation.deform import deform_code_for_logical
from deformation.deform_triangular import deform_logical_to_tri_lattice


def is_x_basis(basis):
    return basis == Pauli.X or str(basis).upper().endswith("X")


def get_PCMS(H):
    H = np.asarray(H, dtype=np.uint8)
    n_rows, n_cols = H.shape
    n_data = n_cols // 2
    n_checks = n_rows // 2
    return H[:n_checks, :n_data], H[n_checks:, n_data:]


def shift_matrix(M, total_cols, offset):
    M = np.asarray(M, dtype=np.uint8)
    out = np.zeros((M.shape[0], total_cols), dtype=np.uint8)
    if M.size:
        out[:, offset:offset + M.shape[1]] = M
    return out


def rows_with_added_qubits(M, row_to_extra_qubits):
    out = np.asarray(M, dtype=np.uint8).copy()
    for row, qubits in row_to_extra_qubits.items():
        for q in qubits:
            out[int(row), int(q)] ^= 1
    return out


def matrix_with_rows(rows, width):
    rows = [np.asarray(row, dtype=np.uint8) for row in rows]
    return np.vstack(rows) if rows else np.zeros((0, width), dtype=np.uint8)


def checks_by_type(tri_def, basis):
    if is_x_basis(basis):
        return {
            "bb_x_checks": tri_def["BB_H_basis"],
            "bb_z_checks": tri_def["BB_H_opposite"],
            "vertex_pauli": "X",
            "cycle_pauli": "Z",
        }
    return {
        "bb_x_checks": tri_def["BB_H_opposite"],
        "bb_z_checks": tri_def["BB_H_basis"],
        "vertex_pauli": "Z",
        "cycle_pauli": "X",
    }


def compile_tri_def(code=None, logical_support=None, basis=Pauli.Z, deformation_result=None, tri_def=None, **tri_kwargs):
    if tri_def is not None:
        return tri_def
    if deformation_result is None:
        if code is None or logical_support is None:
            raise ValueError("Provide tri_def, deformation_result, or code+logical_support")
        deformation_result = deform_code_for_logical(code.matrix, basis, logical_support)
    return deform_logical_to_tri_lattice(deformation_result, basis=basis, plot=False, **tri_kwargs)


def _int_key_dict(d):
    return {int(k): v for k, v in dict(d or {}).items()}


def _as_int_list(value):
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(x) for x in value]


def port_vertex_qubit(tri_def, label):
    label = int(label)
    row_by_label = _int_key_dict(tri_def.get("port_label_to_vertex_row", {}))
    row = row_by_label.get(label)
    if row is None:
        return None
    q_by_row = _int_key_dict(
        tri_def.get("vertex_row_to_check_qubit", tri_def.get("gadget_basis_to_ancilla", {}))
    )
    q = q_by_row.get(int(row))
    return None if q is None else int(q)


def port_measurement_vertex_ids(tri_def):
    """Best-effort map from port label to measurement-graph vertex id."""
    out = {
        int(label): int(node)
        for label, node in _int_key_dict(tri_def.get("port_label_to_measurement_vertex_id", {})).items()
        if node is not None
    }

    vertex_qubit_to_node = {
        int(q): int(node)
        for q, node in _int_key_dict(tri_def.get("vertex_qubit_to_vertex_id", {})).items()
        if node is not None
    }
    G = tri_def.get("measurement_graph", tri_def.get("measurement_graph_networkx"))
    if G is not None:
        for node, data in G.nodes(data=True):
            q = data.get("qubit") if isinstance(data, dict) else None
            if q is not None:
                vertex_qubit_to_node[int(q)] = int(node)

    for label in _int_key_dict(tri_def.get("port_label_to_vertex_row", {})):
        q = port_vertex_qubit(tri_def, label)
        if q in vertex_qubit_to_node:
            out[int(label)] = int(vertex_qubit_to_node[q])
    return out


def _available_port_labels(tri_def):
    labels = set(_int_key_dict(tri_def.get("port_label_to_vertex_row", {})))
    labels |= set(_int_key_dict(tri_def.get("port_label_to_measurement_vertex_id", {})))
    labels |= set(_int_key_dict(tri_def.get("port_label_to_qubit", {})))
    labels |= set(_int_key_dict(tri_def.get("port_label_to_vertex", {})))
    # A merge must update a vertex-check row, so labels without a vertex row are unusable.
    labels &= set(_int_key_dict(tri_def.get("port_label_to_vertex_row", {})))
    return labels


def _old_edge_route_qubits(tri_def, abstract_edge_id):
    old_to_fixed = _int_key_dict(tri_def.get("old_to_fixed_qubit", {}))
    n_data = int(tri_def.get("n_data", 0))
    value = old_to_fixed.get(n_data + int(abstract_edge_id), old_to_fixed.get(int(abstract_edge_id)))
    return _as_int_list(value)


def _port_tree_route_edge_qubits(tri_def, left_label, right_label):
    """Route between port labels using saved skip-tree abstract-edge rows."""
    rows = list(tri_def.get("port_tree_rows", []))
    if not rows:
        return []
    left_label = int(left_label)
    right_label = int(right_label)
    lo, hi = sorted((left_label, right_label))
    if lo == hi or hi > len(rows):
        return []

    support = set()
    for k in range(lo, hi):
        for abstract_edge_id in rows[k]:
            for q in _old_edge_route_qubits(tri_def, int(abstract_edge_id)):
                if q in support:
                    support.remove(q)
                else:
                    support.add(q)
    return sorted(support)


def route_edge_qubits_between_port_labels(tri_def, left_label, right_label):
    ports = port_measurement_vertex_ids(tri_def)
    if int(left_label) in ports and int(right_label) in ports:
        try:
            return route_edge_qubits(tri_def, ports[int(left_label)], ports[int(right_label)])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    route = _port_tree_route_edge_qubits(tri_def, left_label, right_label)
    if route:
        return route
    raise ValueError(
        f"No route inside gadget between port labels {int(left_label)} and {int(right_label)}"
    )


def port_coord(tri_def, label, shift=0.0):
    label = int(label)
    G = tri_def.get("measurement_graph", tri_def.get("measurement_graph_networkx"))
    ports = port_measurement_vertex_ids(tri_def)
    if G is not None and label in ports and int(ports[label]) in G.nodes:
        coord = G.nodes[int(ports[label])].get("coord")
        if coord is not None:
            return float(coord[0]) + shift, float(coord[1])

    coords = tri_def.get("coords", {})
    candidates = [
        port_vertex_qubit(tri_def, label),
        _int_key_dict(tri_def.get("port_label_to_qubit", {})).get(label),
        _int_key_dict(tri_def.get("port_label_to_logical_support_qubit", {})).get(label),
    ]
    for q in candidates:
        if q is not None and int(q) in coords:
            x, y = coords[int(q)]
            return float(x) + shift, float(y)
    return float(label) + shift, 0.0


def port_qubit_coord(tri_def, label, shift=0.0):
    label = int(label)
    coords = tri_def.get("coords", {})
    q = _int_key_dict(tri_def.get("port_label_to_qubit", {})).get(label)
    if q is not None and int(q) in coords:
        x, y = coords[int(q)]
        return float(x) + shift, float(y)
    return port_coord(tri_def, label, shift=shift)


def _port_mean_x(tri_def, labels):
    xs = [port_qubit_coord(tri_def, label)[0] for label in labels]
    return float(np.mean(xs)) if xs else 0.0


def ordered_port_labels(tri_def1, tri_def2):
    labels = sorted(_available_port_labels(tri_def1) & _available_port_labels(tri_def2))
    if not labels:
        raise ValueError("No matching port labels with vertex-check rows in both gadgets")
    return labels


def route_edge_qubits(tri_def, start_vertex_id, end_vertex_id):
    G = tri_def.get("measurement_graph", tri_def.get("measurement_graph_networkx"))
    if G is None:
        raise ValueError("tri_def does not contain a measurement graph")
    path = nx.shortest_path(G, int(start_vertex_id), int(end_vertex_id))
    edge_qubits = []
    for a, b in zip(path, path[1:]):
        edge_qubits.append(int(G.edges[int(a), int(b)]["qubit"]))
    return edge_qubits


def adapter_cycle_rows(tri_def1, tri_def2, labels, adapter_qubits, total_cols, patch2_offset):
    rows = []
    routes = []

    for i, (left_label, right_label) in enumerate(zip(labels, labels[1:])):
        p1_route = route_edge_qubits_between_port_labels(tri_def1, left_label, right_label)
        p2_route = route_edge_qubits_between_port_labels(tri_def2, left_label, right_label)
        row = np.zeros(total_cols, dtype=np.uint8)
        row[int(adapter_qubits[i])] ^= 1
        row[int(adapter_qubits[i + 1])] ^= 1
        for q in p1_route:
            row[int(q)] ^= 1
        for q in p2_route:
            row[int(patch2_offset + q)] ^= 1
        rows.append(row)
        routes.append({
            "labels": (int(left_label), int(right_label)),
            "adapter_qubits": (int(adapter_qubits[i]), int(adapter_qubits[i + 1])),
            "patch1_route_edge_qubits": [int(q) for q in p1_route],
            "patch2_route_edge_qubits": [int(q) for q in p2_route],
        })
    return matrix_with_rows(rows, total_cols), routes


def _fresh_ancilla_map(start, n_rows):
    return {int(row): int(start) + int(row) for row in range(int(n_rows))}


def _offset_row_map(row_to_ancilla, row_offset):
    return {int(row_offset) + int(row): int(ancilla) for row, ancilla in row_to_ancilla.items()}


def _allocate_merged_ancillas(patch1, patch2, n_adapter_cycle_checks, ancilla_start):
    components = [
        ("patch1_bb_x", patch1["n_bb_x_checks"]),
        ("patch1_bb_z", patch1["n_bb_z_checks"]),
        ("patch1_cycle", patch1["n_cycle_checks"]),
        ("patch1_vertex", patch1["n_vertex_checks"]),
        ("patch2_bb_x", patch2["n_bb_x_checks"]),
        ("patch2_bb_z", patch2["n_bb_z_checks"]),
        ("patch2_cycle", patch2["n_cycle_checks"]),
        ("patch2_vertex", patch2["n_vertex_checks"]),
        ("adapter_cycle", n_adapter_cycle_checks),
    ]

    maps = {}
    q = int(ancilla_start)
    for name, n_rows in components:
        maps[name] = _fresh_ancilla_map(q, n_rows)
        q += int(n_rows)
    return maps, list(range(int(ancilla_start), q)), q


def _merged_check_to_ancilla_maps(ancilla_maps, patch1, patch2, n_adapter_cycle_checks, basis):
    x_map = {}
    z_map = {}

    if is_x_basis(basis):
        x_order = [
            ("patch1_bb_x", patch1["n_bb_x_checks"]),
            ("patch2_bb_x", patch2["n_bb_x_checks"]),
            ("patch1_vertex", patch1["n_vertex_checks"]),
            ("patch2_vertex", patch2["n_vertex_checks"]),
        ]
        z_order = [
            ("patch1_bb_z", patch1["n_bb_z_checks"]),
            ("patch2_bb_z", patch2["n_bb_z_checks"]),
            ("patch1_cycle", patch1["n_cycle_checks"]),
            ("patch2_cycle", patch2["n_cycle_checks"]),
            ("adapter_cycle", n_adapter_cycle_checks),
        ]
    else:
        x_order = [
            ("patch1_bb_x", patch1["n_bb_x_checks"]),
            ("patch2_bb_x", patch2["n_bb_x_checks"]),
            ("patch1_cycle", patch1["n_cycle_checks"]),
            ("patch2_cycle", patch2["n_cycle_checks"]),
            ("adapter_cycle", n_adapter_cycle_checks),
        ]
        z_order = [
            ("patch1_bb_z", patch1["n_bb_z_checks"]),
            ("patch2_bb_z", patch2["n_bb_z_checks"]),
            ("patch1_vertex", patch1["n_vertex_checks"]),
            ("patch2_vertex", patch2["n_vertex_checks"]),
        ]

    offset = 0
    for name, n_rows in x_order:
        x_map.update(_offset_row_map(ancilla_maps[name], offset))
        offset += int(n_rows)

    offset = 0
    for name, n_rows in z_order:
        z_map.update(_offset_row_map(ancilla_maps[name], offset))
        offset += int(n_rows)

    return x_map, z_map


def plot_connected_gadgets(merged, ax=None, figsize=(12, 9)):
    """Plot two full compiled triangular gadgets facing each other.

    Patch 1 is drawn in its normal orientation. Patch 2 is mirrored vertically
    so the two port rows face each other, and matching ports are connected by
    one adapter edge qubit.
    """
    from matplotlib.patches import Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    tri1 = merged["patch1"]["tri_def"]
    tri2 = merged["patch2"]["tri_def"]
    labels = [int(x) for x in merged["port_labels"]]
    adapter_gap = float(merged.get("plot_adapter_gap", 2.5))
    x_shift = float(merged.get("plot_patch2_x_shift", _port_mean_x(tri1, labels) - _port_mean_x(tri2, labels)))
    drawn_x = []
    drawn_y = []

    def patch_y_ref(tri_def):
        ys = [port_qubit_coord(tri_def, label)[1] for label in labels]
        if ys:
            return float(np.mean(ys))
        coords = tri_def["fixed_template"].coords
        return max(y for _, y in coords.values())

    lower_port_y = patch_y_ref(tri1)
    upper_port_y = patch_y_ref(tri2)

    def transform_patch1(x, y):
        return float(x), float(y)

    def transform_patch2(x, y):
        return float(x) + x_shift, lower_port_y + adapter_gap + (upper_port_y - float(y))

    def add_visible(points):
        for x, y in points:
            drawn_x.append(float(x))
            drawn_y.append(float(y))

    def draw_rect(template, transform, mirror=False):
        x0, y0, w, h = template.code_patch_rect
        corners = [
            transform(x0, y0),
            transform(x0 + w, y0),
            transform(x0, y0 + h),
            transform(x0 + w, y0 + h),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        ax.add_patch(Rectangle(
            (min(xs), min(ys)),
            max(xs) - min(xs),
            max(ys) - min(ys),
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=1,
        ))
        tx, ty = transform(x0 + w / 2, y0 + h / 2)
        ax.text(tx, ty, template.code_patch_label, ha="center", va="center", fontsize=10, zorder=2)
        add_visible(corners)

    def draw_compiled_patch(tri_def, transform, *, labels_once=True, name="patch"):
        template = tri_def["fixed_template"]
        coords = dict(template.coords)
        coords.update(tri_def.get("coords", {}))
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

        def xy(q):
            return transform(*coords[int(q)])

        draw_rect(template, transform)

        for q, (a, b) in template.edge_qubit_to_vertices.items():
            q = int(q)
            if q in shuttling_edges or q not in coords or int(a) not in coords or int(b) not in coords:
                continue
            x1, y1 = xy(a)
            x2, y2 = xy(b)
            color = "#e8e8e8"
            lw = 0.6
            alpha = 0.55
            z = 1
            if q in floor_used_edges:
                color = "#d62728"
                lw = 2.3
                alpha = 0.95
                z = 4
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=z)
            add_visible([(x1, y1), (x2, y2)])

        def label(text):
            return text if labels_once else None

        def scatter(qubits, *, color, marker="o", size=20, label_text=None, zorder=3, edgecolor="none", alpha=1.0):
            pts = [xy(q) for q in qubits if int(q) in coords]
            if not pts:
                return
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                s=size,
                c=color,
                marker=marker,
                label=label(label_text) if label_text else None,
                zorder=zorder,
                edgecolors=edgecolor,
                alpha=alpha,
            )
            add_visible(pts)

        scatter(template.edge_qubits, color="#93c5fd", size=8, label_text="all floor edge qubits", zorder=2, alpha=0.32)
        scatter(floor_used_edges, color="#1d4ed8", size=14, label_text="used floor edge qubits", zorder=5)
        if shuttling_edges:
            shuttle_sorted = sorted(q for q in shuttling_edges if q in coords)
            transformed = {q: xy(q) for q in shuttle_sorted}
            for x0 in sorted({round(transformed[q][0], 8) for q in shuttle_sorted}):
                column = [q for q in shuttle_sorted if round(transformed[q][0], 8) == x0]
                if len(column) > 1:
                    column.sort(key=lambda q: transformed[q][1])
                    ax.plot(
                        [transformed[q][0] for q in column],
                        [transformed[q][1] for q in column],
                        color="#7c3aed",
                        linewidth=0.7,
                        alpha=0.35,
                        zorder=4,
                    )
            scatter(shuttle_sorted, color="#7c3aed", marker="D", size=22, label_text="shuttling edge qubits", zorder=8, edgecolor="black")
        scatter(template.cycle_check_qubits, color="#fde68a", size=5, label_text="all cycle ancillas", zorder=3, alpha=0.22)
        scatter(template.shuttling_cycle_check_qubits, color="#fef3c7", size=7, label_text="all shuttling-cycle ancillas", zorder=3, alpha=0.45)
        scatter(floor_only_cycle_ancillas, color="#f59e0b", size=18, label_text="floor-cycle ancillas", zorder=6, edgecolor="black")
        scatter(shuttling_only_cycle_ancillas, color="#facc15", marker="o", size=20, label_text="shuttling-cycle ancillas", zorder=7, edgecolor="black")
        scatter(shared_cycle_ancillas, color="#f59e0b", marker="P", size=24, label_text="floor+shuttling cycle ancillas", zorder=7, edgecolor="#7c3aed")
        scatter(template.vertex_check_qubits, color="#9ca3af", size=9, label_text="all vertex ancillas", zorder=4, alpha=0.30)
        scatter(used_vertices, color="#111827", size=24, label_text="used vertex ancillas", zorder=7)
        scatter(template.port_qubits, color="#fca5a5", marker="s", size=18, label_text="all ports", zorder=5)
        scatter(logical_ports, color="#991b1b", marker="s", size=34, label_text="used logical ports", zorder=8)

        if len(logical_ports) <= 24:
            for q in sorted(logical_ports):
                if q in coords:
                    x0, y0 = xy(q)
                    ax.text(x0, y0 + 0.28, str(q), ha="center", va="bottom", fontsize=6, color="#991b1b")

    draw_compiled_patch(tri1, transform_patch1, labels_once=True, name="patch 1")
    draw_compiled_patch(tri2, transform_patch2, labels_once=False, name="patch 2")

    adapter_coords = {}
    for label, q in zip(labels, merged["adapter_qubits"]):
        x1, y1 = transform_patch1(*port_qubit_coord(tri1, label))
        x2, y2 = transform_patch2(*port_qubit_coord(tri2, label))
        ax.plot([x1, x2], [y1, y2], color="#7c3aed", linewidth=2.0, alpha=0.85, zorder=9)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        adapter_coords[int(q)] = (mx, my)
        ax.scatter([mx], [my], s=48, marker="D", color="#a855f7", edgecolor="black", label="adapter edge qubits" if q == merged["adapter_qubits"][0] else None, zorder=10)
        add_visible([(x1, y1), (x2, y2), (mx, my)])

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if drawn_x and drawn_y:
        ax.set_xlim(min(drawn_x) - 1.5, max(drawn_x) + 1.5)
        ax.set_ylim(min(drawn_y) - 1.0, max(drawn_y) + 1.0)
    ax.grid(True, alpha=0.12)
    ax.set_title("Merged logical-measurement gadgets with adapter qubits")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    return fig, ax, adapter_coords


def merge_code_stabilizers(
    code1=None,
    code2=None,
    logical_support1=None,
    logical_support2=None,
    basis1=Pauli.Z,
    basis2=None,
    deformation_result1=None,
    deformation_result2=None,
    tri_def1=None,
    tri_def2=None,
    plot=False,
    **tri_kwargs,
):
    """Merge two compiled triangular logical-measurement gadgets.

    Matching port label ``i`` in patch 1 is connected to matching port label
    ``i`` in patch 2 by one adapter edge qubit.  For each adjacent port pair
    ``i,i+1`` we add one cycle check made from the two adapter qubits and the
    shortest measurement-graph routes between those ports inside each patch.
    """
    basis2 = basis1 if basis2 is None else basis2
    if is_x_basis(basis1) != is_x_basis(basis2):
        raise ValueError("This simple merge expects both gadgets to use the same logical basis")

    tri_def1 = compile_tri_def(code1, logical_support1, basis1, deformation_result1, tri_def1, **tri_kwargs)
    tri_def2 = compile_tri_def(code2, logical_support2, basis2, deformation_result2, tri_def2, **tri_kwargs)

    n_h1 = int(tri_def1["n_h_qubits"])
    n_h2 = int(tri_def2["n_h_qubits"])
    patch2_offset = n_h1
    adapter_start = n_h1 + n_h2
    labels = ordered_port_labels(tri_def1, tri_def2)
    adapter_qubits = list(range(adapter_start, adapter_start + len(labels)))
    total_cols = adapter_start + len(adapter_qubits)

    type1 = checks_by_type(tri_def1, basis1)
    type2 = checks_by_type(tri_def2, basis2)

    p1_extra = {
        int(tri_def1["port_label_to_vertex_row"][label]): [adapter_qubits[i]]
        for i, label in enumerate(labels)
    }
    p2_extra = {
        int(tri_def2["port_label_to_vertex_row"][label]): [adapter_qubits[i]]
        for i, label in enumerate(labels)
    }

    patch1_vertex = rows_with_added_qubits(
        shift_matrix(tri_def1["gadget_H_basis"], total_cols, 0),
        p1_extra,
    )
    patch2_vertex = rows_with_added_qubits(
        shift_matrix(tri_def2["gadget_H_basis"], total_cols, patch2_offset),
        p2_extra,
    )
    patch1_cycle = shift_matrix(tri_def1["gadget_H_opposite"], total_cols, 0)
    patch2_cycle = shift_matrix(tri_def2["gadget_H_opposite"], total_cols, patch2_offset)
    new_cycles, adapter_cycle_routes = adapter_cycle_rows(
        tri_def1,
        tri_def2,
        labels,
        adapter_qubits,
        total_cols,
        patch2_offset,
    )

    patch1 = {
        "bb_x_checks": shift_matrix(type1["bb_x_checks"], total_cols, 0),
        "bb_z_checks": shift_matrix(type1["bb_z_checks"], total_cols, 0),
        "vertex_checks": patch1_vertex,
        "cycle_checks": patch1_cycle,
        "n_edge_qubits": int(tri_def1["n_edges"]),
        "n_bb_x_checks": int(type1["bb_x_checks"].shape[0]),
        "n_bb_z_checks": int(type1["bb_z_checks"].shape[0]),
        "n_vertex_checks": int(patch1_vertex.shape[0]),
        "n_cycle_checks": int(patch1_cycle.shape[0]),
        "offset": 0,
        "tri_def": tri_def1,
    }
    patch2 = {
        "bb_x_checks": shift_matrix(type2["bb_x_checks"], total_cols, patch2_offset),
        "bb_z_checks": shift_matrix(type2["bb_z_checks"], total_cols, patch2_offset),
        "vertex_checks": patch2_vertex,
        "cycle_checks": patch2_cycle,
        "n_edge_qubits": int(tri_def2["n_edges"]),
        "n_bb_x_checks": int(type2["bb_x_checks"].shape[0]),
        "n_bb_z_checks": int(type2["bb_z_checks"].shape[0]),
        "n_vertex_checks": int(patch2_vertex.shape[0]),
        "n_cycle_checks": int(patch2_cycle.shape[0]),
        "offset": patch2_offset,
        "tri_def": tri_def2,
    }

    if is_x_basis(basis1):
        x_checks = np.vstack([patch1["bb_x_checks"], patch2["bb_x_checks"], patch1_vertex, patch2_vertex])
        z_checks = np.vstack([patch1["bb_z_checks"], patch2["bb_z_checks"], patch1_cycle, patch2_cycle, new_cycles])
    else:
        x_checks = np.vstack([patch1["bb_x_checks"], patch2["bb_x_checks"], patch1_cycle, patch2_cycle, new_cycles])
        z_checks = np.vstack([patch1["bb_z_checks"], patch2["bb_z_checks"], patch1_vertex, patch2_vertex])

    n_adapter_cycle_checks = int(new_cycles.shape[0])
    ancilla_start = total_cols
    ancilla_maps, ancilla_qubits, n_total_qubits = _allocate_merged_ancillas(
        patch1,
        patch2,
        n_adapter_cycle_checks,
        ancilla_start,
    )
    x_check_to_ancilla, z_check_to_ancilla = _merged_check_to_ancilla_maps(
        ancilla_maps,
        patch1,
        patch2,
        n_adapter_cycle_checks,
        basis1,
    )

    patch1["bb_x_to_ancilla"] = ancilla_maps["patch1_bb_x"]
    patch1["bb_z_to_ancilla"] = ancilla_maps["patch1_bb_z"]
    patch1["cycle_to_ancilla"] = ancilla_maps["patch1_cycle"]
    patch1["vertex_to_ancilla"] = ancilla_maps["patch1_vertex"]
    patch1["ancilla_qubits"] = (
        list(ancilla_maps["patch1_bb_x"].values())
        + list(ancilla_maps["patch1_bb_z"].values())
        + list(ancilla_maps["patch1_cycle"].values())
        + list(ancilla_maps["patch1_vertex"].values())
    )

    patch2["bb_x_to_ancilla"] = ancilla_maps["patch2_bb_x"]
    patch2["bb_z_to_ancilla"] = ancilla_maps["patch2_bb_z"]
    patch2["cycle_to_ancilla"] = ancilla_maps["patch2_cycle"]
    patch2["vertex_to_ancilla"] = ancilla_maps["patch2_vertex"]
    patch2["ancilla_qubits"] = (
        list(ancilla_maps["patch2_bb_x"].values())
        + list(ancilla_maps["patch2_bb_z"].values())
        + list(ancilla_maps["patch2_cycle"].values())
        + list(ancilla_maps["patch2_vertex"].values())
    )

    n_data1 = int(tri_def1["n_data"])
    n_data2 = int(tri_def2["n_data"])
    qubit_index_ranges = {
        "patch1_data": (0, n_data1),
        "patch1_edges": (n_data1, n_h1),
        "patch2_data": (patch2_offset, patch2_offset + n_data2),
        "patch2_edges": (patch2_offset + n_data2, patch2_offset + n_h2),
        "adapter": (adapter_start, total_cols),
        "ancillas": (ancilla_start, n_total_qubits),
    }

    result = {
        "patch1": patch1,
        "patch2": patch2,
        "adapter_qubits": adapter_qubits,
        "adapter_cycle_checks": new_cycles,
        "adapter_cycle_routes": adapter_cycle_routes,
        "port_labels": labels,
        "n_adapter_qubits": len(adapter_qubits),
        "n_adapter_cycle_checks": n_adapter_cycle_checks,
        "n_h_qubits": total_cols,
        "n_total_h_qubits": total_cols,
        "n_ancilla_qubits": len(ancilla_qubits),
        "n_total_qubits": n_total_qubits,
        "ancilla_start": ancilla_start,
        "ancilla_qubits": ancilla_qubits,
        "ancilla_maps": ancilla_maps,
        "x_check_to_ancilla": x_check_to_ancilla,
        "z_check_to_ancilla": z_check_to_ancilla,
        "check_to_ancilla": {"x": x_check_to_ancilla, "z": z_check_to_ancilla},
        "adapter_cycle_to_ancilla": ancilla_maps["adapter_cycle"],
        "patch1_n_edge_qubits": patch1["n_edge_qubits"],
        "patch2_n_edge_qubits": patch2["n_edge_qubits"],
        "patch1_offset": 0,
        "patch2_offset": patch2_offset,
        "adapter_start": adapter_start,
        "qubit_index_ranges": qubit_index_ranges,
        "x_checks": x_checks.astype(np.uint8),
        "z_checks": z_checks.astype(np.uint8),
        "basis": "X" if is_x_basis(basis1) else "Z",
        "counts": {
            "patch1": {k: patch1[k] for k in ["n_edge_qubits", "n_bb_x_checks", "n_bb_z_checks", "n_vertex_checks", "n_cycle_checks"]},
            "patch2": {k: patch2[k] for k in ["n_edge_qubits", "n_bb_x_checks", "n_bb_z_checks", "n_vertex_checks", "n_cycle_checks"]},
            "adapter": {
                "n_adapter_qubits": len(adapter_qubits),
                "n_adapter_cycle_checks": n_adapter_cycle_checks,
            },
            "ancillas": {
                "n_ancilla_qubits": len(ancilla_qubits),
                "ancilla_start": ancilla_start,
            },
        },
    }

    if plot:
        result["plot_adapter_gap"] = 2.5
        result["plot_patch2_x_shift"] = _port_mean_x(tri_def1, labels) - _port_mean_x(tri_def2, labels)
        fig, ax, adapter_coords = plot_connected_gadgets(result)
        result["fig"] = fig
        result["ax"] = ax
        result["adapter_coords"] = adapter_coords
    else:
        result["plot_adapter_gap"] = 2.5
        result["plot_patch2_x_shift"] = 0.0
        result["fig"] = None
        result["ax"] = None
        result["adapter_coords"] = {}
    return result
