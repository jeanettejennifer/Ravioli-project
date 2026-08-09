from __future__ import annotations

import networkx as nx
import numpy as np
from qldpc.objects import Pauli
from deformation.deform import deform_code_for_logical
from deformation.deform_triangular import deform_logical_to_tri_lattice

def is_x_basis(basis):
    return basis == Pauli.X or str(basis).upper().endswith("X")


def _basis_name(basis):
    return "X" if is_x_basis(basis) else "Z"


def _as_list(value, name):
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"{name} must be a list/tuple, a 1D numpy array, or None")


def _int_dict(d):
    return {int(k): v for k, v in dict(d or {}).items()}


def _matrix_with_rows(rows, width):
    rows = [np.asarray(row, dtype=np.uint8) for row in rows]
    return np.vstack(rows) if rows else np.zeros((0, int(width)), dtype=np.uint8)


def _rows_with_added_qubits(M, row_to_extra_qubits):
    out = np.asarray(M, dtype=np.uint8).copy()
    for row, qubits in row_to_extra_qubits.items():
        for q in qubits:
            out[int(row), int(q)] ^= 1
    return out


def _fresh_ancilla_map(start, n_rows):
    return {int(row): int(start) + int(row) for row in range(int(n_rows))}


def _offset_row_map(row_to_ancilla, row_offset):
    return {int(row_offset) + int(row): int(ancilla) for row, ancilla in row_to_ancilla.items()}


def _edge_width(tri_def):
    return int(tri_def["n_h_qubits"]) - int(tri_def["n_data"])


def _place_local_matrix(M, *, n_data, edge_offset, total_cols):
    """Place a local data+edge matrix into shared-data global columns."""
    M = np.asarray(M, dtype=np.uint8) % 2
    out = np.zeros((M.shape[0], int(total_cols)), dtype=np.uint8)
    data_width = min(int(n_data), M.shape[1])
    out[:, :data_width] ^= M[:, :data_width]
    if M.shape[1] > n_data:
        out[:, int(edge_offset):int(edge_offset) + M.shape[1] - int(n_data)] ^= M[:, int(n_data):]
    return out


def _local_edge_to_global(q, tri_def, edge_offset):
    q = int(q)
    n_data = int(tri_def["n_data"])
    if q < n_data:
        return q
    return int(edge_offset) + q - n_data


def _available_port_labels(tri_def):
    labels = set(_int_dict(tri_def.get("port_label_to_vertex_row", {})))
    labels &= set(_int_dict(tri_def.get("port_label_to_qubit", {}))) | labels
    return sorted(labels)


def _ordered_common_port_labels(left, right):
    labels = sorted(set(_available_port_labels(left)) & set(_available_port_labels(right)))
    if not labels:
        raise ValueError("No matching skip-tree port labels between adjacent gadgets")
    return labels


def _old_edge_route_qubits(tri_def, abstract_edge_id):
    old_to_fixed = _int_dict(tri_def.get("old_to_fixed_qubit", {}))
    n_data = int(tri_def.get("n_data", 0))
    value = old_to_fixed.get(n_data + int(abstract_edge_id), old_to_fixed.get(int(abstract_edge_id)))
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(q) for q in value]


def _port_tree_route_edge_qubits(tri_def, left_label, right_label):
    """Return local edge qubits on the skip-tree route between two port labels."""
    rows = list(tri_def.get("port_tree_rows", []))
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


def _port_measurement_vertex_ids(tri_def):
    """Map port label to local measurement-graph vertex id."""
    out = {
        int(label): int(node)
        for label, node in _int_dict(tri_def.get("port_label_to_measurement_vertex_id", {})).items()
        if node is not None
    }
    if out:
        return out

    row_by_label = _int_dict(tri_def.get("port_label_to_vertex_row", {}))
    row_to_vertex_qubit = _int_dict(tri_def.get("vertex_row_to_check_qubit", tri_def.get("gadget_basis_to_ancilla", {})))
    vertex_qubit_to_id = _int_dict(tri_def.get("vertex_qubit_to_vertex_id", {}))
    for label, row in row_by_label.items():
        vertex_qubit = row_to_vertex_qubit.get(int(row))
        if vertex_qubit in vertex_qubit_to_id:
            out[int(label)] = int(vertex_qubit_to_id[int(vertex_qubit)])
    return out


def _route_edge_qubits_between_port_labels(tri_def, left_label, right_label):
    """Shortest edge-qubit route between two adapter endpoint ports."""
    G = tri_def.get("measurement_graph", tri_def.get("measurement_graph_networkx"))
    ports = _port_measurement_vertex_ids(tri_def)
    if G is not None and int(left_label) in ports and int(right_label) in ports:
        try:
            path = nx.shortest_path(G, int(ports[int(left_label)]), int(ports[int(right_label)]))
            return [int(G.edges[int(a), int(b)]["qubit"]) for a, b in zip(path, path[1:])]
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            pass

    route = _port_tree_route_edge_qubits(tri_def, left_label, right_label)
    if route:
        return route
    raise ValueError(f"No route inside gadget between port labels {int(left_label)} and {int(right_label)}")


def _compile_gadgets(
    *,
    code=None,
    logicals=None,
    logical_indices=None,
    deformation_results=None,
    tri_defs=None,
    basis=Pauli.Z,
    shuttling_threshold=None,
    **tri_kwargs,
):
    if tri_defs is not None:
        out = list(tri_defs)
        if len(out) < 2:
            raise ValueError("At least two gadgets are required")
        return out

    if deformation_results is not None:
        out = [
            deform_logical_to_tri_lattice(
                deformation_result,
                basis=basis,
                plot=False,
                shuttling_threshold=shuttling_threshold,
                **tri_kwargs,
            )
            for deformation_result in deformation_results
        ]
        if len(out) < 2:
            raise ValueError("At least two deformation results are required")
        return out

    if code is None:
        raise ValueError("Provide tri_defs, deformation_results, or code with logicals/logical_indices")

    if logicals is None:
        if logical_indices is None:
            raise ValueError("Provide logicals or logical_indices when using code input")
        all_logicals = code.get_logical_ops(basis)
        logicals = [all_logicals[int(i)] for i in logical_indices]
    else:
        logicals = _as_list(logicals, "logicals")

    out = []
    for logical in logicals:
        deformation_result = deform_code_for_logical(code.matrix, basis, logical)
        out.append(
            deform_logical_to_tri_lattice(
                deformation_result,
                basis=basis,
                plot=False,
                shuttling_threshold=shuttling_threshold,
                **tri_kwargs,
            )
        )
    if len(out) < 2:
        raise ValueError("At least two logicals are required")
    return out


def _check_same_data_block(tri_defs):
    n_data = int(tri_defs[0]["n_data"])
    for i, tri_def in enumerate(tri_defs):
        if int(tri_def["n_data"]) != n_data:
            raise ValueError(
                "Same-codeblock multi-body merge requires every gadget to have the same n_data; "
                f"gadget 0 has {n_data}, gadget {i} has {int(tri_def['n_data'])}"
            )
    return n_data


def _edge_offsets(tri_defs, n_data):
    offsets = []
    q = int(n_data)
    for tri_def in tri_defs:
        offsets.append(q)
        q += _edge_width(tri_def)
    return offsets, q


def _build_adapter_sets_and_cycles(tri_defs, edge_offsets, adapter_start, total_cols):
    adapter_sets = []
    cycle_rows = []
    cycle_routes = []
    q = int(adapter_start)

    for left in range(len(tri_defs) - 1):
        right = left + 1
        labels = _ordered_common_port_labels(tri_defs[left], tri_defs[right])
        adapters = list(range(q, q + len(labels)))
        q += len(adapters)

        adapter_set_index = len(adapter_sets)
        adapter_sets.append({
            "adapter_set_index": int(adapter_set_index),
            "left_gadget": int(left),
            "right_gadget": int(right),
            "port_labels": [int(x) for x in labels],
            "adapter_qubits": [int(a) for a in adapters],
        })

        for k, (label_a, label_b) in enumerate(zip(labels, labels[1:])):
            left_route = _route_edge_qubits_between_port_labels(tri_defs[left], label_a, label_b)
            right_route = _route_edge_qubits_between_port_labels(tri_defs[right], label_a, label_b)
            left_global_route = [
                _local_edge_to_global(local_q, tri_defs[left], edge_offsets[left])
                for local_q in left_route
            ]
            right_global_route = [
                _local_edge_to_global(local_q, tri_defs[right], edge_offsets[right])
                for local_q in right_route
            ]

            row = np.zeros(total_cols, dtype=np.uint8)
            row[int(adapters[k])] ^= 1
            row[int(adapters[k + 1])] ^= 1
            for q_global in left_global_route + right_global_route:
                row[int(q_global)] ^= 1
            cycle_rows.append(row)
            cycle_routes.append({
                "adapter_set_index": int(adapter_set_index),
                "left_gadget": int(left),
                "right_gadget": int(right),
                "endpoint_port_labels": (int(label_a), int(label_b)),
                "adapter_qubits": (int(adapters[k]), int(adapters[k + 1])),
                "left_local_route_edge_qubits": [int(q) for q in left_route],
                "right_local_route_edge_qubits": [int(q) for q in right_route],
                "left_global_route_edge_qubits": [int(q) for q in left_global_route],
                "right_global_route_edge_qubits": [int(q) for q in right_global_route],
            })

    return adapter_sets, _matrix_with_rows(cycle_rows, total_cols), cycle_routes, q


def _gadget_vertex_extras(tri_def, adapter_sets, patch_index):
    extras = {}
    for adapter_set in adapter_sets:
        if adapter_set["left_gadget"] == patch_index:
            labels = adapter_set["port_labels"]
            adapters = adapter_set["adapter_qubits"]
        elif adapter_set["right_gadget"] == patch_index:
            labels = adapter_set["port_labels"]
            adapters = adapter_set["adapter_qubits"]
        else:
            continue
        for label, adapter in zip(labels, adapters):
            row = int(tri_def["port_label_to_vertex_row"][int(label)])
            extras.setdefault(row, []).append(int(adapter))
    return extras


def _patch_data(tri_def, edge_offset, total_cols, adapter_sets, patch_index):
    vertex = _rows_with_added_qubits(
        _place_local_matrix(tri_def["gadget_H_basis"], n_data=tri_def["n_data"], edge_offset=edge_offset, total_cols=total_cols),
        _gadget_vertex_extras(tri_def, adapter_sets, patch_index),
    )
    cycle = _place_local_matrix(
        tri_def["gadget_H_opposite"],
        n_data=tri_def["n_data"],
        edge_offset=edge_offset,
        total_cols=total_cols,
    )
    return {
        "tri_def": tri_def,
        "offset": 0,
        "edge_offset": int(edge_offset),
        "n_data": int(tri_def["n_data"]),
        "n_edge_qubits": int(tri_def["n_edges"]),
        "n_edge_columns": _edge_width(tri_def),
        "n_h_qubits_local": int(tri_def["n_h_qubits"]),
        "vertex_checks": vertex,
        "cycle_checks": cycle,
        "n_vertex_checks": int(vertex.shape[0]),
        "n_cycle_checks": int(cycle.shape[0]),
        "incident_adapter_port_rows": _gadget_vertex_extras(tri_def, adapter_sets, patch_index),
    }


def _combine_bb_matrix(tri_defs, key, edge_offsets, total_cols, n_data):
    """One shared BB check matrix with every gadget's edge decorations included."""
    base = np.asarray(tri_defs[0][key], dtype=np.uint8) % 2
    out = np.zeros((base.shape[0], int(total_cols)), dtype=np.uint8)
    out[:, :n_data] ^= base[:, :n_data]
    for tri_def, edge_offset in zip(tri_defs, edge_offsets):
        M = np.asarray(tri_def[key], dtype=np.uint8) % 2
        if M.shape[0] != base.shape[0]:
            raise ValueError(f"All gadgets must have the same number of {key} rows")
        if M.shape[1] > n_data:
            out[:, int(edge_offset):int(edge_offset) + M.shape[1] - n_data] ^= M[:, n_data:]
    return out


def _allocate_ancillas(n_bb_basis, n_bb_opp, patches, n_adapter_cycles, ancilla_start):
    maps = {
        "bb_basis": _fresh_ancilla_map(ancilla_start, n_bb_basis),
    }
    q = int(ancilla_start) + int(n_bb_basis)
    maps["bb_opposite"] = _fresh_ancilla_map(q, n_bb_opp)
    q += int(n_bb_opp)
    for i, patch in enumerate(patches):
        maps[f"gadget{i}_basis"] = _fresh_ancilla_map(q, patch["n_vertex_checks"])
        q += patch["n_vertex_checks"]
        maps[f"gadget{i}_opposite"] = _fresh_ancilla_map(q, patch["n_cycle_checks"])
        q += patch["n_cycle_checks"]
    maps["adapter_opposite"] = _fresh_ancilla_map(q, n_adapter_cycles)
    q += int(n_adapter_cycles)
    return maps, list(range(int(ancilla_start), q)), q


def _check_to_ancilla_maps(basis, ancilla_maps, patches, n_adapter_cycles):
    basis_map = dict(ancilla_maps["bb_basis"])
    opposite_map = dict(ancilla_maps["bb_opposite"])

    basis_offset = len(basis_map)
    opposite_offset = len(opposite_map)
    for i, patch in enumerate(patches):
        basis_map.update(_offset_row_map(ancilla_maps[f"gadget{i}_basis"], basis_offset))
        basis_offset += patch["n_vertex_checks"]
        opposite_map.update(_offset_row_map(ancilla_maps[f"gadget{i}_opposite"], opposite_offset))
        opposite_offset += patch["n_cycle_checks"]
    opposite_map.update(_offset_row_map(ancilla_maps["adapter_opposite"], opposite_offset))

    if is_x_basis(basis):
        return basis_map, opposite_map
    return opposite_map, basis_map


def _combined_measurement_graph(tri_defs, edge_offsets, adapter_sets):
    G = nx.Graph()
    node_offsets = []
    next_node = 0

    for gadget_index, tri_def in enumerate(tri_defs):
        local = tri_def.get("measurement_graph", tri_def.get("measurement_graph_networkx"))
        node_offsets.append(next_node)
        if local is None:
            continue
        for node, data in local.nodes(data=True):
            attrs = dict(data)
            attrs["gadget"] = int(gadget_index)
            attrs["local_node"] = int(node)
            G.add_node(next_node + int(node), **attrs)
        for a, b, data in local.edges(data=True):
            attrs = dict(data)
            if "qubit" in attrs:
                attrs["local_qubit"] = int(attrs["qubit"])
                attrs["qubit"] = _local_edge_to_global(attrs["local_qubit"], tri_def, edge_offsets[gadget_index])
            attrs["type"] = attrs.get("type", "gadget")
            G.add_edge(next_node + int(a), next_node + int(b), **attrs)
        next_node += int(local.number_of_nodes())

    for adapter_set in adapter_sets:
        left = int(adapter_set["left_gadget"])
        right = int(adapter_set["right_gadget"])
        left_ports = _port_measurement_vertex_ids(tri_defs[left])
        right_ports = _port_measurement_vertex_ids(tri_defs[right])
        for label, adapter in zip(adapter_set["port_labels"], adapter_set["adapter_qubits"]):
            if int(label) not in left_ports or int(label) not in right_ports:
                continue
            G.add_edge(
                node_offsets[left] + int(left_ports[int(label)]),
                node_offsets[right] + int(right_ports[int(label)]),
                qubit=int(adapter),
                type="adapter",
                port_label=int(label),
                left_gadget=left,
                right_gadget=right,
            )
    return G


def merge_multi_body(
    deformation_results=None,
    logicals=None,
    basis=Pauli.Z,
    shuttling_threshold=None,
    *,
    code=None,
    logical_indices=None,
    tri_defs=None,
    plot=False,
    high_weight_threshold=10,
    **tri_kwargs,
):
    """Merge several logical gadgets attached to the same code block.

    Inputs can be one of:
      - ``tri_defs=[...]``: compiled ``deform_logical_to_tri_lattice`` outputs;
      - ``deformation_results=[...]``: outputs of ``deform_code_for_logical``;
      - ``code=..., logicals=[...]`` or ``code=..., logical_indices=[...]``.

    The returned H matrices use shared-codeblock indexing:

        data, gadget0 edges, gadget1 edges, ..., adapters, ancillas

    The adapter chain connects gadget ``i`` to gadget ``i+1`` for all adjacent
    pairs.  Thus three gadgets have two adapter sets.  Each adapter cycle is
    built from two neighboring adapters in one set plus the shortest endpoint
    route in each of the two gadgets, exactly as in the two-logical case.
    """
    tri_defs = _compile_gadgets(
        code=code,
        logicals=logicals,
        logical_indices=logical_indices,
        deformation_results=deformation_results,
        tri_defs=tri_defs,
        basis=basis,
        shuttling_threshold=shuttling_threshold,
        **tri_kwargs,
    )
    n_data = _check_same_data_block(tri_defs)
    edge_offsets, adapter_start = _edge_offsets(tri_defs, n_data)

    adapter_capacity = sum(
        len(_ordered_common_port_labels(tri_defs[i], tri_defs[i + 1]))
        for i in range(len(tri_defs) - 1)
    )
    total_cols = adapter_start + adapter_capacity

    adapter_sets, adapter_cycle_checks, adapter_cycle_routes, next_free = _build_adapter_sets_and_cycles(
        tri_defs,
        edge_offsets,
        adapter_start,
        total_cols,
    )
    if next_free != total_cols:
        raise RuntimeError("Internal adapter allocation mismatch")

    patches = [
        _patch_data(tri_def, edge_offsets[i], total_cols, adapter_sets, i)
        for i, tri_def in enumerate(tri_defs)
    ]

    BB_H_basis = _combine_bb_matrix(tri_defs, "BB_H_basis", edge_offsets, total_cols, n_data)
    BB_H_opposite = _combine_bb_matrix(tri_defs, "BB_H_opposite", edge_offsets, total_cols, n_data)
    gadget_H_basis = np.vstack([patch["vertex_checks"] for patch in patches]).astype(np.uint8)
    gadget_H_opposite = np.vstack([patch["cycle_checks"] for patch in patches] + [adapter_cycle_checks]).astype(np.uint8)

    H_basis = np.vstack([BB_H_basis, gadget_H_basis]).astype(np.uint8)
    H_opposite = np.vstack([BB_H_opposite, gadget_H_opposite]).astype(np.uint8)

    if is_x_basis(basis):
        x_checks = H_basis
        z_checks = H_opposite
    else:
        x_checks = H_opposite
        z_checks = H_basis

    ancilla_start = total_cols
    ancilla_maps, ancilla_qubits, n_total_qubits = _allocate_ancillas(
        BB_H_basis.shape[0],
        BB_H_opposite.shape[0],
        patches,
        adapter_cycle_checks.shape[0],
        ancilla_start,
    )
    x_check_to_ancilla, z_check_to_ancilla = _check_to_ancilla_maps(
        basis,
        ancilla_maps,
        patches,
        adapter_cycle_checks.shape[0],
    )

    basis_to_ancilla = dict(ancilla_maps["bb_basis"])
    opposite_to_ancilla = dict(ancilla_maps["bb_opposite"])
    basis_row = BB_H_basis.shape[0]
    opposite_row = BB_H_opposite.shape[0]
    for i, patch in enumerate(patches):
        patch["basis_to_ancilla"] = ancilla_maps[f"gadget{i}_basis"]
        patch["opposite_to_ancilla"] = ancilla_maps[f"gadget{i}_opposite"]
        patch["ancilla_qubits"] = list(patch["basis_to_ancilla"].values()) + list(patch["opposite_to_ancilla"].values())
        basis_to_ancilla.update(_offset_row_map(patch["basis_to_ancilla"], basis_row))
        basis_row += patch["n_vertex_checks"]
        opposite_to_ancilla.update(_offset_row_map(patch["opposite_to_ancilla"], opposite_row))
        opposite_row += patch["n_cycle_checks"]
    adapter_opposite_to_ancilla = _offset_row_map(ancilla_maps["adapter_opposite"], opposite_row)
    opposite_to_ancilla.update(adapter_opposite_to_ancilla)

    adapter_qubits = [int(q) for adapter_set in adapter_sets for q in adapter_set["adapter_qubits"]]
    qubit_index_ranges = {"data": (0, n_data)}
    for i, patch in enumerate(patches):
        qubit_index_ranges[f"gadget{i}_edges"] = (patch["edge_offset"], patch["edge_offset"] + patch["n_edge_columns"])
    qubit_index_ranges["adapter"] = (adapter_start, total_cols)
    qubit_index_ranges["ancillas"] = (ancilla_start, n_total_qubits)

    counts = {
        "n_data": n_data,
        "n_gadgets": len(patches),
        "n_adapter_qubits": len(adapter_qubits),
        "n_adapter_cycle_checks": int(adapter_cycle_checks.shape[0]),
        "n_h_qubits": total_cols,
        "n_ancilla_qubits": len(ancilla_qubits),
        "n_total_qubits": n_total_qubits,
        "patches": [
            {
                "n_edge_qubits": patch["n_edge_qubits"],
                "n_edge_columns": patch["n_edge_columns"],
                "n_vertex_checks": patch["n_vertex_checks"],
                "n_cycle_checks": patch["n_cycle_checks"],
            }
            for patch in patches
        ],
    }

    result = {
        "tri_defs": tri_defs,
        "logical_indices": None if logical_indices is None else [int(i) for i in logical_indices],
        "measurement_graphs": [tri_def.get("measurement_graph", tri_def.get("measurement_graph_networkx")) for tri_def in tri_defs],
        "combined_measurement_graph": _combined_measurement_graph(tri_defs, edge_offsets, adapter_sets),
        "patches": patches,
        "adapter_sets": adapter_sets,
        "adapter_qubits": adapter_qubits,
        "adapter_cycle_checks": adapter_cycle_checks,
        "adapter_cycle_routes": adapter_cycle_routes,
        "n_data": n_data,
        "n_edge_qubits": sum(patch["n_edge_qubits"] for patch in patches),
        "n_edge_columns": sum(patch["n_edge_columns"] for patch in patches),
        "n_adapter_qubits": len(adapter_qubits),
        "n_adapter_cycle_checks": int(adapter_cycle_checks.shape[0]),
        "n_h_qubits": total_cols,
        "n_total_h_qubits": total_cols,
        "n_ancilla_qubits": len(ancilla_qubits),
        "n_total_qubits": n_total_qubits,
        "ancilla_start": ancilla_start,
        "ancilla_qubits": ancilla_qubits,
        "ancilla_maps": ancilla_maps,
        "BB_H_basis": BB_H_basis,
        "BB_H_opposite": BB_H_opposite,
        "gadget_H_basis": gadget_H_basis,
        "gadget_H_opposite": gadget_H_opposite,
        "H_basis": H_basis,
        "H_opposite": H_opposite,
        "H_basis_def": H_basis,
        "H_opposite_basis_def": H_opposite,
        "x_checks": x_checks,
        "z_checks": z_checks,
        "basis_to_ancilla": basis_to_ancilla,
        "opposite_to_ancilla": opposite_to_ancilla,
        "x_check_to_ancilla": x_check_to_ancilla,
        "z_check_to_ancilla": z_check_to_ancilla,
        "check_to_ancilla": {"x": x_check_to_ancilla, "z": z_check_to_ancilla},
        "adapter_opposite_to_ancilla": adapter_opposite_to_ancilla,
        "patch_edge_offsets": edge_offsets,
        "adapter_start": adapter_start,
        "qubit_index_ranges": qubit_index_ranges,
        "basis": _basis_name(basis),
        "counts": counts,
    }
    result["high_weight_checks"] = _high_weight_check_records(result, high_weight_threshold)
    if plot:
        fig, ax, adapter_coords = plot_multi_body_merge(
            result,
            high_weight_threshold=high_weight_threshold,
        )
        result["fig"] = fig
        result["ax"] = ax
        result["adapter_coords"] = adapter_coords
    return result


def _template_visible_bounds(template):
    qubits = (
        list(template.edge_qubits)
        + list(template.vertex_check_qubits)
        + list(template.cycle_check_qubits)
        + list(template.shuttling_cycle_check_qubits)
        + list(template.port_qubits)
        + list(template.shuttling_edge_qubits)
    )
    pts = [template.coords[int(q)] for q in qubits if int(q) in template.coords]
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    return min(xs), max(xs), min(ys), max(ys)


def _merged_plot_transforms(merged, gap=3.0):
    transforms = []
    x_cursor = 0.0
    for tri_def in merged["tri_defs"]:
        template = tri_def["fixed_template"]
        x_min, x_max, _, _ = _template_visible_bounds(template)
        x_shift = x_cursor - x_min
        transforms.append(lambda x, y, s=x_shift: (float(x) + s, float(y)))
        x_cursor += (x_max - x_min) + float(gap)
    return transforms


def _local_qubit_xy(tri_def, transform, q):
    coords = tri_def["fixed_template"].coords
    x, y = coords[int(q)]
    return transform(x, y)


def _global_edge_to_local(global_q, tri_def, edge_offset):
    n_data = int(tri_def["n_data"])
    global_q = int(global_q)
    if global_q < n_data:
        return global_q
    return n_data + global_q - int(edge_offset)


def _as_qubit_list(value):
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(q) for q in value]


def _plot_qubits(ax, coords, qubits, *, label, color, marker="o", size=20, zorder=4, alpha=1.0, edgecolor="none", seen=None):
    qubits = [int(q) for q in _as_qubit_list(qubits) if int(q) in coords]
    if not qubits:
        return
    label_text = None if seen is not None and label in seen else label
    if seen is not None:
        seen.add(label)
    ax.scatter(
        [coords[q][0] for q in qubits],
        [coords[q][1] for q in qubits],
        s=size,
        c=color,
        marker=marker,
        label=label_text,
        zorder=zorder,
        alpha=alpha,
        edgecolors=edgecolor,
    )



def _check_row_type(merged, family, row):
    row = int(row)
    if family == "basis":
        n_bb = int(merged["BB_H_basis"].shape[0])
        if row < n_bb:
            return f"BB basis {row}"
        local = row - n_bb
        for i, patch in enumerate(merged["patches"]):
            n = int(patch["n_vertex_checks"])
            if local < n:
                return f"gadget {i} vertex {local}"
            local -= n
    else:
        n_bb = int(merged["BB_H_opposite"].shape[0])
        if row < n_bb:
            return f"BB opposite {row}"
        local = row - n_bb
        for i, patch in enumerate(merged["patches"]):
            n = int(patch["n_cycle_checks"])
            if local < n:
                return f"gadget {i} cycle {local}"
            local -= n
        if local < int(merged.get("n_adapter_cycle_checks", 0)):
            route = merged.get("adapter_cycle_routes", [])[local]
            return f"adapter cycle {local} g{route['left_gadget']}-g{route['right_gadget']} ports {route['endpoint_port_labels']}"
    return f"{family} row {row}"


def _high_weight_check_records(merged, threshold):
    if threshold is None:
        return []
    records = []
    for family, key in [("basis", "H_basis"), ("opposite", "H_opposite")]:
        H = np.asarray(merged[key], dtype=np.uint8)
        if H.size == 0:
            continue
        weights = np.sum(H, axis=1).astype(int)
        for row in np.where(weights >= int(threshold))[0]:
            records.append({
                "family": family,
                "row": int(row),
                "weight": int(weights[row]),
                "type": _check_row_type(merged, family, int(row)),
                "support": np.flatnonzero(H[int(row)]).astype(int).tolist(),
            })
    return records


def _data_qubit_plot_coords(n_data, x0, y0, w, h):
    n_data = int(n_data)
    if n_data <= 0:
        return {}
    cols = max(1, int(np.ceil(np.sqrt(n_data))))
    rows = max(1, int(np.ceil(n_data / cols)))
    dx = w / (cols + 1)
    dy = h / (rows + 1)
    return {
        q: (x0 + dx * (q % cols + 1), y0 + h - dy * (q // cols + 1))
        for q in range(n_data)
    }


def _draw_high_weight_checks(ax, merged, global_coords, threshold, seen):
    records = _high_weight_check_records(merged, threshold)
    if not records:
        return records

    colors = {"basis": "#be123c", "opposite": "#0891b2"}
    markers = {"basis": "P", "opposite": "X"}
    offsets = {"basis": 0.16, "opposite": -0.16}
    for rec_i, record in enumerate(records):
        family = record["family"]
        pts = []
        visible_support = []
        for q in record["support"]:
            if int(q) in global_coords:
                pts.append(global_coords[int(q)])
                visible_support.append(int(q))
        if not pts:
            continue

        color = colors[family]
        label = f"high-weight {family} checks" if f"high-weight {family} checks" not in seen else None
        seen.add(f"high-weight {family} checks")
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=78,
            marker=markers[family],
            facecolors="none",
            edgecolors=color,
            linewidths=1.7,
            label=label,
            zorder=30,
        )
        if len(pts) >= 2:
            center = np.mean(np.asarray(pts, dtype=float), axis=0)
            ordered = sorted(
                pts,
                key=lambda p: np.arctan2(float(p[1]) - center[1], float(p[0]) - center[0]),
            )
            closed = ordered + [ordered[0]] if len(ordered) > 2 else ordered
            ax.plot(
                [p[0] + offsets[family] for p in closed],
                [p[1] + offsets[family] for p in closed],
                color=color,
                linewidth=1.8,
                alpha=0.82,
                zorder=29,
            )
        if len(pts) > 0:
            x, y = pts[0]
            ax.text(
                x + 0.22,
                y + 0.22 + 0.18 * (rec_i % 3),
                f"{family[0].upper()}{record['row']} w{record['weight']}",
                color=color,
                fontsize=7,
                zorder=31,
            )
        record["visible_support"] = visible_support
    return records

def plot_multi_body_merge(merged, ax=None, figsize=(15, 8), show_labels=True, high_weight_threshold=10):
    """Visualize a same-codeblock multi-body merge and its adapter sets."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    transforms = _merged_plot_transforms(merged)
    seen = set()
    global_coords = {}
    route_edges_by_patch = {i: set() for i in range(len(merged["tri_defs"]))}
    for route in merged.get("adapter_cycle_routes", []):
        left = int(route["left_gadget"])
        right = int(route["right_gadget"])
        for q in route.get("left_global_route_edge_qubits", []):
            route_edges_by_patch[left].add(_global_edge_to_local(q, merged["tri_defs"][left], merged["patch_edge_offsets"][left]))
        for q in route.get("right_global_route_edge_qubits", []):
            route_edges_by_patch[right].add(_global_edge_to_local(q, merged["tri_defs"][right], merged["patch_edge_offsets"][right]))

    all_x = []
    all_y = []
    for i, tri_def in enumerate(merged["tri_defs"]):
        template = tri_def["fixed_template"]
        transform = transforms[i]
        local_coords = {int(q): transform(*xy) for q, xy in template.coords.items()}
        n_data_local = int(tri_def["n_data"])
        n_h_local = int(tri_def["n_h_qubits"])
        edge_offset = int(merged["patch_edge_offsets"][i])
        for local_q, xy in local_coords.items():
            if n_data_local <= int(local_q) < n_h_local:
                global_coords[edge_offset + int(local_q) - n_data_local] = xy
        all_x.extend(x for x, _ in local_coords.values())
        all_y.extend(y for _, y in local_coords.values())

        used_edges = set(map(int, tri_def.get("fixed_edge_qubits_used", [])))
        shuttles = set(map(int, tri_def.get("shuttling_edge_qubits", [])))
        floor_used_edges = used_edges - shuttles
        adapter_route_edges = route_edges_by_patch[i]

        for q, (a, b) in template.edge_qubit_to_vertices.items():
            q = int(q)
            if q in shuttles:
                continue
            xa, ya = local_coords[int(a)]
            xb, yb = local_coords[int(b)]
            if q in adapter_route_edges:
                color, lw, alpha, z = "#7c3aed", 2.8, 0.95, 7
            elif q in floor_used_edges:
                color, lw, alpha, z = "#d62728", 2.0, 0.90, 5
            else:
                color, lw, alpha, z = "#d7dce2", 0.55, 0.55, 1
            ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=z)

        _plot_qubits(ax, local_coords, template.edge_qubits, label="all floor edge qubits", color="#93c5fd", size=7, zorder=2, alpha=0.28, seen=seen)
        _plot_qubits(ax, local_coords, floor_used_edges, label="used floor edge qubits", color="#1d4ed8", size=13, zorder=6, seen=seen)
        _plot_qubits(ax, local_coords, route_edges_by_patch[i], label="adapter-cycle floor routes", color="#7c3aed", size=18, zorder=8, edgecolor="black", seen=seen)
        _plot_qubits(ax, local_coords, template.cycle_check_qubits, label="all cycle ancillas", color="#fde68a", size=5, zorder=3, alpha=0.25, seen=seen)
        used_cycle_ancillas = [q for qs in tri_def.get("cycle_row_to_check_qubits", {}).values() for q in _as_qubit_list(qs)]
        _plot_qubits(ax, local_coords, used_cycle_ancillas, label="used cycle ancillas", color="#f59e0b", size=18, zorder=7, edgecolor="black", seen=seen)
        _plot_qubits(ax, local_coords, template.vertex_check_qubits, label="all vertex ancillas", color="#9ca3af", size=8, zorder=4, alpha=0.30, seen=seen)
        _plot_qubits(ax, local_coords, tri_def.get("auxiliary_graph_incidence_vertices", []), label="used vertex ancillas", color="#111827", size=22, zorder=8, seen=seen)
        _plot_qubits(ax, local_coords, template.port_qubits, label="all ports", color="#fca5a5", marker="s", size=16, zorder=5, alpha=0.55, seen=seen)
        _plot_qubits(ax, local_coords, tri_def.get("port_qubits", tri_def.get("logical_support_to_port_qubit", {}).values()), label="used logical ports", color="#991b1b", marker="s", size=30, zorder=9, seen=seen)

        label = None
        if merged.get("logical_indices") is not None and i < len(merged["logical_indices"]):
            label = f"{merged['basis']}{merged['logical_indices'][i]}"
        if label:
            x_min, x_max, _, y_max = _template_visible_bounds(template)
            lx, ly = transform((x_min + x_max) / 2, y_max + 0.7)
            ax.text(lx, ly, label, ha="center", va="bottom", fontsize=12, fontweight="bold")

    adapter_coords = {}
    for adapter_set in merged.get("adapter_sets", []):
        left = int(adapter_set["left_gadget"])
        right = int(adapter_set["right_gadget"])
        for label, q in zip(adapter_set["port_labels"], adapter_set["adapter_qubits"]):
            q = int(q)
            left_ports = _int_dict(merged["tri_defs"][left].get("port_label_to_qubit", {}))
            right_ports = _int_dict(merged["tri_defs"][right].get("port_label_to_qubit", {}))
            left_port = int(left_ports[int(label)])
            right_port = int(right_ports[int(label)])
            x1, y1 = _local_qubit_xy(merged["tri_defs"][left], transforms[left], left_port)
            x2, y2 = _local_qubit_xy(merged["tri_defs"][right], transforms[right], right_port)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            adapter_coords[q] = (mx, my)
            global_coords[q] = (mx, my)
            ax.plot([x1, mx, x2], [y1, my, y2], color="#7c3aed", linewidth=1.2, alpha=0.65, zorder=6)
            label_text = None if "adapter edge qubits" in seen else "adapter edge qubits"
            seen.add("adapter edge qubits")
            ax.scatter([mx], [my], s=38, marker="D", color="#a855f7", edgecolors="black", label=label_text, zorder=10)
            if show_labels:
                ax.text(mx, my + 0.25, str(q), ha="center", va="bottom", fontsize=6, color="#581c87", zorder=11)

    if all_x and all_y:
        code_y0 = min(all_y) - 5.8
        code_h = 4.4
        code_x0 = min(all_x)
        code_w = max(all_x) - min(all_x)
        ax.add_patch(Rectangle((code_x0, code_y0), code_w, code_h, facecolor="white", edgecolor="black", linewidth=1.2, zorder=0))
        ax.text(
            code_x0 + code_w / 2,
            code_y0 + code_h / 2,
            f"shared code block\n{merged['n_data']} data qubits",
            ha="center",
            va="center",
            fontsize=10,
            zorder=1,
        )
        data_coords = _data_qubit_plot_coords(merged["n_data"], code_x0, code_y0, code_w, code_h)
        global_coords.update(data_coords)
        high_data = sorted({
            int(q)
            for record in _high_weight_check_records(merged, high_weight_threshold)
            for q in record["support"]
            if int(q) < int(merged["n_data"])
        })
        _plot_qubits(
            ax,
            data_coords,
            high_data,
            label="high-check data qubits",
            color="#111827",
            size=12,
            zorder=28,
            alpha=0.75,
            seen=seen,
        )
        all_y.append(code_y0)
        all_y.append(code_y0 + code_h)

    merged["high_weight_checks"] = _draw_high_weight_checks(ax, merged, global_coords, high_weight_threshold, seen)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.12)
    ax.set_title("Same-codeblock multi-body merge with adapter qubits")
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    return fig, ax, adapter_coords


def plot_z0_z1_z3_test_merge(code, shuttling_threshold=None, **tri_kwargs):
    """Temporary visual smoke test for the same-codeblock Z0 Z1 Z3 merge."""
    return merge_multi_body(
        code=code,
        logical_indices=[0, 1, 3],
        basis=Pauli.Z,
        shuttling_threshold=shuttling_threshold,
        plot=True,
        **tri_kwargs,
    )


__all__ = ["merge_multi_body", "plot_multi_body_merge", "plot_z0_z1_z3_test_merge"]
