"""
Intra-code logical merge sweeps.

Qubit order follows ``merge_multi_body``:
    data, gadget0 edges, gadget1 edges, ..., adapter, ancillas

The same implementation supports Z-product and X-product logical merges via the
``logical_basis`` argument.
"""

import numpy as np
import sinter
import stim
from qldpc.objects import Pauli

from circuits.decoder import bposd_decoder
from circuits.multi_body_merges import merge_multi_body


def is_x_basis(basis):
    return basis == Pauli.X or str(basis).upper().endswith("X")


def basis_name(basis):
    return "X" if is_x_basis(basis) else "Z"


def get_PCMs(H):
    H = np.asarray(H, dtype=np.uint8)
    n_rows, n_cols = H.shape
    n_data = n_cols // 2
    n_checks = n_rows // 2
    return H[:n_checks, :n_data], H[n_checks:, n_data:]


def pauli_string_on_qubits(pauli, qubits, n_qubits):
    chars = ["_"] * int(n_qubits)
    for q in qubits:
        chars[int(q)] = pauli
    return stim.PauliString("+" + "".join(chars))


def put_matrix_on_data(H, width):
    H = np.asarray(H, dtype=np.uint8)
    out = np.zeros((H.shape[0], int(width)), dtype=np.uint8)
    out[:, :H.shape[1]] = H
    return out


def make_intra_merge(BB_code, logical_indices=(0, 1), logical_basis=Pauli.Z, shuttling_threshold=2, plot=False, **merge_kwargs):
    return merge_multi_body(
        code=BB_code,
        logical_indices=list(logical_indices),
        basis=logical_basis,
        shuttling_threshold=shuttling_threshold,
        plot=plot,
        **merge_kwargs,
    )


def prepare_logical_merge_state(BB_code, merged, logical_basis=Pauli.Z):
    """Prepare BB data in the requested logical eigenbasis and edge/adapter qubits in the conjugate basis."""
    Hx, Hz = get_PCMs(BB_code.matrix)
    n_total = int(merged["n_total_qubits"])
    logical_pauli = basis_name(logical_basis)

    stabilizers = []
    for row in Hx:
        stabilizers.append(pauli_string_on_qubits("X", np.flatnonzero(row), n_total))
    for row in Hz:
        stabilizers.append(pauli_string_on_qubits("Z", np.flatnonzero(row), n_total))
    for logical in BB_code.get_logical_ops(logical_basis):
        logical = np.asarray(logical, dtype=np.uint8)
        stabilizers.append(pauli_string_on_qubits(logical_pauli, np.flatnonzero(logical), n_total))

    circuit = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=True,
    ).to_circuit()

    edge_adapter = edge_and_adapter_qubits(merged)
    circuit.append("R", edge_adapter)
    if not is_x_basis(logical_basis):
        circuit.append("H", edge_adapter)
    return circuit


def get_cnot_layers(H):
    H = np.asarray(H, dtype=np.uint8)
    layers = []
    used = []
    for row, check in enumerate(H):
        for q in np.flatnonzero(check):
            q = int(q)
            for layer, used_layer in zip(layers, used):
                if row not in used_layer["rows"] and q not in used_layer["qubits"]:
                    layer.append((row, q))
                    used_layer["rows"].add(row)
                    used_layer["qubits"].add(q)
                    break
            else:
                layers.append([(row, q)])
                used.append({"rows": {row}, "qubits": {q}})
    return layers


def append_noise(circuit, name, targets, p):
    targets = [int(q) for q in targets]
    if p and targets:
        circuit.append(name, targets, p)


def measure_check_matrix(
    circuit,
    H,
    pauli,
    row_to_ancilla,
    p_reset_flip=0.0,
    p_after_clifford_depolarize=0.0,
    p_measurement_flip=0.0,
):
    H = np.asarray(H, dtype=np.uint8)
    ancillas = [int(row_to_ancilla[i]) for i in range(H.shape[0])]
    circuit.append("R", ancillas)
    append_noise(circuit, "X_ERROR", ancillas, p_reset_flip)

    if pauli == "X":
        circuit.append("H", ancillas)
        append_noise(circuit, "DEPOLARIZE1", ancillas, p_after_clifford_depolarize)

    for layer in get_cnot_layers(H):
        targets = []
        for row, q in layer:
            ancilla = ancillas[row]
            targets += [ancilla, int(q)] if pauli == "X" else [int(q), ancilla]
        if targets:
            circuit.append("CX", targets)
            append_noise(circuit, "DEPOLARIZE2", targets, p_after_clifford_depolarize)

    if pauli == "X":
        circuit.append("H", ancillas)
        append_noise(circuit, "DEPOLARIZE1", ancillas, p_after_clifford_depolarize)

    records = []
    for ancilla in ancillas:
        append_noise(circuit, "X_ERROR", [ancilla], p_measurement_flip)
        circuit.append("M", [ancilla])
        records.append(circuit.num_measurements - 1)
    return records


def rec_abs(circuit, measurement_index):
    return stim.target_rec(int(measurement_index) - circuit.num_measurements)


def add_record_comparison_detectors(circuit, previous, current):
    for old, new in zip(previous, current):
        circuit.append("DETECTOR", [rec_abs(circuit, old), rec_abs(circuit, new)])


def normal_bb_checks_and_ancillas(BB_code, merged):
    """Normal BB checks on the shared data block, using merged BB ancillas."""
    Hx, Hz = get_PCMs(BB_code.matrix)
    width = int(merged["n_h_qubits"])
    x_checks = put_matrix_on_data(Hx, width)
    z_checks = put_matrix_on_data(Hz, width)

    x_ancillas = {row: int(merged["x_check_to_ancilla"][row]) for row in range(Hx.shape[0])}
    z_ancillas = {row: int(merged["z_check_to_ancilla"][row]) for row in range(Hz.shape[0])}
    return x_checks, z_checks, x_ancillas, z_ancillas


def qubits_in_range(merged, name):
    start, stop = merged["qubit_index_ranges"][name]
    return list(range(int(start), int(stop)))


def data_qubits(merged):
    return qubits_in_range(merged, "data")


def edge_qubits(merged):
    out = []
    for i in range(int(merged["counts"]["n_gadgets"])):
        out.extend(qubits_in_range(merged, f"gadget{i}_edges"))
    return out


def edge_and_adapter_qubits(merged):
    return edge_qubits(merged) + qubits_in_range(merged, "adapter")


def all_h_qubits(merged):
    return data_qubits(merged) + edge_and_adapter_qubits(merged)


def append_code_capacity_noise(circuit, qubits, p):
    if p and qubits:
        circuit.append("DEPOLARIZE1", [int(q) for q in qubits], p)


def joint_observable_rows(merged):
    """Rows in the logical-basis check matrix whose parity is the multi-logical product."""
    rows = []
    row_offset = int(merged["BB_H_basis"].shape[0])
    for patch in merged["patches"]:
        tri_def = patch["tri_def"]
        for row in tri_def["logical_observable_vertex_rows"]:
            rows.append(row_offset + int(row))
        row_offset += int(patch["n_vertex_checks"])
    return rows


def add_joint_observable(circuit, merged, merged_records, logical_basis=Pauli.Z):
    kind = "x" if is_x_basis(logical_basis) else "z"
    targets = [rec_abs(circuit, merged_records[kind][row]) for row in joint_observable_rows(merged)]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)


def measure_edge_and_adapter_opposite_basis(
    circuit,
    merged,
    logical_basis=Pauli.Z,
    p_after_clifford_depolarize=0.0,
    p_measurement_flip=0.0,
):
    """Measure edge/adapter qubits in the basis that closes opposite-basis cycle checks."""
    qubits = edge_and_adapter_qubits(merged)
    if not is_x_basis(logical_basis):
        circuit.append("H", qubits)
        append_noise(circuit, "DEPOLARIZE1", qubits, p_after_clifford_depolarize)
    append_noise(circuit, "X_ERROR", qubits, p_measurement_flip)
    start = circuit.num_measurements
    circuit.append("M", qubits)
    return {int(q): start + i for i, q in enumerate(qubits)}


def add_final_cycle_detectors(circuit, merged, merged_records, edge_records, logical_basis=Pauli.Z):
    """Close opposite-basis cycle checks using final edge/adapter readout."""
    opposite_kind = "z" if is_x_basis(logical_basis) else "x"
    checks = merged[f"{opposite_kind}_checks"]
    first_cycle_row = int(merged["BB_H_opposite"].shape[0])
    for row in range(first_cycle_row, checks.shape[0]):
        support = [int(q) for q in np.flatnonzero(checks[row])]
        missing = [q for q in support if q not in edge_records]
        if missing:
            raise ValueError(f"{opposite_kind.upper()}-cycle row {row} has non-edge/non-adapter qubits: {missing}")
        targets = [rec_abs(circuit, edge_records[q]) for q in support]
        targets.append(rec_abs(circuit, merged_records[opposite_kind][row]))
        circuit.append("DETECTOR", targets)


def intra_merge_measurement_circuit(
    BB_code,
    logical_indices=(0, 1),
    logical_basis=Pauli.Z,
    bb_rounds=4,
    measurement_rounds=4,
    p=0.0,
    shuttling_threshold=2,
    noise_model="code-capacity",
    plot_merge=False,
    **merge_kwargs,
):
    """Measure a product of several same-basis logicals in one BB code block."""
    merged = make_intra_merge(
        BB_code,
        logical_indices=logical_indices,
        logical_basis=logical_basis,
        shuttling_threshold=shuttling_threshold,
        plot=plot_merge,
        **merge_kwargs,
    )
    circuit = prepare_logical_merge_state(BB_code, merged, logical_basis)

    if noise_model == "code-capacity":
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = 0.0
    elif noise_model == "circuit-level":
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = float(p)
    else:
        raise ValueError("noise_model must be 'code-capacity' or 'circuit-level'")

    normal_x, normal_z, normal_x_ancillas, normal_z_ancillas = normal_bb_checks_and_ancillas(BB_code, merged)
    previous_normal = None
    current_normal = None
    for _ in range(int(bb_rounds)):
        if noise_model == "code-capacity":
            append_code_capacity_noise(circuit, data_qubits(merged), float(p))
        current_normal = {
            "x": measure_check_matrix(circuit, normal_x, "X", normal_x_ancillas, p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
            "z": measure_check_matrix(circuit, normal_z, "Z", normal_z_ancillas, p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
        }
        if previous_normal is None:
            for records in current_normal.values():
                for m in records:
                    circuit.append("DETECTOR", [rec_abs(circuit, m)])
        else:
            add_record_comparison_detectors(circuit, previous_normal["x"], current_normal["x"])
            add_record_comparison_detectors(circuit, previous_normal["z"], current_normal["z"])
        previous_normal = current_normal
        circuit.append("TICK")

    previous_merged = None
    current_merged = None
    n_normal_x = normal_x.shape[0]
    n_normal_z = normal_z.shape[0]
    opposite_kind = "z" if is_x_basis(logical_basis) else "x"
    n_normal_opposite = n_normal_z if opposite_kind == "z" else n_normal_x

    for _ in range(int(measurement_rounds)):
        if noise_model == "code-capacity":
            append_code_capacity_noise(circuit, all_h_qubits(merged), float(p))
        current_merged = {
            "x": measure_check_matrix(circuit, merged["x_checks"], "X", merged["x_check_to_ancilla"], p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
            "z": measure_check_matrix(circuit, merged["z_checks"], "Z", merged["z_check_to_ancilla"], p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
        }
        if previous_merged is None:
            add_record_comparison_detectors(circuit, current_normal["x"], current_merged["x"][:n_normal_x])
            add_record_comparison_detectors(circuit, current_normal["z"], current_merged["z"][:n_normal_z])
            for m in current_merged[opposite_kind][n_normal_opposite:]:
                circuit.append("DETECTOR", [rec_abs(circuit, m)])
        else:
            add_record_comparison_detectors(circuit, previous_merged["x"], current_merged["x"])
            add_record_comparison_detectors(circuit, previous_merged["z"], current_merged["z"])
        previous_merged = current_merged
        circuit.append("TICK")

    if noise_model == "code-capacity":
        append_code_capacity_noise(circuit, edge_and_adapter_qubits(merged), float(p))
    edge_records = measure_edge_and_adapter_opposite_basis(
        circuit,
        merged,
        logical_basis=logical_basis,
        p_after_clifford_depolarize=p_after_clifford_depolarize,
        p_measurement_flip=p_measurement_flip,
    )
    add_final_cycle_detectors(circuit, merged, current_merged, edge_records, logical_basis=logical_basis)
    add_joint_observable(circuit, merged, current_merged, logical_basis=logical_basis)
    return circuit, merged


def estimate_intra_merge_error_rate(
    BB_code,
    p,
    logical_indices=(0, 1),
    logical_basis=Pauli.Z,
    bb_rounds=4,
    measurement_rounds=4,
    shots=1000,
    shuttling_threshold=2,
    noise_model="code-capacity",
    num_workers=1,
    **merge_kwargs,
):
    circuit, merged = intra_merge_measurement_circuit(
        BB_code,
        logical_indices=logical_indices,
        logical_basis=logical_basis,
        bb_rounds=bb_rounds,
        measurement_rounds=measurement_rounds,
        p=float(p),
        shuttling_threshold=shuttling_threshold,
        noise_model=noise_model,
        **merge_kwargs,
    )
    circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)

    _, obs = circuit.compile_detector_sampler().sample(int(shots), separate_observables=True)
    obs = np.asarray(obs).reshape(int(shots), -1)
    raw = float(np.mean(obs[:, 0]))

    task = sinter.Task(
        circuit=circuit,
        decoder="bposd",
        json_metadata={
            "p": float(p),
            "bb_rounds": int(bb_rounds),
            "measurement_rounds": int(measurement_rounds),
            "basis": basis_name(logical_basis),
            "logical_indices": [int(i) for i in logical_indices],
            "noise_model": noise_model,
        },
    )
    stats = sinter.collect(
        tasks=[task],
        max_shots=int(shots),
        num_workers=int(num_workers),
        decoders=[],
        custom_decoders={"bposd": bposd_decoder()},
    )
    decoded = 0.0 if stats[0].shots == 0 else float(stats[0].errors / stats[0].shots)
    return raw, decoded, circuit, merged


def run_intra_merge_sweep(
    BB_code,
    ps,
    logical_indices=(0, 1),
    logical_basis=Pauli.Z,
    bb_rounds=4,
    measurement_rounds=4,
    shots=1000,
    shuttling_threshold=2,
    noise_model="code-capacity",
    num_workers=1,
    **merge_kwargs,
):
    raw = []
    decoded = []
    last_circuit = None
    last_merged = None
    name = basis_name(logical_basis)
    label = "".join(f"{name}{int(i)}" for i in logical_indices)
    for p in ps:
        r, d, last_circuit, last_merged = estimate_intra_merge_error_rate(
            BB_code,
            p=float(p),
            logical_indices=logical_indices,
            logical_basis=logical_basis,
            bb_rounds=bb_rounds,
            measurement_rounds=measurement_rounds,
            shots=shots,
            shuttling_threshold=shuttling_threshold,
            noise_model=noise_model,
            num_workers=num_workers,
            **merge_kwargs,
        )
        raw.append(r)
        decoded.append(d)
        print(f"{label}, p={p:.2e}, raw={r:.4g}, decoded={d:.4g}")
    return {
        "ps": np.asarray(ps, dtype=float),
        "raw": np.asarray(raw, dtype=float),
        "decoded": np.asarray(decoded, dtype=float),
        "bb_rounds": int(bb_rounds),
        "measurement_rounds": int(measurement_rounds),
        "shots": int(shots),
        "logical_indices": [int(i) for i in logical_indices],
        "basis": name,
        "noise_model": noise_model,
        "last_circuit": last_circuit,
        "last_merged": last_merged,
    }


def plot_intra_merge_sweep(result, ax=None):
    import matplotlib.pyplot as plt

    ps = result["ps"]
    raw = result["raw"]
    decoded = result["decoded"]
    label = "".join(f"{result['basis']}{int(i)}" for i in result["logical_indices"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.8))
    else:
        fig = ax.figure

    ax.loglog(ps, raw, "--", color="#7f7f7f", linewidth=1.5, label=f"raw {label}")
    ax.loglog(ps, decoded, "o-", color="#1d4ed8", linewidth=1.8, markersize=4, label=f"decoded {label}")
    ax.loglog(ps, ps, "--", color="red", linewidth=1.8, label=r"$p_L=p$")
    ax.set_title(
        f"Intra-code merge {label}, "
        f"{result['bb_rounds']} BB rounds + {result['measurement_rounds']} merged rounds"
    )
    ax.set_xlabel("physical error probability p")
    ax.set_ylabel("joint logical measurement error probability")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.show()
    return fig, ax


__all__ = [
    "make_intra_merge",
    "prepare_logical_merge_state",
    "intra_merge_measurement_circuit",
    "estimate_intra_merge_error_rate",
    "run_intra_merge_sweep",
    "plot_intra_merge_sweep",
]


