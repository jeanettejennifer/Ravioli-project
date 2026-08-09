"""Stim sweep for a Z logical merge between two BB-code triangular gadgets.

The circuit tests a joint Z logical measurement between logical index 0 of two
BB patches.  Qubit order follows ``merge_code_stabilizers`` as patch1 data, patch1 edges, patch2 data, patch2 edges, adapter, ancillas
"""

import numpy as np
import sinter
import stim
from qldpc.objects import Pauli

from deformation.deform import deform_code_for_logical
from deformation.deform_triangular import deform_logical_to_tri_lattice
from circuits.decoder import bposd_decoder
from circuits.joint_logical_measurement import merge_code_stabilizers


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


def shift_check_matrix(H, width, offset):
    H = np.asarray(H, dtype=np.uint8)
    out = np.zeros((H.shape[0], int(width)), dtype=np.uint8)
    out[:, int(offset):int(offset) + H.shape[1]] = H
    return out


def get_deformed_checks(BB_code, logical_basis=Pauli.Z, logical_index=0, shuttling_threshold=2):
    logical = BB_code.get_logical_ops(logical_basis)[logical_index]
    deformation = deform_code_for_logical(BB_code.matrix, logical_basis, logical)
    return deform_logical_to_tri_lattice(
        deformation,
        basis=logical_basis,
        plot=False,
        shuttling_threshold=shuttling_threshold,
    )


def make_z_merge(BB_code, logical1_index=0, logical2_index=0, shuttling_threshold=2, plot=False):
    tri1 = get_deformed_checks(BB_code, Pauli.Z, logical1_index, shuttling_threshold)
    tri2 = get_deformed_checks(BB_code, Pauli.Z, logical2_index, shuttling_threshold)
    return merge_code_stabilizers(
        tri_def1=tri1,
        tri_def2=tri2,
        basis1=Pauli.Z,
        plot=plot,
    )


def prepare_two_z_logical_patches(BB_code, merged):
    """Prepare both BB patches in |0_L>^k and edge/adapter qubits in |+>."""
    Hx, Hz = get_PCMs(BB_code.matrix)
    n_total = int(merged["n_total_qubits"])
    patch2_offset = int(merged["patch2_offset"])

    stabilizers = []
    for row in Hx:
        q = np.flatnonzero(row)
        stabilizers.append(pauli_string_on_qubits("X", q, n_total))
        stabilizers.append(pauli_string_on_qubits("X", patch2_offset + q, n_total))
    for row in Hz:
        q = np.flatnonzero(row)
        stabilizers.append(pauli_string_on_qubits("Z", q, n_total))
        stabilizers.append(pauli_string_on_qubits("Z", patch2_offset + q, n_total))
    for logical in BB_code.get_logical_ops(Pauli.Z):
        q = np.flatnonzero(np.asarray(logical, dtype=np.uint8))
        stabilizers.append(pauli_string_on_qubits("Z", q, n_total))
        stabilizers.append(pauli_string_on_qubits("Z", patch2_offset + q, n_total))

    circuit = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=True,
    ).to_circuit()

    plus_qubits = []
    r = merged["qubit_index_ranges"]
    for name in ["patch1_edges", "patch2_edges", "adapter"]:
        start, stop = r[name]
        plus_qubits.extend(range(int(start), int(stop)))
    circuit.append("R", plus_qubits)
    circuit.append("H", plus_qubits)
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
    Hx, Hz = get_PCMs(BB_code.matrix)
    width = int(merged["n_h_qubits"])
    patch2_offset = int(merged["patch2_offset"])

    x_checks = np.vstack([
        shift_check_matrix(Hx, width, 0),
        shift_check_matrix(Hx, width, patch2_offset),
    ])
    z_checks = np.vstack([
        shift_check_matrix(Hz, width, 0),
        shift_check_matrix(Hz, width, patch2_offset),
    ])

    x_ancillas = {}
    z_ancillas = {}
    n_x = Hx.shape[0]
    n_z = Hz.shape[0]
    for row in range(n_x):
        x_ancillas[row] = merged["patch1"]["bb_x_to_ancilla"][row]
        x_ancillas[n_x + row] = merged["patch2"]["bb_x_to_ancilla"][row]
    for row in range(n_z):
        z_ancillas[row] = merged["patch1"]["bb_z_to_ancilla"][row]
        z_ancillas[n_z + row] = merged["patch2"]["bb_z_to_ancilla"][row]
    return x_checks, z_checks, x_ancillas, z_ancillas


def qubits_in_ranges(merged, names):
    qubits = []
    ranges = merged["qubit_index_ranges"]
    for name in names:
        start, stop = ranges[name]
        qubits.extend(range(int(start), int(stop)))
    return qubits


def bb_data_qubits(merged):
    return qubits_in_ranges(merged, ["patch1_data", "patch2_data"])


def edge_and_adapter_qubits(merged):
    return qubits_in_ranges(merged, ["patch1_edges", "patch2_edges", "adapter"])


def all_h_qubits(merged):
    return qubits_in_ranges(
        merged,
        ["patch1_data", "patch1_edges", "patch2_data", "patch2_edges", "adapter"],
    )


def append_code_capacity_noise(circuit, qubits, p):
    if p and qubits:
        circuit.append("DEPOLARIZE1", [int(q) for q in qubits], p)


def joint_z_observable_rows(merged):
    """Rows in merged z_checks whose parity is the Z0*Z0 merge outcome."""
    p1 = merged["patch1"]
    p2 = merged["patch2"]
    tri1 = p1["tri_def"]
    tri2 = p2["tri_def"]

    z_offset_patch1_vertex = p1["n_bb_z_checks"] + p2["n_bb_z_checks"]
    z_offset_patch2_vertex = z_offset_patch1_vertex + p1["n_vertex_checks"]

    rows = []
    rows += [z_offset_patch1_vertex + int(r) for r in tri1["logical_observable_vertex_rows"]]
    rows += [z_offset_patch2_vertex + int(r) for r in tri2["logical_observable_vertex_rows"]]
    return rows


def add_joint_z_observable(circuit, merged, merged_records):
    targets = [rec_abs(circuit, merged_records["z"][row]) for row in joint_z_observable_rows(merged)]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)


def measure_edge_and_adapter_x(circuit, merged, p_after_clifford_depolarize=0.0, p_measurement_flip=0.0):
    qubits = edge_and_adapter_qubits(merged)
    circuit.append("H", qubits)
    append_noise(circuit, "DEPOLARIZE1", qubits, p_after_clifford_depolarize)
    append_noise(circuit, "X_ERROR", qubits, p_measurement_flip)
    start = circuit.num_measurements
    circuit.append("M", qubits)
    return {int(q): start + i for i, q in enumerate(qubits)}


def add_final_cycle_detectors(circuit, merged, merged_records, edge_records):
    first_cycle_row = merged["patch1"]["n_bb_x_checks"] + merged["patch2"]["n_bb_x_checks"]
    for row in range(first_cycle_row, merged["x_checks"].shape[0]):
        support = [int(q) for q in np.flatnonzero(merged["x_checks"][row])]
        missing = [q for q in support if q not in edge_records]
        if missing:
            raise ValueError(f"X-cycle row {row} has non-edge/non-adapter qubits: {missing}")
        targets = [rec_abs(circuit, edge_records[q]) for q in support]
        targets.append(rec_abs(circuit, merged_records["x"][row]))
        circuit.append("DETECTOR", targets)


def merged_z_measurement_circuit(
    BB_code,
    rounds=6,
    p=0.0,
    logical1_index=0,
    logical2_index=0,
    shuttling_threshold=2,
    noise_model="code-capacity",
    plot_merge=False,
):
    """Measure Z0*Z0 by d normal BB rounds followed by d merged rounds."""

    merged = make_z_merge(
        BB_code,
        logical1_index=logical1_index,
        logical2_index=logical2_index,
        shuttling_threshold=shuttling_threshold,
        plot=plot_merge,
    )
    circuit = prepare_two_z_logical_patches(BB_code, merged)

    if noise_model == "code-capacity":
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = 0.0
    else:
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = float(p)

    normal_x, normal_z, normal_x_ancillas, normal_z_ancillas = normal_bb_checks_and_ancillas(BB_code, merged)
    previous_normal = None
    current_normal = None
    for _ in range(rounds):
        if noise_model == "code-capacity":
            append_code_capacity_noise(circuit, bb_data_qubits(merged), float(p))
        current_normal = {
            "x": measure_check_matrix(circuit, normal_x, "X", normal_x_ancillas, p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
            "z": measure_check_matrix(circuit, normal_z, "Z", normal_z_ancillas, p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
        }
        if previous_normal is None:
            for recs in current_normal.values():
                for m in recs:
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
    for _ in range(rounds):
        if noise_model == "code-capacity":
            append_code_capacity_noise(circuit, all_h_qubits(merged), float(p))
        current_merged = {
            "x": measure_check_matrix(circuit, merged["x_checks"], "X", merged["x_check_to_ancilla"], p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
            "z": measure_check_matrix(circuit, merged["z_checks"], "Z", merged["z_check_to_ancilla"], p_reset_flip, p_after_clifford_depolarize, p_measurement_flip),
        }
        if previous_merged is None:
            add_record_comparison_detectors(circuit, current_normal["x"], current_merged["x"][:n_normal_x])
            add_record_comparison_detectors(circuit, current_normal["z"], current_merged["z"][:n_normal_z])
            for m in current_merged["x"][n_normal_x:]:
                circuit.append("DETECTOR", [rec_abs(circuit, m)])
        else:
            add_record_comparison_detectors(circuit, previous_merged["x"], current_merged["x"])
            add_record_comparison_detectors(circuit, previous_merged["z"], current_merged["z"])
        previous_merged = current_merged
        circuit.append("TICK")

    if noise_model == "code-capacity":
        append_code_capacity_noise(circuit, edge_and_adapter_qubits(merged), float(p))
    edge_records = measure_edge_and_adapter_x(
        circuit,
        merged,
        p_after_clifford_depolarize=p_after_clifford_depolarize,
        p_measurement_flip=p_measurement_flip,
    )
    add_final_cycle_detectors(circuit, merged, current_merged, edge_records)
    add_joint_z_observable(circuit, merged, current_merged)
    return circuit, merged


def estimate_merge_error_rate(
    BB_code,
    p,
    rounds=6,
    shots=1000,
    logical1_index=0,
    logical2_index=0,
    shuttling_threshold=2,
    noise_model="code-capacity",
    num_workers=1,
):
    circuit, merged = merged_z_measurement_circuit(
        BB_code,
        rounds=rounds,
        p=float(p),
        logical1_index=logical1_index,
        logical2_index=logical2_index,
        shuttling_threshold=shuttling_threshold,
        noise_model=noise_model,
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
            "rounds": int(rounds),
            "basis": "Z",
            "logical1_index": int(logical1_index),
            "logical2_index": int(logical2_index),
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


def run_z_merge_sweep(
    BB_code,
    ps,
    rounds=6,
    shots=1000,
    logical1_index=0,
    logical2_index=0,
    shuttling_threshold=2,
    noise_model="code-capacity",
    num_workers=1,
):
    raw = []
    decoded = []
    last_circuit = None
    last_merged = None
    for p in ps:
        r, d, last_circuit, last_merged = estimate_merge_error_rate(
            BB_code,
            p=float(p),
            rounds=rounds,
            shots=shots,
            logical1_index=logical1_index,
            logical2_index=logical2_index,
            shuttling_threshold=shuttling_threshold,
            noise_model=noise_model,
            num_workers=num_workers,
        )
        raw.append(r)
        decoded.append(d)
        print(f"Z{logical1_index}Z{logical2_index}, p={p:.2e}, raw={r:.4g}, decoded={d:.4g}")
    return {
        "ps": np.asarray(ps, dtype=float),
        "raw": np.asarray(raw, dtype=float),
        "decoded": np.asarray(decoded, dtype=float),
        "rounds": int(rounds),
        "shots": int(shots),
        "logical1_index": int(logical1_index),
        "logical2_index": int(logical2_index),
        "basis": "Z",
        "noise_model": noise_model,
        "last_circuit": last_circuit,
        "last_merged": last_merged,
    }



def plot_z_merge_sweep(result, ax=None):
    import matplotlib.pyplot as plt

    ps = result["ps"]
    raw = result["raw"]
    decoded = result["decoded"]
    label = f"Z{result['logical1_index']}Z{result['logical2_index']}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.8))
    else:
        fig = ax.figure

    ax.loglog(ps, raw, "--", color="#7f7f7f", linewidth=1.5, label=f"raw {label}")
    ax.loglog(ps, decoded, "o-", color="#1d4ed8", linewidth=1.8, markersize=4, label=f"decoded {label}")
    ax.loglog(ps, ps, "--", color="red", linewidth=1.8, label=r"$p_L=p$")
    ax.set_title(f"Gadget merge {label}, {result['rounds']} BB rounds + {result['rounds']} merged rounds")
    ax.set_xlabel("physical error probability p")
    ax.set_ylabel("joint logical measurement error probability")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.show()
    return fig, ax
