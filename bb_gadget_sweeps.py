from stimbposd import BPOSD
import matplotlib.pyplot as plt
from sympy.abc import x, y
from qldpc import codes
from qldpc.objects import Pauli
from graph_helper_functions import deform_code_for_logical
import numpy as np
import stim
import importlib
import deform_triangular_lattice
importlib.reload(deform_triangular_lattice)
from deform_triangular_lattice import deform_logical_to_tri_lattice
import stim
import sinter
import matplotlib.pyplot as plt
from decoder import bposd_decoder

def is_x_basis(basis):
    return basis == Pauli.X or str(basis).upper().endswith('X')

def get_PCMs(H):
    H = np.asarray(H, dtype=np.uint8)
    n_rows, n_cols = H.shape
    n_data = n_cols // 2
    n_checks = n_rows // 2
    Hx = H[:n_checks, :n_data]
    Hz = H[n_checks:, n_data:]
    return Hx, Hz

def pauli_string_from_support(pauli, row):
    return stim.PauliString('+' + ''.join(pauli if b else '_' for b in row))


def get_deformed_logical(BB_code, logical_basis=Pauli.Z, logical_index=0, shuttling_threshold=2):
    logical = BB_code.get_logical_ops(logical_basis)[logical_index]
    deformation = deform_code_for_logical(BB_code.matrix, logical_basis, logical)
    return deform_logical_to_tri_lattice(deformation, basis=logical_basis, plot=False, shuttling_threshold=shuttling_threshold)


def prepare_logical_state(BB_code, logical_basis, n_data, n_edges):
    """Prepare BB data in the requested logical eigenbasis and gadget edges in |+> or |0>."""
    Hx, Hz = get_PCMs(BB_code.matrix)
    stabilizers = []
    stabilizers += [pauli_string_from_support('X', row) for row in Hx]
    stabilizers += [pauli_string_from_support('Z', row) for row in Hz]

    logical_pauli = 'X' if is_x_basis(logical_basis) else 'Z'
    for logical in BB_code.get_logical_ops(logical_basis):
        stabilizers.append(pauli_string_from_support(logical_pauli, logical[:n_data]))

    circuit = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=True,
    ).to_circuit()

    # In the Z-logical gadget, edge qubits start in |+>. In the X-logical
    # gadget, the edge qubits start in |0>.
    if not is_x_basis(logical_basis):
        circuit.append('H', list(range(n_data, n_data + n_edges)))
    return circuit


def append_data_and_edge_depolarize_noise(circuit, n_edges, n_data, p, noise_target="both"):
    n_h = n_data + n_edges
    if p:
        if noise_target == "gadget":
            circuit.append('DEPOLARIZE1', list(range(n_data, n_h)), p)
        elif noise_target == "data":
            circuit.append("DEPOLARIZE1", list(range(n_data)), p)
        else:
            circuit.append('DEPOLARIZE1', list(range(n_h)), p)


def check_groups(def_res, logical_basis):
    """Checks measured each round, in measurement-record order."""
    basis_pauli = 'X' if is_x_basis(logical_basis) else 'Z'
    opposite_pauli = 'Z' if is_x_basis(logical_basis) else 'X'
    return [
        ('basis_bb', basis_pauli, def_res['BB_H_basis']),
        ('opposite_bb', opposite_pauli, def_res['BB_H_opposite']),
        ('cycle', opposite_pauli, def_res['gadget_H_opposite']),
        ('vertex', basis_pauli, def_res['gadget_H_basis']),
    ]

def get_cnot_layers(H):
    # cnot layers without conflict from H = Hz or H = Hx
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
    targets = list(map(int, targets))
    if p and targets:
        circuit.append(name, targets, p)


def _single_ancilla_map(row_to_ancilla, n_rows, name):
    out = []
    for row in range(n_rows):
        ancilla = row_to_ancilla.get(row)
        if ancilla is None:
            raise ValueError(f"Missing ancilla for {name} row {row}")
        out.append(int(ancilla))
    return out


def gadget_ancilla_groups(def_res):
    n_basis = def_res['BB_H_basis'].shape[0]
    n_opposite = def_res['BB_H_opposite'].shape[0]
    n_cycle = def_res['gadget_H_opposite'].shape[0]
    n_vertex = def_res['gadget_H_basis'].shape[0]
    return {
        'basis_bb': _single_ancilla_map(def_res['BB_basis_to_ancilla'], n_basis, 'BB basis'),
        'opposite_bb': _single_ancilla_map(def_res['BB_opposite_to_ancilla'], n_opposite, 'BB opposite'),
        'cycle': _single_ancilla_map(def_res['cycle_row_to_check_qubit'], n_cycle, 'cycle'),
        'vertex': _single_ancilla_map(def_res['gadget_basis_to_ancilla'], n_vertex, 'vertex'),
    }


def measure_checks(
    circuit,
    def_res,
    logical_basis,
    p_reset_flip=0.0,
    p_after_clifford_depolarize=0.0,
    p_measurement_flip=0.0,
):
    records = {}
    ancilla_groups = gadget_ancilla_groups(def_res)
    all_ancillas = [
        q
        for name, _, _ in check_groups(def_res, logical_basis)
        for q in ancilla_groups[name]
    ]

    circuit.append('R', all_ancillas)
    append_noise(circuit, 'X_ERROR', all_ancillas, p_reset_flip)

    for name, pauli, H in check_groups(def_res, logical_basis):
        records[name] = []
        ancillas = ancilla_groups[name]

        if pauli == 'X':
            circuit.append('H', ancillas)
            append_noise(circuit, 'DEPOLARIZE1', ancillas, p_after_clifford_depolarize)

        for layer in get_cnot_layers(H):
            targets = []
            for row, q in layer:
                ancilla = ancillas[row]
                if pauli == 'X':
                    targets += [ancilla, int(q)]
                else:
                    targets += [int(q), ancilla]
            circuit.append('CX', targets)
            append_noise(circuit, 'DEPOLARIZE2', targets, p_after_clifford_depolarize)

        if pauli == 'X':
            circuit.append('H', ancillas)
            append_noise(circuit, 'DEPOLARIZE1', ancillas, p_after_clifford_depolarize)

        for ancilla in ancillas:
            append_noise(circuit, 'X_ERROR', [ancilla], p_measurement_flip)
            circuit.append('M', [ancilla])
            records[name].append(circuit.num_measurements - 1)
    return records


def rec_abs(circuit, measurement_index):
    return stim.target_rec(measurement_index - circuit.num_measurements)

def add_round_detectors(circuit, prev_records, cur_records):
    for name in cur_records:
        for i, (prev, cur) in enumerate(zip(prev_records[name], cur_records[name])):
            circuit.append("DETECTOR", [rec_abs(circuit, prev), rec_abs(circuit, cur)])


def measure_edge_qubits_for_logical(
    circuit,
    n_data,
    n_edges,
    logical_basis,
    p_after_clifford_depolarize=0.0,
    p_measurement_flip=0.0,
):
    """Measure only gadget edge qubits in the opposite basis to the logical basis.

    Z-logical gadget: edge qubits are |+>, close X-cycle checks with final X readout.
    X-logical gadget: edge qubits are |0>, close Z-cycle checks with final Z readout.
    """

    edge_qubits = list(range(n_data, n_data + n_edges))
    if not is_x_basis(logical_basis):
        circuit.append('H', edge_qubits)  # X-basis readout via H then MZ.
        append_noise(circuit, 'DEPOLARIZE1', edge_qubits, p_after_clifford_depolarize)
    append_noise(circuit, 'X_ERROR', edge_qubits, p_measurement_flip)
    start = circuit.num_measurements
    circuit.append('M', edge_qubits)
    return {q: start + (q - n_data) for q in edge_qubits} # This returns a dictionary mapping each edge qubit index to its absolute measurement-record index.


def add_final_cycle_detectors(circuit, def_res, last_records, edge_records, n_data):
    """Close opposite-basis cycle checks using final edge-only readout."""
    for i, row in enumerate(def_res['gadget_H_opposite']):
        support = np.flatnonzero(row)
        targets = [rec_abs(circuit, edge_records[int(q)]) for q in support]
        targets.append(rec_abs(circuit, last_records['cycle'][i]))
        circuit.append('DETECTOR', targets)


def add_logical_measurement_observable(circuit, def_res, last_records):
    """Observable = XOR of final vertex-check rows that span the logical."""
    targets = [
        rec_abs(circuit, last_records['vertex'][int(row)])
        for row in def_res['logical_observable_vertex_rows']
    ]
    circuit.append('OBSERVABLE_INCLUDE', targets, 0)


def logical_measurement_circuit(
    rounds,
    p=0.0,
    logical_basis=Pauli.Z,
    logical_index=0,
    BB_code=None,
    def_res=None,
    noise_model="code-capacity",
    noise_target="gadget",
):
    
    if def_res is None:
        def_res = get_deformed_logical(BB_code, logical_basis, logical_index)

    n_data = int(def_res['n_data'])
    n_h = int(def_res.get('n_h_qubits', def_res['BB_H_basis'].shape[1])) # n_h = n_data + n_edges
    n_edges = n_h - n_data

    circuit = prepare_logical_state(BB_code, logical_basis, n_data, n_edges)
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    if noise_model == "code-capacity":
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = 0.0
    elif noise_model == "circuit-level":
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = p
    else:
        raise ValueError("noise_model must be 'code-capacity' or 'circuit-level'")

    last_records = None
    for _ in range(rounds):
        if noise_model=="code-capacity":
            append_data_and_edge_depolarize_noise(
                circuit,
                n_edges,
                n_data,
                p,
                noise_target=noise_target,
            )
        cur_records = measure_checks(
            circuit,
            def_res,
            logical_basis,
            p_reset_flip=p_reset_flip,
            p_after_clifford_depolarize=p_after_clifford_depolarize,
            p_measurement_flip=p_measurement_flip,
        )
        if last_records is not None:
            add_round_detectors(circuit, last_records, cur_records)
        last_records = cur_records
        circuit.append('TICK')

    edge_records = measure_edge_qubits_for_logical(
        circuit,
        n_data,
        n_edges,
        logical_basis,
        p_after_clifford_depolarize=p_after_clifford_depolarize,
        p_measurement_flip=p_measurement_flip,
    )
    add_final_cycle_detectors(circuit, def_res, last_records, edge_records, n_data)
    add_logical_measurement_observable(circuit, def_res, last_records)
    return circuit


def logical_measurement_code_capacity_circuit(
    rounds,
    p=0.0,
    logical_basis=Pauli.Z,
    logical_index=0,
    BB_code=None,
    def_res=None,
    noise_model="both",
):
    """Backward-compatible wrapper; ``noise_model`` here means noise target."""
    return logical_measurement_circuit(
        rounds=rounds,
        p=p,
        logical_basis=logical_basis,
        logical_index=logical_index,
        BB_code=BB_code,
        def_res=def_res,
        noise_model="code-capacity",
        noise_target=noise_model,
    )


def estimate_logical_measurement_error_rate(
    BB_code,
    def_res,
    logical_basis,
    logical_index,
    p,
    rounds=4,
    shots=100000,
    noise_model="code-capacity",
    noise_target="both",
    num_workers=1,
):
    
    circuit = logical_measurement_circuit(
        rounds=rounds,
        p=float(p),
        logical_basis=logical_basis,
        logical_index=logical_index,
        BB_code=BB_code,
        def_res=def_res,
        noise_model=noise_model,
        noise_target=noise_target,
    )


    dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )

    custom_decoder = {"bposd" : bposd_decoder()}
    
    tasks = []
    tasks.append(sinter.Task(
            circuit=circuit,
            decoder="bposd",
            json_metadata={
                "p": float(p),
                "rounds": int(rounds),
                "basis": "X" if logical_basis == Pauli.X else "Z",
                "logical_index": int(logical_index),
            },
        ))
    _, obs = circuit.compile_detector_sampler().sample(
            shots,
            separate_observables=True,
        )
    raw = float(np.mean(np.asarray(obs).reshape(shots, -1)[:, 0]))
    
    stats = sinter.collect(
        tasks=tasks,
        max_shots=int(shots),
        num_workers=int(num_workers),
        decoders=[],
        custom_decoders=custom_decoder,
    )
    stat = stats[0]
    decoded = 0.0 if stat.shots == 0 else float(stat.errors / stat.shots)

    return raw, decoded
