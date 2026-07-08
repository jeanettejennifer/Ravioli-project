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


def append_code_capacity_noise(circuit, n_edges, n_data, p, noise_model = "both"): 
    n_h = n_data + n_edges
    if p:
        if noise_model == "gadget":
            circuit.append('DEPOLARIZE1', list(range(n_data, n_h)), p)
        elif noise_model == "data":
            circuit.append("DEPOLARIZE1", list(range(n_data)), p)
        else:
            circuit.append('DEPOLARIZE1', list(range(n_h)), p)


def append_mpp_check(circuit, pauli, support):
    targets = []
    for q in np.flatnonzero(support):
        if targets:
            targets.append(stim.target_combiner())
        targets.append(stim.target_x(int(q)) if pauli == 'X' else stim.target_z(int(q)))
    if not targets:
        raise ValueError('Empty check row cannot be measured as an MPP observable')
    circuit.append('MPP', targets)


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


def measure_all_checks(circuit, def_res, logical_basis):
    records = {}
    for name, pauli, H in check_groups(def_res, logical_basis):
        records[name] = []
        for row in H:
            append_mpp_check(circuit, pauli, row)
            records[name].append(circuit.num_measurements - 1)
    return records


def rec_abs(circuit, measurement_index):
    return stim.target_rec(measurement_index - circuit.num_measurements)

def add_round_detectors_excluding_logical_vertices(circuit, prev_records, cur_records, logical_vertex_rows):
    logical_vertex_rows = set(map(int, logical_vertex_rows))

    for name in cur_records:
        for i, (prev, cur) in enumerate(zip(prev_records[name], cur_records[name])):
            if name == "vertex" and i in logical_vertex_rows:
                continue
            circuit.append("DETECTOR", [rec_abs(circuit, prev), rec_abs(circuit, cur)])


def measure_edge_qubits_for_logical(circuit, n_data, n_edges, logical_basis):
    """Measure only gadget edge qubits in the opposite basis to the logical basis.

    Z-logical gadget: edge qubits are |+>, close X-cycle checks with final X readout.
    X-logical gadget: edge qubits are |0>, close Z-cycle checks with final Z readout.
    """

    edge_qubits = list(range(n_data, n_data + n_edges))
    if not is_x_basis(logical_basis):
        circuit.append('H', edge_qubits)  # X-basis readout via H then MZ.
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


def logical_measurement_code_capacity_circuit(
    rounds,
    p=0.0,
    logical_basis=Pauli.Z,
    logical_index=0,
    BB_code=None,
    def_res=None,
    noise_model = "both"
):
    
    if def_res is None:
        def_res = get_deformed_logical(BB_code, logical_basis, logical_index)

    n_data = int(def_res['n_data'])
    n_h = int(def_res.get('n_h_qubits', def_res['BB_H_basis'].shape[1])) # n_h = n_data + n_edges
    n_edges = n_h - n_data

    circuit = prepare_logical_state(BB_code, logical_basis, n_data, n_edges)
    logical_vertex_rows = def_res['logical_observable_vertex_rows']

    # Clean reference round: no errors and no detectors.
    last_records = measure_all_checks(circuit, def_res, logical_basis)
    circuit.append('TICK')

    for _ in range(rounds):
        # One noisy memory interval on BB data + gadget edge qubits, followed by
        # one full deformed syndrome extraction compared to the previous round.
        append_code_capacity_noise(circuit, n_edges, n_data, p, noise_model = noise_model)
        cur_records = measure_all_checks(circuit, def_res, logical_basis)
        add_round_detectors_excluding_logical_vertices(circuit, last_records, cur_records, logical_vertex_rows)
        last_records = cur_records
        circuit.append('TICK')

    # Final readout measures only gadget edge qubits in the X (z) basis for Z(x) logicals. BB data
    # qubits are not measured. There is no noise between the last syndrome
    # round and the final edge readout.
    edge_records = measure_edge_qubits_for_logical(circuit, n_data, n_edges, logical_basis)
    add_final_cycle_detectors(circuit, def_res, last_records, edge_records, n_data)
    add_logical_measurement_observable(circuit, def_res, last_records)
    return circuit


def estimate_logical_measurement_error_rate(
    BB_code,
    def_res,
    logical_basis,
    logical_index,
    p,
    rounds=4,
    shots=100000,
    noise_model = "both",
    num_workers = 1
):
    
    circuit = logical_measurement_code_capacity_circuit(
        rounds=rounds,
        p=float(p),
        logical_basis=logical_basis,
        logical_index=logical_index,
        BB_code=BB_code,
        def_res=def_res,
        noise_model = noise_model
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

