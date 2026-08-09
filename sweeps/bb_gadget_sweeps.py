from stimbposd import BPOSD
import matplotlib.pyplot as plt
from sympy.abc import x, y
from qldpc import codes
from qldpc.objects import Pauli
import numpy as np
import stim
from deformation.deform import deform_code_for_logical
from deformation.deform_triangular import deform_logical_to_tri_lattice
from circuits.decoder import bposd_decoder
import stim
import sinter
import matplotlib.pyplot as plt



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


def prepare_plus_state(n_data, n_edges):
    """Prepare |+> on data+edge qubits, without measuring/projecting a logical."""
    circuit = stim.Circuit()
    circuit.append('R', list(range(n_data + n_edges)))
    circuit.append('H', list(range(n_data + n_edges)))
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
    noisy_qubits,
    p_reset_flip=0.0,
    p_after_clifford_depolarize=0.0,
    p_measurement_flip=0.0,
):
    return measure_named_check_groups(
        circuit,
        check_groups(def_res, logical_basis),
        gadget_ancilla_groups(def_res),
        noisy_qubits=noisy_qubits,
        p_reset_flip=p_reset_flip,
        p_after_clifford_depolarize=p_after_clifford_depolarize,
        p_measurement_flip=p_measurement_flip,
    )

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
    noisy_qubits,
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
        append_noise(circuit, 'DEPOLARIZE1', [q for q in edge_qubits if q in noisy_qubits], p_after_clifford_depolarize)
    append_noise(circuit, 'X_ERROR', [q for q in edge_qubits if q in noisy_qubits], p_measurement_flip)
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


def add_logical_measurement_observable(circuit, def_res, last_records, initial_values=None, projection_record=None):
    """Observable = final gadget logical parity XOR the perfect logical MPP result."""
    targets = [
        rec_abs(circuit, last_records['vertex'][int(row)])
        for row in def_res['logical_observable_vertex_rows']
    ]
    if projection_record is None:
        prepared_bit = initial_values['logical_vertex_parity']['bit']
        if prepared_bit is None:
            raise ValueError("Prepared logical vertex parity is not deterministic")
    else:
        targets.append(rec_abs(circuit, projection_record))
    circuit.append('OBSERVABLE_INCLUDE', targets, 0)


def check_pauli_string(row, pauli, n_qubits):
    chars = ['_'] * int(n_qubits)
    for q in np.flatnonzero(row):
        if int(q) < n_qubits:
            chars[int(q)] = pauli
    return stim.PauliString('+' + ''.join(chars))


def initial_gadget_stabilizer_values(BB_code, def_res, logical_basis=Pauli.Z, logical_index=0, preparation='logical'):
    """Exact first-round check values in the prepared BB+edge product state.

    Returns expectation ``+1``/``-1`` for deterministic checks and ``0`` for
    checks that are random before the first gadget measurement.
    """
    n_data = int(def_res['n_data'])
    n_h = int(def_res.get('n_h_qubits', def_res['BB_H_basis'].shape[1]))
    n_edges = n_h - n_data

    sim = stim.TableauSimulator()
    if preparation == 'logical':
        sim.do(prepare_logical_state(BB_code, logical_basis, n_data, n_edges))
    elif preparation == 'plus':
        sim.do(prepare_plus_state(n_data, n_edges))
    else:
        raise ValueError("preparation must be 'logical' or 'plus'")

    out = {}
    for name, pauli, H in check_groups(def_res, logical_basis):
        expectations = []
        bits = []
        for row in H:
            exp = int(sim.peek_observable_expectation(check_pauli_string(row, pauli, n_h)))
            expectations.append(exp)
            bits.append(0 if exp == 1 else 1 if exp == -1 else None)

        out[name] = {
            'pauli': pauli,
            'matrix': np.asarray(H, dtype=np.uint8),
            'expectations': expectations,
            'bits': bits,
            'deterministic_rows': [i for i, exp in enumerate(expectations) if exp],
            'random_rows': [i for i, exp in enumerate(expectations) if not exp],
        }

    vertex_rows = [int(r) for r in def_res['logical_observable_vertex_rows']]
    vertex_support = np.zeros(n_h, dtype=np.uint8)
    for r in vertex_rows:
        vertex_support ^= np.asarray(def_res['gadget_H_basis'][r], dtype=np.uint8)[:n_h]
    vertex_pauli = 'X' if is_x_basis(logical_basis) else 'Z'
    exp = int(sim.peek_observable_expectation(check_pauli_string(vertex_support, vertex_pauli, n_h)))
    out['logical_vertex_parity'] = {
        'pauli': vertex_pauli,
        'rows': vertex_rows,
        'expectation': exp,
        'bit': 0 if exp == 1 else 1 if exp == -1 else None,
    }
    return out


def print_initial_gadget_stabilizer_values(BB_code, def_res, logical_basis=Pauli.Z, logical_index=0, preparation='logical'):
    values = initial_gadget_stabilizer_values(BB_code, def_res, logical_basis, logical_index, preparation)
    for name, info in values.items():
        if name == 'logical_vertex_parity':
            print(
                f"{name}: rows={info['rows']}, expectation={info['expectation']}, "
                f"bit={info['bit']}"
            )
            continue
        print(
            f"{name} ({info['pauli']}): "
            f"deterministic={len(info['deterministic_rows'])}, "
            f"random={len(info['random_rows'])}"
        )
        print(f"  deterministic rows: {info['deterministic_rows']}")
        print(f"  bits: {info['bits']}")
    return values

def normal_bb_check_groups(BB_code, logical_basis):
    """Normal BB checks, named to line up with the deformed BB checks."""
    Hx, Hz = get_PCMs(BB_code.matrix)
    if is_x_basis(logical_basis):
        return [('basis_bb', 'X', Hx), ('opposite_bb', 'Z', Hz)]
    return [('basis_bb', 'Z', Hz), ('opposite_bb', 'X', Hx)]


def normal_bb_ancilla_groups(def_res, BB_code, logical_basis):
    groups = {name: H for name, _, H in normal_bb_check_groups(BB_code, logical_basis)}
    return {
        'basis_bb': _single_ancilla_map(def_res['BB_basis_to_ancilla'], groups['basis_bb'].shape[0], 'BB basis'),
        'opposite_bb': _single_ancilla_map(def_res['BB_opposite_to_ancilla'], groups['opposite_bb'].shape[0], 'BB opposite'),
    }

def noisy_qubits_for_target(def_res, n_data, n_edges, noise_target):
    n_h = n_data + n_edges
    anc = gadget_ancilla_groups(def_res)

    data = set(range(n_data))
    edge = set(range(n_data, n_h))
    bb_anc = set(anc["basis_bb"]) | set(anc["opposite_bb"])
    gadget_anc = set(anc["cycle"]) | set(anc["vertex"])

    if noise_target == "data":
        return data
    if noise_target == "gadget":
        return edge | gadget_anc
    if noise_target == "both":
        return data | edge | bb_anc | gadget_anc

    raise ValueError("noise_target must be 'data', 'gadget', or 'both'")

def append_cx_layer_with_targeted_noise(circuit, pairs, noisy_qubits, p):
    targets = []
    noisy_targets = []

    for c, t in pairs:
        c = int(c)
        t = int(t)
        targets += [c, t]

        if c in noisy_qubits or t in noisy_qubits:
            noisy_targets += [c, t]

    circuit.append("CX", targets)
    append_noise(circuit, "DEPOLARIZE2", noisy_targets, p)

def measure_named_check_groups(
    circuit,
    groups,
    ancilla_groups,
    noisy_qubits,
    p_reset_flip=0.0,
    p_after_clifford_depolarize=0.0,
    p_measurement_flip=0.0,
):
    records = {}
    all_ancillas = [q for name, _, _ in groups for q in ancilla_groups[name]]

    circuit.append("R", all_ancillas)
    append_noise(circuit, "X_ERROR", [q for q in all_ancillas if q in noisy_qubits], p_reset_flip)

    for name, pauli, H in groups:
        records[name] = []
        ancillas = ancilla_groups[name]

        if pauli == "X":
            circuit.append("H", ancillas)
            append_noise(circuit, "DEPOLARIZE1", [q for q in ancillas if q in noisy_qubits], p_after_clifford_depolarize)

        for layer in get_cnot_layers(H):
            pairs = []
            for row, q in layer:
                ancilla = ancillas[row]
                pairs.append((ancilla, int(q)) if pauli == "X" else (int(q), ancilla))

            append_cx_layer_with_targeted_noise(
                circuit,
                pairs,
                noisy_qubits,
                p_after_clifford_depolarize,
            )

        if pauli == "X":
            circuit.append("H", ancillas)
            append_noise(circuit, "DEPOLARIZE1", [q for q in ancillas if q in noisy_qubits], p_after_clifford_depolarize)

        for ancilla in ancillas:
            if ancilla in noisy_qubits:
                append_noise(circuit, "X_ERROR", [ancilla], p_measurement_flip)
            circuit.append("M", [ancilla])
            records[name].append(circuit.num_measurements - 1)

    return records


def measure_normal_bb_checks(circuit, BB_code, def_res, logical_basis, noisy_qubits, p_reset_flip=0.0, p_after_clifford_depolarize=0.0, p_measurement_flip=0.0):
    return measure_named_check_groups(
        circuit,
        normal_bb_check_groups(BB_code, logical_basis),
        normal_bb_ancilla_groups(def_res, BB_code, logical_basis),
        noisy_qubits=noisy_qubits,
        p_reset_flip=p_reset_flip,
        p_after_clifford_depolarize=p_after_clifford_depolarize,
        p_measurement_flip=p_measurement_flip,
    )


def add_normal_bb_initial_detectors(circuit, normal_records):
    for name in ['basis_bb', 'opposite_bb']:
        for measurement in normal_records[name]:
            circuit.append('DETECTOR', [rec_abs(circuit, measurement)])


def add_bb_to_deformed_transition_detectors(circuit, normal_records, deformed_records):
    for name in ['basis_bb', 'opposite_bb']:
        for old, new in zip(normal_records[name], deformed_records[name]):
            circuit.append('DETECTOR', [rec_abs(circuit, old), rec_abs(circuit, new)])
    for measurement in deformed_records['cycle']:
        circuit.append('DETECTOR', [rec_abs(circuit, measurement)])

def add_normal_bb_rounds_detectors(circuit, normal_records_cur, normal_records_prev):
    for name in ['basis_bb', 'opposite_bb']:
        for prev, cur in zip(normal_records_prev[name], normal_records_cur[name]):
            circuit.append('DETECTOR', [rec_abs(circuit, prev), rec_abs(circuit, cur)])

def logical_measurement_circuit(
    bb_rounds,
    measurement_rounds,
    p=0.0,
    logical_basis=Pauli.Z,
    logical_index=0,
    BB_code=None,
    def_res=None,
    noise_model='code-capacity',
    noise_target='both',
):
    if def_res is None:
        def_res = get_deformed_logical(BB_code, logical_basis, logical_index)

    n_data = int(def_res['n_data'])
    n_h = int(def_res.get('n_h_qubits', def_res['BB_H_basis'].shape[1]))
    n_edges = n_h - n_data

    noisy_qubits = noisy_qubits_for_target(def_res, n_data, n_edges, noise_target)

    if bb_rounds < 1:
        raise ValueError('bb_rounds must be at least 1')
    if measurement_rounds < 1:
        raise ValueError('measurement_rounds must be at least 1')

    circuit = prepare_logical_state(BB_code, logical_basis, n_data, n_edges)

    if noise_model == 'code-capacity':
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = 0.0

    elif noise_model == 'circuit-level':
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = p

    else:
        raise ValueError("noise_model must be 'code-capacity' or 'circuit-level'")

    prev_normal_records = None
    cur_normal_records = None

    for _ in range(bb_rounds):
        if noise_model == 'code-capacity':
            append_data_and_edge_depolarize_noise(circuit, n_edges, n_data, p, noise_target=noise_target)
        cur_normal_records = measure_normal_bb_checks(
            circuit,
            BB_code,
            def_res,
            logical_basis,
            noisy_qubits=noisy_qubits,
            p_reset_flip=p_reset_flip,
            p_after_clifford_depolarize=p_after_clifford_depolarize,
            p_measurement_flip=p_measurement_flip,
        )
        if prev_normal_records is None:
            add_normal_bb_initial_detectors(circuit, cur_normal_records)
        else:
            add_normal_bb_rounds_detectors(circuit, cur_normal_records, prev_normal_records)

        prev_normal_records = cur_normal_records
        circuit.append('TICK')

    initial_values = initial_gadget_stabilizer_values(
        BB_code,
        def_res,
        logical_basis,
        logical_index=logical_index,
        preparation='logical',
    )

    last_records = None
    for _ in range(measurement_rounds):
        if noise_model == 'code-capacity':
            append_data_and_edge_depolarize_noise(circuit, n_edges, n_data, p, noise_target=noise_target)
        cur_records = measure_checks(
            circuit,
            def_res,
            logical_basis,
            noisy_qubits=noisy_qubits,
            p_reset_flip=p_reset_flip,
            p_after_clifford_depolarize=p_after_clifford_depolarize,
            p_measurement_flip=p_measurement_flip,
        )
        if last_records is None:
            add_bb_to_deformed_transition_detectors(circuit, cur_normal_records, cur_records)
        else:
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
        noisy_qubits=noisy_qubits,
    )
    add_final_cycle_detectors(circuit, def_res, last_records, edge_records, n_data)
    add_logical_measurement_observable(circuit, def_res, last_records, initial_values=initial_values)
    return circuit


def estimate_logical_measurement_error_rate(
    BB_code,
    def_res,
    logical_basis,
    logical_index,
    p,
    measurement_rounds,
    bb_rounds,
    shots=100000,
    noise_model="code-capacity",
    noise_target="both",
    num_workers=1,
):
    
    circuit = logical_measurement_circuit(
        measurement_rounds=measurement_rounds,
        bb_rounds=bb_rounds,
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
                "bb_rounds": int(bb_rounds),
                "measurement_rounds" : int(measurement_rounds),
                "basis": "X" if logical_basis == Pauli.X else "Z",
                "logical_index": int(logical_index),
                "noise_model": noise_model,
                "noise_target": noise_target,
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