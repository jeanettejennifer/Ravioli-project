# BB code memory with simple BPOSD decoder
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

def bb_get_PCMS(H):
    H = np.asarray(H, dtype=np.uint8)
    n_rows, n_cols = H.shape
    n_data = n_cols // 2
    n_checks = n_rows // 2
    return H[:n_checks, :n_data], H[n_checks:, n_data:]


def bb_is_x_basis(basis):
    return basis == Pauli.X or str(basis).upper().endswith("X")


def bb_pauli_string(pauli, row):
    return stim.PauliString("+" + "".join(pauli if b else "_" for b in row))


def bb_prepare_logical_memory_state(BB_code, memory_basis=Pauli.Z):
    """Perfectly prepare all logicals in +1 eigenstates of memory_basis."""
    Hx, Hz = bb_get_PCMS(BB_code.matrix)
    logical_pauli = "X" if bb_is_x_basis(memory_basis) else "Z"

    stabilizers = []
    stabilizers += [bb_pauli_string("X", row) for row in Hx]
    stabilizers += [bb_pauli_string("Z", row) for row in Hz]
    stabilizers += [
        bb_pauli_string(logical_pauli, np.asarray(logical, dtype=np.uint8))
        for logical in BB_code.get_logical_ops(memory_basis)
    ]
    return stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=False,
    ).to_circuit()


def bb_append_data_depolarize_noise(circuit, n_data, p):
    if p:
        circuit.append("DEPOLARIZE1", list(range(n_data)), p)
        

def bb_rec_abs(circuit, measurement_index):
    return stim.target_rec(int(measurement_index) - circuit.num_measurements)

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
        
def bb_measure_stabilizer_round(circuit, Hx, Hz, x_ancillas, z_ancillas, p_reset_flip, p_after_clifford_depolarize, p_measurement_flip):
    records = {"x": [], "z": []}
    x_ancillas = [int(q) for q in x_ancillas]
    z_ancillas = [int(q) for q in z_ancillas]
    
    # reset ancillas
    circuit.append("R", x_ancillas)
    circuit.append("R", z_ancillas)
    circuit.append(f"X_ERROR", x_ancillas, p_reset_flip)
    circuit.append(f"X_ERROR", z_ancillas, p_reset_flip)

    # basis change for X ancillas
    circuit.append("h", x_ancillas)
    circuit.append(f"DEPOLARIZE1", x_ancillas, p_after_clifford_depolarize)
    # CX schedule (without ticks) for X checks
    x_layers = get_cnot_layers(Hx)
    z_layers = get_cnot_layers(Hz)

    for layer in x_layers:
        targets = []
        for row, q in layer:
            targets += [x_ancillas[row], q]
        circuit.append("cx", targets)
        circuit.append(f"DEPOLARIZE2", targets, p_after_clifford_depolarize)

    # basis change for X ancillas 
    circuit.append("h", x_ancillas)
    circuit.append(f"DEPOLARIZE1", x_ancillas, p_after_clifford_depolarize)

    # CX schedule for Z checks 
    for layer in z_layers:
        targets = []
        for row, q in layer:
            targets += [q, z_ancillas[row]]
        circuit.append("cx", targets)
        circuit.append(f"DEPOLARIZE2", targets, p_after_clifford_depolarize)

    # measure X ancillas and keep track of record
    for ancilla in x_ancillas:
        circuit.append("X_ERROR", [ancilla], p_measurement_flip)
        circuit.append("M", [ancilla])
        records["x"].append(circuit.num_measurements - 1)
    
    # measure Z ancillas
    for ancilla in z_ancillas:
        circuit.append("X_ERROR", [ancilla], p_measurement_flip)
        circuit.append("M", [ancilla])
        records["z"].append(circuit.num_measurements - 1)
    return records

def bb_add_round_detectors(circuit, previous, current):
    for kind in ["x", "z"]:
        for old, new in zip(previous[kind], current[kind]):
            circuit.append("DETECTOR", [bb_rec_abs(circuit, old), bb_rec_abs(circuit, new)])


def bb_measure_data(circuit, n_data, memory_basis=Pauli.Z, p_measurement_flip=0.0):
    
    start = circuit.num_measurements
    
    if bb_is_x_basis(memory_basis):
        circuit.append(f"Z_ERROR", list(range(n_data)), p_measurement_flip)
        circuit.append("MX", list(range(n_data)))
    else:
        circuit.append(f"X_ERROR", list(range(n_data)), p_measurement_flip)
        circuit.append("M", list(range(n_data)))

    return [start + q for q in range(n_data)]


def bb_add_final_detectors(circuit, Hx, Hz, last_records, data_records, memory_basis=Pauli.Z):
    # Only the stabilizer type matching the final data-measurement basis can be closed.
    checks = Hx if bb_is_x_basis(memory_basis) else Hz
    kind = "x" if bb_is_x_basis(memory_basis) else "z"
    for i, row in enumerate(checks):
        targets = [bb_rec_abs(circuit, data_records[int(q)]) for q in np.flatnonzero(row)]
        targets.append(bb_rec_abs(circuit, last_records[kind][i]))
        circuit.append("DETECTOR", targets)


def bb_add_memory_observables(circuit, BB_code, memory_basis, data_records, logical_indices=None):
    logicals = BB_code.get_logical_ops(memory_basis)
    if logical_indices is None:
        logical_indices = range(len(logicals))
    for obs_i, logical_i in enumerate(logical_indices):
        logical = np.asarray(logicals[logical_i], dtype=np.uint8)
        targets = [bb_rec_abs(circuit, data_records[int(q)]) for q in np.flatnonzero(logical)]
        circuit.append("OBSERVABLE_INCLUDE", targets, obs_i)


def bb_add_initial_detectors(circuit, current):
    for kind in ["x", "z"]:
        for m in current[kind]:
            circuit.append("DETECTOR", [bb_rec_abs(circuit, m)])

def reset_qubits(circuit, qubits, p_reset_flip):
    circuit.append("R", qubits)
    circuit.append("Z_ERROR", qubits, p_reset_flip)

def bb_memory_circuit(BB_code, memory_basis=Pauli.Z, noise_model="code-capacity", rounds=6, p=0.0, logical_indices=None):
    Hx, Hz = bb_get_PCMS(BB_code.matrix)
    n_data = Hx.shape[1]

    circuit = stim.Circuit()
    circuit += bb_prepare_logical_memory_state(BB_code, memory_basis)

    previous = None

    n_data = np.shape(Hx)[1]

    n_x = np.shape(Hx)[0]
    n_z = np.shape(Hz)[0]

    x_ancillas = np.arange(n_data, n_data + n_x)
    z_ancillas = np.arange(n_data + n_x, n_data + n_x + n_z)

    if noise_model == "code-capacity":
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = 0.0
        
    else:
        p_reset_flip = p_after_clifford_depolarize = p_measurement_flip = p

    for _ in range(rounds):

        # One memory interval per round, followed by one full syndrome extraction.
        bb_append_data_depolarize_noise(circuit, n_data, p)

        if noise_model == "code-capacity":
            current = bb_measure_stabilizer_round(circuit = circuit, 
                                                  Hx = Hx, 
                                                  Hz = Hz, 
                                                  x_ancillas = x_ancillas, 
                                                  z_ancillas = z_ancillas, 
                                                  p_after_clifford_depolarize=p_after_clifford_depolarize,
                                                  p_measurement_flip=p_measurement_flip,
                                                  p_reset_flip=p_reset_flip,
                                                  )
        else:
            current = bb_measure_stabilizer_round(circuit = circuit, 
                                                  Hx = Hx, 
                                                  Hz = Hz, 
                                                  x_ancillas = x_ancillas, 
                                                  z_ancillas = z_ancillas, 
                                                  p_after_clifford_depolarize=p,
                                                  p_measurement_flip=p,
                                                  p_reset_flip=p_reset_flip,
                                                  )
        
        if previous is None:
            # Perfect preparation makes the first noiseless syndrome deterministic.
            bb_add_initial_detectors(circuit, current)
        else:
            bb_add_round_detectors(circuit, previous, current)
        previous = current
        circuit.append("TICK")

    # No extra data noise here: final readout immediately closes the last syndrome.
    if noise_model == "code-capacity":
        data_records = bb_measure_data(circuit, n_data, memory_basis=memory_basis, p_measurement_flip=0.0)
    else:
        data_records = bb_measure_data(circuit, n_data, memory_basis=memory_basis, p_measurement_flip=p)

    bb_add_final_detectors(circuit, Hx, Hz, previous, data_records, memory_basis=memory_basis)
    bb_add_memory_observables(circuit, BB_code, memory_basis, data_records, logical_indices=logical_indices)
    return circuit


def bb_estimate_memory_error_rates(
    BB_code,
    memory_basis,
    noise_model,
    p,
    rounds=12,
    shots=1000,
    logical_indices=None,
    max_errors=None,
    num_workers=1,
):
    logicals = BB_code.get_logical_ops(memory_basis)
    if logical_indices is None:
        logical_indices = list(range(len(logicals)))
    logical_indices = list(logical_indices)

    raw = np.zeros(len(logical_indices), dtype=float)
    decoded = np.zeros(len(logical_indices), dtype=float)
    tasks = []

    custom_decoders = {
        "bposd": bposd_decoder()
    }

    for i, logical_i in enumerate(logical_indices):
        circuit = bb_memory_circuit(
            BB_code,
            memory_basis=memory_basis,
            rounds=rounds,
            p=float(p),
            logical_indices=[logical_i],
            noise_model = noise_model,
        )

        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        print(
            f"logical {logical_i}: detectors={circuit.num_detectors}, "
            f"observables={circuit.num_observables}, L0 terms={str(dem).count('L0')}"
        )

        # Sinter reports decoded logical failures. This separate quick sample
        # keeps the old return value for the undecoded/raw observable rate.
        _, obs = circuit.compile_detector_sampler().sample(
            shots,
            separate_observables=True,
        )
        raw[i] = float(np.mean(np.asarray(obs).reshape(shots, -1)[:, 0]))

        tasks.append(sinter.Task(
            circuit=circuit,
            decoder="bposd",
            json_metadata={
                "p": float(p),
                "rounds": int(rounds),
                "basis": "X" if bb_is_x_basis(memory_basis) else "Z",
                "logical_index": int(logical_i),
            },
        ))


    stats = sinter.collect(
        tasks=tasks,
        max_shots=int(shots),
        max_errors=int(shots if max_errors is None else max_errors),
        num_workers=int(num_workers),
        decoders=[],
        custom_decoders=custom_decoders,
    )

    logical_to_row = {int(logical_i): i for i, logical_i in enumerate(logical_indices)}
    for stat in stats:
        row = logical_to_row[int(stat.json_metadata["logical_index"])]
        decoded[row] = 0.0 if stat.shots == 0 else float(stat.errors / stat.shots)

    return raw, decoded

def bb_code_label(BB_code):
    if hasattr(BB_code, "label"):
        return BB_code.label
    n = BB_code.matrix.shape[1] // 2
    k = len(BB_code.get_logical_ops(Pauli.Z))
    return f"BB [[{n},{k},?]]"


def bb_run_memory_sweep(
    BB_code,
    memory_basis=Pauli.Z,
    ps=None,
    rounds=6,
    shots=1000,
    logical_indices=None,
    num_workers = 1,
    noise_model = "code-capacity"
):

    if ps is None:
        ps = np.logspace(-6, -2, 10)

    logicals = BB_code.get_logical_ops(memory_basis)
    
    if logical_indices is None:
        logical_indices = list(range(len(logicals)))
    logical_indices = list(logical_indices)

    raw = np.zeros((len(logical_indices), len(ps)))
    decoded = np.zeros_like(raw)
    basis_name = "X" if bb_is_x_basis(memory_basis) else "Z"

    for j, p in enumerate(ps):
        raw[:, j], decoded[:, j] = bb_estimate_memory_error_rates(
            BB_code,
            memory_basis=memory_basis,
            noise_model = noise_model,
            p=float(p),
            rounds=rounds,
            shots=shots,
            logical_indices=logical_indices,
            num_workers=num_workers
        )
        print(
            f"BB memory {basis_name}, p={p:.2e}, "
            f"avg raw={np.mean(raw[:, j]):.4g}, avg decoded={np.mean(decoded[:, j]):.4g}"
        )

    return {
        "ps": np.asarray(ps, dtype=float),
        "rounds": int(rounds),
        "shots": int(shots),
        "basis": basis_name,
        "logical_indices": logical_indices,
        "raw": raw,
        "decoded": decoded,
        "code_label": bb_code_label(BB_code),
    }