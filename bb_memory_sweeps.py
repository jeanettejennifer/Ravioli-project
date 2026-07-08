# BB code memory: code-capacity MPP stabilizer rounds + simple BPOSD decoder
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


def bb_append_code_capacity_noise(circuit, n_data, p):
    if p:
        circuit.append("DEPOLARIZE1", list(range(n_data)), p)


def bb_append_mpp_check(circuit, pauli, support):
    targets = []
    for q in np.flatnonzero(support):
        if targets:
            targets.append(stim.target_combiner())
        targets.append(stim.target_x(int(q)) if pauli == "X" else stim.target_z(int(q)))
    if not targets:
        raise ValueError("Cannot measure an empty stabilizer row with MPP")
    circuit.append("MPP", targets)


def bb_rec_abs(circuit, measurement_index):
    return stim.target_rec(int(measurement_index) - circuit.num_measurements)


def bb_measure_stabilizer_round(circuit, Hx, Hz):
    records = {"x": [], "z": []}
    for row in Hx:
        bb_append_mpp_check(circuit, "X", row)
        records["x"].append(circuit.num_measurements - 1)
    for row in Hz:
        bb_append_mpp_check(circuit, "Z", row)
        records["z"].append(circuit.num_measurements - 1)
    return records


def bb_add_round_detectors(circuit, previous, current):
    for kind in ["x", "z"]:
        for old, new in zip(previous[kind], current[kind]):
            circuit.append("DETECTOR", [bb_rec_abs(circuit, old), bb_rec_abs(circuit, new)])


def bb_measure_data(circuit, n_data, memory_basis=Pauli.Z):
    # No noise 
    if bb_is_x_basis(memory_basis):
        circuit.append("H", list(range(n_data)))
    start = circuit.num_measurements
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


def bb_memory_circuit(BB_code, memory_basis=Pauli.Z, rounds=12, p=0.0, logical_indices=None):
    Hx, Hz = bb_get_PCMS(BB_code.matrix)
    n_data = Hx.shape[1]

    circuit = stim.Circuit()
    circuit += bb_prepare_logical_memory_state(BB_code, memory_basis)

    previous = None
    for _ in range(rounds):
        # One memory interval per round, followed by one full syndrome extraction.
        bb_append_code_capacity_noise(circuit, n_data, p)
        current = bb_measure_stabilizer_round(circuit, Hx, Hz)
        if previous is None:
            # Perfect preparation makes the first noiseless syndrome deterministic.
            bb_add_initial_detectors(circuit, current)
        else:
            bb_add_round_detectors(circuit, previous, current)
        previous = current
        circuit.append("TICK")

    # No extra data noise here: final readout immediately closes the last syndrome.
    data_records = bb_measure_data(circuit, n_data, memory_basis=memory_basis)
    bb_add_final_detectors(circuit, Hx, Hz, previous, data_records, memory_basis=memory_basis)
    bb_add_memory_observables(circuit, BB_code, memory_basis, data_records, logical_indices=logical_indices)
    return circuit


class _BPOSDSinterCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, dem, **kwargs):
        self.decoder = BPOSD(dem, **kwargs)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        return self.decoder.decode_batch(
            bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


class BPOSDSinterDecoder(sinter.Decoder):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def compile_decoder_for_dem(self, *, dem):
        return _BPOSDSinterCompiledDecoder(dem, **self.kwargs)


def bb_estimate_memory_error_rates(
    BB_code,
    memory_basis,
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
            p=float(p),
            rounds=rounds,
            shots=shots,
            logical_indices=logical_indices,
            num_workers=1
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





# for the logical + gadget