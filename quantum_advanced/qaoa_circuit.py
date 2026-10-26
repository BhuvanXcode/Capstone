"""Genuine QAOA Quantum Circuit implementation using Qiskit.

Constructs a standard gate-level Quantum Approximate Optimization Algorithm (QAOA)
circuit consisting of:
1. Initial equal superposition state: |+>^n via Hadamard (H) gates.
2. Cost Hamiltonian unitary U(C, gamma) = exp(-i * gamma * H_C):
   - Single-qubit Rz(2 * gamma * h_u) rotations for linear terms.
   - Two-qubit Rzz(2 * gamma * J_uv) entangling interactions for quadratic terms.
3. Mixer Hamiltonian unitary U(B, beta) = exp(-i * beta * H_M):
   - Single-qubit Rx(2 * beta) rotations on all qubits.
4. Measurements into classical registers.
"""

from typing import Dict, List, Sequence, Tuple, Union
from qiskit import QuantumCircuit
import numpy as np

try:
    from .qubo_formulation import RoutingQUBO
except ImportError:
    from qubo_formulation import RoutingQUBO


def build_qaoa_circuit(
    qubo: RoutingQUBO,
    gamma: Union[float, Sequence[float]],
    beta: Union[float, Sequence[float]],
    p_layers: int = 1,
    insert_barriers: bool = True,
    measure: bool = True,
) -> QuantumCircuit:
    """Build a genuine gate-level QAOA quantum circuit.

    Args:
        qubo: RoutingQUBO containing Ising coefficients (h, J).
        gamma: Cost unitary parameter(s). Float if p=1 or list of length p.
        beta: Mixer unitary parameter(s). Float if p=1 or list of length p.
        p_layers: Number of QAOA alternating operator layers (p >= 1).
        insert_barriers: If True, adds visual barriers between cost and mixer blocks.
        measure: If True, appends measurement operations on all qubits.

    Returns:
        Qiskit QuantumCircuit ready for statevector simulation or Aer measurement.
    """
    n_qubits = qubo.n_qubits

    # Normalize parameters to lists
    if isinstance(gamma, (int, float, np.floating)):
        gamma_list = [float(gamma)]
    else:
        gamma_list = [float(g) for g in gamma]

    if isinstance(beta, (int, float, np.floating)):
        beta_list = [float(beta)]
    else:
        beta_list = [float(b) for b in beta]

    if len(gamma_list) != p_layers or len(beta_list) != p_layers:
        raise ValueError(
            f"Expected {p_layers} parameters for gamma and beta, "
            f"got {len(gamma_list)} and {len(beta_list)}."
        )

    qc = QuantumCircuit(n_qubits, name=f"QAOA_Routing_p{p_layers}")

    # --- Step 1: Initial State Preparation (Equal Superposition) ---
    qc.h(range(n_qubits))
    if insert_barriers:
        qc.barrier(label="Hadamard Init")

    # --- Step 2: Alternating QAOA Layers ---
    for layer in range(p_layers):
        g = gamma_list[layer]
        b = beta_list[layer]

        # 2a. Problem Cost Unitary: U(C, gamma) = exp(-i * gamma * H_C)
        # Single-qubit Z rotations: Rz(theta) = exp(-i * theta/2 * Z)
        # We need exp(-i * gamma * h_u * Z_u) => theta = 2 * gamma * h_u
        for u in range(n_qubits):
            if abs(qubo.h[u]) > 1e-7:
                qc.rz(2.0 * g * qubo.h[u], u)

        # Two-qubit ZZ interactions: Rzz(theta) = exp(-i * theta/2 * Z (x) Z)
        # We need exp(-i * gamma * J_uv * Z_u * Z_v) => theta = 2 * gamma * J_uv
        for u in range(n_qubits):
            for v in range(u + 1, n_qubits):
                if abs(qubo.J[u, v]) > 1e-7:
                    qc.rzz(2.0 * g * qubo.J[u, v], u, v)

        if insert_barriers:
            qc.barrier(label=f"Cost Layer {layer+1}")

        # 2b. Mixer Unitary: U(B, beta) = exp(-i * beta * sum X_u)
        # Rx(theta) = exp(-i * theta/2 * X) => theta = 2 * beta
        for u in range(n_qubits):
            qc.rx(2.0 * b, u)

        if insert_barriers:
            qc.barrier(label=f"Mixer Layer {layer+1}")

    # --- Step 3: Measurement ---
    if measure:
        qc.measure_all()

    return qc


def get_circuit_summary(qc: QuantumCircuit) -> Dict[str, Union[int, dict, str]]:
    """Extract circuit metrics: depth, qubit count, gate operations breakdown, and ASCII diagram."""
    ops = dict(qc.count_ops())
    # Exclude barrier and measure from pure logical gate depth calculation
    decomposed_qc = qc.decompose()

    return {
        "n_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "decomposed_depth": decomposed_qc.depth(),
        "gate_counts": ops,
        "ascii_diagram": qc.draw(output="text").__str__(),
    }
