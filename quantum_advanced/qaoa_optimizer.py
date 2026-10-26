"""QAOA Optimization and AerSimulator Execution Module.

Executes a genuine classical-quantum hybrid variational loop to optimize QAOA
parameters (gamma, beta) using Scipy and Qiskit Statevector, then simulates physical
quantum measurement sampling using Qiskit Aer (AerSimulator) with real shots.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

try:
    from .data_loader import ProblemInstance
    from .qubo_formulation import RoutingQUBO
    from .qaoa_circuit import build_qaoa_circuit, get_circuit_summary
except ImportError:
    from data_loader import ProblemInstance
    from qubo_formulation import RoutingQUBO
    from qaoa_circuit import build_qaoa_circuit, get_circuit_summary


@dataclass
class QAOAExperimentResult:
    """Full results from a QAOA optimization and AerSimulator sampling run."""
    p_layers: int
    n_qubits: int
    circuit_depth: int
    gate_counts: Dict[str, int]
    initial_parameters: List[float]
    optimized_parameters: List[float]
    initial_expected_energy: float
    optimized_expected_energy: float
    optimizer_success: bool
    iterations_count: int
    convergence_history: List[Dict[str, float]]  # List of {"iteration": i, "energy": E, "gamma": g, "beta": b}
    shots: int
    simulator_name: str
    measurement_counts: Dict[str, int]
    measurement_probabilities: Dict[str, float]
    optimization_time_seconds: float
    simulation_time_seconds: float


class QAOAOptimizer:
    """Hybrid Variational Optimizer and Aer Simulator Executor."""

    def __init__(self, qubo: RoutingQUBO, p_layers: int = 1):
        self.qubo = qubo
        self.p_layers = p_layers
        self.n_qubits = qubo.n_qubits
        self.hamiltonian = qubo.hamiltonian
        self.simulator = AerSimulator()

    def run_optimization(
        self,
        initial_params: Optional[List[float]] = None,
        max_iter: int = 60,
        shots: int = 5000,
        seed: int = 42,
    ) -> QAOAExperimentResult:
        """Execute the variational parameter optimization and AerSimulator sampling.

        Args:
            initial_params: Starting [gamma_1, ..., gamma_p, beta_1, ..., beta_p].
            max_iter: Maximum classical optimizer evaluations (COBYLA).
            shots: Number of measurement shots for AerSimulator.
            seed: Random seed for simulation reproducibility.

        Returns:
            QAOAExperimentResult with complete circuit, optimization, and measurement statistics.
        """
        if initial_params is None:
            # Heuristic standard initialization for QAOA
            initial_params = [0.05] * self.p_layers + [0.35] * self.p_layers

        history = []
        iteration_counter = 0

        # Objective function for classical optimizer: <psi(gamma, beta)| H_C |psi(gamma, beta)>
        def expectation_objective(params: np.ndarray) -> float:
            nonlocal iteration_counter
            gammas = params[:self.p_layers]
            betas = params[self.p_layers:]

            if self.n_qubits <= 16:
                # Fast exact statevector expectation (< 0.1s for <= 16 qubits)
                qc = build_qaoa_circuit(
                    self.qubo,
                    gamma=gammas,
                    beta=betas,
                    p_layers=self.p_layers,
                    insert_barriers=False,
                    measure=False,
                )
                state = Statevector.from_instruction(qc)
                expected_energy = float(state.expectation_value(self.hamiltonian).real)
            else:
                # Shot-based expectation on AerSimulator (matches real QC execution, avoids 512MB RAM bandwidth choke)
                eval_shots = 300
                qc = build_qaoa_circuit(
                    self.qubo,
                    gamma=gammas,
                    beta=betas,
                    p_layers=self.p_layers,
                    insert_barriers=False,
                    measure=True,
                )
                tqc = transpile(qc, self.simulator, optimization_level=0, seed_transpiler=seed)
                job = self.simulator.run(tqc, shots=eval_shots, seed_simulator=seed + iteration_counter)
                cnts = job.result().get_counts()
                total_e = 0.0
                for s, count in cnts.items():
                    x_vec = np.array([int(s[self.n_qubits - 1 - u]) for u in range(self.n_qubits)], dtype=float)
                    total_e += self.qubo.evaluate_bitstring_qubo(x_vec) * count
                expected_energy = float(total_e / eval_shots)

            history.append({
                "iteration": iteration_counter,
                "energy": round(expected_energy, 4),
                "params": [round(float(p), 5) for p in params],
            })
            iteration_counter += 1
            return expected_energy

        # Record initial baseline energy
        initial_energy = expectation_objective(np.array(initial_params))

        # Run classical optimization (COBYLA is standard for noisy/variational quantum surfaces)
        opt_start = time.perf_counter()
        opt_res = minimize(
            expectation_objective,
            x0=np.array(initial_params),
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.15},
        )
        opt_time = time.perf_counter() - opt_start

        best_params = opt_res.x.tolist()
        best_gammas = best_params[:self.p_layers]
        best_betas = best_params[self.p_layers:]
        best_energy = float(opt_res.fun)

        # Build final measured circuit with optimal parameters
        final_qc = build_qaoa_circuit(
            self.qubo,
            gamma=best_gammas,
            beta=best_betas,
            p_layers=self.p_layers,
            insert_barriers=True,
            measure=True,
        )

        circuit_summary = get_circuit_summary(final_qc)

        # Transpile and execute on Qiskit AerSimulator
        sim_start = time.perf_counter()
        transpiled_qc = transpile(final_qc, self.simulator, optimization_level=0, seed_transpiler=seed)
        sim_job = self.simulator.run(transpiled_qc, shots=shots, seed_simulator=seed)
        counts = sim_job.result().get_counts()
        sim_time = time.perf_counter() - sim_start

        # Calculate measurement probabilities
        probabilities = {k: v / shots for k, v in counts.items()}

        return QAOAExperimentResult(
            p_layers=self.p_layers,
            n_qubits=self.n_qubits,
            circuit_depth=circuit_summary["depth"],
            gate_counts=circuit_summary["gate_counts"],
            initial_parameters=[round(float(p), 5) for p in initial_params],
            optimized_parameters=[round(float(p), 5) for p in best_params],
            initial_expected_energy=round(initial_energy, 4),
            optimized_expected_energy=round(best_energy, 4),
            optimizer_success=bool(opt_res.success),
            iterations_count=iteration_counter,
            convergence_history=history,
            shots=shots,
            simulator_name=self.simulator.name,
            measurement_counts=counts,
            measurement_probabilities=probabilities,
            optimization_time_seconds=round(opt_time, 4),
            simulation_time_seconds=round(sim_time, 4),
        )
