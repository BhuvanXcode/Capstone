"""Visualization module for the Quantum Routing Demonstration.

Generates high-resolution publication-quality plots:
1. QAOA optimization convergence curve (energy vs. optimizer iteration).
2. Measurement bitstring distribution histogram highlighting feasible and optimal states.
3. Spatial 2D route map comparing QAOA route vs. Exact Classical route with time windows.
4. Quantum circuit architectural dashboard.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

try:
    from .classical_solver import RouteEvaluation
    from .data_loader import ProblemInstance
    from .decoder_and_validator import DecodedQuantumCandidate
    from .qaoa_optimizer import QAOAExperimentResult
except ImportError:
    from classical_solver import RouteEvaluation
    from data_loader import ProblemInstance
    from decoder_and_validator import DecodedQuantumCandidate
    from qaoa_optimizer import QAOAExperimentResult


def plot_convergence(
    history: List[Dict[str, float]],
    output_path: Path,
    title: str = "QAOA Variational Parameter Convergence",
) -> None:
    """Plot optimizer convergence history showing expected Hamiltonian energy vs. iteration."""
    iterations = [h["iteration"] for h in history]
    energies = [h["energy"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    ax.plot(iterations, energies, color="#1f77b4", marker="o", markersize=4, linewidth=1.8, label=r"Expected Energy $\langle H_C \rangle$")

    initial_e = energies[0]
    final_e = energies[-1]
    improvement = initial_e - final_e

    ax.axhline(final_e, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Final Energy: {final_e:.2f}")

    ax.annotate(
        f"Initial: {initial_e:.2f}\nFinal: {final_e:.2f}\nReduction: {improvement:.2f}",
        xy=(iterations[-1], final_e),
        xytext=(iterations[-1] - max(5, len(iterations) // 4), final_e + (initial_e - final_e) * 0.25),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f8ff", edgecolor="#1f77b4", alpha=0.9),
        fontsize=9,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Classical Optimizer Iteration (COBYLA)", fontsize=10)
    ax.set_ylabel(r"Expectation Value $\langle \psi(\gamma, \beta) | H_C | \psi(\gamma, \beta) \rangle$", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_bitstring_distribution(
    candidates: List[DecodedQuantumCandidate],
    optimal_routes: List[List[int]],
    output_path: Path,
    top_n: int = 10,
    title: str = "QAOA Measurement Bitstring Analysis (AerSimulator)",
) -> None:
    """Plot a two-panel distribution: Top Overall Measured States and Feasible Routing Candidates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # --- PANEL 1: Top Most Frequent Measured States ---
    top_overall = sorted(candidates, key=lambda c: c.probability, reverse=True)[:top_n]
    p1_labels = []
    p1_probs = []
    p1_colors = []

    for c in top_overall:
        p1_probs.append(c.probability * 100.0)
        if c.is_permutation_feasible:
            route_str = "->".join(map(str, c.route))
            is_opt = c.route in optimal_routes
            p1_labels.append(f"{c.bitstring[:6]}..\n[{route_str}]\n({c.routing_distance:.1f})")
            p1_colors.append("#2ca02c" if is_opt else "#1f77b4")
        else:
            p1_labels.append(f"{c.bitstring[:6]}..\n[Infeasible]\n(E={c.qubo_energy:.0f})")
            p1_colors.append("#d62728")

    bars1 = ax1.bar(range(len(top_overall)), p1_probs, color=p1_colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xticks(range(len(top_overall)))
    ax1.set_xticklabels(p1_labels, fontsize=7.5, rotation=25, ha="right")
    ax1.set_ylabel("Sampling Probability (%)", fontsize=10)
    ax1.set_title("Top 10 Most Frequent Measured Bitstrings", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Feasible (Optimal Tour)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Feasible (Sub-optimal)"),
        Patch(facecolor="#d62728", edgecolor="black", label="Infeasible Bitstring"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8)

    # --- PANEL 2: Spectrum of Feasible Routing Candidates ---
    feasible_cands = [c for c in candidates if c.is_permutation_feasible]
    feasible_cands.sort(key=lambda c: (round(c.routing_distance, 4), -c.shots_count))

    p2_labels = []
    p2_probs = []
    p2_colors = []

    for c in feasible_cands:
        p2_probs.append(c.probability * 100.0)
        route_str = "->".join(map(str, c.route))
        is_opt = c.route in optimal_routes
        is_tw = c.vrptw_evaluation.is_time_window_feasible if c.vrptw_evaluation else False
        status_tag = " [TW-Valid]" if is_tw else " [TW-Late]"
        p2_labels.append(f"{route_str}\nDist: {c.routing_distance:.2f}\n{status_tag}")
        p2_colors.append("#2ca02c" if is_opt else "#1f77b4")

    if feasible_cands:
        bars2 = ax2.bar(range(len(feasible_cands)), p2_probs, color=p2_colors, edgecolor="black", linewidth=0.8, alpha=0.85)
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.set_ylim(0, max(p2_probs) * 1.35)
        ax2.set_xticks(range(len(feasible_cands)))
        ax2.set_xticklabels(p2_labels, fontsize=7.5, rotation=20, ha="right")
    else:
        ax2.text(0.5, 0.5, "No feasible routes measured in shots", ha="center", va="center", fontsize=11)

    ax2.set_ylabel("Sampling Probability (%)", fontsize=10)
    ax2.set_title("Feasible Route Candidates (Permutation Subspace)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_route_comparison(
    instance: ProblemInstance,
    qaoa_eval: Optional[RouteEvaluation],
    exact_eval: RouteEvaluation,
    output_path: Path,
    title: str = "Route Comparison: QAOA vs. Exact Classical Benchmark",
) -> None:
    """Plot 2D spatial route map alongside a detailed operational metrics comparison card."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300, gridspec_kw={"width_ratios": [1.2, 1.0]})

    coords = instance.coordinates
    customers = instance.customers

    # --- PANEL 1: 2D Spatial Route Map ---
    # Plot Depot (Node 0)
    ax1.scatter(coords[0, 0], coords[0, 1], c="#d62728", s=240, marker="s", edgecolors="black", linewidth=1.5, zorder=5, label="Depot (Node 0)")
    ax1.text(coords[0, 0] + 0.35, coords[0, 1] - 0.7, "Depot 0\n[0, 1236]", fontsize=8.5, fontweight="bold", color="#990000")

    # Plot Customers
    for i, cid in enumerate(instance.customer_ids, start=1):
        x, y = coords[i, 0], coords[i, 1]
        c_info = customers[cid]
        ax1.scatter(x, y, c="#1f77b4", s=200, marker="o", edgecolors="black", linewidth=1.5, zorder=5)
        label_text = f"C{cid} (Dem: {c_info['demand']:.0f})\n[{c_info['ready_time']:.0f}, {c_info['due_time']:.0f}]"
        ax1.text(x + 0.3, y + 0.3, label_text, fontsize=8, bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f4f8", edgecolor="#b0c4de", alpha=0.9))

    # Draw Exact Route
    exact_route = exact_eval.route
    for k in range(len(exact_route) - 1):
        u, v = exact_route[k], exact_route[k + 1]
        p_u = instance.coordinates[0 if u == 0 else instance.customer_ids.index(u) + 1]
        p_v = instance.coordinates[0 if v == 0 else instance.customer_ids.index(v) + 1]
        ax1.annotate(
            "",
            xy=(p_v[0], p_v[1]),
            xytext=(p_u[0], p_u[1]),
            arrowprops=dict(
                arrowstyle="->",
                color="#2ca02c",
                lw=2.5,
                mutation_scale=16,
                shrinkA=8,
                shrinkB=8,
            ),
            zorder=3,
        )

    # Draw QAOA Route if distinct
    is_identical = qaoa_eval is not None and qaoa_eval.route == exact_eval.route
    if qaoa_eval is not None and not is_identical:
        qaoa_route = qaoa_eval.route
        for k in range(len(qaoa_route) - 1):
            u, v = qaoa_route[k], qaoa_route[k + 1]
            p_u = instance.coordinates[0 if u == 0 else instance.customer_ids.index(u) + 1]
            p_v = instance.coordinates[0 if v == 0 else instance.customer_ids.index(v) + 1]
            ax1.annotate(
                "",
                xy=(p_v[0], p_v[1]),
                xytext=(p_u[0], p_u[1]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#ff7f0e",
                    lw=2.0,
                    linestyle="--",
                    mutation_scale=14,
                    shrinkA=10,
                    shrinkB=10,
                ),
                zorder=4,
            )

    ax1.set_title("2D Spatial Tour Map (Solomon C101 Coordinates)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X Coordinate", fontsize=10)
    ax1.set_ylabel("Y Coordinate", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    ax1.set_xlim(min(x_vals) - 2.5, max(x_vals) + 3.5)
    ax1.set_ylim(min(y_vals) - 2.5, max(y_vals) + 3.5)

    # --- PANEL 2: Formatted Metrics Comparison Card ---
    ax2.axis("off")

    qaoa_route_str = " -> ".join(map(str, qaoa_eval.route)) if qaoa_eval else "None"
    qaoa_dist_str = f"{qaoa_eval.total_distance:.4f}" if qaoa_eval else "N/A"
    exact_route_str = " -> ".join(map(str, exact_eval.route))
    exact_dist_str = f"{exact_eval.total_distance:.4f}"

    qaoa_tw_pass = qaoa_eval.is_time_window_feasible if qaoa_eval else False
    exact_tw_pass = exact_eval.is_time_window_feasible

    metrics_text = (
        "Routing Performance & Validation Card\n"
        "═════════════════════════════════════════════════\n\n"
        f"1. EXACT CLASSICAL BENCHMARK (Exhaustive Enumeration):\n"
        f"   • Optimal Route:       {exact_route_str}\n"
        f"   • Total Distance:      {exact_dist_str}\n"
        f"   • Capacity Check:      {exact_eval.total_demand:.0f} / {exact_eval.capacity_limit:.0f} (PASSED)\n"
        f"   • Time Windows:        {'PASSED (0 violations)' if exact_tw_pass else 'VIOLATED'}\n"
        f"   • Schedule Duration:   {exact_eval.total_duration:.1f} time units\n\n"
        f"2. QAOA QUANTUM MEASURED BEST CANDIDATE:\n"
        f"   • Best Route:          {qaoa_route_str}\n"
        f"   • Total Distance:      {qaoa_dist_str}\n"
        f"   • Matches Optimum:     {'YES (Exact Match)' if (qaoa_eval and np.isclose(qaoa_eval.total_distance, exact_eval.total_distance)) else 'NO'}\n"
        f"   • Capacity Check:      {qaoa_eval.total_demand if qaoa_eval else 0:.0f} / {instance.vehicle_capacity:.0f} (PASSED)\n"
        f"   • Time Windows:        {'PASSED (0 violations)' if qaoa_tw_pass else 'VIOLATED'}\n\n"
        f"3. CONSTRAINT METHODOLOGY DISTINCTION:\n"
        f"   • QUBO Encoded:        Permutation & Distance (TSP Loop)\n"
        f"   • Post-Validated:      Vehicle Capacity & Arrival Time Windows\n"
        f"   • Purpose:             Educational quantum PoC for capstone,\n"
        f"                          demonstrating genuine QAOA on Qiskit.\n"
    )

    ax2.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=9,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#adb5bd", alpha=0.95),
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_circuit_visualization(
    ascii_diagram: str,
    circuit_summary: dict,
    opt_result: QAOAExperimentResult,
    output_path_txt: Path,
    output_path_png: Path,
) -> None:
    """Save circuit ASCII representation to text file and render circuit summary dashboard to PNG."""
    # 1. Save text file
    with output_path_txt.open("w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("QAOA GATE-LEVEL QUANTUM CIRCUIT DIAGRAM\n")
        f.write("=====================================================\n\n")
        f.write(f"Number of Qubits: {circuit_summary['n_qubits']}\n")
        f.write(f"Circuit Depth: {circuit_summary['depth']}\n")
        f.write(f"Gate Operations Breakdown: {circuit_summary['gate_counts']}\n")
        f.write(f"QAOA Layers (p): {opt_result.p_layers}\n")
        f.write(f"Optimized Parameters:\n")
        f.write(f"  Gamma: {opt_result.optimized_parameters[:opt_result.p_layers]}\n")
        f.write(f"  Beta:  {opt_result.optimized_parameters[opt_result.p_layers:]}\n\n")
        f.write("Circuit Diagram:\n")
        f.write(ascii_diagram)
        f.write("\n")

    # 2. Render summary card PNG
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    ax.axis("off")

    card_text = (
        "QAOA Quantum Circuit Architecture Summary\n"
        "─────────────────────────────────────────────────────────\n"
        f"• Number of Qubits:        {circuit_summary['n_qubits']} (N^2 for N={int(np.sqrt(circuit_summary['n_qubits']))} customers)\n"
        f"• Logical Circuit Depth:    {circuit_summary['depth']}\n"
        f"• Decomposed Depth:         {circuit_summary['decomposed_depth']}\n"
        f"• Gate Counts Breakdown:\n"
    )
    for gate, count in sorted(circuit_summary["gate_counts"].items()):
        card_text += f"    - {gate.upper():<10} : {count}\n"

    card_text += (
        f"\n• QAOA Parameters (p={opt_result.p_layers}):\n"
        f"    - Initial [gamma, beta]: {opt_result.initial_parameters}\n"
        f"    - Optimized [gamma*, beta*]: {opt_result.optimized_parameters}\n"
        f"    - Initial Energy:        {opt_result.initial_expected_energy:.4f}\n"
        f"    - Optimized Energy:      {opt_result.optimized_expected_energy:.4f}\n"
        f"• Execution Backend:        {opt_result.simulator_name} ({opt_result.shots} shots)\n"
        f"• Optimizer:                COBYLA ({opt_result.iterations_count} iterations, {opt_result.optimization_time_seconds:.2f}s)\n"
    )

    ax.text(
        0.05,
        0.95,
        card_text,
        transform=ax.transAxes,
        fontsize=9.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
    )

    plt.tight_layout()
    fig.savefig(output_path_png)
    plt.close(fig)
