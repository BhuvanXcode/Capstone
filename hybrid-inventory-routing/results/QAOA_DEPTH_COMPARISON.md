# QAOA Depth Comparison

## Problem
Inventory allocation with four customers.

- Customer demands: 10, 30, 10, 10
- Available inventory: 60
- Number of simulated qubits: 10
- Classical optimum energy: -4
- Optimal allocation: All four customers

## Experimental Results

| Metric | QAOA p=1 | QAOA p=2 |
|---|---:|---:|
| Expected energy | 33537.302890 | 31673.576734 |
| Feasible probability | 0.043011 | 0.045624 |
| Optimal probability | 0.017237 | 0.002888 |
| Best feasible energy | -4 | -4 |

## Observations

1. Both QAOA depths measured the classical optimal solution.
2. Depth p=2 achieved lower expected energy.
3. Depth p=2 achieved slightly higher feasible probability.
4. Depth p=1 achieved a higher probability of measuring the exact optimum.
5. Increasing QAOA depth does not guarantee better optimal-solution probability.

## Conclusion

The experiments demonstrate that QAOA circuit depth affects the
distribution of measured solutions. Further parameter optimization
and repeated experiments are needed before drawing conclusions
about the best circuit depth.
