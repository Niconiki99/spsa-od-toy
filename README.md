# spsa-od-toy

Work in progress.

A toy traffic simulator to experiment with OD matrix correction from
sensor counts. The idea is to treat the simulator as a black box (no
gradients) and test gradient-free optimisation on top of it, starting
with SPSA.

For now the repo only contains the simulator:

- `odtoy/network.py`: synthetic grid network with random link capacities
- `odtoy/assignment.py`: BPR costs + MSA assignment
- `odtoy/simulator.py`: `Simulator(od, seed) -> link flows`, counts its own calls

Route choice is all-or-nothing on the shortest path, so the OD -> flows
map is not differentiable. The seed changes link capacities slightly, as
a stand-in for day-to-day variability.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python examples/run_simulator.py
```

## Next

- generate synthetic "days" (true OD, sensor subset, noisy counts)
- a loss between simulated and measured flows
- a first SPSA run directly on the OD cells
