# spsa-od-toy

Work in progress.

A toy traffic simulator to experiment with OD matrix correction from
sensor counts. The idea is to treat the simulator as a black box (no
gradients) and test gradient-free optimisation on top of it, starting
with SPSA.

What is in the package so far:

- `odtoy/network.py`: synthetic grid network with random link capacities
- `odtoy/assignment.py`: BPR costs + MSA assignment
- `odtoy/simulator.py`: `Simulator(od, seed) -> link flows`, counts its own calls
- `odtoy/days.py`: gravity-model mean OD, daily variations, biased prior
- `odtoy/sensors.py`: sensor placement, noisy counts, fit/holdout split
- `odtoy/scenario.py`: `make_scenario(seed)` bundles network, simulator,
  sensors, prior and a list of days (true OD hidden, counts observed)
- `odtoy/metrics.py`: relative count loss, GEH statistic, OD recovery errors

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
python examples/make_days.py
```

## Next

- a first SPSA run directly on the OD cells
