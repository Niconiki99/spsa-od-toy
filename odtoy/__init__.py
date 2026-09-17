"""odtoy: a small traffic-assignment toy used as a black-box simulator.

The package mimics the structure of a real OD-calibration problem:
an origin-destination matrix goes into a simulator that is *not*
differentiable (route choice is discrete), and what comes out are link
flows that can be compared with sensor counts.
"""

from .network import Network, grid_network
from .assignment import assign, bpr_time
from .simulator import Simulator

__all__ = ["Network", "grid_network", "assign", "bpr_time", "Simulator"]
