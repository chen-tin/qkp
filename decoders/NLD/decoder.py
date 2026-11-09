"""Nearest-lattice decoder (memoryless baseline).

The implementation lives in simulator/sim.py so the simulator's own checks
can use it; it is re-exported here so all decoders sit under decoders/.
"""

from simulator.sim import NearestLattice

__all__ = ["NearestLattice"]
