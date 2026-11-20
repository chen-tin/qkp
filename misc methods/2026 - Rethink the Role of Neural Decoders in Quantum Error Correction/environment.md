Use Python 3.11 in an isolated environment unless the upstream package requires otherwise. The shared runner itself uses only Python standard-library modules.

Install the pinned upstream requirements, when supplied, before its model code. On macOS, PyTorch Geometric compiled extensions and OpenMP may need separate installation; TensorFlow code should use its compatible Keras version.

No production environment has been validated for this paper yet. The runner records interpreter, platform and pip freeze for each execution. Never reuse or alter the active qkp training environment for dependency experiments.

Verified resolver conflict: upstream numpy==2.2.6 and brevitas==0.12.1 cannot resolve together. Local requirements use numpy==1.26.4. This is a compatibility deviation, not the original paper environment.

The SciPy 1.15.3 macOS wheel failed to load (dyld thread_bss error). The isolated smoke environment uses 1.16.3; record that production-environment deviation.
