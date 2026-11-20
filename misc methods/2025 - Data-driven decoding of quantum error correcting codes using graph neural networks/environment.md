Use Python 3.11 in an isolated environment unless the upstream package requires otherwise. The shared runner itself uses only Python standard-library modules.

Install the pinned upstream requirements, when supplied, before its model code. On macOS, PyTorch Geometric compiled extensions and OpenMP may need separate installation; TensorFlow code should use its compatible Keras version.

No production environment has been validated for this paper yet. The runner records interpreter, platform and pip freeze for each execution. Never reuse or alter the active qkp training environment for dependency experiments.
