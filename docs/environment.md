# Environments

The core package requires NumPy, PyTorch, pandas and scikit-learn. Training additionally uses LibAUC 1.4.0 and a matching torchvision installation. Data export uses OpenCV/Pillow; JAX point hosts have a separate upstream dependency set.

The recorded remote point/TrackNet experiment stack included Python 3.10, PyTorch 2.11.0+cu128, NumPy 2.2.6, scikit-learn 1.7.2 and LibAUC 1.4.0. The recorded native point stack included JAX/jaxlib 0.6.2, dm-haiku 0.0.17, optax 0.2.8 and chex 0.1.90. `requirements-reference.txt` and `requirements-points.txt` preserve these known package pins; they are not a complete operating-system lockfile.

The local core-package smoke test used Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, NumPy 2.5.2, pandas 3.0.5 and scikit-learn 1.9.0. This validates package execution; it is not a claim that the paper was retrained under this stack.

Install PyTorch/torchvision as a matching pair from the PyTorch distribution for your platform. For JAX native export, follow the upstream CUDA installation instructions. Keep the Keras weight-conversion environment separate from the current NumPy/PyTorch environment.

Training uses deterministic seeding and disables TF32. Released numerical model weights and readouts are provided without optimizer states. Small differences across GPU, BLAS, library and video-decoder versions may affect candidates near decision thresholds.
