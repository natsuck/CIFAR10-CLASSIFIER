# CIFAR-10 Classifier

Small PyTorch project with a simple CNN for CIFAR-10 classification.

**Contents**
- `models/` — model definitions (e.g. `cnn.py`)
- `src/` — training and utility scripts (`train.py`)
- `outputs/` — saved model checkpoints (created automatically by `train.py`)
- `requirements.txt` — Python dependencies

**Prerequisites**
- Windows with Python 3.11+ installed.
- A virtual environment (recommended) for the project.
- NVIDIA GPU and drivers that support the CUDA version you install for PyTorch (optional but recommended for training performance).

Quick setup (PowerShell)

```powershell
# from project root
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Note: The repository may use PyTorch wheels built for a specific CUDA version (for example `+cu118`). If you need a specific CUDA build, follow instructions at https://pytorch.org to install the matching wheel for your CUDA version.

Verify PyTorch + GPU availability

```powershell
# run inside the virtualenv
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); import sys; print('python', sys.executable)"
# If cuda is available you can also run:
# nvidia-smi
```

Running training

```powershell
cd src
# Run training (this script will create ../outputs if missing)
python train.py
```

- Training prints device information on startup. If CUDA is available and the environment is configured, training will run on GPU.
- Models are saved to `outputs/best_model.pth` by default.

Common issues & troubleshooting

- ModuleNotFoundError: No module named 'models'
  - Run the script from project root or `src` depending on how your project is structured. The project contains a small path fix inside `src/train.py` that adds the project root to `sys.path` so `from models.cnn import SimpleCNN` works when running from `src`.
  - Alternative: run with `python -m src.train` from project root or make `models` an installable package.

- RuntimeError: Parent directory ../outputs does not exist
  - `train.py` now ensures `outputs/` exists before saving. If you see this, ensure the script has permission to create directories in the project root.

- CUDA not available or wrong driver
  - Make sure your NVIDIA driver supports the CUDA runtime reported by `torch.version.cuda`.
  - Run `nvidia-smi` to inspect driver version.
  - If needed, update GPU drivers from NVIDIA and reinstall matching PyTorch wheels.
