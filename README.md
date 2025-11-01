# Deep-Reinforcement-Learning-Hands-On-Second-Edition
Deep-Reinforcement-Learning-Hands-On-Second-Edition, published by Packt

## Code branches
The repository is maintained to keep dependency versions up-to-date. 
This requires efforts and time to test all the examples on new versions, so, be patient.

The logic is following: there are several branches of the code, corresponding to 
major pytorch version code was tested. Due to incompatibilities in pytorch and other components,
**code in the printed book might differ from the code in the repo**.

At the moment, there are the following branches available:
* `master`: contains the code with the latest pytorch which was tested. At the moment, it is pytorch 1.7.
* `torch-1.3-book`: code printed in the book with minor bug fixes. Uses pytorch=1.3 which 
is available only on conda repos.
* `torch-1.7`: pytorch 1.7. This branch was tested and merged into master.
* `torch-2.6`: pytorch 2.6.0 with Python 3.11. Includes platform-specific environment files for macOS (MPS support) and Linux/Windows (CUDA support). This branch maintains compatibility with modern dependencies.

**Note**: Older branches use python 3.7. The `torch-2.6` branch uses python 3.11.

## Dependencies installation

### Option 1: Using Conda Environment (Recommended)

We provide platform-specific environment files to ensure smooth installation:

**For macOS (Apple Silicon) - MPS GPU acceleration included:**
```bash
# Clone and navigate to the repository
git clone https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition.git
cd Deep-Reinforcement-Learning-Hands-On-Second-Edition

# Create conda environment (works on macOS, Linux, Windows - CPU only)
conda env create -f environment.yml

# Activate the environment
conda activate rlbook
```

**For Linux/Windows with NVIDIA GPU - CUDA support included:**
```bash
# Use the CUDA-enabled environment file
conda env create -f environment-cuda.yml

# Activate the environment
conda activate rlbook
```

**Note:** 
- `environment.yml`: Works on all platforms. Includes PyTorch without CUDA (CPU-only on Linux/Windows, MPS-enabled on macOS)
- `environment-cuda.yml`: For Linux/Windows with NVIDIA GPUs. Includes CUDA 12.4 support. **Do not use on macOS** - CUDA is not available on macOS.
- **macOS users**: MPS (Metal Performance Shaders) GPU acceleration is **automatically available** with PyTorch 2.6.0+. No additional installation needed! See the [GPU Acceleration](#gpu-acceleration) section below.

### Option 2: Manual Installation

If you prefer manual installation, use conda/pip as follows:

* change directory to book repository dir: `cd Deep-Reinforcement-Learning-Hands-On-Second-Edition`
* create virtual environment with `conda create -n rlbook python=3.11`
* activate it: `conda activate rlbook`
* install pytorch:
  * **Linux/Windows (NVIDIA GPU)**: `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
  * **macOS/Linux/Windows (CPU-only)**: `conda install pytorch torchvision torchaudio -c pytorch` (MPS will be available on macOS automatically)
* install rest of dependencies: `pip install -r requirements.txt`

Now you're ready to launch and experiment with examples!

## GPU Acceleration

### Apple Silicon (M1/M2/M3/M4) Macs

**Yes!** Regular PyTorch from conda/pip includes built-in support for **Metal Performance Shaders (MPS)**, which provides GPU acceleration on Apple Silicon Macs. There's no special "Metal version" needed - it's built into the standard PyTorch distribution.

As mentioned in [Apple's PyTorch documentation](https://developer.apple.com/metal/pytorch/), MPS support is included in PyTorch 1.12+ and is automatically enabled - no additional installation required!

**To use MPS in your code**, you can use the provided utility function:

```python
from utils import get_device

# Automatically selects MPS on macOS, CUDA on Linux/Windows, or CPU as fallback
device = get_device()
# Or explicitly prefer CUDA if available:
device = get_device(cuda=True)

# Use the device
model = YourModel().to(device)
```

**Alternative**: Manually check for MPS:

```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

**Note**: Some examples in this repository use `--cuda` flags. On macOS, you can modify them to use MPS instead, or use the `utils.get_device()` function which handles this automatically.

**Test MPS availability**: After setting up the environment, run `python test_gpu.py` to verify GPU acceleration is working.

### Linux/Windows (NVIDIA GPU)

For NVIDIA GPUs, install CUDA support as described in the installation steps above, then use:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
