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

All the branches uses python 3.7, more recent versions weren't tested.

## Dependencies installation

### Option 1: Using Conda Environment (Recommended)

The easiest way to set up all dependencies is using the provided conda environment file:

```bash
# Clone and navigate to the repository
git clone https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition.git
cd Deep-Reinforcement-Learning-Hands-On-Second-Edition

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate rlbook
```

### Option 2: Manual Installation

If you prefer manual installation, use conda/pip as follows:

* change directory to book repository dir: `cd Deep-Reinforcement-Learning-Hands-On-Second-Edition`
* create virtual environment with `conda create -n rlbook python=3.11`
* activate it: `conda activate rlbook`
* install pytorch (update CUDA version according to your CUDA): `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
* install rest of dependencies: `pip install -r requirements.txt`

Now you're ready to launch and experiment with examples!
