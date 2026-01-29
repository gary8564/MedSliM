# Med-SliM: Slice-wise Mamba pretraining for medical CT and MRI volumetric medical images from 2D foundation features.

## Prerequisite
1. [uv]()
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. [mise]()
```bash
curl https://mise.run | sh
```

## Installaion
```bash
git clone && cd MedSliM
mise trust
uv venv --python=3.12
source .venv/bin/activate
uv pip install torch==2.6.0 setuptools packaging wheel numpy==2.2.5 hatchling editables
uv sync --no-build-isolation
uv pip install -e .
```

**Note:**

`causal-conv1d` and `mamba-ssm` installation may face issues. If so, try to set up the environment variables as follows:
```bash
# If running on HPC cluster, depending on the cluster arrangement, load a CUDA module that provides nvcc might be needed
# module load CUDA/12.4 || module load cuda/12.4
# Set up environment variables
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

If `flash-attn` installation fails, try to install the matching pre-built wheel instead. 
For Linux user:
```bash
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
You can find the matching wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1).