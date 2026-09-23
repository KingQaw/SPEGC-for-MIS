#!/usr/bin/env bash
#
# Rebuild the SPEGC Python environment from scratch.
#
# Tested on: WSL2 (Ubuntu 26.04 userspace) / Python 3.9 / torch 1.9.1+cu111 /
#            detectron2 0.5+cu111, host GPU = 2x RTX 2070 (driver 591.86).
#
# Usage:
#     bash tools/setup_env.sh              # creates ./.venv
#     ENV_PREFIX=/some/path bash tools/setup_env.sh
#
# The environment is deliberately created INSIDE the repository (./.venv),
# which .gitignore already excludes.  Nothing outside the repo is written
# except conda's own package cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.venv}"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
CACHE_DIR="$REPO_ROOT/.cache"

PYPI_TORCH_INDEX="https://download.pytorch.org/whl/cu111/torch_stable.html"
D2_WHEEL_INDEX="https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"

echo "== repo      : $REPO_ROOT"
echo "== env prefix: $ENV_PREFIX"

# ---------------------------------------------------------------------------
# 0. sanity
# ---------------------------------------------------------------------------
command -v "$CONDA" >/dev/null 2>&1 || {
    echo "ERROR: conda not found at $CONDA (set CONDA_EXE=...)"; exit 1;
}
# This machine has no C compiler, so every dependency must ship a wheel.
if ! command -v gcc >/dev/null 2>&1; then
    echo "NOTE: no C compiler found -- only prebuilt wheels can be installed."
fi

# ---------------------------------------------------------------------------
# 1. Python 3.9 base environment
#    -p <prefix>       : prefix env, keeps everything inside the repo
#    --no-plugins      : the conda_anaconda_tos plugin aborts non-interactively
#    --solver=classic  : --no-plugins also disables the libmamba solver plugin
# ---------------------------------------------------------------------------
if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "== reusing existing interpreter at $ENV_PREFIX"
else
    echo "== creating Python 3.9 environment"
    CONDA_PKGS_DIRS="$CACHE_DIR/conda-pkgs" "$CONDA" --no-plugins create \
        -p "$ENV_PREFIX" python=3.9 -y --solver=classic
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
mkdir -p "$CACHE_DIR"

echo "== python: $($PY --version 2>&1)"

# ---------------------------------------------------------------------------
# 2. PyTorch 1.9.1 + CUDA 11.1 runtime
#    (cu111 wheels bundle their own CUDA runtime; no system CUDA toolkit and
#     no compiler are required)
# ---------------------------------------------------------------------------
echo "== installing torch / torchvision / torchaudio (cu111)"
"$PIP" install -q -U pip setuptools wheel
"$PIP" install "torch==1.9.1+cu111" "torchvision==0.10.1+cu111" \
    -f "$PYPI_TORCH_INDEX"
"$PIP" install --no-deps "torchaudio==0.9.1" -f "$PYPI_TORCH_INDEX"

# ---------------------------------------------------------------------------
# 3. Pinned versions that must be overridden (see docs/ENVIRONMENT_SETUP.md)
#    numpy  : requirements pins 1.21.6, but opencv-python 4.8.1.78 uses PEP-585
#             generics (numpy.dtype[...]) requiring >= 1.23; numpy must also
#             stay < 1.24 because evaluation/dice_metric.py:106 uses np.bool.
#    scipy  : 1.7.3 declares numpy<1.23.0, so bump to a version compatible
#             with numpy 1.23.5 (only ndimage/spatial/optimize are used).
# ---------------------------------------------------------------------------
echo "== pinning numpy / Pillow / scipy"
"$PIP" install "numpy==1.23.5" "Pillow==9.5.0" "scipy==1.9.3"

# ---------------------------------------------------------------------------
# 4. Everything else from requirements.txt, minus the packages that come from
#    the special indexes above.
# ---------------------------------------------------------------------------
echo "== installing remaining requirements"
grep -vE "^(torch|torchvision|torchaudio|detectron2)==" "$REPO_ROOT/requirements.txt" \
    > "$CACHE_DIR/req-bulk.txt"
"$PIP" install -r "$CACHE_DIR/req-bulk.txt"

# ---------------------------------------------------------------------------
# 5. detectron2 0.5 (cu111 / torch1.9).  It is NOT on PyPI; the wheel index is
#    the only source.  --no-deps so the resolver cannot undo step 3.
# ---------------------------------------------------------------------------
echo "== installing detectron2 0.5+cu111"
"$PIP" install --no-deps "detectron2==0.5+cu111" -f "$D2_WHEEL_INDEX"

# ---------------------------------------------------------------------------
# 6. distutils compatibility shim.
#    torch 1.9.1 does `from setuptools import distutils` followed by
#    `distutils.version.LooseVersion` (attribute access on a submodule that
#    nothing imports).  detectron2's default_writers() -> TensorboardXWriter
#    therefore raises AttributeError on any modern setuptools.  Importing the
#    submodule at interpreter start-up fixes it without patching packages.
# ---------------------------------------------------------------------------
SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "== installing distutils shim into $SITE_PACKAGES"
cp "$REPO_ROOT/tools/env_compat/sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"

# ---------------------------------------------------------------------------
# 7. verify
# ---------------------------------------------------------------------------
echo "== verifying"
"$PY" - <<'PYEOF'
import numpy, cv2, torch, torchvision, detectron2
from torch.utils.tensorboard import SummaryWriter  # distutils shim check
print("  torch      ", torch.__version__)
print("  torchvision", torchvision.__version__)
print("  detectron2 ", detectron2.__version__)
print("  numpy      ", numpy.__version__)
print("  opencv     ", cv2.__version__)
print("  cuda avail ", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())
PYEOF
"$PIP" check

cat <<EOF

Environment ready.

  Activate : conda activate $ENV_PREFIX     (or use $PY directly)
  Data     : $PY tools/make_synthetic_fundus.py
  Train    : $PY train_net.py --num-gpus 1 --config configs/smoke_baseline.yaml
  Evaluate : $PY train_net.py --eval-only --num-gpus 1 \\
                 --config configs/smoke_baseline.yaml \\
                 MODEL.WEIGHTS output/smoke_baseline/model_final.pth
EOF
