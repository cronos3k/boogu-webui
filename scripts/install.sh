#!/usr/bin/env bash
# One-shot installer for the Boogu-Image web UI (Linux).
#  - clones the official boogu-project/Boogu-Image repo
#  - creates a venv and installs torch + deps + (optional) flash-attn
#  - drops the UI files (app.py, run_boogu.sh, ui.sh, scripts/) into the checkout
#
# Usage:   bash scripts/install.sh [--cuda cu126] [--no-flash]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CUDA="cu126"; FLASH=1
while [ $# -gt 0 ]; do case "$1" in
  --cuda) CUDA="$2"; shift 2;;
  --no-flash) FLASH=0; shift;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done

cd "$HERE"
if [ ! -d Boogu-Image ]; then
  echo "== cloning boogu-project/Boogu-Image =="
  git clone --depth 1 https://github.com/boogu-project/Boogu-Image.git Boogu-Image
fi
cd Boogu-Image

echo "== creating venv =="
python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip

echo "== installing torch ($CUDA) =="
pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

echo "== installing deps =="
pip install -q -r "$HERE/requirements.txt" || pip install -q \
  "diffusers[torch]>=0.35.2,<0.39" "transformers[torch]>=4.57.3,<6" accelerate \
  "torchao>=0.15,<0.18" "cache-dit>=1.3,<2" "kernels>=0.14,<0.15" einops numpy \
  pillow scipy "webdataset>=1.0,<2" python-dotenv omegaconf "huggingface_hub>=0.34" gradio

if [ "$FLASH" = 1 ]; then
  echo "== installing flash-attn (Linux + Ampere/Ada; skip on Turing/Windows) =="
  ABI=$(python -c "import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  PYV=$(python -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abi${ABI}-${PYV}-${PYV}-linux_x86_64.whl"
  pip install "$WHL" || echo "!! flash-attn wheel didn't match; UI still works via SDPA (slower)."
fi

echo "== installing UI files into the checkout =="
cp "$HERE/app.py" "$HERE/run_boogu.sh" "$HERE/ui.sh" .
mkdir -p scripts && cp "$HERE/scripts/download_weights.py" scripts/
chmod +x run_boogu.sh ui.sh

cat <<EOF

Done. Next:
  cd "$HERE/Boogu-Image"
  . .venv/bin/activate
  python scripts/download_weights.py all      # downloads Base + Turbo + Edit (~115 GB)
  ./ui.sh start                               # launches the web UI; prints the URL
EOF
