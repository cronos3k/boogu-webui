#!/usr/bin/env bash
# run_boogu.sh - launcher for Boogu-Image on a multi-GPU server.
# Auto-selects a free flash-attn-capable GPU (Ampere/Ada; skips the Turing RTX 8000).
#
# Usage:
#   ./run_boogu.sh base  "a fox astronaut on the moon, cinematic"
#   ./run_boogu.sh turbo "a fox astronaut on the moon"                 # fast, 4 steps
#   IMAGE=input_image_examples/03.jpg ./run_boogu.sh edit "put a red hat on the dog"
#
# Optional env overrides: H, W (resolution), ICFG (image cfg for edit),
#   STEPS, CFG, GPU (force a specific CUDA index), OFFLOAD=0 to disable cpu offload.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1
# make CUDA device indices match nvidia-smi indices (default FASTEST_FIRST reorders them!)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${1:-base}"; PROMPT="${2:-}"
[ -n "$PROMPT" ] || { echo "usage: $0 <base|turbo|edit> \"prompt\""; exit 1; }

# pick GPU with most free VRAM among flash-attn-capable cards (exclude Turing RTX 8000)
if [ -z "${GPU:-}" ]; then
  GPU=$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
        | grep -viE "Quadro RTX 8000|Tesla K|Tesla P" \
        | sort -t',' -k3 -n -r | head -1 | awk -F',' '{gsub(/ /,"",$1);print $1}')
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export device="cuda:0"
echo "[run_boogu] GPU idx=$GPU ($(nvidia-smi -i "$GPU" --query-gpu=name,memory.free --format=csv,noheader))"

case "$MODEL" in
  base)  SCRIPT=inference.py;       DEF_STEPS=28; DEF_CFG=4.0; MP=models/Boogu-Image-0.1-Base  ;;
  turbo) SCRIPT=inference_turbo.py; DEF_STEPS=4;  DEF_CFG=1.0; MP=models/Boogu-Image-0.1-Turbo ;;
  edit)  SCRIPT=inference.py;       DEF_STEPS=28; DEF_CFG=5.0; MP=models/Boogu-Image-0.1-Edit  ;;
  *) echo "unknown model '$MODEL' (use base|turbo|edit)"; exit 1 ;;
esac
STEPS="${STEPS:-$DEF_STEPS}"; CFG="${CFG:-$DEF_CFG}"
OUT="outputs/${MODEL}/out_$(date +%Y%m%d_%H%M%S).png"; mkdir -p "$(dirname "$OUT")"

ARGS=(--pretrained_pipeline_name_or_path "$MP" --instruction "$PROMPT"
      --num_inference_steps "$STEPS" --height "${H:-1024}" --width "${W:-1024}"
      --text_guidance_scale "$CFG" --output_image_path "$OUT" --device cuda:0)
[ "${OFFLOAD:-1}" = 1 ] && ARGS+=(--enable_model_cpu_offload_flag True)
if [ "$MODEL" = edit ]; then
  [ -n "${IMAGE:-}" ] || { echo "edit needs IMAGE=<path>"; exit 1; }
  ARGS+=(--input_image_paths "$IMAGE" --image_guidance_scale "${ICFG:-2.5}")
fi

echo "[run_boogu] $MODEL steps=$STEPS cfg=$CFG offload=${OFFLOAD:-1} -> $OUT"
python "$SCRIPT" "${ARGS[@]}"
echo "[run_boogu] DONE -> $OUT"
