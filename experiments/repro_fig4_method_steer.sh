#!/usr/bin/env bash
# FIGURE 4 -- figures/fig_method_steer.{pdf,png}
#
#     bash experiments/repro_fig4_method_steer.sh           # capture (GPU) + plot
#     bash experiments/repro_fig4_method_steer.sh --plot    # plot only, reuse the capture
#
# The methods schematic: the (u, r) chart drawn on the loop's real projected shape, and the
# intervention drawn as the forward pass it is. Nothing in it is a measurement.
#
# ITS FIELD IS FIGURE 1'S FIELD. figures/llama_polar70_stop_fp16.npz is the same file
# repro_fig1_combined_llama.sh captures for panel C, so whichever script runs first, the
# other's capture is a no-op. Only the centroids, mu and the two token positions are read
# from it -- the measured fields are not touched -- but taking them from the same file the
# measurement came from is what keeps the schematic honest about what was measured.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
export PYTHONPATH="${PYTHONPATH:-src:experiments}"
# Weights are left wherever huggingface_hub keeps them (~/.cache/huggingface) unless you
# export HF_HOME; this work was done with them on a separate volume. Llama-3.1-8B is a
# gated repo either way, so `huggingface-cli login` or an already-populated cache is required.
if [[ -n "${HF_HOME:-}" ]]; then export HF_HOME; fi
# Online by default, so a fresh clone can fetch what it is missing. Export HF_HUB_OFFLINE=1
# once the weights are cached: a silent multi-GB re-download is the failure worth preventing,
# and it is the setting every committed figure was actually produced under.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

FIELD="figures/llama_polar70_stop_fp16.npz"
CONTEXT="figures/readout_context.json"

if [[ "${1:-}" != "--plot" ]]; then
  if [[ -f "$FIELD" ]]; then
    echo "[fig4] $FIELD exists (figure 1's capture), skipping"
  else
    echo "[fig4] capture: the polar Jacobian field  ->  $FIELD"   # ~1-2 h on one 24 GB card
    $PY experiments/jacobian_polar_field.py \
        --config configs/llama31_8b_fp16.json --formulation me \
        --patch-layer 2 --readout-layer 28 --readout-pos 8 --n-prompts 70 \
        --out "$FIELD"
  fi
fi

[[ -f "$FIELD" ]] || { echo "[fig4] missing $FIELD -- run without --plot to capture it" >&2; exit 1; }

# The token strip is tracked, so this only needs rebuilding if the prompt templates change:
#   $PY experiments/readout_context.py
[[ -f "$CONTEXT" ]] || { echo "[fig4] missing $CONTEXT -- run experiments/readout_context.py" >&2; exit 1; }

echo "[fig4] plate"
$PY experiments/figure_method_steer.py \
    --npz "$FIELD" --context "$CONTEXT" \
    --out figures/fig_method_steer.pdf --also-png

echo "[fig4] done"
