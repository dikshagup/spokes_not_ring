#!/usr/bin/env bash
# FIGURE 1 -- figures/fig_combined_llama.{pdf,png}
#
#     bash experiments/repro_fig1_combined_llama.sh           # capture (GPU) + plot
#     bash experiments/repro_fig1_combined_llama.sh --plot    # plot only, reuse the captures
#
# Two independent routes to one claim: a finite-difference walk round the ring (A, B) and an
# autodiff Jacobian over the day disc (C).
#
# BOTH HALVES ARE float16 AND THAT IS LOAD-BEARING -- in bfloat16 the rounding floor is 69%
# of the measured step and fills in the troughs the figure is about. Running either capture
# against a bfloat16 config gives a different figure without erroring. No md5 is pinned here;
# see README.md for why a float16 plate cannot carry a portable one.
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

CASCADE="figures/ladder_llama_best.npz"
FIELD="figures/llama_polar70_stop_fp16.npz"

if [[ "${1:-}" != "--plot" ]]; then
  # ---- panels A and B: the ring walk ------------------------------------------------
  # The mention family ("It was Monday.", 20 templates x 7 days) read at the FULL STOP,
  # position 8. Mention prompts ask nothing of the day, so the OUTPUT barely moves however
  # the day is steered -- that is what makes them the right family for a readout measurement
  # and why the readout is the full stop rather than an answer slot.
  #
  # Everything not passed here is a default that IS the published setting: --layers
  # 2,6,10,14,18,22,26,28,30,31 (the 10 steer layers), --n-u 141 (which puts alpha = 0 exactly
  # on the grid, so the identity check is exact), --param arclength-global (uniform in
  # DISTANCE, so the walk does not measure the input slowing at the knots) and --step-k 3,9.
  # ~2-3 h on one 24 GB card.
  echo "[fig1] capture 1/2: the ring walk  ->  $CASCADE"
  # --readout-group input --sites weekday are what the mention family resolves to on its
  # own; they are written out because this is the published command and a reader should
  # not have to know the defaulting rule to see what was measured.
  $PY experiments/alpha_ladder_sites.py \
      --config configs/llama31_8b_fp16.json --formulation me \
      --readout-group input --sites weekday \
      --readout-pos 8 --out "$CASCADE"

  # ---- panel C: the polar Jacobian field --------------------------------------------
  # Same model, same family, same readout position, steered at L2 and read at L28 -- the
  # four things figure_combined.py asserts. 56 angles x 17 radii x 70 background prompts,
  # ||J||_F by Hutchinson with 6 probes each. ~1-2 h on the same card.
  echo "[fig1] capture 2/2: the polar Jacobian field  ->  $FIELD"
  $PY experiments/jacobian_polar_field.py \
      --config configs/llama31_8b_fp16.json --formulation me \
      --patch-layer 2 --readout-layer 28 --readout-pos 8 --n-prompts 70 \
      --out "$FIELD"
fi

for f in "$CASCADE" "$FIELD"; do
  [[ -f "$f" ]] || { echo "[fig1] missing $f -- run without --plot to capture it" >&2; exit 1; }
done

# --layout tight is the default and is the published cut: three panels in one row. The
# cumulative panel is the speed panel integrated, and discs E-G resolve the norm by
# direction, so the tight cut keeps one panel from each of the three things this measures
# (output, residual, derivative) and drops the ones that restate them. --layout full draws
# all seven.
echo "[fig1] plate"
$PY experiments/figure_combined.py \
    --cascade "$CASCADE" --field "$FIELD" \
    --out figures/fig_combined_llama.pdf --also-png

echo "[fig1] done"
