#!/usr/bin/env bash
# FIGURE 6 -- the two plates the appendix figure is composed from:
#     figures/fig_combined_arith_ad.{pdf,png}   panels A-D, the sweep
#     figures/fig_combined_arith_eh.{pdf,png}   panels E-H, the Jacobian discs
#
#     bash experiments/repro_fig6_combined_arith.sh           # captures (GPU) + plot
#     bash experiments/repro_fig6_combined_arith.sh --plot    # plot only
#
# The mention-family measurements repeated on the INTERROGATIVE family: where the weekday
# answer lives and when (A-D), and the Jacobian over the day disc (E-H).
#
# A DIFFERENT PROMPT FAMILY FROM FIGURES 1, 4 AND 5, and getting it wrong does not error --
# it silently measures the wrong site. The interrogative weekday sits at position 9 and the
# readout at 12, against 2 and 8 for the mention family, so --readout-pos is left at its
# default of -1 here where figure 1 pins it to 8. The formulation comes from the config.
#
# TWO PLATES, COMPOSED IN THE MANUSCRIPT: A-D are finite differences over a sweep and E-H is
# autodiff over a grid. They share no arithmetic, and one output file would hide that.
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

CFG="configs/llama31_8b_fp16_interrogative.json"
RINGS="figures/rings.npz"
SWEEP="figures/alpha_ladder_sites.npz"
# Panels E-H. Named for the family, not the figure: it is the interrogative counterpart of
# figure 1's llama_polar70_stop_fp16.npz and nothing else reads it.
FIELD="figures/llama_polar70_interrogative_fp16.npz"

if [[ "${1:-}" != "--plot" ]]; then
  # ---- panel A: the ring geometry itself ----------------------------------------------
  # --dump-rings writes the day CENTROIDS at every (layer, site) and exits before the sweep.
  # Panel A draws the real rings, and the sweep only ever stored the splines fitted to them,
  # so without this the figure would have to rebuild the setup -- and a second copy of it
  # could drift from the one that was actually measured. Minutes.
  echo "[fig6] capture 1/2: ring centroids  ->  $RINGS"
  $PY experiments/alpha_ladder_sites.py --config "$CFG" \
      --sites weekday,answer --dump-rings "$RINGS"

  # ---- panels B, C, D: the sweep --------------------------------------------------------
  # Ten patch layers x 49 prompts (7 days x 7 offsets) x 141 ring samples, both sites, two
  # readings per forward pass -- the residual's position on the readout ring, and the
  # Hellinger distance of the restricted next-token distribution from the same prompt's
  # unsteered one. ~3-4 h on one 24 GB card.
  echo "[fig6] capture 2/2: the sweep  ->  $SWEEP"
  $PY experiments/alpha_ladder_sites.py --config "$CFG" \
      --sites weekday,answer --out "$SWEEP"
fi

for f in "$RINGS" "$SWEEP"; do
  [[ -f "$f" ]] || { echo "[fig6] missing $f -- run without --plot to capture it" >&2; exit 1; }
done

# All defaults. --max-offset 1 restricts panel B to single-day offsets, which keeps the
# model's own arithmetic ceiling out of a measurement about steering.
echo "[fig6] plate 1/2: panels A-D"
$PY experiments/figure_combined_arith.py --rings "$RINGS" --npz "$SWEEP" \
    --out figures/fig_combined_arith_ad.pdf --also-png

# ---- panels E-H: the Jacobian on the interrogative family ------------------------------
# The same jacobian_polar_field.py run figures 1, 4 and 5 use, at the same grid, on the
# interrogative config. --readout-pos is NOT passed: this family reads at the final ":",
# which is the script's own default of -1, where the mention family pins 8.
if [[ -f "$FIELD" ]]; then
  echo "[fig6] $FIELD exists, skipping capture"
elif [[ "${1:-}" != "--plot" ]]; then
  echo "[fig6] capture 3/3: the interrogative polar field  ->  $FIELD"   # ~1-2 h GPU
  $PY experiments/jacobian_polar_field.py --config "$CFG" \
      --patch-layer 2 --readout-layer 28 --n-prompts 70 --out "$FIELD"
fi

if [[ -f "$FIELD" ]]; then
  echo "[fig6] plate 2/2: panels E-H"
  $PY experiments/figure_jac_discs.py --npz "$FIELD" \
      --out figures/fig_combined_arith_eh.pdf --n-layers 32 \
      --context figures/readout_context.json --also-png
else
  echo "[fig6] no $FIELD -- panels E-H not built; run without --plot to capture it" >&2
fi

echo "[fig6] done -- compose fig_combined_arith from _ad and _eh in the manuscript"
