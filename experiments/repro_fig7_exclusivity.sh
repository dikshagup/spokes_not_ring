#!/usr/bin/env bash
# FIGURE 7 -- figures/exclusivity_figure.{pdf,png}
#
#     bash experiments/repro_fig7_exclusivity.sh           # captures (GPU) + plot
#     bash experiments/repro_fig7_exclusivity.sh --plot    # plot only
#     bash experiments/repro_fig7_exclusivity.sh --verify  # plot, then checksum
#
# The weekday subspace is privileged but not exclusive. No captures of its own: this script
# delegates its whole capture stage to repro_fig3_arc_occupancy.sh, so CORPUS= and RAWDIR=
# must match between the two.
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

# The same two paths repro_fig3_arc_occupancy.sh uses, defaulted identically so that
# delegating to it below lands on the captures this script then reads. Redirect both at a
# volume with room if the user cache has none.
_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/weekday-manifold"
CORPUS="${CORPUS:-$_CACHE/corpus_v2}"
RAWDIR="${RAWDIR:-$_CACHE/corpus_v2_raw}"
LIST="$CORPUS/capture_list_n256.npz"
RAW="$RAWDIR/raw_L28_n16.npz"
SWAP="$RAWDIR/swap_L28_n16.npz"

MODE="${1:-}"
if [[ "$MODE" != "--plot" && "$MODE" != "--verify" ]]; then
  echo "[fig7] captures are figure 3's -- delegating to repro_fig3_arc_occupancy.sh"
  CORPUS="$CORPUS" RAWDIR="$RAWDIR" PY="$PY" bash experiments/repro_fig3_arc_occupancy.sh
fi

for f in "$RAW" "$SWAP" "$LIST"; do
  [[ -f "$f" ]] || { echo "[fig7] missing $f -- run without --plot/--verify to capture it" >&2; exit 1; }
done

# --plane centroid is the default and is the published choice: the 2-D view is the plane of
# the seven day centroids. --plane rawpca fits the plane by unsupervised PCA on the positives
# instead, which is a different question and a different figure.
#
# SIZED FOR PRINT. The plotting script's figsize is the actual size on the page (7.5 in wide,
# inside A4's margins) so matplotlib's point sizes ARE the printed point sizes. Authoring at
# 12 in and letting the page scale it down would turn 8pt type into 5pt -- which is why there
# is no --scale here to match figure 2's.
echo "[fig7] 2x2 plate"
$PY experiments/figure_exclusivity.py \
    --raw "$RAW" --swap "$SWAP" --list "$LIST" \
    --out figures/exclusivity_figure.pdf --also-png

if [[ "$MODE" == "--verify" ]]; then
  echo "[fig7] verifying"
  # PNG only: the PDF carries a /CreationDate and differs run to run. The panel D subsample is
  # drawn from a --seed 0 generator, so the PNG is byte-identical across runs.
  md5sum -c --strict - <<'EOF'
4d21388a3a5eff947b56dec1f4f70441  figures/exclusivity_figure.png
EOF
fi

echo "[fig7] done"
