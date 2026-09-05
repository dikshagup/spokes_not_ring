#!/usr/bin/env bash
# FIGURE 3 -- figures/arc_occupancy_main_ab.{pdf,png}
#              (+ figures/arc_occupancy_appendix.{pdf,png}, off the same run)
#
#     bash experiments/repro_fig3_arc_occupancy.sh           # corpus + captures (GPU) + plot
#     bash experiments/repro_fig3_arc_occupancy.sh --plot    # plot only
#     bash experiments/repro_fig3_arc_occupancy.sh --verify  # plot, then checksum
#
# Weekday mentions in natural text form a ring (A), and that ring is seven clumps rather than
# a populated loop (B).
#
# BOTH PLATES COME OFF ONE REFIT of the geometry from the raw 4096-d activations, so the
# appendix md5 doubles as an isolation check: change a main panel and it must still pass.
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

# Both of these are large and neither belongs in the working tree: the corpus is ~2.3 GB
# and the raw 4096-d captures larger still (raw 978 MB, swap 652 MB). They default under
# the user cache so a clone works without root, and are the two paths worth redirecting at
# a volume with room -- CORPUS=/mnt/big/corpus_v2 RAWDIR=/mnt/big/corpus_v2_raw bash ...
_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/weekday-manifold"
CORPUS="${CORPUS:-$_CACHE/corpus_v2}"
RAWDIR="${RAWDIR:-$_CACHE/corpus_v2_raw}"
LIST="$CORPUS/capture_list_n256.npz"
PROJ="$CORPUS/projections_n16.npz"
RAW="$RAWDIR/raw_L28_n16.npz"
SWAP="$RAWDIR/swap_L28_n16.npz"

MODE="${1:-}"
if [[ "$MODE" != "--plot" && "$MODE" != "--verify" ]]; then
  # ---- 1. stream FineWeb and pick the windows -----------------------------------------
  # CPU only, no model weights, no activations. The four window classes are defined in
  # select_corpus_windows.py and are not decided here. ~35 min; the published selection
  # scanned 116,587 documents before its targets were met.
  #
  # HF_HUB_OFFLINE IS FORCED OFF FOR THIS STAGE, whatever the caller set. This stage
  # STREAMS FineWeb from the Hub, so under HF_HUB_OFFLINE=1 -- the setting recommended once
  # the weights are cached -- it cannot reach the dataset at all, and the script dies here
  # on its first line of real work. It is pinned per-stage rather than run-wide because
  # this is the one stage that is supposed to hit the network: leaving the run online
  # would let a missing local checkpoint turn into a silent multi-gigabyte download at
  # stage 4 instead of an error.
  echo "[fig3] 1/6: select corpus windows  ->  $CORPUS"
  HF_HUB_OFFLINE=0 $PY experiments/select_corpus_windows.py \
      --max-docs 120000 --n-max 256 --out "$CORPUS"

  # ---- 2. near-duplicate removal, per dump ---------------------------------------------
  echo "[fig3] 2/6: dedup"          # ~3 min, CPU
  $PY experiments/dedup_corpus_windows.py --in "$CORPUS" --n-max 256

  # ---- 3. the exact capture list --------------------------------------------------------
  # Tokenizer only. Token ids come from SELECTION, never from re-tokenising the stored text:
  # a window is defined as exactly n_max tokens ending at a specific capture token, and every
  # class decision was computed on those ids. ~1 min.
  echo "[fig3] 3/6: build capture list  ->  $LIST"
  $PY experiments/build_capture_list.py --in "$CORPUS" --n-max 256

  # ---- 4. the 6-D projections ------------------------------------------------------------
  # Needed here not for its own numbers but for its FIT/SCORE SPLIT, which steps 5 and 6 read
  # rather than recompute -- so the plane is fitted on exactly the windows the published
  # figure fits on, and a reseeding or a --fit-frac change cannot silently desynchronise the
  # two passes. ~28 min on GPU.
  # --no-cache because this is a reproduction. The prompt-library capture inside this
  # stage is keyed by (model, flags, sites, prompts) and cached under data/activations
  # INSIDE THE REPO (~642 MB); left enabled, a second run reads that file and reports a
  # successful reproduction of a stage it never actually recomputed.
  echo "[fig3] 4/6: projections  ->  $PROJ"
  $PY experiments/capture_corpus_windows.py --in "$CORPUS" --n 16 --no-cache

  # ---- 5. the raw 4096-d activations at L28 -----------------------------------------------
  # THE GPU STEP. capture_corpus_windows.py projects inside the hook because 118k windows x 33
  # layers is ~64 GB; this keeps ONE layer, so the same windows cost ~0.96 GB in fp16. That
  # buys the one thing the projections cannot answer: every cached quantity lives inside a
  # 6-D subspace that was FITTED ON THE DAY CENTROIDS, so any PCA of the cache is a PCA of
  # something the labels already built.
  echo "[fig3] 5/6: raw L28 capture  ->  $RAW"
  mkdir -p "$RAWDIR"
  $PY experiments/capture_raw_layer.py --in "$CORPUS" --n-max 256 --n 16 --layer 28 --out "$RAW"

  # ---- 6. the weekday-swap control ---------------------------------------------------------
  # Within-context contrast: hold the context fixed and vary ONLY the weekday. All weekday-class
  # windows end in one of 14 single-token weekday forms, so a swap is a one-for-one substitution
  # of the last token -- no length change, and attention is causal, so tokens 1..15 are
  # bit-identical across all seven variants of a family. swap_capture_l28.py is deliberately a
  # near-clone of the pass above (same hook, same site, same fp16 store, same batch size):
  # swap_analyze asserts that each family's SELF-swap reproduces the corresponding row of the
  # raw capture, and that assertion is only meaningful if the two passes differ in nothing but
  # the input list. ~5 min.
  echo "[fig3] 6/6: weekday-swap set + capture  ->  $SWAP"
  $PY experiments/swap_build_set.py --in "$CORPUS" --n-max 256 --n 16 --out "$RAWDIR/swap_set_n16.npz"
  $PY experiments/swap_capture_l28.py --set "$RAWDIR/swap_set_n16.npz" --layer 28 --out "$SWAP"
fi

# None of the three inputs is tracked -- they are captures, and *.npz is gitignored by policy.
for f in "$RAW" "$SWAP" "$LIST"; do
  [[ -f "$f" ]] || { echo "[fig3] missing $f -- run without --plot/--verify to capture it" >&2; exit 1; }
done

# The tokenizer load is for the appendix only -- it decodes the four prompt templates in its
# panel B -- and it is offline: the Llama-3.1-8B files are already in HF_HOME.
# THE MAIN-TEXT CUT IS THE 2x2. --panels abcd is the plotting script's default and writes
# arc_occupancy_main_abcd: the ring plane and the ring itself on the top row, and the two
# weekday-swap panels -- previously the appendix plate's B and C -- on the bottom. The
# subspace-energy violin is NOT in it, and the appendix plate is no longer rendered by
# default -- --verify still renders it, because its md5 below is the check on this stage.
# `--panels ab` is the previously published two-panel row, and `--panels abc` the
# three-panel row; the latter is what the pinned md5 below covers, and it is kept because
# that hash is the isolation check on this stage.
echo "[fig3] arc_occupancy_main_abcd"
# The four --pick-context strings are the panel C exemplars OF THE PUBLISHED FIGURE,
# pinned by their text. They are not reachable through the FAMILIES regexes: those take
# at most one window per family, and two of these four ("...is open 6 days a week," and
# "...The winery is open") are both `opening hours`. Pinning also makes the panel stable
# against a FineWeb re-upload -- load_dataset pins no revision, so the stream that
# selected these windows is not guaranteed to be the one a later run sees. Any string
# that stops matching is REPORTED by the plotting script rather than silently replaced.
$PY experiments/figure_arc_occupancy_split.py \
    --raw "$RAW" --swap "$SWAP" --list "$LIST" \
    --pick-context "published the series in a" \
    --pick-context "is open 6 days a week" \
    --pick-context "the winery is open" \
    --pick-context "in a statement on" \
    --panels abcd --outdir figures --also-png

if [[ "$MODE" == "--verify" ]]; then
  echo "[fig3] verifying"
  # PNG only. The PDFs carry a /CreationDate and so differ run to run; the PNGs are
  # byte-identical across runs -- no timestamp chunk, and every subsample is drawn from a
  # --seed 0 generator -- so a mismatch below is a real difference in the figure.
  #
  # THE THREE-PANEL ROW IS RE-RENDERED HERE PURELY TO CHECK IT. Its md5 is the one that was
  # verified from scratch on this tree, and it shares every panel with the two-panel cut,
  # so it is still the strongest available check on this stage -- but it is not the
  # published plate, and figures/arc_occupancy_main.png is not a figure of the paper.
  $PY experiments/figure_arc_occupancy_split.py \
      --raw "$RAW" --swap "$SWAP" --list "$LIST" \
      --panels abc --appendix --outdir figures --also-png
  md5sum -c --strict - <<'EOF'
97f24974980c57e8331fec214d69ab66  figures/arc_occupancy_main.png
bdb62cb3cb1a60c6f1a79a2cd95cce9e  figures/arc_occupancy_appendix.png
EOF
  # arc_occupancy_main_abcd HAS NO PIN. It is a new layout, and two of its four panels
  # are float16 captures rendered through a newer matplotlib, so a hash would be a check
  # on this tree alone; pin one here once the layout has settled.
  echo "[fig3] note: arc_occupancy_main_abcd carries no pinned md5 -- see README"
fi

echo "[fig3] done"
