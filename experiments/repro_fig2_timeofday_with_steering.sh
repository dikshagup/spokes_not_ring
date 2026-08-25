#!/usr/bin/env bash
# FIGURE 2 -- figures/timeofday_with_steering.{pdf,png}
#
#     bash experiments/repro_fig2_timeofday_with_steering.sh           # capture (GPU) + plot
#     bash experiments/repro_fig2_timeofday_with_steering.sh --plot    # plot only
#     bash experiments/repro_fig2_timeofday_with_steering.sh --verify  # plot, then checksum
#
# Time is carried orthogonally to the weekday ring (A), modifiers cause only small in-plane
# shifts (B), and rotating a state along the ring does not change the time the model names (C).
#
# THE EXPORT ARGUMENTS BELOW ARE NOT THE PLOTTING SCRIPT'S DEFAULTS, and running it without
# them produces a different figure rather than an error. They are the authority on what this
# figure is; the md5 at the bottom covers the plot stage only. See README.md.
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

NPZ="experiments/results/timeofday_L28.npz"
STEERDIR="experiments/results/steer_timeofday/meta-llama_Llama-3.1-8B"
ROWS="$STEERDIR/clamp_gated/matched_pairs/rows.csv"

MODE="${1:-}"
if [[ "$MODE" != "--plot" && "$MODE" != "--verify" ]]; then
  # ---- panels A and B: the 364-prompt carrier-2 capture ------------------------------
  # Every prompt is "It was {t} on {day}." read at the WEEKDAY token (position 6). The time
  # phrase PRECEDES the weekday, so under causal attention the weekday token has already read
  # the hour; the reverse ordering would make time-of-day structure impossible at this site.
  # Three modifier families in the {t} slot: 23 clock hours, coarse word times, and
  # weather/mood placebos with no time content as the null band. Re-running is cheap and
  # idempotent -- the script reuses the npz unless the modifier set has changed. Its --out is
  # a first-draft plate that is NOT one of the three figures, so it goes to a scratch path.
  echo "[fig2] capture 1/3: carrier-2 daytime  ->  $NPZ"
  DRAFT="$(mktemp -d)"
  $PY experiments/capture_timeofday.py --layer 28 --device cuda \
      --npz "$NPZ" --out "$DRAFT/timeofday_draft.pdf"
  rm -rf "$DRAFT"

  # ---- panel C, part 1: the carrier screen that fixes the 12 carriers ----------------
  # Panel C's cells are (carrier x day), and WHICH twelve carriers is decided here, by
  # screening for ones whose time distribution is not already concentrated. Only meta.json's
  # `carriers_kept` is read downstream, but the screen is stage E0 of this run and the run is
  # one pass, so it is run whole. ~1 h.
  echo "[fig2] capture 2/3: the steering run  ->  $STEERDIR/meta.json"
  $PY experiments/steer_timeofday.py

  # ---- panel C, part 2: the edit-size-matched pairs ----------------------------------
  # 84 cells, each contributing one in-plane edit and one off-plane edit solved to the SAME
  # ||Delta|| to within 0.5%. ||Delta|| is `displacement` -- the norm the hook moves the state
  # BY -- not `maintenance`, the drift-correction cost a closed-loop clamp pays; comparing one
  # family's maintenance against the other's displacement is the error that once made a
  # matched control look impossible. The dose is secant-iterated rather than inverted, because
  # displacement(alpha) is a bf16 staircase, and each probe re-runs the 60 ambiguous prefixes
  # so the probe's batch shape matches the scoring's -- probing at batch 1 solved the dose to
  # 0.1% and still missed the recorded value by up to 5%. ~30 min on one 24 GB card.
  echo "[fig2] capture 3/3: the 84-cell matched pairs  ->  $ROWS"
  $PY experiments/steer_clamp_matched_pairs.py
fi

for f in "$NPZ" "$ROWS"; do
  [[ -f "$f" ]] || { echo "[fig2] missing $f -- run without --plot/--verify to capture it" >&2; exit 1; }
done

# Panels A and B are the raw-pca talk plate. Panel C is the matched-pairs run read through the
# extended time readout on the weekday-token band only: offsets 1 and 2 have no valid ring at
# 24 of 54 sites and are not steered.
#
# --a-xlabel and panel C's shorter y label are part of the fit, not tidying. The full "in-plane
# angular shift from the day's own centroid" is wider than its own panel and met panel C's
# ticks; panel C's y label is ROTATED, so its length is measured against the axes HEIGHT and
# the long form ran into panel B's title. Both say the same quantity.
echo "[fig2] three-panel plate"
$PY experiments/figure_carrier2_daytime_talk.py \
    --npz "$NPZ" \
    --theme slide-white --scale 2.15 --text-boost 1.15 --figsize 25.5x9.5 --opaque --also-png \
    --basis raw-pca --pca-rows time \
    --steer-pairs "$ROWS" \
    --title "" --no-subtitle --cap-labels --no-gridlines \
    --fit-titles --align-titles --a-tight-rows --a-row-size 0.88 --cube-text 1.10 --check-overlap \
    --wspace 0.60 --bottom 0.15 --top 0.855 --pairs-width 1.95 --cb-shrink 0.78 --pad-inches 0.15 \
    --a-xlabel $'In-plane rotation from centroid\n(degrees)' \
    --swap-ab --centre-titles --b-balance --b-panel-shift 0.032 \
    --b-label-pad 4.3 --b-axis-labels pc --b-rotate-labels --b-labelpad -7 \
    --b-label-pad-day 5.8 4.3 3.5 5.8 5.8 5.8 4.1 \
    --title-a "Time is represented orthogonally" \
    --title-b "Modifiers cause small shifts" \
    --title-c "Shifts on ring don't change time" \
    --out figures/timeofday_with_steering.pdf

if [[ "$MODE" == "--verify" ]]; then
  # PNG only. The PDF carries a /CreationDate and so differs run to run; the PNG has no
  # timestamp chunk and is byte-identical across runs, so a mismatch here is a real
  # difference in the figure.
  echo "[fig2] verifying"
  md5sum -c --strict - <<'SUMS'
35b0063a7708503902d0ac721df43ea0  figures/timeofday_with_steering.png
SUMS

  # --steer-pairs must APPEND panel C and touch nothing else. Re-rendering the two-panel
  # plate from the same script must still produce the md5 pinned before panel C existed. If
  # this fails, panel C is not an isolated addition -- fix the code, do not re-pin.
  echo "[fig2] verifying that --steer-pairs changed nothing else"
  TMP="$(mktemp -d)"
  $PY experiments/figure_carrier2_daytime_talk.py \
      --npz "$NPZ" \
      --theme slide-white --scale 1.25 --opaque --also-png \
      --basis raw-pca --pca-rows time \
      --out "$TMP/two_panel.pdf" >/dev/null
  echo "730cc68ecdd97eede77bfa75bc82b5f2  $TMP/two_panel.png" | md5sum -c --strict -
  rm -rf "$TMP"
fi

echo "[fig2] done"
