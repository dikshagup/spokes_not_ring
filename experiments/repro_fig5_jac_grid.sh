#!/usr/bin/env bash
# FIGURE 5 -- the three per-model rows the appendix grid is composed from:
#     figures/fig_jac_llama.{pdf,png}    + _caption.md
#     figures/fig_jac_mistral.{pdf,png}  + _caption.md
#     figures/fig_jac_gpt2xl.{pdf,png}   + _caption.md
#
#     bash experiments/repro_fig5_jac_grid.sh              # captures (GPU) + plot, all three
#     bash experiments/repro_fig5_jac_grid.sh --plot       # plot only
#     bash experiments/repro_fig5_jac_grid.sh --only llama # one model (llama|mistral|gpt2xl)
#
# THREE PLATES, COMPOSED IN THE MANUSCRIPT into figures/fig_jac_grid. Each row is rendered
# separately because the model is not a flag: it is which field npz is passed, and the field
# records the model, the depths and the positions the caption is written from.
#
# LLAMA'S FIELD IS FIGURE 1'S FIELD, so whichever script runs first the other's Llama capture
# is a no-op. All three steer at L2 and read at 87.5% of the stack, each at the best precision
# it can run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
export PYTHONPATH="${PYTHONPATH:-src:experiments}"
# Online by default, so a fresh clone can fetch what it is missing. Export HF_HUB_OFFLINE=1
# once the weights are cached: a silent multi-GB re-download is the failure worth preventing.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# One cache variable per model. They default to the same place -- huggingface_hub's own
# cache, or HF_HOME if you export one -- and are separate only because the three sets of
# weights need not live together: on the pod this was developed on Llama sat on the
# container disk and the other two on a network volume, which was where the room was.
# HF_GPT2XL is also where GPT-2 XL lands if it has to be fetched, so give it 7 GB.
_HF_DEFAULT="${HF_HOME:-$HOME/.cache/huggingface}"
HF_LLAMA="${HF_LLAMA:-$_HF_DEFAULT}"
HF_MISTRAL="${HF_MISTRAL:-$_HF_DEFAULT}"
HF_GPT2XL="${HF_GPT2XL:-$_HF_DEFAULT}"

MODE="${1:-}"; ONLY="${2:-}"
[[ "$MODE" == "--only" ]] && { ONLY="${2:?--only needs llama|mistral|gpt2xl}"; MODE=""; }

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# ---- the readout context strip -------------------------------------------------------
# figure_jac_discs.py checks that the position it is about to call the weekday site really
# holds a weekday, against the token list in this file, and refuses to draw if it does not.
# The file is tracked, so this only needs re-running if the prompt templates change.
#   $PY experiments/readout_context.py
CONTEXT="figures/readout_context.json"

capture() {  # name config readout_layer out extra...
  local name="$1" cfg="$2" ro="$3" out="$4"; shift 4
  if [[ -f "$out" ]]; then echo "[fig5] $name: $out exists, skipping capture"; return; fi
  echo "[fig5] $name: capture -> $out"
  $PY experiments/jacobian_polar_field.py \
      --config "$cfg" --formulation me \
      --patch-layer 2 --readout-layer "$ro" --readout-pos 8 --n-prompts 70 \
      --out "$out" "$@"
}

plot() {     # npz out n_layers
  echo "[fig5] plate -> $2"
  $PY experiments/figure_jac_discs.py --npz "$1" --out "$2" --n-layers "$3" \
      --context "$CONTEXT" --also-png
}

LLAMA_NPZ="figures/llama_polar70_stop_fp16.npz"
MISTRAL_NPZ="figures/mistral_polar70_stop.npz"
GPT2XL_NPZ="figures/gpt2xl_polar70_L2.npz"

if [[ "$MODE" != "--plot" ]]; then
  # Llama-3.1-8B, float16. All 20 mention templates come to 12 tokens under its BPE with the
  # full stop at 8, which is what makes a scalar --readout-pos meaningful, so no length filter.
  want llama   && HF_HOME="$HF_LLAMA" capture "Llama-3.1-8B" \
                     configs/llama31_8b_fp16.json 28 "$LLAMA_NPZ"

  # Mistral-7B-v0.1, float16 -- the model Engels et al. 2025 made the circular-features claim
  # in. --require-seq-len 12 is NOT optional here: SentencePiece splits the same 20 templates
  # into 12/13/14/15 tokens, so a fixed index would read the full stop in some prompts and a
  # content word in others. Verified on this pod: the filter keeps 105 of 140 prompts, 15 per
  # day, and drops whole templates so the days stay balanced.
  want mistral && HF_HOME="$HF_MISTRAL" capture "Mistral-7B-v0.1" \
                     configs/mistral7b_mention.json 28 "$MISTRAL_NPZ" --require-seq-len 12

  # GPT-2 XL, float32, 48 blocks -- read at L42 to match 87.5% of the stack. No length filter:
  # its BPE is expected to lay the templates out as Llama's does, and if it does not the run
  # dies on a torch.cat shape mismatch rather than silently reading the wrong site. If that
  # happens, run the length check and pass --require-seq-len with what it reports.
  #
  # THIS ONE IS ALLOWED TO DOWNLOAD. Unlike Llama and Mistral, GPT-2 XL is not assumed to be
  # sitting in a local cache -- it is 6.4 GB and it is small enough to fetch, so a machine
  # that has never run this repo can still build the third panel. The offline pin stays on
  # for the two large models, where a silent 16 GB re-download is the failure mode worth
  # preventing; here it is lifted only if the weights are genuinely absent, so a populated
  # cache still resolves locally and hits the network not at all.
  if want gpt2xl; then
    if [[ -d "$HF_GPT2XL/hub/models--gpt2-xl" ]]; then
      HF_HOME="$HF_GPT2XL" capture "GPT-2 XL" configs/gpt2xl_mention.json 42 "$GPT2XL_NPZ"
    else
      echo "[fig5] gpt2-xl not in $HF_GPT2XL -- fetching it (~6.4 GB; set HF_GPT2XL= to"
      echo "       put it somewhere with room, and check that somewhere has 7 GB free)"
      HF_HOME="$HF_GPT2XL" HF_HUB_OFFLINE=0 capture "GPT-2 XL" \
          configs/gpt2xl_mention.json 42 "$GPT2XL_NPZ"
    fi
  fi
fi

want llama   && { [[ -f "$LLAMA_NPZ"   ]] || { echo "[fig5] missing $LLAMA_NPZ" >&2; exit 1; }; \
                  plot "$LLAMA_NPZ"   figures/fig_jac_llama.pdf   32; }
want mistral && { [[ -f "$MISTRAL_NPZ" ]] || { echo "[fig5] missing $MISTRAL_NPZ" >&2; exit 1; }; \
                  plot "$MISTRAL_NPZ" figures/fig_jac_mistral.pdf 32; }
want gpt2xl  && { [[ -f "$GPT2XL_NPZ"  ]] || { echo "[fig5] missing $GPT2XL_NPZ" >&2; exit 1; }; \
                  plot "$GPT2XL_NPZ"  figures/fig_jac_gpt2xl.pdf  48; }

echo "[fig5] done"
