# weekday-manifold

Code for **"Spokes, not a Ring: Testing Continuity of Representation Manifolds."**

Large language models place the seven weekdays on a closed loop in residual-stream space.
This repository asks whether that loop is a *continuum* — a coordinate whose intermediate
points mean something — or seven discrete states arranged in a circle. It measures two
things: how sensitively the readout responds as an activation is steered around the loop,
and what, if anything, occupies the arcs between the days.

Three models (Llama-3.1-8B, Mistral-7B-v0.1, GPT-2 XL), loaded through TransformerLens.

## What is here

**Only what produces a figure.** Every module in `src/` and every script in `experiments/`
is on the path from raw model to one of the seven plates below — that is checked, not
claimed. Nothing generated is tracked: no activations, no sweeps, and no rendered figures.
Each plate is rebuilt by one command.

| # | Plate | Command | Cost |
|---|-------|---------|------|
| 1 | `fig_combined_llama` | `bash experiments/repro_fig1_combined_llama.sh` | ~4 h GPU |
| 2 | `timeofday_with_steering` | `bash experiments/repro_fig2_timeofday_with_steering.sh` | ~2 h GPU |
| 3 | `arc_occupancy_main_ab` | `bash experiments/repro_fig3_arc_occupancy.sh` | ~40 min CPU + ~55 min GPU |
| 4 | `fig_method_steer` | `bash experiments/repro_fig4_method_steer.sh` | free if fig 1 has run |
| 5 | `fig_jac_{llama,mistral,gpt2xl}` | `bash experiments/repro_fig5_jac_grid.sh` | ~2-4 h GPU |
| 6 | `fig_combined_arith_{ad,eh}` | `bash experiments/repro_fig6_combined_arith.sh` | ~5 h GPU |
| 7 | `exclusivity_figure` | `bash experiments/repro_fig7_exclusivity.sh` | free if fig 3 has run |

Each script runs its capture stages and then its plotting stage. `--plot` skips to plotting
against captures already on disk. Figures 2, 3 and 7 also take `--verify`, which re-renders
and checksums the result against the md5 pinned at the bottom of the script.

**Two plates are composed in the manuscript, not here.** Figure 5 is three per-model rows,
because the model is not a flag — it is which field is passed, and the field is what the
caption is written from. Figure 6 is two plates because A–D are finite differences over a
sweep and E–H is autodiff over a grid; they share no arithmetic, and one output file would
hide that. `main.tex` expects `fig_jac_grid` and `fig_combined_arith`, so it needs either a
subfigure block over those inputs or a pre-composed file.

**Nothing is captured twice.** Figure 4's field and figure 5's Llama field *are* figure 1's
panel-C field, the same file — whichever script runs first, the others are a no-op. Figure 7
delegates its whole capture stage to figure 3's script.

## Quick start

```bash
git clone https://github.com/dikshagup/weekday-manifold.git
cd weekday-manifold
python -m venv .venv && . .venv/bin/activate     # Python >= 3.9
pip install -e '.[dev]'
pytest -q                                        # CPU only, no GPU, no weights

huggingface-cli login                            # Llama-3.1-8B is a gated repo
bash experiments/repro_fig3_arc_occupancy.sh     # the cheapest figure to try first
```

The scripts set `PYTHONPATH=src:experiments` themselves and `cd` to the repo root. They
default to a bare `python`; pass another with `PY=/path/to/python bash experiments/...`.

### Requirements

Llama-3.1-8B in float16 needs roughly 16 GB of GPU memory, plus headroom for the JVPs in
figure 1. Weights come through `transformer_lens` / `huggingface-hub`, so
`huggingface-cli login` or a populated `HF_HOME` is required.

**Budget ~10 GB of disk beyond the model weights** for a full run of all seven. Large
intermediates default under `${XDG_CACHE_HOME:-~/.cache}/weekday-manifold` rather than the
working tree; redirect them with `CORPUS=` and `RAWDIR=` if that volume is small.

| Path | Size | Written by |
|------|------|------------|
| `figures/ladder_llama_best.npz` | 223 MB | figure 1's ladder |
| `~/.cache/weekday-manifold/corpus_v2/` | ~2.3 GB | figure 3's window selection |
| `~/.cache/weekday-manifold/corpus_v2_raw/` | ~1.6 GB | figure 3's raw + swap captures |
| GPT-2 XL weights | 6.4 GB | figure 5, if not already cached |

`HF_HUB_OFFLINE` defaults to `0` so a fresh clone can fetch what it is missing. **Export
`HF_HUB_OFFLINE=1` once the weights are cached** — that is the setting every published
figure was produced under, and it turns a missing checkpoint into an error rather than a
silent multi-gigabyte re-download. Figure 3's FineWeb stage forces it back off for that one
stage, which is the only stage meant to hit the network.

Figure 5 reads three cache variables — `HF_LLAMA`, `HF_MISTRAL`, `HF_GPT2XL` — which all
default to the same place. They are separate because the three sets of weights need not live
on one volume. `HF_GPT2XL` is also where GPT-2 XL lands if it has to be fetched, so give it
7 GB.

### The environment the published figures came from

`pyproject.toml` gives lower bounds; these are the exact versions behind every published
figure, read off the interpreter that produced them. The bounds *are* these versions, so an
install resolving to anything newer is untested here.

| Package | Version | | Package | Version |
|---------|---------|---|---------|---------|
| Python | 3.9.25 | | `transformers` | 4.57.6 |
| `numpy` | 1.26.4 | | `huggingface-hub` | 0.36.2 |
| `scipy` | 1.13.1 | | `matplotlib` | 3.9.4 |
| `torch` | 2.8.0 (CUDA 12.8) | | `pytest` | 8.4.2 |
| `transformer-lens` | 2.18.0 | | | |

Transitive pins that matter, because a resolver choosing differently is the usual reason a
capture stops matching: `tokenizers` 0.22.2, `safetensors` 0.7.0, `einops` 0.8.2,
`accelerate` 1.10.1, `jaxtyping` 0.2.36, `fancy-einsum` 0.0.3, `sentencepiece` 0.2.2,
`pandas` 2.0.3, `datasets` 4.5.0.

Two of those are not preferences:

* **`torch` >= 2.7 on a CUDA 12.8 build if the GPU is Blackwell.** These ran on an RTX PRO
  4000 Blackwell, sm_120. A torch whose arch list stops at sm_90 still returns `True` from
  `torch.cuda.is_available()` and then raises `no kernel image is available for execution on
  the device` on the first real kernel, well after load.
* **`transformer-lens` must stay on 2.x** — hence `~=2.0`. 3.x pulls `transformers` 5.x,
  deprecates `HookedTransformer.from_pretrained`, and its load path exceeds the container
  memory these captures were sized for: the process is SIGKILLed after "Loading weights"
  with no traceback, which reads like a crash rather than an OOM. 2.18 loads Llama-3.1-8B at
  about 27 GB peak RSS.

## Two things the code depends on that are easy to get wrong

**float16 is load-bearing for figures 1 and 6.** Their speed measure divides a difference of
two nearby residuals by a small step, and the rounding error in that difference does not
shrink as the step does. In bfloat16 the floor was 69% of the measured step — constant round
the ring, so it filled in exactly the troughs the figures are about, and pulled the
day-to-day ratio from 2.52 to 1.58. The Jacobian never subtracts two nearly equal vectors
and has none of this: Llama's field moved bf16 → fp16 and did not budge. Running a capture
against a bfloat16 config produces a different figure without erroring.

**The two prompt families read at different sites.** Figures 1, 4 and 5 use the **mention**
family — *"On Monday, she missed her train."* — which asks nothing of the day and is read at
the full stop, position 8, with the weekday at 2. Figure 6 uses the **interrogative**
family — *"Q: What day is one days after Monday?\nA:"* — which makes the model *compute* a
day: the weekday sits at 9 and the readout at 12. Swapping them measures the wrong site and
reports nothing wrong.

A third trap, in figure 5: Mistral's `--require-seq-len 12` is not optional. SentencePiece
splits the 20 mention templates into 12/13/14/15 tokens, so a fixed index would read the
full stop in some prompts and a content word in others. The filter keeps 105 of 140 prompts,
15 per day, dropping whole templates so the days stay balanced. GPT-2 XL takes no filter; if
its BPE disagrees, `torch.cat` fails on a shape mismatch rather than reading the wrong site.

## What is verified, and what is not

| # | Plate | Verified |
|---|-------|----------|
| 1 | `fig_combined_llama` | captures re-run here; arguments confirmed. No md5 — see below |
| 2 | `timeofday_with_steering` | PNG md5 of the **plot stage only** |
| 3 | `arc_occupancy_main_ab` | the three-panel variant matches its md5; the two-panel cut has no pin yet |
| 4 | `fig_method_steer` | wiring only — a schematic, no measured content |
| 5 | `fig_jac_*` | render-equivalence against the original, plus wiring |
| 6 | `fig_combined_arith_*` | wiring of its inputs |
| 7 | `exclusivity_figure` | PNG md5 |

**Figures 3 and 7 are the strong form.** Both were re-run from scratch on this tree — a
fresh FineWeb stream, fresh captures, nothing carried over — and matched their pinned PNG
md5s byte for byte on different hardware. So did `arc_occupancy_appendix`.

**Figure 3's published cut has no pin yet.** `--panels ab` is a layout this branch
introduces, so no hash from the published run exists for it. `--verify` re-renders the
three-panel variant, whose md5 *is* from that run and which shares every panel with the
two-panel cut, so it remains the strongest available check on the stage. Pin `_ab` once it
has been rendered.

**Float16 figures cannot carry a portable md5.** Figures 1, 2, 6 and figure 5's Llama and
Mistral rows are captured in float16, whose values drift by ~1e-4 relative between hardware
generations — the fp16 rounding scale, not an error. Figure 5's GPT-2 XL row is float32 and
*does* match byte for byte, which is the control identifying dtype as the cause. Figures 3
and 7 are float16 too and still match, because their chain collapses to 6-D projections and
rank statistics before anything reaches a pixel. Treat a pinned md5 as a check on the
plotting stage and on your own tree, not as a portable statement about the captures.

**Figure 1 carries no md5 and should not be given one.** Re-rendering came out at identical
pixel dimensions and 3.9/255 mean absolute difference from the published plate — the largest
drift here, and expected: panel A is the finite-difference measure described above, so a
pinned hash would fail on any machine but the one that wrote it.

**Figure 2's md5 covers its plot stage.** Re-rendering from the committed captures
reproduces the hash, including the isolation check that `--steer-pairs` appends panel C and
touches nothing else. Re-running the *captures* does not: the figure drifts at ~1e-3 meanAbs
at identical pixel dimensions, for the float16 reason above.

**Figure 5 is verified by render-equivalence.** Its plotting script is master's
`figure_main.py` reduced to the `--panels discs` cut, so the check that matters is that the
reduction changed nothing: rendered against the original on synthetic fields, in both the
three-disc and four-disc cases, the PNGs are byte-identical and the captions identical
strings. Figure 4 is the `--panels method` cut of the same file, ported the same way.

`tests/test_figure_wiring.py` builds figures 1 and 5's captures at the real key schema but
tiny and renders the plates, so a broken import or a missing helper cannot hide behind those
GPU hours.

## Layout

```
configs/
  llama31_8b_fp16.json                Llama-3.1-8B, float16, mention        figs 1, 4, 5
  llama31_8b_fp16_interrogative.json  the same in the interrogative family  fig 6
  mistral7b_mention.json              Mistral-7B-v0.1, float16              fig 5
  gpt2xl_mention.json                 GPT-2 XL, float32, 48 blocks          fig 5

experiments/
  repro_fig{1..7}_*.sh          the seven entry points, and the authority on
                                  what each figure's arguments are

  alpha_ladder_sites.py         figs 1, 6: the ring walk, every steer layer in one pass
  jacobian_polar_field.py       figs 1, 4, 5, 6: the polar Jacobian field over the day disc
  figure_combined.py            fig 1: the plate
  cascade_row.py                fig 1: panels A and B   (drawing only, no main)
  polar_disc.py                 figs 1, 5: one disc     (drawing only, no main)

  capture_timeofday.py          fig 2: the clock-family prompts at L28
  steer_timeofday.py            fig 2: the steering run
  steer_clamp.py                fig 2: frames and clamp hooks
  steer_clamp_dists.py          fig 2: the clean-state reference
  time_readout.py               fig 2: the predicted hour, as a circular mean
  steer_clamp_matched_pairs.py  fig 2: the 84-cell edit-size-matched run
  figure_carrier2_daytime_talk.py  fig 2: the three-panel plate

  select_corpus_windows.py      fig 3: stream FineWeb, pick windows (CPU)
  dedup_corpus_windows.py       fig 3: near-duplicate removal, per dump
  build_capture_list.py         fig 3: deduplicated windows -> the exact capture list
  capture_corpus_windows.py     fig 3: the 6-D projections, and the fit/score split
  capture_raw_layer.py          fig 3: the raw 4096-d L28 dump
  swap_build_set.py             fig 3: the one-token weekday-swap control set
  swap_capture_l28.py           fig 3: that set, at the same layer and settings
  arc_geometry.py               figs 3, 7: foot points along the spline (drawing only)
  figure_arc_occupancy_split.py fig 3: the main plate and its appendix
  figure_exclusivity.py         fig 7: the four-panel plate

  method_schematic.py           fig 4: the chart and the intervention  (drawing only)
  figure_method_steer.py        fig 4: the plate
  readout_context.py            figs 4, 5: the token strip -> figures/readout_context.json
  figure_jac_discs.py           figs 5, 6: the disc row, one per field
  figure_combined_arith.py      fig 6: panels A-D

src/weekday_manifold/           the library the above import
tests/                          CPU tests, plus the figure 1 and 5 wiring tests
  synthetic_captures.py         the capture schemas, for figures that cannot be re-rendered
```

`cascade_row.py`, `polar_disc.py`, `arc_geometry.py` and `method_schematic.py` are drawing
modules, not figure scripts. Each holds functions lifted verbatim from a script on `master`
that also built a *different* figure; the function bodies are unchanged, so what they draw
is what was published, and only the unrelated `main()`s are gone.

**What is shared, and why.** `polar_disc.py` draws figure 1's panel C and every disc in
figures 5 and 6; `arc_geometry.py`'s `spline_op` builds both figure 3's foot points and
figure 7's ring, which are the same spline through the same knots. In each case the
alternative was two copies of one renderer that could drift apart without either figure
erroring. Per-figure *style* helpers are deliberately not shared: they differ between plates,
and sharing them would couple the figures' appearance rather than their arithmetic.

## Known rough edges

* `ManifoldConfig.group_by` is read by nothing. It is kept because all four configs set it
  and dropping the field would make them fail to load, but do not reason from its value —
  three of them say `"input"`, which is not one of the two values its own docstring
  documents. It is a *different* knob from `alpha_ladder_sites.py --group-by`, which takes
  `answer|input|auto` and is read.
* `ManifoldConfig.plot_3d_dims`, `out_dir`, `seq_run_lengths` and `seq_separators` are
  likewise unread on this branch, and `DEFAULT_LAYER_SWEEP` names blocks past the depth of
  every model here.
* Some kept scripts carry flags for analyses that are not on this branch. They are inert,
  and removing them would mean editing measurement code to no reproductive benefit.

## Citing

See [`CITATION.cff`](CITATION.cff). Released under the [MIT License](LICENSE).
