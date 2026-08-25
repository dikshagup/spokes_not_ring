"""Guards for the things that stopped the repro scripts running at their own defaults.

Each test here corresponds to a failure found by re-running every figure from scratch.
They are cheap and CPU-only on purpose: the bugs they cover all sat behind a model load,
so nothing caught them until someone spent the GPU hours to reach them.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------- the capture layer set is clamped
def test_capture_layers_never_exceeds_the_model_depth():
    """The defaults asked for blocks 0..39 of a 32-block model and died on the first hook."""
    from steer_timeofday import capture_layers
    caps = capture_layers(list(range(32)), 28, [1, 3, 5, 9], 32)
    assert caps == list(range(32)), caps
    assert max(caps) < 32


def test_capture_layers_still_reaches_past_the_swept_layers():
    """Clamping must not eat the band reach it exists to provide -- only the overflow."""
    from steer_timeofday import capture_layers
    caps = capture_layers([10, 13], 28, [1, 3, 5, 9], 32)
    assert 21 in caps, "a band of 9 from L13 reaches L21; it must be captured"
    assert max(caps) == 28, caps


# ------------------------------------------- the mention family runs at the plain defaults
@pytest.mark.parametrize("formulation,group,sites",
                         [("me", "input", ["weekday"]),
                          ("mention_early", "input", ["weekday"]),
                          ("interrogative", "answer", ["weekday", "answer", "both"])])
def test_ladder_defaults_resolve_by_formulation(formulation, group, sites):
    """`--formulation me` used to fail two asserts on its OWN defaults, after the load."""
    import argparse

    import alpha_ladder_sites as als
    args = argparse.Namespace(formulation=formulation, readout_group=None, sites=None)
    als.resolve_family_defaults(args)
    assert args.readout_group == group
    assert [s for s in args.sites.split(",") if s.strip()] == sites
    assert all(s in als.SITES for s in sites)


def test_ladder_keeps_an_explicit_choice():
    """Resolution fills a blank; it must not overwrite a flag the caller passed."""
    import argparse

    import alpha_ladder_sites as als
    args = argparse.Namespace(formulation="me", readout_group="answer", sites="both")
    als.resolve_family_defaults(args)
    assert args.readout_group == "answer" and args.sites == "both", (
        "the guard downstream only means something if a wrong choice survives to it")


def test_ladder_leaves_the_family_defaults_unset():
    """The resolution above is only reachable if argparse does not pre-fill these."""
    src = open(os.path.join(REPO, "experiments", "alpha_ladder_sites.py")).read()
    assert '"--readout-group", choices=["answer", "input"], default=None' in src
    assert '"--sites", default=None' in src


# ---------------------------------------------------------- the offline pin has an opening
def test_fig3_lets_the_fineweb_stage_reach_the_hub():
    """Stage 1 streams FineWeb; the script-wide HF_HUB_OFFLINE=1 blocked it outright."""
    src = open(os.path.join(REPO, "experiments", "repro_fig3_arc_occupancy.sh")).read()
    line = [l for l in src.splitlines() if "select_corpus_windows.py" in l]
    assert line, "stage 1 disappeared"
    assert any("HF_HUB_OFFLINE=0" in l for l in src.splitlines()
               if "select_corpus_windows.py" in l or l.strip().startswith("HF_HUB_OFFLINE=0"))


def test_fig3_recaptures_rather_than_reading_the_activation_cache():
    """A reproduction that reads data/activations has not reproduced the GPU stage."""
    src = open(os.path.join(REPO, "experiments", "repro_fig3_arc_occupancy.sh")).read()
    stage4 = [l for l in src.splitlines() if "capture_corpus_windows.py" in l]
    assert stage4 and any("--no-cache" in l for l in stage4), stage4


def test_capture_corpus_windows_exposes_no_cache():
    out = subprocess.run([sys.executable, "experiments/capture_corpus_windows.py", "--help"],
                         cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": "src:experiments"})
    assert "--no-cache" in out.stdout, out.stdout + out.stderr


# ------------------------------------------------- the token strip belongs to one tokeniser
def test_readout_context_names_a_config_that_exists():
    """It pointed at a config pruned when this branch was cut, so it could not be rebuilt."""
    import readout_context
    ap = [l for l in open(readout_context.__file__).read().splitlines()
          if 'ap.add_argument("--config"' in l]
    assert ap, "the config is still hard-coded"
    default = ap[0].split('default=')[1].strip().strip('),').strip('"')
    assert os.path.exists(os.path.join(REPO, default)), default


def test_discs_refuse_another_models_token_strip(tmp_path, monkeypatch):
    """A (patch_pos, read_pos) pair is not unique across tokenisers; the model must match."""
    import numpy as np

    pytest.importorskip("matplotlib")
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from synthetic_captures import polar_field

    p = tmp_path / "other.npz"
    polar_field(p)
    z = dict(np.load(p, allow_pickle=True))
    meta = json.loads(str(z["meta"]))
    meta["model"] = "some-other-org/Sevenish-7B"
    z["meta"] = json.dumps(meta)
    np.savez(p, **z)

    # A context entry that matches on positions but is stamped for a different model.
    ctx = tmp_path / "ctx.json"
    json.dump({"x": {"model": "meta-llama/Llama-3.1-8B", "tokens": ["<bos>"] * 13,
                     "patch_pos": meta["day_pos"], "read_pos": meta["ro_pos"]}},
              open(ctx, "w"))

    import figure_jac_discs
    out = tmp_path / "other.pdf"
    monkeypatch.setattr(sys, "argv",
                        ["x", "--npz", str(p), "--out", str(out), "--context", str(ctx)])
    figure_jac_discs.main()
    assert out.exists(), "the plate must still render, just without borrowed tokens"


# --------------------------------------------------- a half-written cache is a cache miss
def test_timeofday_reuse_check_survives_a_truncated_npz(tmp_path):
    """A disk that filled mid-savez left a truncated zip; the reuse check died reading it."""
    import numpy as np

    good = tmp_path / "t.npz"
    np.savez(good, lab=np.array(["a", "b"]), A=np.zeros((2, 3)))
    raw = good.read_bytes()
    bad = tmp_path / "truncated.npz"
    bad.write_bytes(raw[: len(raw) // 2])

    # This is the exact shape of the guard in capture_timeofday.main(): anything that
    # cannot be parsed must fall through to "re-capture", never out of the process.
    for path, expect_labs in ((good, ["a", "b"]), (bad, None)):
        labs = None
        try:
            labs = sorted(set(np.load(path, allow_pickle=True)["lab"].astype(str)))
        except Exception:
            labs = None
        assert labs == expect_labs, path

    src = open(os.path.join(REPO, "experiments", "capture_timeofday.py")).read()
    head = src[src.index('if os.path.exists(args.npz):'):]
    assert "try:" in head[:1200] and "except Exception" in head[:1200], (
        "the reuse check reads the npz without a guard again")
