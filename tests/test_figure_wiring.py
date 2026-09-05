"""The figures whose captures cost GPU hours render from synthetic captures.

WHY THESE EXIST. Figures 1 and 5 are built from captures that are gitignored and take hours
on a card, so unlike figures 2 and 3 they cannot be re-rendered on demand to check a
change. These build both captures at the real key schema but tiny, and render the plates.

They check WIRING, not numbers: that every key each panel reads is a key the capture writes,
that the cross-file asserts fire on agreeing inputs, and that the shared drawing modules
define everything they call. That last one is not hypothetical -- polar_disc.py was first
extracted without `measurement_inset`, which `disc` calls only when a panel is measured along
a named direction, and this is the check that caught it.
"""
import sys

import pytest

pytest.importorskip("matplotlib")

from synthetic_captures import polar_field, ring_walk  # noqa: E402

CONTEXT = "figures/readout_context.json"


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + [str(a) for a in argv])
    module.main()


# --------------------------------------------------------------------- figure 1: the plate
def _combined(tmp_path, monkeypatch, **over):
    import figure_combined
    out = tmp_path / "fig_combined_llama.pdf"
    argv = ["--cascade", ring_walk(tmp_path / "walk.npz"),
            "--field", polar_field(tmp_path / "field.npz"),
            "--out", out, "--also-png"]
    for k, v in over.items():
        argv += ["--" + k.replace("_", "-"), v]
    _run(figure_combined, argv, monkeypatch)
    return out


def test_combined_tight_layout_renders(tmp_path, monkeypatch):
    """The published cut: three panels in one row, and it is the default."""
    out = _combined(tmp_path, monkeypatch)
    assert out.stat().st_size > 0
    assert out.with_suffix(".png").stat().st_size > 0


def test_combined_full_layout_renders(tmp_path, monkeypatch):
    """--layout full draws all seven panels."""
    assert _combined(tmp_path, monkeypatch, layout="full").stat().st_size > 0


def test_combined_refuses_a_mismatched_steer_layer(tmp_path, monkeypatch):
    """The plate's claim is that both halves measure ONE intervention, so it asserts it."""
    with pytest.raises(AssertionError, match="the field steers at"):
        _combined(tmp_path, monkeypatch, fix_steer="6")


# ----------------------------------------------------------------- figure 5: the disc plate
def _discs(tmp_path, monkeypatch, name="fig_jac.pdf", **field_kw):
    import figure_jac_discs
    out = tmp_path / name
    _run(figure_jac_discs,
         ["--npz", polar_field(tmp_path / (name + ".npz"), **field_kw),
          "--out", out, "--context", CONTEXT, "--also-png"],
         monkeypatch)
    return out


def test_discs_three_measures_render(tmp_path, monkeypatch):
    """fro, gain_t, gain_r -- the three the published plates carry, lettered B, C, D."""
    out = _discs(tmp_path, monkeypatch)
    assert out.stat().st_size > 0
    assert out.with_suffix(".png").stat().st_size > 0


def test_discs_pick_up_the_off_direction_when_present(tmp_path, monkeypatch):
    """A fourth disc appears iff the field carries gain_off, so the cut follows the data."""
    three = _discs(tmp_path, monkeypatch, name="three.pdf")
    four = _discs(tmp_path, monkeypatch, name="four.pdf", with_off=True)
    assert four.stat().st_size != three.stat().st_size


def test_discs_write_their_caption(tmp_path, monkeypatch):
    """Every number in the caption is computed by the figure, so it cannot drift from it."""
    out = _discs(tmp_path, monkeypatch)
    cap = out.with_name(out.stem + "_caption.md")
    text = cap.read_text()
    assert "Llama-3.1-8B" in text and "layer 28" in text and "layer 2" in text


def test_discs_refuse_a_patch_site_that_is_not_a_weekday(tmp_path, monkeypatch):
    """The token strip is checked, not trusted: a day_pos landing off the weekday is fatal."""
    import json

    import numpy as np
    p = tmp_path / "bad.npz"
    polar_field(p)
    z = dict(np.load(p, allow_pickle=True))
    meta = json.loads(str(z["meta"]))
    meta["day_pos"] = 5                      # " is", not a weekday
    z["meta"] = json.dumps(meta)
    np.savez(p, **z)
    import figure_jac_discs
    with pytest.raises(AssertionError, match="not a weekday"):
        _run(figure_jac_discs,
             ["--npz", p, "--out", tmp_path / "bad.pdf", "--context", CONTEXT],
             monkeypatch)
