"""ManifoldConfig: layer resolution (presets/frac/index) + JSON round-trip."""

import json

from weekday_manifold.manifold.config import LATE_LAYER_PRESETS, ManifoldConfig


def test_default_preset_resolves_to_late_block():
    cfg = ManifoldConfig().resolve(n_layers=48)
    # late_78 -> round(0.78 * 48) = 37
    assert cfg.layer_index == 37
    assert cfg.hook_name == "blocks.37.hook_resid_post"


def test_layer_frac_overrides_preset():
    cfg = ManifoldConfig(layer_frac=0.5, layer_preset="late_78").resolve(48)
    assert cfg.layer_index == 24


def test_explicit_index_wins():
    cfg = ManifoldConfig(layer_index=10, layer_frac=0.9, layer_preset="late_85").resolve(48)
    assert cfg.layer_index == 10


def test_index_clamped_into_range():
    assert ManifoldConfig(layer_index=999).resolve(48).layer_index == 47
    assert ManifoldConfig(layer_index=-5).resolve(48).layer_index == 0


def test_all_presets_resolve():
    for name, frac in LATE_LAYER_PRESETS.items():
        cfg = ManifoldConfig(layer_preset=name, layer_index=None).resolve(48)
        assert 0 <= cfg.layer_index < 48
        assert cfg.layer_index == round(frac * 48)


def test_json_round_trip(tmp_path):
    cfg = ManifoldConfig(formulation="seq", offset_style="word", n_pca_dims=32)
    path = tmp_path / "cfg.json"
    cfg.to_json(str(path))
    loaded = ManifoldConfig.from_json(str(path))
    assert loaded == cfg


def test_unknown_key_rejected():
    try:
        ManifoldConfig.from_dict({"bogus": 1})
        assert False, "expected ValueError"
    except ValueError:
        pass
