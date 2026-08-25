"""Unit tests for config: layer resolution, position semantics, JSON I/O.

No model needed — these are pure parametrization checks.
"""

import json

import pytest

from weekday_manifold.plateau.config import (
    LAYER_PRESETS,
    PlateauConfig,
    hook_name_for_layer,
    resolve_layer_index,
)


def test_layer_index_precedence_explicit_wins():
    # Explicit index beats frac/preset.
    idx = resolve_layer_index(48, layer_frac=0.5, layer_preset="mid", layer_index=7)
    assert idx == 7


def test_layer_frac_resolves_against_depth():
    assert resolve_layer_index(48, layer_frac=0.25) == 12
    assert resolve_layer_index(48, layer_frac=0.5) == 24
    # paper_early ~ 1/12 of 48 ~ 4
    assert resolve_layer_index(48, layer_preset="paper_early") == 4


def test_layer_index_is_clamped_into_range():
    assert resolve_layer_index(48, layer_frac=2.0) == 47  # clamp high
    assert resolve_layer_index(48, layer_frac=-1.0) == 0  # clamp low


def test_layer_presets_are_fractions_in_unit_interval():
    for name, frac in LAYER_PRESETS.items():
        assert 0.0 <= frac <= 1.0, name


def test_resolve_requires_some_selector():
    with pytest.raises(ValueError):
        resolve_layer_index(48)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_layer_index(48, layer_preset="does_not_exist")


def test_hook_name_format():
    assert hook_name_for_layer(4) == "blocks.4.hook_resid_post"


def test_default_layer_index_is_zero():
    # Default patches after the first block (papers' literal "after first layer").
    cfg = PlateauConfig()
    assert cfg.layer_index == 0
    assert cfg.resolve(n_layers=48).hook_name == "blocks.0.hook_resid_post"


def test_preset_resolves_only_when_layer_index_is_none():
    # layer_index takes precedence, so the preset is used only when it's None.
    cfg = PlateauConfig(layer_preset="early_mid", layer_index=None)
    assert cfg.layer_index is None
    resolved = cfg.resolve(n_layers=48)
    assert resolved.layer_index == 12
    assert resolved.hook_name == "blocks.12.hook_resid_post"
    # original is untouched (dataclasses.replace returns a copy)
    assert cfg.layer_index is None


def test_hook_name_before_resolve_raises():
    cfg = PlateauConfig(layer_index=None)
    with pytest.raises(ValueError):
        _ = cfg.hook_name


def test_config_json_roundtrip(tmp_path):
    cfg = PlateauConfig(prompt_1="a b c", prompt_2="d e f", layer_index=3)
    path = tmp_path / "cfg.json"
    cfg.to_json(str(path))
    loaded = PlateauConfig.from_json(str(path))
    assert loaded == cfg
    # JSON is human-readable and self-describing
    raw = json.loads(path.read_text())
    assert raw["prompt_1"] == "a b c"
    assert raw["layer_index"] == 3


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        PlateauConfig.from_dict({"prompt_1": "x", "bogus": 1})


def test_cache_key_excludes_t_range_but_includes_prompts():
    base = PlateauConfig(layer_index=4)
    same_t_change = base.__class__(**{**base.to_dict(), "t_steps": 999, "t_max": 5.0})
    assert base.cache_key() == same_t_change.cache_key()  # t-range excluded

    diff_prompt = base.__class__(**{**base.to_dict(), "prompt_1": "different"})
    assert base.cache_key() != diff_prompt.cache_key()  # prompts included


def test_cache_key_requires_resolved_layer():
    with pytest.raises(ValueError):
        PlateauConfig(layer_index=None).cache_key()


def test_fixed_position_is_int_last_is_str():
    cfg = PlateauConfig()
    assert cfg.token_position == "last"
    cfg2 = PlateauConfig(token_position=3)
    assert isinstance(cfg2.token_position, int)
