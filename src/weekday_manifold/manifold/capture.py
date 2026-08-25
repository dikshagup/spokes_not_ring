"""Capture resid_post activations at one (layer, position) for many prompts."""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Sequence

import numpy as np
import torch

from weekday_manifold.manifold.behavior import restrict_to_concept
from weekday_manifold.manifold.days import PromptSpec


def _to_ids(model, text: str, prepend_bos: bool) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=prepend_bos)


def resolve_capture_position(model, spec: PromptSpec, prepend_bos: bool) -> int:
    """0-based index (into the full prompt's tokens) of the capture site."""
    full = _to_ids(model, spec.text, prepend_bos)[0]
    cap = _to_ids(model, spec.capture_text, prepend_bos)[0]
    n_cap = cap.shape[0]
    assert n_cap >= 1, f"empty capture_text for prompt {spec.text!r}."
    assert n_cap <= full.shape[0], (
        f"capture_text tokenizes longer than text for {spec.text!r}."
    )
    assert torch.equal(cap, full[:n_cap]), (
        "capture_text is not a token-prefix of text (BPE merge shifted at the "
        f"boundary) for prompt {spec.text!r}; check trailing spaces."
    )
    return int(n_cap - 1)


def _cache_key(model_name: str, hook_name: str, prepend_bos: bool,
               flags: dict, specs: Sequence[PromptSpec]) -> str:
    payload = {
        "model_name": model_name,
        "hook_name": hook_name,
        "prepend_bos": prepend_bos,
        "flags": flags,
        "prompts": [(s.text, s.capture_text) for s in specs],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    safe = model_name.replace("/", "_")
    layer = hook_name.split(".")[1]
    return f"manifold_{safe}_L{layer}_{digest}"


def capture_manifold_activations(
    model,
    specs: Sequence[PromptSpec],
    config,
    use_cache: bool = True,
) -> dict:
    """Capture the (layer, answer-pos) activation for every prompt in ``specs``."""
    hook_name = config.hook_name
    flags = {
        "dtype": config.dtype,
        "fold_ln": config.fold_ln,
        "center_writing_weights": config.center_writing_weights,
        "center_unembed": config.center_unembed,
    }
    os.makedirs(config.cache_dir, exist_ok=True)
    key = _cache_key(config.model_name, hook_name, config.prepend_bos, flags, specs)
    cache_path = os.path.join(config.cache_dir, key + ".npz")

    if use_cache and os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=True)
        return {
            "activations": blob["activations"],
            "answer_days": blob["answer_days"],
            "positions": blob["positions"],
            "hook_name": str(blob["hook_name"]),
            "layer_index": int(blob["layer_index"]),
            "texts": list(blob["texts"]),
        }

    acts: List[np.ndarray] = []
    positions: List[int] = []
    answer_days: List[int] = []
    texts: List[str] = []
    for spec in specs:
        pos = resolve_capture_position(model, spec, config.prepend_bos)
        tokens = _to_ids(model, spec.text, config.prepend_bos)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=hook_name, return_type=None,
            )
        act = cache[hook_name][0, pos, :].detach().to(torch.float32).cpu().numpy()
        acts.append(act)
        positions.append(pos)
        answer_days.append(int(spec.answer_day))
        texts.append(spec.text)

    out = {
        "activations": np.stack(acts).astype(np.float32),
        "answer_days": np.asarray(answer_days, dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.int64),
        "hook_name": hook_name,
        "layer_index": int(config.layer_index),
        "texts": texts,
    }
    if use_cache:
        np.savez(
            cache_path,
            activations=out["activations"],
            answer_days=out["answer_days"],
            positions=out["positions"],
            hook_name=hook_name,
            layer_index=out["layer_index"],
            texts=np.array(texts, dtype=object),
        )
    return out


def _resolve_site_position(model, spec: PromptSpec, site: str, prepend_bos: bool) -> int:
    """Position of one capture ``site`` for ``spec`` (via its ``meta["sites"]`` prefix)."""
    cap_text = spec.meta["sites"].get(site, spec.meta["sites"]["last_token"])
    probe = PromptSpec(text=spec.text, answer_day=spec.answer_day,
                       capture_text=cap_text, formulation=spec.formulation)
    return resolve_capture_position(model, probe, prepend_bos)


def _alllayer_cache_key(model_name: str, prepend_bos: bool, flags: dict,
                        sites: Sequence[str], specs: Sequence[PromptSpec]) -> str:
    payload = {
        "model_name": model_name, "prepend_bos": prepend_bos, "flags": flags,
        "sites": list(sites),
        "prompts": [(s.text, tuple(sorted(s.meta["sites"].items()))) for s in specs],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"allsite_{model_name.replace('/', '_')}_{digest}"


def capture_manifold_activations_all_layers(
    model,
    specs: Sequence[PromptSpec],
    config,
    sites: Sequence[str] = ("last_token", "summary_token"),
    use_cache: bool = True,
) -> dict:
    """Capture EVERY layer's ``resid_post`` at MULTIPLE sites, one forward per prompt."""
    sites = list(sites)
    n_layers = model.cfg.n_layers
    flags = {"dtype": config.dtype, "fold_ln": config.fold_ln,
             "center_writing_weights": config.center_writing_weights,
             "center_unembed": config.center_unembed}
    os.makedirs(config.cache_dir, exist_ok=True)
    key = _alllayer_cache_key(config.model_name, config.prepend_bos, flags, sites, specs)
    cache_path = os.path.join(config.cache_dir, key + ".npz")

    if use_cache and os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=True)
        return {
            "acts": {s: blob[f"acts__{s}"] for s in sites},
            "layers": list(range(n_layers)),
            "labels": blob["labels"],
            "site_pos": {s: blob[f"pos__{s}"] for s in sites},
            "texts": list(blob["texts"]),
        }

    N = len(specs)
    acts = {s: np.empty((n_layers, N, model.cfg.d_model), dtype=np.float32) for s in sites}
    site_pos = {s: np.empty(N, dtype=np.int64) for s in sites}
    labels = np.empty(N, dtype=np.int64)
    texts: List[str] = []

    def is_resid_post(name: str) -> bool:
        return name.endswith("hook_resid_post")

    for i, spec in enumerate(specs):
        positions = {s: _resolve_site_position(model, spec, s, config.prepend_bos)
                     for s in sites}
        tokens = _to_ids(model, spec.text, config.prepend_bos)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=is_resid_post, return_type=None)
        for l in range(n_layers):
            h = cache[f"blocks.{l}.hook_resid_post"][0]  # [P, d_model]
            for s in sites:
                acts[s][l, i, :] = h[positions[s], :].detach().to(torch.float32).cpu().numpy()
        for s in sites:
            site_pos[s][i] = positions[s]
        labels[i] = int(spec.answer_day)
        texts.append(spec.text)

    if use_cache:
        save = {"labels": labels, "texts": np.array(texts, dtype=object)}
        for s in sites:
            save[f"acts__{s}"] = acts[s]
            save[f"pos__{s}"] = site_pos[s]
        np.savez(cache_path, **save)

    return {"acts": acts, "layers": list(range(n_layers)),
            "labels": labels, "site_pos": site_pos, "texts": texts}


def _dist_cache_key(model_name: str, prepend_bos: bool,
                    concept_ids: Sequence[int], specs: Sequence[PromptSpec]) -> str:
    payload = {
        "model_name": model_name,
        "prepend_bos": prepend_bos,
        "concept_ids": list(int(i) for i in concept_ids),
        "prompts": [(s.text, s.capture_text) for s in specs],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    safe = model_name.replace("/", "_")
    return f"behavior_{safe}_{digest}"


def capture_output_distributions(
    model,
    specs: Sequence[PromptSpec],
    config,
    concept_ids: Sequence[int],
    use_cache: bool = True,
) -> dict:
    """Capture the restricted OUTPUT distribution at each prompt's answer position."""
    concept_ids = [int(i) for i in concept_ids]
    os.makedirs(config.cache_dir, exist_ok=True)
    key = _dist_cache_key(config.model_name, config.prepend_bos, concept_ids, specs)
    cache_path = os.path.join(config.cache_dir, key + ".npz")

    if use_cache and os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=True)
        return {
            "distributions": blob["distributions"],
            "answer_days": blob["answer_days"],
            "positions": blob["positions"],
            "concept_ids": list(blob["concept_ids"]),
            "texts": list(blob["texts"]),
        }

    dists: List[np.ndarray] = []
    positions: List[int] = []
    answer_days: List[int] = []
    texts: List[str] = []
    for spec in specs:
        pos = resolve_capture_position(model, spec, config.prepend_bos)
        tokens = _to_ids(model, spec.text, config.prepend_bos)
        with torch.no_grad():
            logits = model(tokens, return_type="logits")[0, pos, :]  # [vocab]
        probs = torch.softmax(logits.to(torch.float32), dim=-1).cpu().numpy()
        dists.append(restrict_to_concept(probs, concept_ids).astype(np.float32))
        positions.append(pos)
        answer_days.append(int(spec.answer_day))
        texts.append(spec.text)

    out = {
        "distributions": np.stack(dists).astype(np.float32),
        "answer_days": np.asarray(answer_days, dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.int64),
        "concept_ids": concept_ids,
        "texts": texts,
    }
    if use_cache:
        np.savez(
            cache_path,
            distributions=out["distributions"],
            answer_days=out["answer_days"],
            positions=out["positions"],
            concept_ids=np.asarray(concept_ids, dtype=np.int64),
            texts=np.array(texts, dtype=object),
        )
    return out


def _text_cache_key(model_name: str, hook_name: str, prepend_bos: bool,
                    flags: dict, texts: Sequence[str]) -> str:
    payload = {
        "model_name": model_name, "hook_name": hook_name,
        "prepend_bos": prepend_bos, "flags": flags, "texts": list(texts),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    layer = hook_name.split(".")[1]
    return f"text_{model_name.replace('/', '_')}_L{layer}_{digest}"


def capture_text_activations(
    model,
    texts: Sequence[str],
    config,
    include_bos: bool = False,
    use_cache: bool = True,
) -> dict:
    """Capture per-token ``resid_post`` at layer L for arbitrary strings (Lens A)."""
    hook_name = config.hook_name
    flags = {"dtype": config.dtype, "fold_ln": config.fold_ln,
             "center_writing_weights": config.center_writing_weights,
             "center_unembed": config.center_unembed, "include_bos": include_bos}
    os.makedirs(config.cache_dir, exist_ok=True)
    key = _text_cache_key(config.model_name, hook_name, config.prepend_bos, flags, texts)
    cache_path = os.path.join(config.cache_dir, key + ".npz")
    if use_cache and os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=True)
        return {k: blob[k] for k in ("activations", "token_ids", "text_idx", "pos")} | {
            "token_strs": list(blob["token_strs"]),
            "hook_name": str(blob["hook_name"]), "layer_index": int(blob["layer_index"]),
        }

    acts, tok_ids, tok_strs, text_idx, pos = [], [], [], [], []
    for ti, text in enumerate(texts):
        tokens = _to_ids(model, text, config.prepend_bos)          # [1, P]
        strs = model.to_str_tokens(text, prepend_bos=config.prepend_bos)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name, return_type=None)
        h = cache[hook_name][0].detach().to(torch.float32).cpu().numpy()  # [P, d_model]
        start = 1 if (config.prepend_bos and not include_bos) else 0
        for p in range(start, h.shape[0]):
            acts.append(h[p]); tok_ids.append(int(tokens[0, p]))
            tok_strs.append(strs[p]); text_idx.append(ti); pos.append(p)

    out = {
        "activations": np.stack(acts).astype(np.float32),
        "token_ids": np.asarray(tok_ids, dtype=np.int64),
        "token_strs": tok_strs,
        "text_idx": np.asarray(text_idx, dtype=np.int64),
        "pos": np.asarray(pos, dtype=np.int64),
        "hook_name": hook_name, "layer_index": int(config.layer_index),
    }
    if use_cache:
        np.savez(cache_path, activations=out["activations"], token_ids=out["token_ids"],
                 token_strs=np.array(tok_strs, dtype=object), text_idx=out["text_idx"],
                 pos=out["pos"], hook_name=hook_name, layer_index=out["layer_index"])
    return out
