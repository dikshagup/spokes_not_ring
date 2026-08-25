"""Synthetic stand-ins for the captures the figure scripts read, at the real key schemas.

Figures 1, 5 and 6 are built from captures that cost GPU hours and are gitignored, so they
are the figures that cannot be re-rendered on demand -- which means a broken import or a
missing helper in their drawing code would go unnoticed until someone spent those hours.
These builders exist so the wiring can be tested without them.

They are stand-ins for the SCHEMA, not for the data: shapes, dtypes, key names and the
invariants the figures assert against (knots on the grid, mean-centred centroids, agreeing
metadata). Nothing here should be read as a fixture of what the model does.
"""
import json

import numpy as np

MODEL = "meta-llama/Llama-3.1-8B"
FORMULATION = "mention_early"
RO_POS, DAY_POS, PATCH_LAYER, READOUT_LAYER = 8, 2, 2, 28
LAYERS = [2, 6, 10, 14, 18, 22, 26, 28, 30, 31]
N_DAYS = 7


def polar_field(path, n_u=56, n_r=17, n_bg=6, d=64, with_off=False, seed=0,
                readout_layer=READOUT_LAYER):
    """jacobian_polar_field.py's npz: the Jacobian measured over the day disc.

    The centroids are mean-centred because the real capture's are -- fig_jac_discs quotes
    ||mu|| in its caption as evidence that r = 0 is a prompt with its weekday removed rather
    than an average weekday, and that claim is checked against these numbers.
    """
    rng = np.random.default_rng(seed)
    ang = np.arange(N_DAYS) / N_DAYS * 2 * np.pi
    C = np.zeros((N_DAYS, d))
    C[:, 0] = np.cos(ang) * (1 + 0.2 * np.sin(3 * ang))     # a deliberately irregular loop
    C[:, 1] = np.sin(ang)
    C += 0.05 * rng.standard_normal((N_DAYS, d))
    C -= C.mean(0)
    kw = dict(
        centroids=C.astype(np.float32), mu=np.zeros(d, np.float32),
        us=np.linspace(0.0, 1.0, n_u, endpoint=False),
        rs=np.linspace(0.0, 1.6, n_r),
        fro=(rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32),
        gain_t=(rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32),
        gain_r=(rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32),
        meta=json.dumps(dict(model=MODEL, formulation=FORMULATION, ro_pos=RO_POS,
                             day_pos=DAY_POS, patch_layer=PATCH_LAYER,
                             readout_layer=readout_layer, n_hutch=6)),
    )
    if with_off:
        kw["gain_off"] = (rng.random((n_bg, n_u, n_r)) + 1.0).astype(np.float32)
    np.savez(path, **kw)
    return str(path)


def ring_walk(path, n_prompts=14, n_u=29, n_layers=32, seed=0):
    """alpha_ladder_sites.py's npz: the finite-difference walk round the ring.

    n_u = 29 keeps the seven knots exactly on the grid at indices 0, 4, ... 24, the way
    n_u = 141 does at 0, 20, ... 120 in the real run. The plate reads knot positions out of
    the npz rather than assuming k*m, so a grid that did not carry them would test nothing.
    """
    rng = np.random.default_rng(seed)
    n_l, n_s = len(LAYERS), 1
    np.savez(
        path,
        us=np.linspace(0.0, 1.0, n_u),
        layers=np.array(LAYERS),
        d_layer=rng.random((n_s, n_l, n_prompts, n_u, n_layers, 2)).astype(np.float32),
        step_layer=rng.random((n_s, n_l, n_prompts, n_u - 1, n_layers, 2)).astype(np.float32),
        resid_scale=(rng.random((n_layers, 2)) + 1.0).astype(np.float32),
        dists=rng.random((n_s, n_l, n_prompts, n_u, N_DAYS + 1)).astype(np.float32),
        stride_in=rng.random((n_s, n_l, n_u - 1)) + 1.0,
        knot_index=np.tile(np.arange(N_DAYS) * ((n_u - 1) // N_DAYS), (n_l, 1)),
        prompt_days=np.arange(n_prompts) % N_DAYS,
        readout_layer=READOUT_LAYER, ro_pos=RO_POS,
        model=MODEL, formulation=FORMULATION,
    )
    return str(path)
