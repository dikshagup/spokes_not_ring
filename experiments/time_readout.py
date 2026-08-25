#!/usr/bin/env python
"""Read a predicted hour off the next-token distribution as a circular mean. No main."""
from __future__ import annotations

import numpy as np
import torch

MINUTES = (0, 15, 30, 45)


def _hh_ap(h):
    return (12 if h % 12 == 0 else h % 12), ("am" if h < 12 else "pm")


def unambiguous_forms():
    """[(string, hour_value)] -- every entry pins the hour on its own."""
    out = []
    for h in range(24):
        hh, ap = _hh_ap(h)
        AP, DOT, DOTU = ap.upper(), f"{ap[0]}.{ap[1]}.", f"{ap[0].upper()}.{ap[1].upper()}."
        for m in MINUTES:
            t = "" if m == 0 else f":{m:02d}"
            v = h + m / 60.0
            out += [(f" {hh}{t}{ap}", v), (f" {hh}{t} {ap}", v), (f" {hh}{t} {DOT}", v),
                    (f" {hh}{t}{AP}", v), (f" {hh}{t} {AP}", v), (f" {hh}{t} {DOTU}", v)]
    # 24-hour clock, evening only: 13-23 cannot be read as a 12-hour time
    for h in range(13, 24):
        for m in MINUTES:
            out.append((f" {h}:{m:02d}", h + m / 60.0))
    # the words, which the model prefers at these two bins
    out += [(" noon", 12.0), (" midday", 12.0), (" 12 noon", 12.0),
            (" midnight", 0.0), (" 12 midnight", 0.0)]
    seen, uniq = set(), []
    for s, v in out:
        if s not in seen:
            seen.add(s); uniq.append((s, v))
    return uniq


def ambiguous_prefixes():
    """[(string, clock_face_value)] -- forms that do NOT pin the hour."""
    out = [(f" {hh}", float(hh)) for hh in range(1, 13)]
    out += [(f" {hh}:{m:02d}", hh + m / 60.0)
            for hh in range(1, 13) for m in MINUTES if m]
    out += [(f" {hh} o'clock", float(hh)) for hh in range(1, 13)]
    return out


class TimeReadout:
    """Teacher-forced scoring of the extended candidate set, with a coverage accounting."""

    def __init__(self, model, prepend_bos=True, chunk=320):
        self.m = model
        self.bos = prepend_bos
        self.chunk = chunk
        self.forms = unambiguous_forms()
        self.values = np.array([v for _, v in self.forms])
        amb = ambiguous_prefixes()
        self.amb = [a for a, _ in amb]
        self.amb_val = np.array([v for _, v in amb])
        tok = lambda s: model.to_tokens(s, prepend_bos=False)[0].tolist()
        self._ids = [tok(s) for s, _ in self.forms]
        self._amb_ids = [tok(s) for s in self.amb]
        # which scored strings extend which ambiguous prefix (for the residual)
        self._ext = [[j for j, (s, _) in enumerate(self.forms) if s.startswith(a)]
                     for a in self.amb]

    @torch.no_grad()
    def _logp(self, pre, cont_ids, fwd_hooks=None):
        out = np.full(len(cont_ids), -np.inf)
        dev, n = self.m.cfg.device, len(pre)
        for lo in range(0, len(cont_ids), self.chunk):
            part = cont_ids[lo:lo + self.chunk]
            seqs = [pre + c for c in part]
            mx = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), mx), dtype=torch.long, device=dev)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s, device=dev)
            logits = (self.m.run_with_hooks(ids, return_type="logits", fwd_hooks=fwd_hooks)
                      if fwd_hooks else self.m(ids, return_type="logits"))
            # positions n-1 .. mx-2 predict continuation tokens n .. mx-1; nothing else is
            # read, so the float log_softmax is taken on that slice only
            lg = torch.log_softmax(logits[:, n - 1:mx - 1, :].float(), -1)
            for j, s in enumerate(seqs):
                k = len(s) - n
                tgt = torch.tensor(s[n:], device=dev).unsqueeze(-1)
                out[lo + j] = float(lg[j, :k].gather(-1, tgt).sum())
        return out

    def score(self, text, fwd_hooks=None):
        pre = self.m.to_tokens(text, prepend_bos=self.bos)[0].tolist()
        lp = self._logp(pre, self._ids, fwd_hooks)
        lpa = self._logp(pre, self._amb_ids, fwd_hooks)
        p, pa = np.exp(lp), np.exp(lpa)

        captured = float(p.sum())
        # residual per ambiguous prefix: mass entering it that no scored string explains
        resid = np.array([max(float(pa[i] - p[self._ext[i]].sum()), 0.0)
                          for i in range(len(self.amb))])
        # a bare hour's residual already contains its own ":mm" and "o'clock" residuals
        bare = {f" {h}": i for i, h in zip(range(len(self.amb)), range(1, 13))}
        for i, a in enumerate(self.amb):
            for b, bi in bare.items():
                if a != b and a.startswith(b):
                    resid[bi] = max(resid[bi] - resid[i], 0.0)
        ambiguous = float(resid.sum())

        w = p / max(captured, 1e-30)
        ang = 2 * np.pi * self.values / 24.0
        R = complex(float((w * np.cos(ang)).sum()), float((w * np.sin(ang)).sum()))
        cm = (np.arctan2(R.imag, R.real) % (2 * np.pi)) * 24 / (2 * np.pi)

        bins = np.zeros(24)
        for k, v in enumerate(self.values):
            bins[int(round(v)) % 24] += w[k]
        am = float(w[(self.values >= 0) & (self.values < 12)].sum())
        return dict(p=bins, circ_mean=float(cm), conc=abs(R), logodds=float(
            np.log(max(am, 1e-12)) - np.log(max(1 - am, 1e-12))),
            captured=captured, ambiguous=ambiguous, resid=resid, logp=lp, weights=w)

    def bounds(self, res):
        """Circular mean if ALL the ambiguous mass were am, and if it were all pm."""
        mass = np.concatenate([res["weights"] * res["captured"], res["resid"]])
        tot = mass.sum()
        base = self.amb_val % 12.0
        out = {}
        for side, off in (("am", 0.0), ("pm", 12.0)):
            vv = np.concatenate([self.values, base + off])
            ang = 2 * np.pi * vv / 24.0
            ww = mass / max(tot, 1e-30)
            R = complex(float((ww * np.cos(ang)).sum()), float((ww * np.sin(ang)).sum()))
            out[side] = float((np.arctan2(R.imag, R.real) % (2 * np.pi)) * 24 / (2 * np.pi))
        return out

    def score_strings(self, text, strings, fwd_hooks=None):
        """log P of an arbitrary string list -- used to test against the old scorer at its OWN
        batch shape, so a logic difference is not confounded with bf16 batching noise."""
        pre = self.m.to_tokens(text, prepend_bos=self.bos)[0].tolist()
        ids = [self.m.to_tokens(s, prepend_bos=False)[0].tolist() for s in strings]
        keep, self.chunk = self.chunk, max(len(ids), 1)
        try:
            return self._logp(pre, ids, fwd_hooks)
        finally:
            self.chunk = keep
