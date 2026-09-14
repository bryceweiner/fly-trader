"""PolicyBrain: the trained connectome policy driving the live runner.

Loads a policy checkpoint (brain_snapshots kind='policy'): the pending/live snapshot if it is a policy,
else the best evaluated one. Builds the same observation the offline env used (standardised features with
the checkpoint's statistics + [position fraction, unrealised return, log1p(beats held)]), carries the
hidden state per slot across beats, and returns target exposure fractions. Deterministic (mean action).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .. import config
from ..brain.policy import ConnectomePolicy, SubConnectome
from ..db.connection import transaction

log = logging.getLogger(__name__)


class PolicyBrain:
    def __init__(self, connectome, batch: int, snapshot_id: int | None = None):
        self.c = connectome
        self.B = batch
        self.dev = connectome.device
        path, sid, meta = self._pick(snapshot_id)
        ck = torch.load(path, map_location="cpu", weights_only=False) if path else None
        subgraph = (ck or {}).get("subgraph", "central")
        self.graph = SubConnectome(connectome, exclude=("VISUAL",)) if subgraph == "central" else connectome
        obs_dim = int((ck or {}).get("obs_dim", 48))
        self.policy = ConnectomePolicy(self.graph, obs_dim=obs_dim, k_steps=int((ck or {}).get("k_steps", 4)), device=self.dev)
        if ck:
            self.policy.load_state_dict(ck["state_dict"])
            self.obs_mean = np.asarray(ck["obs_mean"], np.float32); self.obs_std = np.asarray(ck["obs_std"], np.float32)
        else:
            self.obs_mean = np.zeros(obs_dim - 3, np.float32); self.obs_std = np.ones(obs_dim - 3, np.float32)
        self.policy.eval()
        self.snapshot_id = sid
        self.h = self.policy.init_hidden(batch)
        log.info("policy brain: snapshot=%s path=%s %s", sid, path, self.policy.describe())

    def _pick(self, snapshot_id: int | None) -> tuple[Path | None, int | None, dict]:
        with transaction() as conn:
            if snapshot_id is None:
                st = conn.execute("SELECT live_snapshot_id, pending_snapshot_id FROM brain_state WHERE singleton").fetchone()
                for cand in ((st or {}).get("pending_snapshot_id"), (st or {}).get("live_snapshot_id")):
                    if cand:
                        r = conn.execute("SELECT id, path, kind FROM brain_snapshots WHERE id = %s AND kind = 'policy'", (cand,)).fetchone()
                        if r:
                            snapshot_id = int(r["id"]); break
            if snapshot_id is None:
                rows = conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = 'policy' ORDER BY id DESC LIMIT 50").fetchall()
                best = None
                for r in rows:
                    try:
                        m = json.loads(r["note"] or "{}")
                    except Exception:
                        m = {}
                    ev = m.get("eval_net_sol")
                    if ev is not None and (best is None or ev > best[0]):
                        best = (ev, int(r["id"]))
                if best:
                    snapshot_id = best[1]
                elif rows:
                    snapshot_id = int(rows[0]["id"])
            if snapshot_id is None:
                return None, None, {}
            r = conn.execute("SELECT id, path, note FROM brain_snapshots WHERE id = %s", (snapshot_id,)).fetchone()
            conn.execute("UPDATE brain_state SET live_snapshot_id = %s, pending_snapshot_id = NULL, updated_at = now() WHERE singleton", (snapshot_id,))
        return Path(r["path"]), int(r["id"]), {}

    def reset_columns(self, cols: list[int]) -> None:
        if cols:
            self.h[:, cols] = 0.0

    def observe(self, feats: np.ndarray, masks: np.ndarray, active: np.ndarray, pos_frac: np.ndarray, unreal: np.ndarray,
                held_beats: np.ndarray) -> np.ndarray:
        z = np.clip((feats.astype(np.float32) - self.obs_mean) / self.obs_std, -5, 5)
        z[~active] = 0.0
        port = np.stack([pos_frac, np.clip(unreal, -1, 5), np.log1p(held_beats)], axis=1).astype(np.float32)
        return np.concatenate([z, port], axis=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (target fraction [B], value [B], mu [B]); inactive slots → 0."""
        o = torch.tensor(obs, device=self.dev)
        out = self.policy(o, self.h)
        self.h = out.h
        a = self.policy.to_action(out.mu).cpu().numpy()
        a = np.where(active, a, 0.0).astype(np.float32)
        return a, out.value.cpu().numpy().astype(np.float32), out.mu.cpu().numpy().astype(np.float32)
