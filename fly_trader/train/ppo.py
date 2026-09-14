"""PPO for the ConnectomePolicy on the offline TradingEnv (and reusable online).

Recurrent PPO with truncated backprop through the connectome: rollouts over a window of beats with all
tokens as the batch; hidden states snapshotted at chunk boundaries; updates run each chunk forward again from
its snapshot (gradient checkpointing per beat keeps memory bounded) and apply the clipped surrogate,
value loss and entropy bonus. Reward is the net P&L in percent of the max position (env.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint as ckpt

from .. import config
from ..brain.policy import ConnectomePolicy, SubConnectome
from ..db.apilog import record_event
from ..db.connection import transaction
from .dataset import Dataset
from .env import TradingEnv
from . import progress as prog

log = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    window: int = 400            # beats per rollout
    chunk: int = 16              # beats per BPTT chunk
    token_batch: int = 64        # tokens per minibatch
    epochs: int = 2
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.003
    lr_graph: float = 1e-4
    lr_heads: float = 3e-4
    max_grad_norm: float = 1.0
    k_steps: int = 4
    eval_every: int = 5
    iterations: int = 40
    subgraph: str = "central"    # central (no optic lobes) | whole
    seed: int = 0


class PPOTrainer:
    def __init__(self, ds: Dataset, cfg: PPOConfig, connectome=None, device=None):
        from ..brain.lif import Connectome
        self.ds, self.cfg = ds, cfg
        c = connectome or Connectome.load()
        self.dev = torch.device(device) if device else c.device
        self.graph = SubConnectome(c, exclude=("VISUAL",)) if cfg.subgraph == "central" else c
        self.obs_dim = ds.D + 3
        self.policy = ConnectomePolicy(self.graph, obs_dim=self.obs_dim, k_steps=cfg.k_steps, device=self.dev)
        self.opt = torch.optim.Adam(self.policy.param_groups(cfg.lr_graph, cfg.lr_heads))
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        split = int(ds.T * 0.8)
        self.train_range = (0, split)
        self.eval_range = (split, ds.T)
        self.history: list[dict] = []
        log.info("PPO: graph N=%d edges=%d obs_dim=%d params=%s train beats=%d eval beats=%d device=%s",
                 self.graph.N, self.graph.indices.shape[1], self.obs_dim, self.policy.describe()["params"], split, ds.T - split, self.dev)

    # ---- rollout ----
    def rollout(self, t_start: int, t_end: int, deterministic: bool = False, collect: bool = True) -> dict:
        env = TradingEnv(self.ds, t_start, t_end)
        M = env.M
        obs = env.reset()
        h = self.policy.init_hidden(M)
        buf = {"obs": [], "u": [], "logp": [], "value": [], "reward": [], "h0": {}}
        total = np.zeros(M, np.float64)
        self.policy.eval()
        label = "eval rollout" if deterministic else "rollout"
        with torch.no_grad():
            t = t_start
            while True:
                k = t - t_start
                if k % 50 == 0:
                    prog.update(label, k, t_end - t_start)
                if collect and k % self.cfg.chunk == 0:
                    buf["h0"][k] = h.clone()
                o = torch.tensor(obs, device=self.dev)
                out = self.policy(o, h)
                u, logp = self.policy.sample(out.mu, out.log_std, deterministic=deterministic)
                a = self.policy.to_action(u).cpu().numpy()
                obs_next, r, done, info = env.step(a)
                total += r
                if collect:
                    buf["obs"].append(o); buf["u"].append(u); buf["logp"].append(logp); buf["value"].append(out.value)
                    buf["reward"].append(torch.tensor(r, device=self.dev))
                h = out.h
                obs = obs_next
                t += 1
                if done:
                    break
        self.policy.train()
        res = {"net_sol": float(total.sum()) / 100.0 * config.MAX_POSITION_SOL, "reward_mean": float(total.mean()),
               "trades": info["trades"], "turnover": info["turnover"], "fees": info["fees"], "beats": t - t_start, "tokens": M}
        if collect:
            buf["obs"] = torch.stack(buf["obs"]); buf["u"] = torch.stack(buf["u"]); buf["logp"] = torch.stack(buf["logp"])
            buf["value"] = torch.stack(buf["value"]); buf["reward"] = torch.stack(buf["reward"])
            res["buf"] = buf
        return res

    # ---- advantages ----
    def gae(self, rewards: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, M = rewards.shape
        adv = torch.zeros_like(rewards)
        last = torch.zeros(M, device=self.dev)
        for t in reversed(range(T)):
            v_next = values[t + 1] if t + 1 < T else torch.zeros(M, device=self.dev)
            delta = rewards[t] + self.cfg.gamma * v_next - values[t]
            last = delta + self.cfg.gamma * self.cfg.lam * last
            adv[t] = last
        ret = adv + values
        return adv, ret

    # ---- update ----
    def update(self, buf: dict, adv: torch.Tensor, ret: torch.Tensor) -> dict:
        cfg = self.cfg
        T, M = adv.shape
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "clipfrac": 0.0, "n": 0}
        starts = list(range(0, T, cfg.chunk))
        total_mb = cfg.epochs * len(starts) * math.ceil(M / cfg.token_batch)
        done_mb = 0
        for _ in range(cfg.epochs):
            self.rng.shuffle(starts)
            for c0 in starts:
                c1 = min(c0 + cfg.chunk, T)
                perm = self.rng.permutation(M)
                for g in range(0, M, cfg.token_batch):
                    if prog.should_stop():
                        break
                    done_mb += 1
                    if done_mb % 5 == 0:
                        prog.update("ppo update", done_mb, total_mb, loss_pi=stats["loss_pi"] / max(stats["n"], 1), loss_v=stats["loss_v"] / max(stats["n"], 1))
                    idx = torch.tensor(perm[g:g + cfg.token_batch], device=self.dev)
                    h = buf["h0"][c0][:, idx].detach()
                    logps, values, ents = [], [], []
                    for t in range(c0, c1):
                        o = buf["obs"][t, idx]
                        def step(o_, h_):
                            out = self.policy(o_, h_)
                            return out.mu, out.value, out.h
                        mu, v, h = ckpt.checkpoint(step, o, h, use_reentrant=False)
                        logps.append(self.policy.log_prob(mu, self.policy.log_std.expand_as(mu), buf["u"][t, idx]))
                        values.append(v); ents.append(self.policy.entropy(self.policy.log_std).expand_as(mu))
                    logp_new = torch.stack(logps); v_new = torch.stack(values); ent = torch.stack(ents)
                    logp_old = buf["logp"][c0:c1, idx]
                    a = adv_n[c0:c1, idx]; r = ret[c0:c1, idx]
                    ratio = torch.exp(logp_new - logp_old)
                    loss_pi = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                    loss_v = (v_new - r).pow(2).mean()
                    loss = loss_pi + cfg.c_value * loss_v - cfg.c_entropy * ent.mean()
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    self.opt.step()
                    stats["loss_pi"] += float(loss_pi); stats["loss_v"] += float(loss_v); stats["entropy"] += float(ent.mean())
                    stats["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().mean()); stats["n"] += 1
        n = max(stats.pop("n"), 1)
        return {k: v / n for k, v in stats.items()}

    # ---- checkpoint ----
    def save(self, tag: str, metrics: dict, run_id: str | None = None) -> tuple[Path, int]:
        root = config.BRAIN_DIR / "policies"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"policy_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{tag}.pt"
        torch.save({"state_dict": self.policy.state_dict(), "obs_mean": self.ds.mean, "obs_std": self.ds.std, "obs_dim": self.obs_dim,
                    "k_steps": self.cfg.k_steps, "subgraph": self.cfg.subgraph, "describe": self.policy.describe(),
                    "connectome_sha256": getattr(self.graph, "content_sha256", None), "metrics": metrics, "cfg": self.cfg.__dict__}, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with transaction() as conn:
            row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,'policy',%s) RETURNING id",
                               (run_id, str(path), sha, json.dumps(metrics, default=str)[:900])).fetchone()
        return path, int(row["id"])

    # ---- main loop ----
    def train(self, run_id: str | None = None) -> list[dict]:
        cfg = self.cfg
        t0_all = time.time()
        best = None
        for it in range(cfg.iterations):
            if prog.should_stop():
                log.info("stop requested; ending PPO after %d iterations", it)
                break
            t_it = time.time()
            prog.update("ppo iteration", it, cfg.iterations, iteration=it, force=True)
            lo, hi = self.train_range
            start = int(self.rng.integers(lo, max(lo + 1, hi - cfg.window)))
            ro = self.rollout(start, min(start + cfg.window, hi))
            adv, ret = self.gae(ro["buf"]["reward"], ro["buf"]["value"])
            st = self.update(ro["buf"], adv, ret)
            rec = {"iter": it, "train_net_sol": ro["net_sol"], "reward_mean": ro["reward_mean"], "trades": ro["trades"], "turnover": ro["turnover"],
                   "fees": ro["fees"], **st, "log_std": float(self.policy.log_std), "secs": time.time() - t_it}
            if cfg.eval_every and (it % cfg.eval_every == cfg.eval_every - 1 or it == cfg.iterations - 1):
                ev = self.rollout(*self.eval_range, deterministic=True, collect=False)
                rec.update({"eval_net_sol": ev["net_sol"], "eval_trades": ev["trades"], "eval_fees": ev["fees"], "eval_turnover": ev["turnover"]})
                path, sid = self.save(f"it{it}", rec, run_id)
                rec["snapshot_id"] = sid
                if best is None or ev["net_sol"] > best[0]:
                    best = (ev["net_sol"], sid)
            self.history.append(rec)
            log.info("ppo %s", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            record_event("info", "ppo", f"iteration {it}", rec)
            prog.update("ppo iteration", it + 1, cfg.iterations, last_iteration=rec, force=True)
        log.info("ppo done in %.0fs; best eval net %s", time.time() - t0_all, best)
        return self.history


def main(iterations: int | None = None, window: int | None = None, dataset: str | None = None, subgraph: str | None = None,
         eval_every: int | None = None, imitate_epochs: int = 0, init_from: int | None = None, stop_event=None) -> None:
    import glob, uuid
    from ..logging_setup import setup
    from ..ops.reset import reset_training_stats
    setup("ppo")
    reset_training_stats(reason="ppo training")
    prog.clear(); prog.set_stop_event(stop_event)
    prog.update("loading dataset", force=True)
    path = dataset or sorted(glob.glob(str(config.DATA_DIR / "train" / "obs_*.npz")))[-1]
    ds = Dataset(Path(path))
    cfg = PPOConfig()
    if iterations: cfg.iterations = iterations
    if window: cfg.window = window
    if subgraph: cfg.subgraph = subgraph
    if eval_every is not None: cfg.eval_every = eval_every
    run_id = str(uuid.uuid4())
    with transaction() as conn:
        conn.execute("INSERT INTO runs (run_id, kind, config, corpus, status) VALUES (%s,'ppo',%s,%s,'running')",
                     (run_id, json.dumps({**config.summary(), "ppo": cfg.__dict__}, default=str), str(path)))
    tr = PPOTrainer(ds, cfg)
    prog.update("ready", force=True, graph=tr.policy.describe(), dataset=str(path), train_beats=tr.train_range[1], eval_beats=ds.T - tr.train_range[1], run_id=run_id)
    if init_from:
        with transaction() as conn:
            r = conn.execute("SELECT path FROM brain_snapshots WHERE id = %s", (init_from,)).fetchone()
        ck = torch.load(r["path"], map_location="cpu", weights_only=False)
        tr.policy.load_state_dict(ck["state_dict"])
        log.info("initialised from snapshot %s", init_from)
    if imitate_epochs:
        from .imitation import imitate
        im = imitate(tr, epochs=imitate_epochs)
        path, sid = tr.save("imitation", {"stage": "imitation", **im["history"][-1]}, run_id)
        record_event("info", "ppo", "imitation done", {**im["history"][-1], "snapshot_id": sid})
        log.info("imitation checkpoint %s (%s)", sid, path)
    hist = tr.train(run_id) if cfg.iterations > 0 else []
    with transaction() as conn:
        conn.execute("UPDATE runs SET ended_at = now(), status = %s, metrics = %s WHERE run_id = %s",
                     ("stopped" if prog.should_stop() else "done", json.dumps({"history": hist}, default=str), run_id))
    prog.update("stopped" if prog.should_stop() else "done", force=True)
    print(json.dumps(hist[-1], indent=1, default=str))
