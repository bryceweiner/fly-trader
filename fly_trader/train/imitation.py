"""Imitation warm-start: train the ConnectomePolicy to reproduce the expert's target exposures on the training
window (binary cross-entropy on σ(mu) vs expert target, truncated BPTT through the connectome), FlyGM stage 1."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import torch.utils.checkpoint as ckpt

from .. import config
from .expert import run_expert
from . import progress as prog
from ..db.apilog import record_event

log = logging.getLogger(__name__)


def dagger_rollout(trainer, t_start: int, t_end: int, beta: float) -> dict:
    """Roll the env with a mixture of expert (prob beta) and student (deterministic) actions; label every
    visited state with the expert's action computed on that state (DAgger, Ross et al. 2011)."""
    from .env import TradingEnv
    from .expert import ExpertPolicy
    ds, pol, dev = trainer.ds, trainer.policy, trainer.dev
    env = TradingEnv(ds, t_start, t_end)
    obs = env.reset()
    ex = ExpertPolicy(env)
    h = pol.init_hidden(env.M)
    O, A, total = [], [], np.zeros(env.M)
    rng = np.random.default_rng(1)
    pol.eval()
    with torch.no_grad():
        while True:
            a_exp = ex.act()
            out = pol(torch.tensor(obs, device=dev), h); h = out.h
            a_stu = pol.to_action(out.mu).cpu().numpy()
            a_stu = np.where(env.tradable(), a_stu, 0.0)
            use_exp = rng.random(env.M) < beta
            a = np.where(use_exp, a_exp, a_stu)
            O.append(obs.astype(np.float32)); A.append(a_exp.astype(np.float32))
            obs, r, done, info = env.step(a)
            ex.after_step()
            total += r
            if done:
                break
    pol.train()
    return {"obs": np.stack(O), "act": np.stack(A), "net_sol": float(total.sum()) / 100.0 * config.MAX_POSITION_SOL, "trades": info["trades"]}


def imitate(trainer, epochs: int = 3, chunk: int = 16, token_batch: int = 64, lr: float = 3e-4, dagger: bool = True,
            chunk_frac: float = 0.5) -> dict:
    ds, pol, dev = trainer.ds, trainer.policy, trainer.dev
    lo, hi = trainer.train_range
    prog.update("expert rollout (train window)", force=True)
    ex = run_expert(ds, lo, hi, collect=True)
    log.info("expert on train window: net %+.3f SOL, %d trades, fees %.3f", ex["net_sol"], ex["trades"], ex["fees"])
    prog.update("expert rollout (eval window)", force=True, expert_train_net=ex["net_sol"], expert_train_trades=ex["trades"])
    ev = run_expert(ds, *trainer.eval_range, collect=False)
    log.info("expert on eval window: net %+.3f SOL, %d trades", ev["net_sol"], ev["trades"])
    O = torch.tensor(ex["obs"], device=dev); A = torch.tensor(ex["act"], device=dev)   # [T, M, obs], [T, M]
    T, M = A.shape
    y = (A > 0.05).float()
    opt = torch.optim.Adam(pol.param_groups(lr * 0.3, lr))
    rng = np.random.default_rng(0)
    pol.train()
    hist = []
    for ep in range(epochs):
        if prog.should_stop():
            break
        if dagger and ep > 0:   # DAgger: aggregate expert labels on the student's own trajectories
            beta = max(0.0, 1.0 - ep / max(epochs - 1, 1))
            prog.update(f"dagger rollout (epoch {ep}, beta {beta:.2f})", force=True)
            dg = dagger_rollout(trainer, lo, hi, beta)
            O = torch.cat([O, torch.tensor(dg["obs"], device=dev)], dim=1)
            A = torch.cat([A, torch.tensor(dg["act"], device=dev)], dim=1)
            y = (A > 0.05).float()
            T, M = A.shape
            log.info("dagger round %d (beta %.2f): mixture rollout net %+.3f SOL, %d trades; dataset now %d x %d", ep, beta, dg["net_sol"], dg["trades"], T, M)
        losses, accs, n = 0.0, 0.0, 0
        starts = list(range(0, T, chunk)); rng.shuffle(starts)
        starts = starts[: max(1, int(len(starts) * chunk_frac))]   # a random half of the chunks per epoch keeps epochs short
        h_all = pol.init_hidden(M)
        total_mb = len(starts) * int(np.ceil(M / token_batch)); done_mb = 0
        # transitions (the expert entering or exiting) are rare and expensive to miss: weight them up
        trans = torch.zeros_like(y); trans[1:] = (y[1:] != y[:-1]).float()
        for c0 in starts:
            c1 = min(c0 + chunk, T)
            perm = rng.permutation(M)
            for g in range(0, M, token_batch):
                if prog.should_stop():
                    break
                done_mb += 1
                if done_mb % 5 == 0:
                    prog.update(f"imitation epoch {ep + 1}/{epochs}", done_mb, total_mb, bce=losses / max(n, 1), acc=accs / max(n, 1))
                idx = torch.tensor(perm[g:g + token_batch], device=dev)
                h = h_all[:, idx].detach()
                mus = []
                for t in range(c0, c1):
                    def step(o_, h_):
                        out = pol(o_, h_)
                        return out.mu, out.h
                    mu, h = ckpt.checkpoint(step, O[t, idx], h, use_reentrant=False)
                    mus.append(mu)
                mu = torch.stack(mus)
                yy = y[c0:c1, idx]
                pos_w = (yy.numel() / max(yy.sum().item(), 1.0)) ** 0.5   # mild rebalancing: expert is flat most of the time
                w = 1.0 + 9.0 * trans[c0:c1, idx]                            # 10x on entry/exit beats
                loss = (Fn.binary_cross_entropy_with_logits(mu, yy, pos_weight=torch.tensor(pos_w, device=dev), reduction="none") * w).sum() / w.sum()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
                losses += float(loss); accs += float(((mu > 0).float() == yy).float().mean()); n += 1
        rec = {"epoch": ep, "bce": losses / max(n, 1), "acc": accs / max(n, 1), "expert_train_net": ex["net_sol"], "expert_eval_net": ev["net_sol"]}
        prog.update(f"imitation eval (epoch {ep + 1})", force=True)
        evr = trainer.rollout(*trainer.eval_range, deterministic=True, collect=False)
        rec.update({"student_eval_net": evr["net_sol"], "student_eval_trades": evr["trades"]})
        hist.append(rec)
        log.info("imitation %s", json.dumps(rec))
        record_event("info", "ppo", f"imitation epoch {ep}", rec)
        prog.update(f"imitation epoch {ep + 1} done", force=True, last_imitation=rec)
    pol.eval()
    return {"history": hist, "expert_train_net": ex["net_sol"], "expert_eval_net": ev["net_sol"]}
