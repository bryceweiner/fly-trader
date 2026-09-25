"""Mutation test for FlyVault.sol: apply one mutation at a time (flipped comparisons and arithmetic, dropped
modifiers, swapped roles and states, deleted statements), run the forge suite, print one JSON line per mutant.
A mutant the suite still passes is "SURVIVED". Declarations whose deletion does not compile are "nocompile".

It edits src/FlyVault.sol in place, so run it on a copy (one per worker; restores the file when done):

    rsync -a --exclude out --exclude cache contracts/ /tmp/mut0/
    python3 contracts/test/mutation/mutate.py /tmp/mut0 0 1 > mutants.jsonl     # worker 0 of 1

2026-09-25: 115 mutants; 84 killed, 29 nocompile, 2 survived (deleting __AccessControl_init / __Pausable_init,
which are empty in OZ 5.7, so those two are equivalent mutants).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
wi, nw = int(sys.argv[2]), int(sys.argv[3])
src = root / "src/FlyVault.sol"
orig = src.read_text()
lines = orig.split("\n")

# body only: from `contract FlyVault` to the end
start = next(i for i, l in enumerate(lines) if l.startswith("contract FlyVault"))

OPS = [
    (r" < ", " <= "), (r" <= ", " < "), (r" > ", " >= "), (r" >= ", " > "),
    (r" != ", " == "), (r" == ", " != "),
    (r" \+= ", " -= "), (r" -= ", " += "), (r" \+ ", " - "), (r" - ", " + "),
    (r"\+\+\$", "$"),
    (r" whenNotPaused", ""), (r" nonReentrant", ""), (r" onlyRole\(PAUSER_ROLE\)", ""),
    (r" onlyRole\(UPGRADER_ROLE\)", ""), (r"RequestState\.Cancelled", "RequestState.Pending"),
    (r"RequestState\.Withdrawn", "RequestState.Pending"), (r"_grantRole\(UPGRADER_ROLE, timelock\)", "_grantRole(UPGRADER_ROLE, pauser)"),
    (r"_grantRole\(DEFAULT_ADMIN_ROLE, timelock\)", "_grantRole(DEFAULT_ADMIN_ROLE, pauser)"),
    (r"_grantRole\(PAUSER_ROLE, pauser\)", "_grantRole(PAUSER_ROLE, timelock)"),
    (r"safeTransfer\(user, amount\)", "safeTransfer(user, amount + 1)"),
    (r"lockedAfter = available - amount", "lockedAfter = available"),
    (r"\|\| ", "&& "),
    (r"_pause\(\)", "_unpause()"), (r"_unpause\(\)", "_pause()"),
    (r"block\.timestamp \+ WITHDRAW_DELAY", "block.timestamp"),
    (r"readyAt: readyAt", "readyAt: 0"),
    (r"user: user", "user: address(0)"),
]

mutants = []
for i in range(start, len(lines)):
    l = lines[i]
    s = l.strip()
    if not s or s.startswith("//") or s.startswith("///") or s.startswith("*") or s.startswith("event ") or s.startswith("error "):
        continue
    for pat, rep in OPS:
        for m in re.finditer(pat, l):
            new = l[: m.start()] + re.sub(pat, rep, m.group(0)) + l[m.end():]
            if new != l:
                mutants.append((i, new, f"{pat!r}->{rep!r}"))
    # statement deletion
    if s.endswith(";") and not s.startswith("return") and "=" in s or s.startswith(("emit ", "$.", "fly.", "_grantRole", "__", "if (", "r.state", "$.token")):
        if s.endswith(";"):
            mutants.append((i, re.sub(r"\S.*", "// deleted", l), "delete"))

mine = [m for k, m in enumerate(mutants) if k % nw == wi]
env = {**os.environ, "FOUNDRY_FUZZ_RUNS": "128", "FOUNDRY_INVARIANT_RUNS": "24", "FOUNDRY_INVARIANT_DEPTH": "64"}
results = []
try:
    for i, new, what in mine:
        mutated = lines.copy()
        mutated[i] = new
        src.write_text("\n".join(mutated))
        r = subprocess.run(["forge", "test", "--no-match-contract", "ForkTest"], cwd=root, env=env, capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        if "Compiler run failed" in out or "Error (" in out and "failed" in out and "Suite result" not in out:
            status = "nocompile"
        elif r.returncode != 0:
            status = "killed"
        else:
            status = "SURVIVED"
        results.append({"line": i + 1, "what": what, "code": new.strip(), "status": status})
        print(json.dumps(results[-1]), flush=True)
finally:
    src.write_text(orig)
