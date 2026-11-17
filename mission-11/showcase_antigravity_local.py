"""
Mission 11 - Part 5, Track B, local run: the instructor's Antigravity workflow against a local runtime
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

`andromeda_branch_swarm.py` (the instructor's Track B script) is loaded unchanged. Google Antigravity
supplies `phase`, `agent` and `parallel` at run time; here a small local runtime supplies them instead:

  phase(name)                    prints a phase header
  agent(prompt, schema, role, workspace)
                                 a role-based worker. The "language model" is replaced by deterministic
                                 Python: the Lead Architect lists the four station modules, each Bridge
                                 Station Verifier really runs its module, and the Security Sentinel reads the
                                 reports. `workspace="branch"` copies the mission folder into a fresh
                                 temporary directory, so each verifier works on an isolated clone.
  parallel(items, fn)            runs fn(item) for every item concurrently and keeps the input order
  JSON-Schema `finish` contract  every agent result is validated against the schema given to `agent`; a result that
                                 does not match is rejected

What is real: the module executions (in separate subprocesses, in isolated copies), the isolation check (the
original folder is hashed before and after) and the schema validation. What is local: the runtime and the roles.
This is NOT Antigravity: it produces no Subagents Auxiliary Pane. To produce that, run
`andromeda_branch_swarm.py` with Antigravity's `run_workflow` tool (or `/teamwork`) and capture the pane.

Usage:
    python showcase_antigravity_local.py
"""

import ast
import asyncio
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
STATIONS = {
    "swarm_state_graph_architect.py": ("Amdahl Speedup", "reducers merge 3 parallel branches without losing an update; S = T_seq / T_crit"),
    "handoff_contract_protocol.py": ("reduction=", "the scoped context never contains chat history or the sensor dump; hops above max_hops are refused"),
    "blackboard_debate_consensus.py": ("P_maj(N=3", "a stale writer gets CONCURRENCY_CONFLICT; the log-odds quorum overrules two correlated clones"),
    "deadlock_byzantine_sentinel.py": ("Ship Anomaly 11 Resolved", "the PayloadBuilder -> Astrogation edge is pre-empted and the wait-for graph is acyclic"),
}
RUN_LOG: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# JSON-Schema subset used by the finish contracts
# ---------------------------------------------------------------------------
def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    errors: List[str] = []
    types = {"string": str, "boolean": bool, "array": list, "object": dict, "number": (int, float), "integer": int}
    t = schema.get("type")
    if t and not isinstance(instance, types[t]) or (t in ("number", "integer") and isinstance(instance, bool)):
        return [f"{path}: expected {t}, got {type(instance).__name__}"]
    if t == "object":
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required field '{req}'")
        for k, sub in schema.get("properties", {}).items():
            if k in instance:
                errors += validate(instance[k], sub, f"{path}.{k}")
    if t == "array" and "items" in schema:
        for i, item in enumerate(instance):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


# ---------------------------------------------------------------------------
# the local runtime
# ---------------------------------------------------------------------------
def tree_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def run_module_in_branch(module: str) -> Dict[str, Any]:
    """Copy the mission folder into a fresh temp directory (the isolated branch) and run one module there."""
    branch = Path(tempfile.mkdtemp(prefix="branch_"))
    try:
        for p in HERE.glob("*.py"):
            shutil.copy2(p, branch / p.name)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, module], cwd=branch, capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        seconds = time.perf_counter() - t0
        out = proc.stdout.strip().splitlines()
        marker = STATIONS[module][0]
        metric = next((l.strip() for l in out if marker in l), out[-1] if out else proc.stderr.strip()[-120:])
        passed = proc.returncode == 0 and any("[PASS]" in l for l in out)
        RUN_LOG.append({"module": module, "branch": branch.name, "seconds": round(seconds, 2), "status": "PASS" if passed else "FAIL"})
        return {"module": module, "status": "PASS" if passed else "FAIL", "key_metric": metric[:160]}
    finally:
        shutil.rmtree(branch, ignore_errors=True)


class AgentHandle:
    """Result of `agent(...)`: awaitable for a one-shot call, callable for a stateful session (the Sentinel)."""

    def __init__(self, prompt: str, schema, role: str, workspace):
        self.prompt, self.schema, self.role, self.workspace = prompt, schema, role, workspace

    async def _run(self, prompt: str, schema) -> Any:
        if self.role == "Lead Systems Architect":
            result: Any = [f"{m}: {inv}" for m, (_, inv) in STATIONS.items()]
        elif self.role == "Bridge Station Verifier":
            module = next(m for m in STATIONS if m in prompt)
            assert self.workspace == "branch", "verifiers must run in an isolated branch workspace"
            result = await asyncio.to_thread(run_module_in_branch, module)
        elif self.role == "Security Sentinel":
            reports = ast.literal_eval(re.search(r"(\[\{.*\}\])", prompt, re.S).group(1))
            ok = all(r["status"] == "PASS" for r in reports) and len(reports) == len(STATIONS)
            result = {"approved": ok, "summary": f"{sum(r['status'] == 'PASS' for r in reports)}/{len(reports)} station modules passed their self-checks in isolated branches"}
        else:
            raise ValueError(f"unknown role {self.role!r}")
        if schema is not None:
            problems = validate(result, schema)
            if problems:
                raise ValueError(f"finish contract violated by {self.role}: {problems}")
        return result

    def __await__(self):
        return self._run(self.prompt, self.schema).__await__()

    def __call__(self, prompt: str, schema=None):
        return self._run(prompt, schema or self.schema)


def make_runtime() -> Dict[str, Any]:
    def phase(name: str) -> None:
        print(f"\n=== phase: {name} ===")

    def agent(prompt: str, schema=None, role: str = "", workspace=None, model: str = ""):
        return AgentHandle(prompt, schema, role, workspace)

    async def parallel(items, fn):
        return list(await asyncio.gather(*(fn(i) for i in items)))
    return {"phase": phase, "agent": agent, "parallel": parallel}


def load_instructor_script():
    spec = importlib.util.spec_from_file_location("andromeda_branch_swarm", HERE / "andromeda_branch_swarm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def main() -> int:
    before = tree_hash(HERE)
    mod = load_instructor_script()
    for name, fn in make_runtime().items():
        setattr(mod, name, fn)
    print(f"Local runtime for the instructor's Track B script | python {sys.version.split()[0]} | roles are deterministic Python, modules run for real")
    t0 = time.perf_counter()
    await mod.run_andromeda_workflow()
    wall = time.perf_counter() - t0

    print("\n--- branch execution log ---")
    for r in RUN_LOG:
        print(f"  {r['module']:<34} branch={r['branch']:<16} {r['seconds']:>5.2f}s  {r['status']}")
    seq = sum(r["seconds"] for r in RUN_LOG)
    print(f"  4 verifiers ran concurrently: {wall:.2f}s wall clock against {seq:.2f}s if run one after another")

    print("\n--- checks ---")
    assert len(RUN_LOG) == 4 and all(r["status"] == "PASS" for r in RUN_LOG)
    assert len({r["branch"] for r in RUN_LOG}) == 4, "each verifier must get its own branch"
    assert tree_hash(HERE) == before, "the original folder must be untouched by the branch runs"
    assert wall < seq
    bad = validate({"module": "x", "status": 1}, {"type": "object", "properties": {"module": {"type": "string"}, "status": {"type": "string"}}, "required": ["module", "status", "key_metric"]})
    assert len(bad) == 2                                          # wrong type and missing field are both rejected
    print("4 branches, 4 distinct isolated directories, original folder hash unchanged, finish contracts validated.")
    print("[PASS] Part 5 Track B (local runtime): the instructor's Antigravity script executed against the real modules.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
