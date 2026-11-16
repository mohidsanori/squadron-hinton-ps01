"""
Mission 08 - Part 1: Systems Architect
ReWOO (Reasoning Without Observation) decoupled DAG planner, worker and token-cost profiler
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula), Subsystem 08

ReAct interleaves Thought -> Action -> Observation and re-reads the whole growing prefix at every
step, so K tool calls cost  C_ReAct(K) = K*L0 + K(K-1)/2 * (L_step + L_obs)  input tokens: O(K^2).
ReWOO (Xu et al., 2023) separates the work into three stages:

    Planner   one LLM call emits the whole plan:  #E1 = Tool[arg], #E2 = Tool[... #E1 ...], ...
    Worker    runs the tools in dependency order, substituting #E variables with their outputs
    Solver    one LLM call turns the collected evidence into the answer

    C_ReWOO(K) = 2*L0 + K*(L_step + L_obs)                                     : O(K)

Steps whose arguments do not mention each other's variables form one topological level and can
run in parallel. If a tool reports an error (an "ERR_" output or an exception) the worker asks a
fallback re-planner for a replacement; steps that depend on an unrecoverable step are skipped
instead of being fed error text.

Usage:
    python rewoo_planner.py
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

Step = Tuple[str, str, str, str]          # (var_name, plan_desc, tool_name, raw_arg)


@dataclass
class PlanStep:
    """A single symbolic node #E_i = ToolName[raw_arg] in a ReWOO DAG."""
    var_name: str
    plan_desc: str
    tool_name: str
    raw_arg: str
    dependencies: List[str] = field(default_factory=list)


class ReWOOEngine:
    """Planner output -> topological levels -> worker execution -> evidence for the solver."""

    VAR_PATTERN = re.compile(r"#E\d+")
    _STEP_LINE = re.compile(r"^(#E\d+)\s*=\s*(\w+)\[(.*)\]$")

    def __init__(self, tools: Dict[str, Callable[[str], str]]):
        self.tools = tools
        self.last_report: Dict[str, object] = {}

    # -- planner output -------------------------------------------------------------
    def parse_blueprint(self, blueprint_text: str) -> List[Step]:
        """
        Parse lines `Plan: <reasoning>` and `#E1 = ToolName[argument]` into
        (var_name, plan_desc, tool_name, raw_arg) tuples. A malformed `#E` line, a duplicate
        variable or a tool-step before any `Plan:` line raises ValueError.
        """
        steps: List[Step] = []
        seen: Set[str] = set()
        current_plan = ""
        for number, line in enumerate(blueprint_text.strip().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Plan:"):
                current_plan = line[len("Plan:"):].strip()
            elif line.startswith("#E"):
                match = self._STEP_LINE.match(line)
                if not match:
                    raise ValueError(f"line {number}: malformed step {line!r}; expected '#E<n> = Tool[arg]'")
                var_name, tool_name, raw_arg = match.groups()
                if var_name in seen:
                    raise ValueError(f"line {number}: variable {var_name} defined twice")
                seen.add(var_name)
                steps.append((var_name, current_plan, tool_name, raw_arg))
        return steps

    def to_plan_steps(self, steps: List[Step]) -> List[PlanStep]:
        """Structured PlanStep nodes with their #E dependencies extracted."""
        nodes: List[PlanStep] = []
        for var_name, plan_desc, tool_name, raw_arg in steps:
            deps = sorted(set(self.VAR_PATTERN.findall(raw_arg)), key=lambda v: int(v[2:]))
            nodes.append(PlanStep(var_name, plan_desc, tool_name, raw_arg, deps))
        return nodes

    def build_dependency_levels(self, steps: List[Step]) -> List[List[str]]:
        """
        Kahn-style grouping into topological levels; every variable in a level depends only on
        earlier levels. Raises ValueError for a cycle (including a self-reference) or for a
        reference to a variable the plan never defines.
        """
        defined = {s[0] for s in steps}
        remaining: Dict[str, Set[str]] = {}
        order: Dict[str, int] = {}
        for idx, (var_name, _, _, raw_arg) in enumerate(steps):
            deps = set(self.VAR_PATTERN.findall(raw_arg))
            undefined = deps - defined
            if undefined:
                raise ValueError(f"{var_name} references undefined variable(s) {sorted(undefined)}")
            remaining[var_name] = deps
            order[var_name] = idx

        resolved: Set[str] = set()
        levels: List[List[str]] = []
        while remaining:
            ready = sorted((v for v, deps in remaining.items() if deps <= resolved), key=order.get)
            if not ready:
                raise ValueError(f"Cyclic dependency in ReWOO DAG among {sorted(remaining)}")
            levels.append(ready)
            for var in ready:
                resolved.add(var)
                del remaining[var]
        return levels

    # -- worker ---------------------------------------------------------------------
    def _substitute(self, raw_arg: str, evidence: Dict[str, str]) -> str:
        """Replace every #E<n> in one pass (so #E1 never clobbers #E10 and outputs are not re-scanned)."""
        return self.VAR_PATTERN.sub(lambda m: evidence[m.group(0)], raw_arg)

    def _run_tool(self, tool_name: str, arg: str) -> str:
        try:
            return str(self.tools[tool_name](arg))
        except Exception as exc:  # noqa: BLE001 - a tool crash is a runtime fault like any other
            return f"ERR_TOOL_EXCEPTION: {type(exc).__name__}: {exc}"

    def execute_dag(
        self,
        steps: List[Step],
        fallback_replan_fn: Optional[Callable[[str, str], str]] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, str]:
        """
        Run the plan level by level and return {#E_i: output}. With `parallel=True` the tools of
        one level run in a thread pool (results are still stored in plan order).

        If an output contains "ERR_" and `fallback_replan_fn` is given, it is called as
        fallback_replan_fn(var_name, error_output) and its return value replaces the output.
        A step that is still in error after that is recorded as failed, and every step that
        depends on a failed or skipped step is skipped. Details are in `self.last_report`.
        """
        lookup = {s[0]: s for s in steps}
        unknown = sorted({s[2] for s in steps} - set(self.tools))
        if unknown:
            raise KeyError(f"Unknown tool(s) referenced in plan: {unknown}")
        levels = self.build_dependency_levels(steps)

        evidence: Dict[str, str] = {}
        failed: List[str] = []
        skipped: List[str] = []
        replanned: List[str] = []
        tool_calls = 0

        for level in levels:
            jobs: List[Tuple[str, str, str]] = []
            for var in level:
                _, _, tool_name, raw_arg = lookup[var]
                deps = set(self.VAR_PATTERN.findall(raw_arg))
                bad = sorted(d for d in deps if d in failed or d in skipped)
                if bad:
                    evidence[var] = f"SKIPPED_DEPENDENCY_FAILED({','.join(bad)})"
                    skipped.append(var)
                    continue
                jobs.append((var, tool_name, self._substitute(raw_arg, evidence)))

            if parallel and len(jobs) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    outputs = list(pool.map(lambda j: self._run_tool(j[1], j[2]), jobs))
            else:
                outputs = [self._run_tool(tool, arg) for _, tool, arg in jobs]
            tool_calls += len(jobs)

            for (var, _, _), output in zip(jobs, outputs):
                if "ERR_" in output and fallback_replan_fn is not None:
                    output = str(fallback_replan_fn(var, output))
                    replanned.append(var)
                evidence[var] = output
                if "ERR_" in output:
                    failed.append(var)

        # keep plan order in the returned dictionary
        evidence = {s[0]: evidence[s[0]] for s in steps}
        self.last_report = {"levels": levels, "tool_calls": tool_calls, "failed": failed,
                            "skipped": skipped, "replanned": replanned}
        return evidence

    def solve(self, blueprint_text: str, solver_fn: Callable[[Dict[str, str]], str], **execute_kwargs) -> str:
        """Planner output -> worker -> solver in one call (the solver is the second LLM call)."""
        steps = self.parse_blueprint(blueprint_text)
        return solver_fn(self.execute_dag(steps, **execute_kwargs))

    # -- cost model -----------------------------------------------------------------
    @staticmethod
    def compare_token_cost(k_steps: int, base_tokens: int, step_tokens: int, obs_tokens: int) -> Dict[str, float]:
        """
        Cumulative input tokens for K tool calls:
          C_ReAct = sum_{t=1..K} (L0 + (t-1)(L_step + L_obs)) = K*L0 + K(K-1)/2 * (L_step + L_obs)
          C_ReWOO = 2*L0 + K*(L_step + L_obs)
        Returns react_tokens, rewoo_tokens, savings_pct (negative when ReWOO costs more) and speedup_ratio.
        """
        if min(k_steps, base_tokens, step_tokens, obs_tokens) < 0:
            raise ValueError("k_steps and token counts must be non-negative")
        growth = step_tokens + obs_tokens
        react_tokens = sum(base_tokens + (t - 1) * growth for t in range(1, k_steps + 1))
        rewoo_tokens = 2 * base_tokens + k_steps * growth
        savings_pct = (react_tokens - rewoo_tokens) / react_tokens * 100.0 if react_tokens > 0 else 0.0
        speedup_ratio = react_tokens / rewoo_tokens if rewoo_tokens > 0 else 1.0
        return {
            "react_tokens": float(react_tokens),
            "rewoo_tokens": float(rewoo_tokens),
            "savings_pct": round(savings_pct, 2),
            "speedup_ratio": round(speedup_ratio, 2),
        }

    @staticmethod
    def break_even_steps(base_tokens: int, step_tokens: int, obs_tokens: int, k_max: int = 1000) -> Optional[int]:
        """Smallest K at which ReWOO's total input tokens drop below ReAct's (None if never up to k_max)."""
        for k in range(1, k_max + 1):
            m = ReWOOEngine.compare_token_cost(k, base_tokens, step_tokens, obs_tokens)
            if m["rewoo_tokens"] < m["react_tokens"]:
                return k
        return None


# ==============================================================================
# Verification harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 1): ReWOO DAG Planner & Token Profiler Verification ===")

    BLUEPRINT = """
    Plan: Scan Cygnus X-1 hard X-ray flare flux from primary telemetry array.
    #E1 = TelemetryScan[Cygnus_X1_Corona_Sector_08]
    Plan: Measure Veil Nebula shockwave plasma density in parallel.
    #E2 = PlasmaProbe[Veil_Nebula_Shock_Front]
    Plan: Compute Hohmann escape burn vector combining X-ray flux #E1 and plasma density #E2.
    #E3 = HohmannCalc[flux=#E1, density=#E2]
    Plan: Modulate deflector shield harmonics for burn trajectory #E3.
    #E4 = ShieldModulate[vector=#E3]
    """

    mock_tools: Dict[str, Callable[[str], str]] = {
        "TelemetryScan": lambda arg: f"XRayFlux({arg})=4.2e37_erg/s",
        "PlasmaProbe": lambda arg: f"IonDensity({arg})=14.8_cm^-3",
        "HohmannCalc": lambda arg: f"DeltaV({arg})=18.4_km/s",
        "ShieldModulate": lambda arg: f"HarmonicLock({arg})=STABLE",
    }

    engine = ReWOOEngine(mock_tools)
    steps = engine.parse_blueprint(BLUEPRINT)
    levels = engine.build_dependency_levels(steps)
    evidence = engine.execute_dag(steps)

    print(f"Parsed Steps ({len(steps)}): {[s[0] for s in steps]}")
    print(f"Topological Parallel Execution Levels: {levels}")
    for k, v in evidence.items():
        print(f"  {k} -> {v}")

    assert levels == [["#E1", "#E2"], ["#E3"], ["#E4"]], f"Unexpected DAG levels: {levels}"
    assert "4.2e37_erg/s" in evidence["#E3"] and "14.8_cm^-3" in evidence["#E3"]
    assert evidence["#E4"].startswith("HarmonicLock(vector=DeltaV(")

    print("\n--- ReAct vs. ReWOO Cumulative Input Token Benchmark (L0=1200, L_step=120, L_obs=280) ---")
    for k in (3, 5, 8):
        metrics = ReWOOEngine.compare_token_cost(k_steps=k, base_tokens=1200, step_tokens=120, obs_tokens=280)
        print(
            f"  K={k} tools | ReAct={int(metrics['react_tokens']):6d} tok | "
            f"ReWOO={int(metrics['rewoo_tokens']):5d} tok | "
            f"Savings={metrics['savings_pct']:5.2f}% ({metrics['speedup_ratio']:.2f}x)"
        )
    m5 = ReWOOEngine.compare_token_cost(5, 1200, 120, 280)
    assert m5["react_tokens"] == 10000.0 and m5["rewoo_tokens"] == 4400.0

    # ---- extra checks ----------------------------------------------------------------
    # closed form equals the explicit sum, and ReWOO is linear while ReAct is quadratic
    for k in range(0, 40):
        m = ReWOOEngine.compare_token_cost(k, 1200, 120, 280)
        assert m["react_tokens"] == k * 1200 + k * (k - 1) // 2 * 400
        assert m["rewoo_tokens"] == 2 * 1200 + k * 400
    be = ReWOOEngine.break_even_steps(1200, 120, 280)
    assert be == 3 and ReWOOEngine.compare_token_cost(2, 1200, 120, 280)["savings_pct"] < 0
    print(f"ReWOO only costs fewer tokens than ReAct from K = {be} tool calls (K = 2 is {ReWOOEngine.compare_token_cost(2, 1200, 120, 280)['savings_pct']}%).")

    # parallel execution gives the same evidence as sequential
    assert engine.execute_dag(steps, parallel=True) == evidence

    # #E1 must not clobber #E10 (single-pass substitution)
    ten = {f"T{i}": (lambda a, i=i: f"v{i}") for i in range(1, 12)}
    ten["Join"] = lambda a: a
    bp = "\n".join(f"Plan: s{i}\n#E{i} = T{i}[x]" for i in range(1, 12)) + "\nPlan: join\n#E12 = Join[#E1|#E10|#E11]"
    assert ReWOOEngine(ten).execute_dag(ReWOOEngine(ten).parse_blueprint(bp))["#E12"] == "v1|v10|v11"

    # malformed plans are rejected instead of silently dropped
    for bad, why in [("Plan: x\n#E1 = Tool(arg)", "malformed"),
                     ("Plan: x\n#E1 = A[x]\n#E1 = B[y]", "twice"),
                     ("Plan: x\n#E1 = A[#E2]\nPlan: y\n#E2 = A[#E1]", "Cyclic"),
                     ("Plan: x\n#E1 = A[#E1]", "Cyclic"),
                     ("Plan: x\n#E1 = A[#E9]", "undefined")]:
        try:
            e = ReWOOEngine({"A": lambda a: a})
            e.build_dependency_levels(e.parse_blueprint(bad))
            raise SystemExit(f"expected ValueError for {bad!r}")
        except ValueError as exc:
            assert why in str(exc), (why, str(exc))
    try:
        ReWOOEngine({}).execute_dag([("#E1", "p", "Ghost", "x")])
        raise SystemExit("unknown tool must raise")
    except KeyError:
        pass

    # runtime fault: ERR_ output -> fallback re-planner repairs it; unrecoverable -> dependants skipped
    flaky = dict(mock_tools)
    flaky["TelemetryScan"] = lambda arg: "ERR_CORONA_SPIKE: sensor saturated"
    e2 = ReWOOEngine(flaky)
    out = e2.execute_dag(steps, fallback_replan_fn=lambda var, err: "XRayFlux(backup_array)=3.9e37_erg/s")
    assert "3.9e37_erg/s" in out["#E3"] and e2.last_report["replanned"] == ["#E1"] and not e2.last_report["failed"]
    out = e2.execute_dag(steps)                                   # no fallback: error stays
    assert e2.last_report["failed"] == ["#E1"] and e2.last_report["skipped"] == ["#E3", "#E4"]
    assert out["#E3"].startswith("SKIPPED_DEPENDENCY_FAILED") and e2.last_report["tool_calls"] == 2
    crashing = dict(mock_tools)
    crashing["PlasmaProbe"] = lambda arg: 1 / 0
    e3 = ReWOOEngine(crashing)
    assert "ERR_TOOL_EXCEPTION: ZeroDivisionError" in e3.execute_dag(steps)["#E2"]
    print("Fallback re-planning recovered ERR_CORONA_SPIKE; an unrecoverable fault skipped #E3 and #E4.")
    print("[OK] Part 1 (Systems Architect) verification passed.")
