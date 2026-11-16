"""
Mission 08 - Part 2: Communications & Protocol Officer
Three-actor Reflexion loop (verbal reinforcement learning) with a traceback surgeon
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula), Subsystem 08

Reflexion (Shinn et al., NeurIPS 2023) improves an agent without weight updates:

    Actor M_a            writes a candidate patch, conditioned on the reflections in memory
    Evaluator M_e        runs deterministic unit tests: an external oracle, not the model's opinion
    Self-reflection M_sr turns the failure traceback into a short critique r_k for trial k+1

    M_ep^(k+1) = Tail_Omega( M_ep^(k) + [r_k] ),   Omega = 3

Huang et al. (2023) showed that a model asked to correct itself with no external signal does not
reliably improve; here every reflection is grounded in the oracle's traceback. Two guards make
the loop halt-able: a hard `max_trials`, and a SHA-256 hash of each whitespace-normalised patch,
so an actor that proposes a patch it has already tried is stopped instead of burning trials.
An optional `patch_guard` (the AST sandbox from Part 4) can veto a patch before it is run.

Usage:
    python reflexion_surgeon.py
"""

import hashlib
import re
import traceback
from typing import Callable, Dict, List, Optional, Set, Tuple


class TracebackParser:
    """Extract (file, line_number, exception_type, fault_summary) from a Python traceback."""

    LINE_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')
    EXC_PATTERN = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Iteration))(?::\s*(.*))?$")

    @classmethod
    def parse(cls, stderr_text: str) -> Dict[str, str]:
        file_name, line_number = "unknown.py", "-1"
        exception_type, fault_summary = "RuntimeError", stderr_text.strip()
        for ln in (l.strip() for l in stderr_text.strip().splitlines() if l.strip()):
            m_loc = cls.LINE_PATTERN.search(ln)
            if m_loc:                                  # the last frame is the innermost one
                file_name, line_number = m_loc.group(1), m_loc.group(2)
            m_exc = cls.EXC_PATTERN.match(ln)
            if m_exc:
                exception_type, fault_summary = m_exc.group(1), (m_exc.group(2) or "")
        return {"file": file_name, "line_number": line_number,
                "exception_type": exception_type, "fault_summary": fault_summary}


def normalize_code(code: str) -> str:
    """Strip each line and drop blank lines, so re-indented or re-spaced copies hash the same."""
    return "\n".join(line.strip() for line in code.strip().splitlines() if line.strip())


class ReflexionSurgeon:
    """Actor / Evaluator / Self-reflection loop grounded in a unit-test oracle."""

    def __init__(self, max_trials: int = 4, buffer_cap: int = 3):
        if max_trials < 1 or buffer_cap < 1:
            raise ValueError("max_trials and buffer_cap must be >= 1")
        self.max_trials = max_trials
        self.buffer_cap = buffer_cap
        self.episodic_buffer: List[str] = []
        self.patch_hashes: Set[str] = set()

    def parse_traceback(self, stderr_text: str) -> Dict[str, str]:
        return TracebackParser.parse(stderr_text)

    def generate_verbal_reflection(self, diag: Dict[str, str], failed_code: str) -> str:
        """
        Reflection r_k: root cause plus a concrete constraint for trial k+1. The offending source
        line (looked up from the traceback's line number) is quoted. The reflection is appended to
        the episodic buffer, which keeps only the newest `buffer_cap` entries.
        """
        exc = diag.get("exception_type", "Exception")
        line_no = diag.get("line_number", "?")
        summary = diag.get("fault_summary", "")

        if exc == "ZeroDivisionError":
            prescription = "Guard denominator against zero before dividing telemetry flux."
        elif exc == "KeyError":
            prescription = f"Use dict.get() or verify key {summary} exists in telemetry packet."
        elif exc == "IndexError":
            prescription = "Check array bounds before indexing shockwave sample buffer."
        elif exc == "TypeError":
            prescription = "Validate argument types before the operation."
        elif exc == "SandboxViolation":
            prescription = "Remove the forbidden construct; patches may use only pure computation."
        else:
            prescription = f"Resolve {exc} ({summary}) with explicit input validation."

        offending = ""
        if line_no.lstrip("-").isdigit() and 1 <= int(line_no) <= len(failed_code.splitlines()):
            offending = f" Offending line: `{failed_code.splitlines()[int(line_no) - 1].strip()}`."

        reflection = (f"[Reflection Line {line_no} | {exc}]: Fault '{summary}'.{offending} "
                      f"Constraint for next trial: {prescription}")
        self.episodic_buffer.append(reflection)
        if len(self.episodic_buffer) > self.buffer_cap:
            self.episodic_buffer.pop(0)
        return reflection

    def is_duplicate_patch(self, candidate_code: str) -> bool:
        """True if the whitespace-normalised patch was seen before; otherwise record its SHA-256 and return False."""
        digest = hashlib.sha256(normalize_code(candidate_code).encode("utf-8")).hexdigest()
        if digest in self.patch_hashes:
            return True
        self.patch_hashes.add(digest)
        return False

    def run_repair_loop(
        self,
        initial_code: str,
        evaluator_fn: Callable[[str], Tuple[bool, str]],
        actor_patch_fn: Callable[[str, List[str]], str],
        patch_guard: Optional[Callable[[str], Tuple[bool, List[str]]]] = None,
        reflection_gate: Optional[Callable[[str], Tuple[bool, str]]] = None,
    ) -> Dict[str, object]:
        """
        Up to `max_trials` rounds of evaluate -> reflect -> patch.

        evaluator_fn(code) -> (passed, stderr)         the oracle; exceptions inside it count as failures
        actor_patch_fn(code, reflections) -> new code  the actor; sees the bounded episodic buffer
        patch_guard(code) -> (is_safe, violations)     optional sandbox; an unsafe patch is not executed
        reflection_gate(text) -> (allowed, reason)     optional memory-provenance audit; a rejected
                                                       reflection is not added to the episodic buffer

        Returns a report with success, trials_used, final_code, episodic_buffer, history and, when
        the loop stopped early, `halt_reason` ("DUPLICATE_PATCH_CYCLE" or "ACTOR_ERROR").
        """
        self.is_duplicate_patch(initial_code)            # register the starting code
        current_code = initial_code
        history: List[Dict[str, object]] = []

        def report(success: bool, trials: int, halt: Optional[str] = None) -> Dict[str, object]:
            out: Dict[str, object] = {"success": success, "trials_used": trials, "final_code": current_code,
                                      "episodic_buffer": list(self.episodic_buffer), "history": history}
            if halt:
                out["halt_reason"] = halt
            return out

        for trial in range(1, self.max_trials + 1):
            safe, violations = (True, []) if patch_guard is None else patch_guard(current_code)
            if not safe:
                passed, stderr = False, ("Traceback (most recent call last):\n"
                                         '  File "patch.py", line 1, in <module>\n'
                                         f"SandboxViolation: {'; '.join(violations)}")
            else:
                try:
                    passed, stderr = evaluator_fn(current_code)
                except Exception:  # noqa: BLE001 - a crashing oracle is a failed trial, not a crashed loop
                    passed, stderr = False, traceback.format_exc()

            if passed:
                history.append({"trial": trial, "passed": True, "duplicate": False, "reflection": "PASS"})
                return report(True, trial)

            diag = self.parse_traceback(stderr)
            reflection = self.generate_verbal_reflection(diag, current_code)
            entry: Dict[str, object] = {"trial": trial, "passed": False, "duplicate": False,
                                        "exception": diag["exception_type"], "line": diag["line_number"],
                                        "reflection": reflection, "blocked_by_sandbox": not safe}
            if reflection_gate is not None:
                allowed, reason = reflection_gate(reflection)
                entry["memory_audit"] = reason
                if not allowed:
                    self.episodic_buffer.remove(reflection)
            history.append(entry)

            if trial == self.max_trials:
                break
            try:
                next_code = actor_patch_fn(current_code, list(self.episodic_buffer))
            except Exception as exc:  # noqa: BLE001
                history.append({"trial": trial, "passed": False, "halt": f"actor raised {type(exc).__name__}: {exc}"})
                return report(False, trial, "ACTOR_ERROR")
            if self.is_duplicate_patch(next_code):       # cycle: the actor repeats an earlier patch
                history.append({"trial": trial + 1, "passed": False, "duplicate": True,
                                "reflection": "DUPLICATE_PATCH_CYCLE: actor re-proposed an earlier patch; halting"})
                return report(False, trial, "DUPLICATE_PATCH_CYCLE")
            current_code = next_code

        return report(False, self.max_trials)


# ==============================================================================
# Verification harness for Part 2 (Communications & Protocol Officer)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 2): Reflexion Surgeon & Traceback Parser Verification ===")

    surgeon = ReflexionSurgeon(max_trials=4, buffer_cap=3)

    BUGGY_CODE_V1 = """
def normalize_xray_flux(packet):
    return packet['flux'] / packet['window_sec']
"""

    PATCH_V2 = """
def normalize_xray_flux(packet):
    if packet['window_sec'] == 0:
        return 0.0
    return packet['flux'] / packet['window_sec']
"""

    PATCH_V3 = """
def normalize_xray_flux(packet):
    flux = packet.get('flux', 0.0)
    window = packet.get('window_sec', 1.0)
    if window == 0:
        return 0.0
    return flux / window
"""

    def mock_unit_test_evaluator(code_str: str) -> Tuple[bool, str]:
        if "packet.get('flux'" not in code_str and "if packet['window_sec'] == 0" not in code_str:
            return (False, 'Traceback (most recent call last):\n'
                           '  File "telemetry_parser.py", line 3, in normalize_xray_flux\n'
                           'ZeroDivisionError: division by zero when window_sec=0')
        if "packet.get('flux'" not in code_str:
            return (False, 'Traceback (most recent call last):\n'
                           '  File "telemetry_parser.py", line 5, in normalize_xray_flux\n'
                           "KeyError: 'flux'")
        return True, ""

    def mock_actor(curr_code: str, reflections: List[str]) -> str:
        latest = reflections[-1] if reflections else ""
        if "ZeroDivisionError" in latest:
            return PATCH_V2
        if "KeyError" in latest:
            return PATCH_V3
        return curr_code

    report = surgeon.run_repair_loop(BUGGY_CODE_V1, mock_unit_test_evaluator, mock_actor)
    print(f"Repair Success: {report['success']} in {report['trials_used']} trials")
    for step in report["history"]:  # type: ignore[union-attr]
        print(f"  Trial {step['trial']}: passed={step['passed']} | {step['reflection']}")

    assert report["success"] is True and report["trials_used"] == 3
    assert surgeon.is_duplicate_patch(PATCH_V3) is True, "Duplicate patch SHA-256 guard failed!"

    # ---- extra checks ----------------------------------------------------------------
    # traceback parser: innermost frame, exception without a message, dotted exception names
    tb = ('Traceback (most recent call last):\n  File "main.py", line 10, in <module>\n    run()\n'
          '  File "lib.py", line 42, in run\n    x = d["k"]\nKeyError: \'k\'')
    assert TracebackParser.parse(tb) == {"file": "lib.py", "line_number": "42", "exception_type": "KeyError", "fault_summary": "'k'"}
    assert TracebackParser.parse("Traceback...\n  File \"a.py\", line 1, in f\nStopIteration")["exception_type"] == "StopIteration"
    assert TracebackParser.parse("something odd")["exception_type"] == "RuntimeError"

    # the reflection quotes the offending source line, and the buffer is capped at Omega = 3
    s = ReflexionSurgeon(buffer_cap=3)
    r = s.generate_verbal_reflection({"exception_type": "ZeroDivisionError", "line_number": "3", "fault_summary": "x"}, BUGGY_CODE_V1)
    assert "Offending line: `return packet['flux'] / packet['window_sec']`" in r
    for i in range(6):
        s.generate_verbal_reflection({"exception_type": "ValueError", "line_number": str(i), "fault_summary": f"f{i}"}, "")
    assert len(s.episodic_buffer) == 3 and "f5" in s.episodic_buffer[-1] and "f2" not in " ".join(s.episodic_buffer)

    # whitespace-normalised hashing
    h = ReflexionSurgeon()
    assert h.is_duplicate_patch("def f():\n    return 1") is False
    assert h.is_duplicate_patch("\n\ndef f():\n        return 1   \n") is True
    assert h.is_duplicate_patch("def f():\n    return 2") is False

    # a cycling actor is halted early, not after max_trials evaluations
    calls = {"n": 0}
    def counting_eval(code):
        calls["n"] += 1
        return False, 'File "x.py", line 1, in f\nValueError: still broken'
    cyc = ReflexionSurgeon(max_trials=6).run_repair_loop("v0", counting_eval, lambda c, r: "v1" if c == "v0" else "v0")
    assert cyc["halt_reason"] == "DUPLICATE_PATCH_CYCLE" and cyc["success"] is False and calls["n"] == 2

    # budget exhaustion, crashing oracle and crashing actor are all survived
    never = ReflexionSurgeon(max_trials=3).run_repair_loop("a", lambda c: (False, "ValueError: x"), lambda c, r: c + "!")
    assert never["success"] is False and never["trials_used"] == 3
    boom = ReflexionSurgeon(max_trials=2).run_repair_loop("a", lambda c: 1 / 0, lambda c, r: c + "!")
    assert boom["history"][0]["exception"] == "ZeroDivisionError"
    bad_actor = ReflexionSurgeon(max_trials=3).run_repair_loop("a", lambda c: (False, "ValueError: x"), lambda c, r: 1 / 0)
    assert bad_actor["halt_reason"] == "ACTOR_ERROR"

    # patch_guard: an unsafe patch is never executed; reflection_gate keeps a rejected reflection out of memory
    executed: List[str] = []
    def eval_rec(code):
        executed.append(code)
        return (code == "safe", "ValueError: nope")
    g = ReflexionSurgeon(max_trials=3).run_repair_loop(
        "bad", eval_rec, lambda c, r: "evil" if c == "bad" else "safe",
        patch_guard=lambda c: (c != "evil", ["Forbidden import 'os'"]),
        reflection_gate=lambda t: (False, "REJECTED") if "SandboxViolation" in t else (True, "OK"))
    assert g["success"] and executed == ["bad", "safe"], executed
    assert g["history"][1]["blocked_by_sandbox"] is True and all("SandboxViolation" not in r for r in g["episodic_buffer"])
    print("Cycle guard halted a repeating actor after 2 evaluations (budget 6); unsafe patch was never executed.")
    print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")
