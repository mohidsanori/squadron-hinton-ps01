"""
Mission 06 - Part 2: Communications & Protocol Officer
Tool contracts, JSON-Schema validation and an exception boundary around tool execution
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula), Subsystem 06

An LLM's tool call is untrusted input. Before anything touches a ship system the dispatcher
  1. parses the payload (a JSON string or a dict) -- malformed JSON becomes an observation,
  2. validates the arguments against the tool's JSON-Schema contract (own validator, no dependency),
  3. runs the tool inside an error boundary.
Every failure is returned as the string "Tool Execution Error: <msg>" so it can be appended to
the agent's context and corrected in the next deliberation cycle. Nothing here raises.

Usage:
    python 02_Tool_Protocol_Dispatcher.py
"""

import json
import math
from typing import Any, Callable, Dict, List, Tuple, Union

ERROR_PREFIX = "Tool Execution Error: "
OBSERVATION_PREFIX = "Observation: "


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------
def navigate_warp_vector(x: float, y: float, z: float) -> str:
    """Sets the course vector through the nebula."""
    return f"Warp navigational coordinates successfully aligned to vector [{x:.2f}, {y:.2f}, {z:.2f}]."


def adjust_plasma_shields(intensity: int) -> str:
    """Modulates magnetic plasma shield intensity (0 to 100 %)."""
    if not (0 <= intensity <= 100):
        raise ValueError(f"Shield intensity {intensity}% is out of bounds! Valid range: [0, 100].")
    return f"Plasma shield harmonics adjusted to {intensity}% capacity."


SENSOR_TARGETS = {
    "homunculus_core": "Severe plasma flare detected. High density stellar dust at azimuth 142.",
    "eta_carinae_b": "Secondary O-type binary companion emission: 100x solar luminosity.",
    "escape_corridor": "Clear gravitational corridor detected along vector [-0.85, 0.52, 0.05].",
}


def query_sensor_array(target: str) -> str:
    """Scans telemetry for anomalies or radiation spikes."""
    return SENSOR_TARGETS.get(target.lower(), f"Sensor array reports: Target '{target}' unresolved.")


TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "navigate_warp_vector": navigate_warp_vector,
    "adjust_plasma_shields": adjust_plasma_shields,
    "query_sensor_array": query_sensor_array,
}

# ---------------------------------------------------------------------------
# 2. Schema contracts (JSON Schema subset: object / number / integer / string / boolean)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "navigate_warp_vector": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
        "required": ["x", "y", "z"],
        "additionalProperties": False,
    },
    "adjust_plasma_shields": {
        "type": "object",
        "properties": {"intensity": {"type": "integer"}},
        "required": ["intensity"],
        "additionalProperties": False,
    },
    "query_sensor_array": {
        "type": "object",
        "properties": {"target": {"type": "string", "minLength": 1}},
        "required": ["target"],
        "additionalProperties": False,
    },
}


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_type(value: Any, expected: str) -> bool:
    # bool is a subclass of int in Python but is never a JSON number
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    raise ValueError(f"unsupported schema type '{expected}'")


def validate_against_schema(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Return a list of human-readable violations (empty list = valid)."""
    errors: List[str] = []
    expected = schema.get("type")
    if expected and not _check_type(instance, expected):
        return [f"{path}: expected {expected}, got {_type_name(instance)} ({instance!r})"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if expected in ("number", "integer"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    if expected == "string" and "minLength" in schema and len(instance) < schema["minLength"]:
        errors.append(f"{path}: string shorter than minLength {schema['minLength']}")
    if expected == "object":
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required argument '{req}'")
        for key, val in instance.items():
            if key in props:
                errors.extend(validate_against_schema(val, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unexpected argument '{key}'")
    return errors


# ---------------------------------------------------------------------------
# 3. Dispatcher with error boundary
# ---------------------------------------------------------------------------
class ActionDispatcher:
    """Validates tool calls against their schemas and executes them inside an error boundary."""

    def __init__(self, registry: Dict[str, Callable[..., str]] = None,
                 schemas: Dict[str, Dict[str, Any]] = None):
        self.registry = registry if registry is not None else TOOL_REGISTRY
        self.schemas = schemas if schemas is not None else TOOL_SCHEMAS
        self.call_log: List[Dict[str, Any]] = []

    def dispatch_detailed(self, tool_name: Any, arguments: Union[str, Dict[str, Any], None]) -> Tuple[bool, str]:
        """Returns (ok, observation). Never raises."""
        ok, text = self._dispatch(tool_name, arguments)
        self.call_log.append({"tool": tool_name, "arguments": arguments, "ok": ok, "observation": text})
        return ok, text

    def dispatch(self, tool_name: Any, arguments: Union[str, Dict[str, Any], None]) -> str:
        return self.dispatch_detailed(tool_name, arguments)[1]

    def _dispatch(self, tool_name: Any, arguments: Any) -> Tuple[bool, str]:
        try:
            if not isinstance(tool_name, str) or tool_name not in self.registry:
                return False, f"{ERROR_PREFIX}Unknown tool {tool_name!r}. Available: {sorted(self.registry)}."

            # a) parse the payload
            if isinstance(arguments, (str, bytes)):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    return False, f"{ERROR_PREFIX}Invalid JSON payload for '{tool_name}': {exc}."
            if arguments is None:
                arguments = {}

            # b) validate against the contract
            violations = validate_against_schema(arguments, self.schemas[tool_name])
            if violations:
                return False, f"{ERROR_PREFIX}Schema validation failed for '{tool_name}': " + "; ".join(violations)

            # c) execute inside the error boundary
            try:
                result = self.registry[tool_name](**arguments)
            except Exception as exc:  # noqa: BLE001 - the boundary must catch everything a tool can raise
                return False, f"{ERROR_PREFIX}{exc}"
            return True, f"{OBSERVATION_PREFIX}{result}"
        except Exception as exc:  # noqa: BLE001 - last resort: the dispatcher itself must not crash the agent
            return False, f"{ERROR_PREFIX}Dispatcher fault: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    d = ActionDispatcher()
    print("=== Subsystem 06: Tool Dispatcher & Error Boundary ===")
    cases = [
        ("adjust_plasma_shields", {"intensity": 85}),
        ("adjust_plasma_shields", {"intensity": -10}),            # tool ValueError -> boundary
        ("adjust_plasma_shields", {"intensity": "high"}),         # wrong type
        ("adjust_plasma_shields", {"intensity": True}),           # bool is not an integer
        ("navigate_warp_vector", {"x": 1, "y": 2}),               # missing z
        ("navigate_warp_vector", {"x": 1, "y": 2, "z": float("nan")}),
        ("navigate_warp_vector", '{"x": -0.85, "y": 0.52, "z": 0.05}'),  # JSON string payload
        ("navigate_warp_vector", '{"x": 1, "y": '),               # malformed JSON
        ("query_sensor_array", {"target": "escape_corridor"}),
        ("query_sensor_array", {"target": "x", "debug": True}),   # extra argument
        ("self_destruct", {}),                                    # unknown tool
    ]
    for name, args in cases:
        print(f"{name}({args!r})\n    -> {d.dispatch(name, args)}")
