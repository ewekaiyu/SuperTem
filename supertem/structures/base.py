"""
supertem.structures.base

Dataclass-based structures for TEM automation, covering:
  - settings (beam / projection / detector / stage / acquisition outputs)
  - microscope state snapshots
  - image metadata containers
  - executable requests (control-plane objects)

This module supports two distinct contexts:
  (1) data-plane: ingestion and storage of imperfect data (metadata/state/logs)
  (2) control-plane: strict, safe execution of microscope commands

===============================================================================
I. Object Lifecycle (End-to-End)
===============================================================================

The intended lifecycle for objects in this module is:

  1) Ingest (untrusted input)
     - Source: vendor SDK returns, JSON logs, user configs, network payloads
     - Entry:  Class.from_dict(payload, mode=LENIENT or STRICT)
     - Mode propagation: The `mode` provided to a top-level `from_dict(...)` is
       treated as the default for the entire object graph. Nested `from_dict(...)`
       calls MUST pass the same effective mode to children (unless a field
       explicitly overrides it), so a single boundary choice (LENIENT vs STRICT)
       controls all descendants consistently.
     - Goal:   Capture data without crashing, even if imperfect.

  2) Normalize (Integrity of Structure)
     - Happens during __post_init__ via `FieldParser`.
     - Goal:   Produce a well-typed internal representation (Type Safety).
     - Action: Coerce types (str->int), populate structural defaults (None->[]),
               and park unparseable garbage in Extras.raw.
     - Note:   Does NOT check logic. Invalid values (e.g., width=-100) are
               preserved here to ensure data fidelity during ingestion.

  3) Validate (Integrity of Meaning)
     - Happens in validate(mode=...) via `Validator`.
     - Goal:   Enforce domain constraints and logical invariants (Logic Safety).
     - Action (STRICT): Raises exceptions immediately on violation.
     - Action (LENIENT): Records violation in Extras.notes and "Heals" the object
               (e.g., reset width=-100 -> 512, or disable the specific feature).
     - Result: The object is now guaranteed to be logically consistent.

  4) Serialize (JSON-capable representation)
     - Happens in to_dict()
     - Goal:   Produce JSON-serializable output for logging/storage/transport.

  5) Execute (Gatekeeping)
     - Only applicable to control-plane objects (requests/settings).
     - Policy: Validate STRICTLY immediately before hardware interaction.
     - Goal:   Ensure commands applied to the microscope are within hardware limits.

A key rule:
  Objects created via from_dict() are "Type-Safe" but "Logically Unverified."
  They MUST NOT be executed until validate() has been called and passed.

===============================================================================
II. ParseMode and Context
===============================================================================

ParseMode.LENIENT (data-plane default)
  Intended for metadata/state/log ingestion where completeness is not guaranteed.

  Behavior at Construction (__post_init__):
    - Prioritizes survival: construction will not fail due to malformed types.
    - Captures unparseable data in Extras.raw / Extras.notes.
    - Result: Object is type-safe but may contain logically unsafe values (e.g. exposure=-5).

  Behavior at Validation (.validate()):
    - "Heals" invalid logic: unsafe values are reset to defaults or None.
    - Records the intervention in Extras.notes.
    - Result: Object becomes safe for use.

ParseMode.STRICT (control-plane default)
  Intended for objects that will be applied to hardware (requests/settings).

  Behavior at Construction (__post_init__):
    - Prioritizes correctness: raises immediately on malformed types or missing structure.

  Behavior at Validation (.validate()):
    - Enforces constraints: raises note_or_raise(...) on any logical violation.
    - Result: Guaranteed safe to execute, or raises Exception.

===============================================================================
III. The Normalization, Validation, and Gatekeeping Rulebook
===============================================================================

To maintain safety without sacrificing robustness, this module enforces a strict
separation of concerns across three distinct lifecycles:

1. Normalization (__post_init__)
-------------------------------------------------------------------------------
   GOAL:    Integrity of Structure (Type & Shape Safety)
   INPUT:   "Dirty" data (Strings, Nones, Dicts, Missing Keys)
   OUTPUT:  "Clean" data (Correct Python Types, Structurally Complete)
   TOOL:    FieldParser

   Rules:
   A. Coercion is Normalization.
      Convert inputs to their target types.
      (e.g., "128" -> 128, "10 nm" -> Quantity(10, 'nm'))

   B. Structural Defaults are Normalization.
      If a field is `None` but required for the object to exist (e.g., to prevent
      AttributeError later), set a safe default here via `p.model(..., default=...)`.

   C. Structural Patching is Normalization.
      If a required value exists elsewhere in the object graph (e.g., copying
      an ID from an inner object to a missing outer field), perform the copy
      here to complete the structure.

   D. DO NOT Check Logic.
      Do not check if a number is positive, finite, or consistent with other
      fields. If the type is right, let it pass.
      (e.g., `width=-100` is a valid integer. Leave it for validation.)

2. Validation (validate)
-------------------------------------------------------------------------------
   GOAL:    Integrity of Meaning (Internal Logic & Self-Consistency)
   TOOL:    Validator

   Rules:
   A. Trust the Types.
      Do not check `isinstance` or try/except AttributeErrors. If `__post_init__`
      did its job, variables have the correct type. Focus on *values*.

   B. Domain Constraints are Validation.
      Check physical and logical bounds of the object itself.
      (e.g., `width > 0`, `min_limit <= max_limit`).

   C. Internal Cross-Field Consistency.
      Check if two fields within the *same* object or hierarchy contradict each other.
      - "If axis is enabled (`can_tilt=True`), limits MUST be defined (`tilt_limits!=None`)."
      - "If `default_id` is set, it MUST exist in `available_ids`."

   D. Healing is Validation (Lenient Mode Only).
      If a value is structurally sound but logically invalid (e.g., `width=-50`):
        - STRICT Mode: Raise an Exception.
        - LENIENT Mode: "Heal" it to a safe value or disable the feature.

3. Gatekeeping (is_safe_... / is_supported)
-------------------------------------------------------------------------------
   GOAL:    Integrity of Action (Runtime Safety & Hardware Compatibility)
   INPUT:   An external "Request" object (e.g., StagePosition, BeamSettings)
   OUTPUT:  Transient `SafetyCheck` result (Allowed/Rejected + Reasons).

   Rules:
   A. Configs are Guardrails, Requests are Intent.
      The SystemSettings object acts as the Gatekeeper. It validates *external*
      requests against its *internal* limits.

   B. NO State Modification (Anti-Leak Policy).
      Gatekeeping checks happen frequently (e.g. inside UI polling loops).
      Writing failure logs to `self.extra.notes` would cause infinite memory growth.
      Safety methods MUST return a transient result object and MUST NOT modify
      the persistent settings object.

   C. Specificity over Genericity.
      Use specific method names that describe the risk:
      - `is_safe_move(target)`: Checks collision/travel limits (Stage).
      - `is_safe_beam(target)`: Checks voltage/optical limits (Beam).
      - `is_supported(settings)`: Checks driver capabilities (Detector).

===============================================================================
IV. Handling Defaults & None (The 4 Categories)
===============================================================================

To prevent ingestion crashes while ensuring runtime safety, `__post_init__` acts
as a "loading dock" that accepts flexible inputs (including None). However, the
final internal state depends on the object's semantic category，controlled by the
`default` parameter in parser methods.

Universal Rule: All domain-data input fields are typed as `Optional[T] = None`.
(Internal control fields like `_mode` and `extra` are excluded from this rule).

Category A: Configuration & Mandatory Parameters (Strict Runtime)
  - Definition: Operational constants required for logic (timeouts, image formats).
  - Action:     Aggressive Defaulting.
  - Pattern:    `self.val = p.int(self.val, "name", default=10)`

Category B: Structural Containers (Strict Runtime)
  - Definition: Fields holding nested objects/lists.
  - Action:     Structural Defaulting. Never leave as None.
  - Pattern:    `self.child = p.model(Child, self.child, "name", default=Child())`

Category C: Measured State & Optional Boundaries (Nullable Runtime)
  - Definition: Snapshots of reality OR permissive boundaries.
  - Action:     Preserve None.
  - Pattern:    `self.val = p.int(self.val, "name", default=None)`

Category D: Requests & Intents (Nullable Runtime)
  - Definition: Commands to change specific settings (Tristate logic).
  - Action:     Preserve None.
  - Pattern:    `self.val = p.int(self.val, "name", default=None)`

The "Zero Trap" Warning:
  Avoid: `self.val = p.int(...) or 10`  <-- UNSAFE. Input 0 becomes 10.
  Use:   `self.val = p.int(..., default=10)` <-- SAFE. Parser handles 0 vs None.

===============================================================================
V. Extras: Preservation vs. Transient Logic
===============================================================================

`Extras` is the structured container for **persistent** non-canonical information:
  - vendor: vendor-specific extension payloads (namespaced by vendor key)
  - unknown: unknown top-level keys swept during from_dict (forward compatibility)
  - raw: raw values replaced/rejected during normalization
  - notes: structured diagnostics produced by **normalization** or **validation**

Conventions:
  - Use namespaced note keys, e.g. "DetectorSettings.roi_invalid_shape".
  - LENIENT mode should preserve information rather than discard it.

Anti-Pattern Warning:
  Do NOT use `Extras` to store transient errors from high-frequency checks
  (e.g., safety polling). `Extras` travels with the object lifecycle; filling it
  with runtime logs causes memory leaks. Use `SafetyCheck` return values instead.

===============================================================================
VI. Serialization Contract: to_dict Must Be JSON-Capable
===============================================================================

All to_dict() methods must return JSON-serializable output:
  - dataclasses -> dict
  - enums -> str
  - tuples -> lists
  - quantities/units -> plain numbers.

  *AUTOMATION*: Unit stripping and key renaming are handled by `_auto_to_dict`.
  - Standard Rule: Fields with units are stripped to floats, and the key is
    suffixed with the unit (e.g., field `x` + unit `nm` -> key `x_nm`).
  - Override Rule: Specific keys can be manually renamed via `_KEYS` if the
    standard suffix pattern is insufficient.

If a value cannot be expressed safely as JSON, preserve a safe representation in
Extras.raw and record a diagnostic note.

===============================================================================
VII. Implementation Conventions & Usage
===============================================================================

1. Internal Implementation Patterns
-------------------------------------------------------------------------------
   To ensure consistent behavior across 20+ structures, every lifecycle method
   must follow a strict pattern using the shared helper functions.

   A. __post_init__ (Normalization)
      - Instantiate: `p = FieldParser(self, self._mode, "ClassName")`
      - Normalize: `self.field = p.type(self.field, "name", default=...)`
      - Constraint: DO NOT perform logic checks here. Only ensure types.
      - Exception: Lightweight value objects (e.g., Point) may skip Extras/Lifecycle
        overhead for performance, but must still normalize fields.

   B. validate(mode=...) (Logic & Healing)
      - Use `v = Validator(self, mode)`
      - Define constraints using `v.check(condition, key, error, heal=lambda: ...)`
      - Standard helpers: `v.check_gt_zero`, `v.check_finite`, `v.check_nested`.
      - Return `v.valid`.

   C. to_dict() (Serialization)
      - Mechanism: Use `_auto_to_dict(self, unit_map=..., key_map=...)`.
      - Configuration: Define `_UNITS` dict mapping fields to target units.
        (e.g. `{"x": Units.NM}` results in key `x_nm` and float value).
      - Overrides: Define `_KEYS` dict only if renaming fields beyond standard
        suffixes (e.g. `{"settle_time": "settle_time_s"}`).
      - Behavior: Automatically handles recursion, list serialization, and extras injection.

   D. from_dict(data, mode=...) (Ingestion)
      - Mechanism: Use `_auto_from_dict(cls, data, mode, alias_map=...)`.
      - Configuration: Define `alias_map` to map incoming keys to internal field names.
        (e.g., `{"x": "x_nm"}` allows input key `x_nm` to populate field `x`).
        Supports strings (1:1 alias) or lists (priority fallback aliases).
      - Behavior:
        1. Introspects dataclass fields to identify valid targets.
        2. Validates input type and harvests unknown keys into `Extras.unknown`.
        3. STRICT Precedence: Direct Key > Alias > Missing.
           (Explicit `None` in input is preserved and not overwritten by an alias).
        4. Passes arguments to constructor; `__post_init__` handles default application.

2. External Usage: Immutability by Policy
-------------------------------------------------------------------------------
   To guarantee type safety, external code MUST treat these objects as immutable.

   - DO NOT modify attributes directly:
       `settings.beam.voltage = "300 kV"`  <-- UNSAFE. Bypasses normalization.
                                               Result is a string, not a Quantity.

   - ALWAYS use `dataclasses.replace()`:
       `new_beam = replace(settings.beam, voltage="300 kV")` <-- SAFE.
       This triggers `__init__` and `__post_init__` again, ensuring the string
       is correctly parsed into a Quantity object.

===============================================================================
VIII. The Request Pattern (Control Plane Intents)
===============================================================================

While `Settings` objects define "Configuration" (static parameters) and `State`
objects define "Snapshot" (telemetry), `Request` objects define "Intent."

They are the vehicles for the Control Plane. A Request encapsulates a user's
command to change the microscope's state.

1. Anatomy of a Request
-------------------------------------------------------------------------------
   A Request typically consists of three components:
   A. Target Identity (Mandatory)
      - "Which hardware device?" (e.g., `detector_id`, `aperture_id`).
      - Unlike Settings objects (which can be generic/reusable presets), Requests
        MUST bind to a specific physical device before execution.

   B. The Payload (The "What")
      - Usually reuses a Domain Object (e.g., `StagePosition`, `DetectorSettings`).
      - Fields are `Optional`. A value of `None` means "Do not change."
      - This allows for precise partial updates (e.g., "Change only Exposure").

   C. Modifiers (The "How")
      - Execution flags that don't fit in the domain object.
      - Examples: `relative=True`, `wait_for_settle=True`, `force=False`.

2. The Usage Contract
-------------------------------------------------------------------------------
   A. Construct Strict by Default
      Requests are code-driven commands. They default to `ParseMode.STRICT` because
      ambiguity in a command is dangerous.

   B. Validate Before Sending
      A Request is invalid until `validate()` returns True.
      - It must have a Target ID.
      - It must have a non-empty Payload (no-op requests are rejected).
      - It must satisfy cross-field logic (e.g. relative moves need coordinates).

   C. Ephemeral Lifecycle
      Requests are transient. They are created, validated, executed by the
      Manager/Driver, and then discarded. They are rarely stored long-term.

3. How to Write a New Request
-------------------------------------------------------------------------------
   1. Define the class with `_mode: ParseMode = ParseMode.STRICT`.
   2. Include the ID field (e.g. `beam_id`) and validate its presence.
   3. Include a `payload` or specific fields (use Category D: default=None).
   4. Implement `validate()` to ensure the intent is executable.

   Example:
     req = StageMoveRequest(target=StagePosition(x=Q_(10, 'um')), relative=True)
     if req.validate():
         microscope.stage.move(req)

===============================================================================
IX. Architecture Overview (The "Noun-Verb" Topology)
===============================================================================

The module implements a strict Command-Query Separation (CQS) architecture.
Interaction is divided into "Nouns" (Data/State) and "Verbs" (Intents/Requests).

1. The Data Plane (Nouns)
   - State Objects: Telemetry snapshots (e.g. `MicroscopeState`, `BeamState`).
   - Settings Objects: Configuration payloads (e.g. `BeamSettings`, `ProjectionSettings`).
   - System Settings: Hardware capabilities & limits (e.g. `StageSystemSettings`).

2. The Control Plane (Verbs)
   - Request Objects: Executable commands that wrap Settings with an intent.
   - Naming Convention: `*ControlRequest` for hardware state management,
     `*MoveRequest` for coordinate navigation, and `AcquisitionRequest` for data.

3. Component Map
   -----------------------------------------------------------------------------
   Subsystem             | Configuration (Noun)   | Execution (Verb)
   -----------------------------------------------------------------------------
   Stage (Motion)        | StagePosition          | StageMoveRequest (Nav)
                         |                        | StageControlRequest (Stop)
   -----------------------------------------------------------------------------
   Beam (Illumination)   | BeamSettings           | BeamControlRequest
   (Gun/Condenser)       |                        |
   -----------------------------------------------------------------------------
   Projection (Imaging)  | ProjectionSettings     | ProjectionControlRequest
   (Obj/Projector)       |                        |
   -----------------------------------------------------------------------------
   Scan (STEM)           | ScanSettings           | ScanControlRequest
   -----------------------------------------------------------------------------
   Detector              | DetectorSettings       | AcquisitionRequest (Capture)
                         |                        | DetectorControlRequest (Mech)
   -----------------------------------------------------------------------------
   Vacuum                | VacuumSettings         | VacuumControlRequest
   -----------------------------------------------------------------------------
   Aperture              | Aperture               | ApertureControlRequest
   -----------------------------------------------------------------------------

===============================================================================
X. Data Organization (The Hierarchy)
===============================================================================

The data plane is organized into two primary trees: Configuration (Static/Limits)
and Telemetry (Dynamic/Snapshot).

1. Configuration Tree (`MicroscopeSettings`)
   Defines how the machine *should* behave and what it is *capable* of.

   MicroscopeSettings
   ├── system: SystemSettings
   │   ├── stage_system: StageSystemSettings (Travel limits, Max speeds)
   │   ├── beam_system: BeamSystemSettings (Voltage limits, Safety checks)
   │   ├── projection_system: ProjectionSystemSettings (Mag ranges, Cam lengths)
   │   ├── scan_system: ScanSystemSettings (Dwell time limits, Scan modes)
   │   ├── detector_system: DetectorSystemSettings (Registry of cameras)
   │   └── info: SystemInfo (Static hardware IDs, IP addresses)
   ├── image: ImageOutputSettings (File formats, Save paths)
   └── protocol: Dict (User-defined automation scripts)

2. Telemetry Tree (`MicroscopeState`)
   Defines what the machine is *currently doing*. Used for logs and metadata.

   MicroscopeState
   ├── stage_position: StagePosition (x, y, z, tilt)
   ├── beam: BeamState (Voltage, current, spot size)
   ├── projection: ProjectionSettings (Defocus, magnification, optical mode)
   ├── scan: ScanSettings (Active dwell time, grid resolution)
   ├── vacuum: VacuumSettings (Valve states, pressures)
   ├── apertures: Dict[str, Aperture] (State of all inserted apertures)
   └── detectors: Dict[str, DetectorState] (State of all active cameras)

===============================================================================
Rationale
===============================================================================

TEM automation consumes heterogeneous and imperfect inputs (vendor SDKs, partial
configs, historical logs). Strict parsing everywhere makes metadata pipelines
brittle; lenient parsing everywhere makes control paths unsafe. This module
provides a consistent lifecycle to be both robust (LENIENT ingestion/storage)
and safe (STRICT validation/execution).
"""
import datetime
import json
import os
import ipaddress
import dataclasses
from dataclasses import dataclass, field, replace, is_dataclass
from pathlib import Path
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, TypeVar, Type
from collections.abc import Mapping
import numpy as np
from PIL import Image
import tifffile as tff

# =============================================================================
# Versioning & Imports
# =============================================================================

try:
    from supertem.config import METADATA_VERSION  # type: ignore
except Exception:
    METADATA_VERSION = "1"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("supertem")
except PackageNotFoundError:
    try:
        __version__ = version("SuperTem")
    except PackageNotFoundError:
        __version__ = "unknown"

try:
    from pint import UnitRegistry
except ImportError as e:
    raise ImportError(
        "Dependency missing: 'pint' is required. Install it with `pip install pint`."
    ) from e

# Initialize central registry
ureg = UnitRegistry()
Q_ = ureg.Quantity

# Quantity type import is version-dependent across Pint releases.
try:  # Pint >= 0.20 often exposes Quantity at top-level
    from pint import Quantity  # type: ignore
except Exception:
    try:
        from pint.facets.plain.quantity import Quantity  # type: ignore
    except ImportError:
        # Fallback for very old/new structures if facets path changes
        Quantity = type(Q_(1, "nm"))

# =============================================================================
# Constants & Enums
# =============================================================================

class ParseMode(str, Enum):
    """Defines the strictness level for data ingestion."""
    STRICT = "strict"  # Raise errors immediately (Control Plane / Execution)
    LENIENT = "lenient"  # Log errors to Extras and continue (Data Plane / Logging)

class StageDriveType(str, Enum):
    """Defines the mechanism used for stage movement."""
    DEFAULT = "default"  # Logic decided by driver (usually mechanical for large, piezo for small)
    MECHANICAL = "mechanical"  # Coarse, large range, backlash prone
    PIEZO = "piezo"  # Fine, small range, hysteresis free
    HYBRID = "hybrid"  # Combined movement

class Units:
    """Centralized definition of physical units."""
    NM = "nm"
    UM = "um"
    MM = "mm"
    KV = "kV"
    NA = "nA"
    UA = "uA"
    MRAD = "mrad"
    DEG = "deg"
    MS = "ms"
    US = "us"
    SEC = "s"
    PA = "Pa"
    HZ = "Hz"

def as_parse_mode(mode: Union["ParseMode", str, None]) -> "ParseMode":
    if isinstance(mode, ParseMode): return mode
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m == "lenient": return ParseMode.LENIENT
        if m == "strict": return ParseMode.STRICT
    return ParseMode.STRICT

def is_strict(mode: Union["ParseMode", str, None]) -> bool:
    """Helper to check if the effective mode is STRICT."""
    return as_parse_mode(mode) == ParseMode.STRICT

T = TypeVar("T")

# =============================================================================
# Extras & Core Helpers
# =============================================================================

@dataclass
class Extras:
    """
    Structured container for non-standard data.

    This class supports the 'Lenient Parsing' philosophy. Any data that doesn't
    fit the strict schema ends up here for later inspection instead of causing a crash.

    Attributes:
    vendor: Namespaced storage for vendor-specific extensions.
    unknown: Storage for JSON keys not recognized by the schema.
    raw: Original raw values that failed type coercion/validation.
    notes: Error messages or warnings generated during parsing.
    """
    vendor: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unknown: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.vendor or self.unknown or self.raw or self.notes)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe payload representation."""

        # We define EXACTLY what to keep
        def _keep(val: Any) -> bool:
            # 1. Always toss None
            if val is None: return False

            # 2. Toss empty collections (dict, list, string)
            if isinstance(val, (dict, list, tuple, str)) and len(val) == 0:
                return False

            # 3. Keep everything else! (Numbers, Booleans, Objects)
            return True

        # Now 0 and False will survive because they are not None and not collections
        return _jsonable({k: deepcopy(v) for k, v in self.__dict__.items() if _keep(v)})

    @staticmethod
    def from_any(value: Any, *, owner: str = "unknown") -> "Extras":
        """Intelligently parse 'extra' fields from various inputs."""
        if isinstance(value, Extras): return value
        ex = Extras()

        if not isinstance(value, dict):
            if value is not None:
                ex.raw[f"{owner}.extra"] = repr(value)
            return ex

        # Smart merge logic
        known = {"vendor", "unknown", "raw", "notes"}
        keys = set(value.keys())

        if keys and keys.issubset(known):
            # It is a structured Extras dict
            for k in known:
                if k in value and isinstance(value[k], dict):
                    setattr(ex, k, deepcopy(value[k]))
            # Handle vendor special case (vendor keys might not be dicts)
            if "vendor" in value and isinstance(value["vendor"], dict):
                for vend, payload in value["vendor"].items():
                    ex.vendor[str(vend)] = deepcopy(payload) if isinstance(payload, dict) else {
                        "_value": deepcopy(payload)}
        else:
            # It is a flat dict (unknown data) -> Check partial matches
            for k in known:
                if k in value: setattr(ex, k, deepcopy(value[k]))

            # Remaining goes to unknown
            for k, v in value.items():
                if k not in known:
                    try:
                        ex.unknown[str(k)] = deepcopy(v)
                    except:
                        ex.raw[f"{owner}.extra.{k}"] = repr(v)
        return ex

@dataclass
class SafetyCheck:
    """
    Transient result of a safety or capability check.
    Does not modify persistent state (Extras) to prevent memory leaks during polling.
    """
    allowed: bool
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allows direct boolean usage: if settings.is_safe(...):"""
        return self.allowed

    @classmethod
    def success(cls) -> "SafetyCheck":
        return cls(allowed=True)

    @classmethod
    def failure(cls, reason: str) -> "SafetyCheck":
        return cls(allowed=False, reasons=[reason])

    def add_reason(self, reason: str):
        self.allowed = False
        self.reasons.append(reason)

def _jsonable(obj: Any) -> Any:
    """Recursively convert object to JSON-safe primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (list, tuple)): return [_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Quantity): return {"magnitude": float(obj.magnitude), "unit": str(obj.units)}
    try:
        if isinstance(obj, Path): return str(obj)
        if hasattr(obj, "tolist"): return obj.tolist()  # Numpy array
        if hasattr(obj, "item"): return obj.item()  # Numpy scalar
    except Exception:
        pass
    return str(obj)

def note_or_raise(extra: Optional[Extras], key: str, exc: Exception, *, mode: Union[ParseMode, str],
                  raw: Any = None) -> None:
    """Handle validation error based on Strict/Lenient mode."""
    if as_parse_mode(mode) == ParseMode.STRICT: raise exc
    if extra is None: return
    try:
        if raw is not None: extra.raw[key] = _jsonable(raw)
        extra.notes.setdefault(key, []).append({"error": str(exc), "type": type(exc).__name__})
    except Exception:
        pass

def _extra_put_raw(extra: Any, key: str, value: Any) -> None:
    if extra is None: return
    if isinstance(extra, Extras):
        extra.raw[key] = _jsonable(value)
    elif isinstance(extra, dict):
        extra[f"{key}_raw"] = _jsonable(value)

def ensure_quantity(value: Any, unit: str) -> Optional["Quantity"]:
    """
    Coerce arbitrary input into a Pint Quantity with the target unit.

    Returns:
        Quantity: If conversion is successful.
        None: If the input value is None.

    Raises:
        TypeError: If input is boolean or incompatible type.
        ValueError: If string parsing fails.
        pint.errors.UndefinedUnitError: If units are unknown (e.g. 'lightyears').
        pint.errors.DimensionalityError: If units are incompatible (e.g. '10 ms' for 'nm').
    """
    if value is None: return None
    # Explicitly reject booleans (True == 1.0, but semantically invalid for physical quantities)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Cannot coerce boolean {value} to Quantity")

    if isinstance(value, (int, float, np.number)): return Q_(float(value), unit)

    # Remove the outer try/except block to let specific errors bubble up
    if isinstance(value, Quantity):
        return Q_(value.magnitude, str(value.units)).to(unit)

    if isinstance(value, dict):
        mag = value.get("magnitude", value.get("value"))
        u = value.get("unit", value.get("units"))
        if mag is None: raise ValueError(f"Quantity dict missing 'magnitude': {value}")
        return Q_(mag, u or unit).to(unit)

    if isinstance(value, str):
        s = value.strip()
        if not s: return None
        # This might raise pint.UndefinedUnitError or ValueError
        try:
            return Q_(float(s), unit).to(unit)
        except ValueError:
            # If standard float conversion fails, try parsing as a Quantity string (e.g., "10 nm")
            return Q_(s).to(unit)

    # Fallback for unexpected types
    return Q_(float(value), unit).to(unit)

def serialize_quantity(q: Optional["Quantity"], target_unit: str) -> Optional[float]:
    """Convert a Quantity to a plain float magnitude in the target unit. Raises Exception on unit mismatch.

    This strips the unit information for safe JSON serialization.
    Example: serialize_quantity(Q_(300, 'kV'), 'V') -> 300000.0
    """
    if q is None: return None
    if not isinstance(q, Quantity): return float(q)
    return float(q.to(target_unit).magnitude)  # Let pint errors bubble up

def _check_data_format(data: np.ndarray) -> bool:
    """Validate if numpy array is a valid 2D image (uint8/uint16)."""
    if data.ndim == 3:
        if data.shape[0] == 1: data = data[0]
        elif data.shape[2] == 1: data = data[:, :, 0]
    return (data.ndim == 2) and (data.dtype.kind == "u") and (data.dtype.itemsize in (1, 2))

def drop_none_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def add_extra_if_any(out: Dict[str, Any], extra: Any) -> Dict[str, Any]:
    if extra is None: return out
    ex_obj = extra if isinstance(extra, Extras) else Extras.from_any(extra)
    if ex_obj.is_empty(): return out
    payload = ex_obj.to_dict()
    if payload:
        out["extra"] = payload
    return out

def _finish_to_dict(payload: Dict[str, Any], extra: Any) -> Dict[str, Any]:
    add_extra_if_any(payload, extra)
    return _jsonable(drop_none_keys(payload))

def normalize_extra(extra: Any) -> Extras:
    if extra is None: return Extras()
    if isinstance(extra, Extras): return extra
    if isinstance(extra, dict): return Extras.from_any(extra)
    raise TypeError(f"extra must be Extras, dict, or None, got {type(extra)}")

def normalize_extra_lenient(extra: Any, owner: str) -> Extras:
    try: return normalize_extra(extra)
    except Exception:
        ex = Extras()
        ex.raw[f"{owner}.extra"] = repr(extra)
        return ex

def _setup_from_dict(cls: Type, data: Any, mode: Union[ParseMode, str, None], known_keys: Iterable[str] = (),
                     aliases: Iterable[str] = ()) -> Tuple[Optional[Dict[str, Any]], ParseMode, Optional[Extras]]:
    mode = as_parse_mode(mode)
    if isinstance(data, cls): return None, mode, None
    if not isinstance(data, dict):
        if is_strict(mode): raise TypeError(f"{cls.__name__} expects dict")
        # Treat whole input as unknown/extra
        return {}, mode, Extras.from_any(data, owner=cls.__name__)

    # Collect extras from unknown keys
    extra = Extras.from_any(data.get("extra"), owner=cls.__name__)
    known_set = set(known_keys) | set(aliases) | {"extra"}
    for k, v in data.items():
        if k not in known_set:
            try:
                extra.unknown[str(k)] = deepcopy(v)
            except:
                extra.raw[f"{cls.__name__}.unknown.{k}"] = repr(v)
    return data, mode, extra

def _setup_validate(obj_mode: Any, override_mode: Any) -> Tuple[ParseMode, bool]:
    mode = as_parse_mode(obj_mode if override_mode is None else override_mode)
    return mode, is_strict(mode)

# =============================================================================
# FieldParser, Validator and auto to_dict/from_dict helper functions
# =============================================================================

class FieldParser:
    """
    Context-aware parser that normalizes input data into typed fields.
    Consolidates strict/lenient logic, default handling, and error recording.
    """

    def __init__(self, obj: Any, mode: Any, owner_name: str):
        self.mode = as_parse_mode(mode)
        self.strict = is_strict(self.mode)
        self.owner = owner_name

        # Initialize Extras immediately
        raw_extra = getattr(obj, "extra", None)
        if self.strict:
            self.extra = normalize_extra(raw_extra)
        else:
            self.extra = normalize_extra_lenient(raw_extra, owner_name)

        # Attach the normalized container back to the object
        obj.extra = self.extra

    def _key(self, name: str) -> str:
        return f"{self.owner}.{name}"

    def _record(self, name: str, value: Any, exc: Optional[Exception] = None):
        if exc:
            note_or_raise(self.extra, self._key(name), exc, mode=self.mode, raw=value)
        elif self.extra is not None:
            _extra_put_raw(self.extra, self._key(name), value)

    # --- Primitives ---

    def bool(self, val: Any, name: str, default: Optional[bool] = None) -> Optional[bool]:
        """Parses bool. Use default=False for Flags, default=None for Tristate."""
        if val is None: return default
        if isinstance(val, str) and not val.strip(): return default
        if isinstance(val, bool): return val

        try:
            if isinstance(val, (int, float, np.number)): return bool(val)
            if isinstance(val, str):
                s = val.strip().lower()
                if s in {"1", "true", "t", "yes", "y", "on"}: return True
                if s in {"0", "false", "f", "no", "n", "off"}: return False
            raise ValueError(f"Invalid boolean: {val!r}")
        except Exception as e:
            self._record(name, val, e)
            return default

    def int(self, val: Any, name: str, default: Optional[int] = None) -> Optional[int]:
        """Parses int. Use default=10 for Config, default=None for State."""
        if val is None: return default
        if isinstance(val, bool):
            self._record(name, val, TypeError(f"{name} cannot be bool"))
            return default
        try:
            if isinstance(val, (int, np.integer)): return int(val)
            if isinstance(val, (float, np.floating)):
                f = float(val)
                if f.is_integer(): return int(f)
                raise ValueError("Float is not integer-like")
            if isinstance(val, str):
                s = val.strip()
                if not s: return default
                f = float(s)
                if f.is_integer(): return int(f)
                raise ValueError("String is not integer-like")
            raise TypeError(f"Cannot parse int from {type(val)}")
        except Exception as e:
            self._record(name, val, e if isinstance(e, (TypeError, ValueError)) else ValueError(str(e)))
            return default

    def float(self, val: Any, name: str, unit: Optional[str] = None, default: Optional[float] = None) -> Optional[
        float]:
        """Parses float. Use default=0.0 for Config, default=None for State."""
        if val is None: return default
        if isinstance(val, (bool, np.bool_)):
            self._record(name, val, TypeError(f"{name} cannot be bool"))
            return default

        if unit:
            try:
                q = ensure_quantity(val, unit)
                if q is not None: return float(q.magnitude)
            except Exception:
                # If quantity parsing fails (e.g. "10 lightyears"), ignore and fall through
                # to standard float parsing below, which will catch the error and record it.
                pass

        try:
            if isinstance(val, Quantity) and not unit:
                raise ValueError(f"{name} is a Quantity but no target unit defined.")
            if isinstance(val, (int, float, np.number)): return float(val)
            if isinstance(val, str):
                s = val.strip()
                if not s: return default
                return float(s)
            raise TypeError(f"Cannot parse float from {type(val)}")
        except Exception as e:
            self._record(name, val, e)
            return default

    def str(self, val: Any, name: str, default: Optional[str] = None) -> Optional[str]:
        """Parses string. Use default=None to convert Empty/Missing -> None."""
        if val is None: return default
        if self.strict and not isinstance(val, str):
            self._record(name, val, TypeError(f"{name} must be str"))
            return default
        if isinstance(val, bool):
            self._record(name, val, TypeError(f"{name} cannot be bool"))
            return default
        try:
            s = str(val).strip()
            return s or default
        except Exception:
            self._record(name, val)
            return default

    def id(self, val: Any, name: str) -> Optional[str]:
        """Parses identifier. Logs specific error for empty strings."""
        if val is None: return None
        if isinstance(val, str) and not val.strip():
            if self.strict:
                raise ValueError(f"Field '{name}' cannot be an empty string")
            self._record(name, val)
            if hasattr(self.extra, "notes"):
                self.extra.notes.setdefault(f"{self._key(name)}.empty", []).append({
                    "error": f"Field '{name}' is an empty string",
                    "type": "ValueError"
                })
            return None
        return self.str(val, name)

    # --- Quantities ---

    def qty(self, val: Any, name: str, unit: str, default: Optional[Quantity] = None) -> Optional[Quantity]:
        """
        Parses Pint Quantity.

        Captures underlying Pint/ValueError exceptions and records them in
        Extras.notes before returning the default value.
        """
        try:
            q = ensure_quantity(val, unit)
            if q is not None: return q
        except Exception as e:
            # Capture the specific error (e.g. UndefinedUnitError, DimensionalityError)
            self._record(name, val, exc=e)
            return default

        # Handle the case where val was not None but ensure_quantity returned None (e.g., empty string)
        if val is not None:
            self._record(name, val, ValueError(f"'{name}' must be {unit}, got {val!r}"))

        return default

    def pair_qty(self, val: Any, name: str, unit: str) -> Optional[Tuple[Quantity, Quantity]]:
        """Parses (Quantity, Quantity)."""
        if val is None: return None
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            self._record(name, val, TypeError(f"{name} must be (val, val)"))
            return None
        q1 = self.qty(val[0], f"{name}[0]", unit)
        q2 = self.qty(val[1], f"{name}[1]", unit)
        if q1 is not None and q2 is not None: return (q1, q2)
        return None

    # --- Structures ---

    def pair_int(self, val: Any, name: str) -> Optional[Tuple[int, int]]:
        if val is None: return None
        try:
            if not isinstance(val, (list, tuple)) or len(val) != 2: raise TypeError
            a = self.int(val[0], f"{name}[0]")
            b = self.int(val[1], f"{name}[1]")
            if a is None or b is None: raise ValueError
            return (a, b)
        except Exception:
            self._record(name, val, ValueError(f"{name} must be (int, int)"))
            return None

    def pair_float(self, val: Any, name: str) -> Optional[Tuple[float, float]]:
        if val is None: return None
        try:
            if not isinstance(val, (list, tuple)) or len(val) != 2: raise TypeError
            a = self.float(val[0], f"{name}[0]")
            b = self.float(val[1], f"{name}[1]")
            if a is None or b is None: raise ValueError
            return (a, b)
        except Exception:
            self._record(name, val, ValueError(f"{name} must be (float, float)"))
            return None

    def list_str(self, val: Any, name: str) -> List[str]:
        if val is None: return []
        if not isinstance(val, (list, tuple)):
            self._record(name, val, TypeError(f"{name} must be list"))
            return []
        out = []
        for i, item in enumerate(val):
            s = self.str(item, f"{name}[{i}]")
            if s is not None: out.append(s)
        return out

    def dict(self, val: Any, name: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if val is None: return default
        if isinstance(val, dict): return val
        self._record(name, val, TypeError(f"{name} must be dict"))
        return default

    def model(self, cls: Type[T], val: Any, name: str, default: Optional[T] = None) -> Optional[T]:
        """Parses nested Dataclass."""
        if val is None: return default
        # Identity
        try:
            if isinstance(val, cls):
                if is_dataclass(val): return replace(val, _mode=self.mode)  # type: ignore
                return val
        except TypeError:
            pass
        # Structure Check
        if not isinstance(val, (dict, list, tuple)):
            self._record(name, val, TypeError(f"expected structured data, got {type(val)}"))
            return default
        # Instantiate
        try:
            if hasattr(cls, "from_dict"):
                return cls.from_dict(val, mode=self.mode)  # type: ignore
            elif is_dataclass(cls) and isinstance(val, dict):
                return cls(**val)  # type: ignore
        except Exception as e:
            self._record(name, val, e)
            return default
        return default


    def map_model(self, cls: Type[T], val: Any, name: str, id_field: Optional[str] = None) -> Dict[str, T]:
        """Parses {key: Object} map."""
        out: Dict[str, T] = {}
        if val is None: return out
        if not isinstance(val, dict):
            self._record(name, val, TypeError(f"{name} must be dict, got {type(val).__name__}"))
            return out
        for k, v in val.items():
            key_norm = self.id(k, f"{name}.key")
            if key_norm is None: continue

            obj_key = f"{name}.{key_norm}"
            obj = self.model(cls, v, obj_key)

            if obj is None:
                self._record(obj_key, v, TypeError(f"Invalid object for key '{key_norm}'"))
                if not self.strict:
                    try:
                        obj = cls(_mode=self.mode)  # type: ignore
                    except:
                        pass

            if obj is not None:
                if id_field and hasattr(obj, id_field):
                    internal_id = getattr(obj, id_field, None)
                    if internal_id is None:
                        setattr(obj, id_field, key_norm)
                    elif internal_id != key_norm:
                        self._record(f"{obj_key}.id_mismatch", v, ValueError("Key!=ID"))
                        if not self.strict: setattr(obj, id_field, key_norm)
                out[key_norm] = obj
        return out

class Validator:
    """
    Context-aware validation helper.
    Encapsulates mode checking (Strict/Lenient), error recording, and healing logic.
    """
    def __init__(self, obj: Any, mode_override: Union[ParseMode, str, None] = None):
        self.obj = obj
        self.mode, self.strict = _setup_validate(obj._mode, mode_override)
        self.extra = obj.extra
        self.valid = True
        self.owner = obj.__class__.__name__

    def _key(self, name: str) -> str:
        return f"{self.owner}.{name}"

    def check(self, condition: bool, key_suffix: str, exc: Union[Exception, str],
              raw: Any = None, heal: Optional[callable] = None) -> bool:
        """
        Generic assertion.
        If False: raises in STRICT, logs + heals in LENIENT.
        """
        if condition:
            return True

        # Prepare Exception
        e = exc if isinstance(exc, Exception) else ValueError(str(exc))

        # 1. STRICT: Raise immediately (Fail Fast)
        if self.strict:
            self.valid = False  # Optional, but good for state hygiene before crashing
            raise e

        # 2. LENIENT: Log and Continue
        note_or_raise(self.extra, self._key(key_suffix), e, mode=self.mode, raw=raw)

        # Heal
        if heal:
            try:
                heal()
                # If heal succeeds, we consider the object "repaired" (validity preserved)
                return True
            except Exception:
                # If healing crashes, the object is definitely broken
                self.valid = False
        else:
            # If there is no way to heal the violation, the object remains invalid
            self.valid = False

        return False

    # --- Common specialized checks ---

    def check_nested(self, child: Any) -> bool:
        """Validates a child object if it exists."""
        if child and hasattr(child, 'validate'):
            if not child.validate(mode=self.mode):
                self.valid = False
                return False
        return True

    def check_nested_map(self, children: Dict[str, Any]) -> bool:
        """Validates a dictionary of child objects."""
        if not children: return True
        all_valid = True
        for child in children.values():
            if not self.check_nested(child):
                all_valid = False
        return all_valid

    def check_gt_zero(self, val: Any, name: str, unit_aware: bool = False,
                      reset_to: Any = None) -> bool:
        """Check value > 0."""
        if val is None: return True
        is_valid = val.magnitude > 0 if (unit_aware and isinstance(val, Quantity)) else val > 0
        return self.check(
            is_valid, name, f"{name} must be > 0", raw=val,
            heal=lambda: setattr(self.obj, name.split('.')[-1], reset_to)
        )

    def check_ge_zero(self, val: Any, name: str, unit_aware: bool = False,
                      reset_to: Any = None) -> bool:
        """Check value >= 0."""
        if val is None: return True
        is_valid = val.magnitude >= 0 if (unit_aware and isinstance(val, Quantity)) else val >= 0
        return self.check(
            is_valid, name, f"{name} must be >= 0", raw=val,
            heal=lambda: setattr(self.obj, name.split('.')[-1], reset_to)
        )

    def check_finite(self, val: Any, name: str, reset_to: Any = None) -> bool:
        """Check numbers are not NaN or Inf."""
        if val is None: return True
        is_valid = np.isfinite(val.magnitude) if isinstance(val, Quantity) else np.isfinite(val)
        return self.check(
            is_valid, name, f"{name} must be finite", raw=val,
            heal=lambda: setattr(self.obj, name.split('.')[-1], reset_to)
        )

def _auto_to_dict(obj: Any, unit_map: Dict[str, str] = None, key_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Automatically converts a dataclass to a dict using introspection and mapping rules.

    Args:
        obj: The dataclass instance.
        unit_map: Dict mapping field names to target units (e.g. {"x": "nm"}).
                  Resulting key will be "{field}_{unit}" (e.g. "x_nm").
        key_map:  Dict for explicit renaming (e.g. {"exposure_min": "exposure_ms_min"}).
                  Overrides default unit suffix naming if present.
    """
    if unit_map is None: unit_map = {}
    if key_map is None: key_map = {}

    is_strict_mode = hasattr(obj, "_mode") and is_strict(obj._mode)

    out = {}

    # Iterate over all fields defined in the dataclass
    for field in dataclasses.fields(obj):
        name = field.name
        val = getattr(obj, name)

        if name.startswith("_") or name == "extra" or val is None:
            continue

        key = key_map.get(name, name)
        target_unit = unit_map.get(name)

        # 1. Happy Path: Field is in _UNITS
        if target_unit:
            key = f"{key}_{target_unit}" if name not in key_map else key
            try:
                # Try strict conversion
                if isinstance(val, (list, tuple)):
                    out[key] = [serialize_quantity(v, target_unit) for v in val]
                else:
                    out[key] = serialize_quantity(val, target_unit)
            except Exception as e:
                # CONVERSION FAILED
                if is_strict_mode:
                    raise ValueError(f"Serialization failed for {name}: {e}")

                # Lenient Fallback: Keep original structure + Warning
                out[key] = _jsonable(val)  # Dump as {magnitude: x, unit: y}
                if hasattr(obj, "extra") and isinstance(obj.extra, Extras):
                    obj.extra.notes.setdefault("serialization_errors", []).append(
                        f"Field '{name}' failed conversion to {target_unit}: {e}"
                    )

        # 2. The Guard: Field is a Quantity BUT MISSING from _UNITS
        elif isinstance(val, Quantity):
            msg = f"Serialization Error: Field '{obj.__class__.__name__}.{name}' is a Quantity but is missing from _UNITS."

            if is_strict_mode:
                # Ruthless: Crash the control plane
                raise ValueError(msg)
            else:
                # Lenient: Record the shame, save as object, and continue
                if hasattr(obj, "extra") and isinstance(obj.extra, Extras):
                    obj.extra.notes.setdefault("serialization_warnings", []).append(msg)

                # Fallback to {magnitude: x, unit: y}
                out[key] = _jsonable(val)

        # 3. Standard Recursion
        elif hasattr(val, "to_dict"):
            out[key] = val.to_dict()
        elif isinstance(val, (list, tuple)):
            out[key] = [v.to_dict() if hasattr(v, "to_dict") else _jsonable(v) for v in val]
        elif isinstance(val, dict):
            out[key] = {k: (v.to_dict() if hasattr(v, "to_dict") else _jsonable(v)) for k, v in val.items()}
        else:
            out[key] = _jsonable(val)

    return _finish_to_dict(out, getattr(obj, "extra", None))

def _auto_from_dict(
        cls: Type[T],
        data: Any,
        mode: Union[ParseMode, str, None],
        alias_map: Optional[Dict[str, Union[str, List[str]]]] = None
) -> T:
    """
    Automates instantiation from a dict with strict precedence rules:
    1. Direct key present (even if None) -> Use it.
    2. Direct key missing -> Check aliases.
    3. Both missing -> Pass nothing (let __post_init__ apply defaults).
    """
    alias_map = alias_map or {}

    # 1. Introspect class to find all valid field names
    cls_fields = {f.name for f in dataclasses.fields(cls) if not f.name.startswith('_')}

    # 2. Flatten aliases for known-key exclusions
    all_aliases = set()
    for v in alias_map.values():
        if isinstance(v, str):
            all_aliases.add(v)
        else:
            all_aliases.update(v)

    # 3. Setup / Validation / Harvesting Unknowns
    d_dict, mode, extra = _setup_from_dict(
        cls, data, mode,
        known_keys=cls_fields,
        aliases=all_aliases
    )

    if d_dict is None:
        return replace(data, _mode=mode)

    # 4. Build Arguments
    kwargs = {}
    for name in cls_fields:
        if name == "extra": continue

        # PRIORITY 1: Exact Match
        # We check existence (name in d_dict) to preserve explicit None values.
        if name in d_dict:
            kwargs[name] = d_dict[name]

        # PRIORITY 2: Alias Match (Only if primary key is COMPLETELY MISSING)
        elif name in alias_map:
            param = alias_map[name]
            if isinstance(param, str):
                # Single alias
                if param in d_dict:
                    kwargs[name] = d_dict[param]
            else:
                # List of aliases (first found wins)
                for alias in param:
                    if alias in d_dict:
                        kwargs[name] = d_dict[alias]
                        break

    # Note: If a field is missing from kwargs, the dataclass __init__
    # uses its defined default (usually None), which FieldParser then handles.
    return cls(**kwargs, extra=extra, _mode=mode)

# =============================================================================
# Structures (Dataclasses)
# =============================================================================

@dataclass
class Point:
    """
    Simple 3D point used for geometry and alignments.

    Role:     Structure
    Context:  Both (Data-plane / Control-plane)
    Category: A (Config)

    Attributes:
        x (Optional[float]): X coordinate.
            None Behavior: Defaulted to 0.0.
        y (Optional[float]): Y coordinate.
            None Behavior: Defaulted to 0.0.
        z (Optional[float]): Z coordinate.
            None Behavior: Defaulted to 0.0.
        name (Optional[str]): Label for this point.
            None Behavior: Preserved as None.
    """
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    name: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "Point")
        # Category A: Apply Defaults (0.0) safely
        self.x = p.float(self.x, "x", default=0.0)
        self.y = p.float(self.y, "y", default=0.0)
        self.z = p.float(self.z, "z", default=0.0)
        self.name = p.str(self.name, "name")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        for axis in ["x", "y", "z"]:
            v.check_finite(getattr(self, axis), f"coordinates.{axis}", reset_to=0.0)
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "Point":
        if isinstance(d, (list, tuple)) and len(d) in (2, 3):
            return Point(x=d[0], y=d[1], z=d[2] if len(d) == 3 else 0.0, _mode=as_parse_mode(mode))
        return _auto_from_dict(Point, d, mode)

@dataclass
class ROI:
    """
    Region of Interest specified in pixel coordinates for a detector sensor.

    Role:     Gatekeeper (Hardware Window)
    Context:  Control-plane
    Category: A (Config)

    Attributes:
        x, y (Optional[int]): Top-left corner of the window. Units: px.
            None Behavior: Defaulted to 0.
        width, height (Optional[int]): Dimensions of the window. Units: px.
            None Behavior: Defaulted to 512.
    """
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ROI")
        # Category A: Apply Defaults
        self.x = p.int(self.x, "x", default=0)
        self.y = p.int(self.y, "y", default=0)
        self.width = p.int(self.width, "width", default=512)
        self.height = p.int(self.height, "height", default=512)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check(self.x >= 0, "x", "ROI.x must be >= 0", raw=self.x, heal=lambda: setattr(self, 'x', 0))
        v.check(self.y >= 0, "y", "ROI.y must be >= 0", raw=self.y, heal=lambda: setattr(self, 'y', 0))
        v.check(self.width > 0, "width", "ROI.width must be > 0", raw=self.width,
                heal=lambda: setattr(self, 'width', 512))
        v.check(self.height > 0, "height", "ROI.height must be > 0", raw=self.height,
                heal=lambda: setattr(self, 'height', 512))
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ROI":
        if isinstance(d, (list, tuple)) and len(d) == 4:
            m = as_parse_mode(mode)
            ex = Extras() if is_strict(m) else normalize_extra_lenient(None, "ROI")
            return ROI(x=d[0], y=d[1], width=d[2], height=d[3], extra=ex, _mode=m)

        return _auto_from_dict(ROI, d, mode, alias_map={
            "width": "w", "height": "h"
        })

@dataclass
class StagePosition:
    """
    Stage coordinate representation for both telemetry (State) and movement (Intent).

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Both (Data-plane / Control-plane)
    Category: C / D (Measured State / Intent)

    Attributes:
        name (Optional[str]): Label for this position (e.g. "Sample Center").
            None Behavior: Preserved as None.
        x, y, z (Optional[Quantity]): Translation coordinates. Units: nm.
            None Behavior: Snapshot (Unknown) | Intent (No Change/Wildcard).
        r (Optional[Quantity]): Rotation coordinate. Units: deg.
            None Behavior: Snapshot (Unknown) | Intent (No Change/Wildcard).
        tilt_x, tilt_y (Optional[Quantity]): Alpha/Beta tilts. Units: deg.
            None Behavior: Snapshot (Unknown) | Intent (No Change/Wildcard).
        coordinate_system (Optional[str]): Reference frame ID.
            None Behavior: Preserved as None.

    Behavior:
        1. Snapshot (Telemetry): Represents the physical location of the stage.
           Fields capable of being read are populated; others are None.
        2. Intent (Movement): Represents a target or a delta.
           - Absolute Move: Fields set to None are ignored (axes do not move).
           - Relative Move: Fields set to None imply 0 delta.
        3. Arithmetic: Supports vector math (pos_a + pos_b, target - current).
           Math operations propagate None (None + 10 = None).
    """
    name: Optional[str] = None
    x: Optional["Quantity"] = None
    y: Optional["Quantity"] = None
    z: Optional["Quantity"] = None
    r: Optional["Quantity"] = None
    tilt_x: Optional["Quantity"] = None
    tilt_y: Optional["Quantity"] = None
    coordinate_system: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    _UNITS = {
        "x": Units.NM, "y": Units.NM, "z": Units.NM,
        "r": Units.DEG, "tilt_x": Units.DEG, "tilt_y": Units.DEG
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "StagePosition")
        # Category C: Preserve None
        self.name = p.str(self.name, "name")
        self.coordinate_system = p.str(self.coordinate_system, "coordinate_system")
        self.x = p.qty(self.x, "x", Units.NM)
        self.y = p.qty(self.y, "y", Units.NM)
        self.z = p.qty(self.z, "z", Units.NM)
        self.r = p.qty(self.r, "r", Units.DEG)
        self.tilt_x = p.qty(self.tilt_x, "tilt_x", Units.DEG)
        self.tilt_y = p.qty(self.tilt_y, "tilt_y", Units.DEG)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        for axis in ["x", "y", "z", "r", "tilt_x", "tilt_y"]:
            v.check_finite(getattr(self, axis), axis)  # Default heal is no-op (preserved as None)
        return v.valid

    def __add__(self, other: 'StagePosition') -> 'StagePosition':
        if not isinstance(other, StagePosition): return NotImplemented
        def add(a, b):
            return (a + b) if (a is not None and b is not None) else None
        return StagePosition(
            name=self.name,
            x=add(self.x, other.x), y=add(self.y, other.y), z=add(self.z, other.z),
            r=add(self.r, other.r), tilt_x=add(self.tilt_x, other.tilt_x), tilt_y=add(self.tilt_y, other.tilt_y),
            coordinate_system=self.coordinate_system,
            _mode=self._mode
        )

    def __sub__(self, other: 'StagePosition') -> 'StagePosition':
        if not isinstance(other, StagePosition): return NotImplemented
        def sub(a, b):
            return (a - b) if (a is not None and b is not None) else None
        return StagePosition(
            name=self.name,
            x=sub(self.x, other.x), y=sub(self.y, other.y), z=sub(self.z, other.z),
            r=sub(self.r, other.r), tilt_x=sub(self.tilt_x, other.tilt_x), tilt_y=sub(self.tilt_y, other.tilt_y),
            coordinate_system=self.coordinate_system,
             _mode=self._mode
        )

    def is_close(self, other: 'StagePosition', tol_nm: float = 1.0, tol_deg: float = 1e-3) -> bool:
        """
        Checks if the 'other' position satisfies the constraints defined by 'self'.

        This is an asymmetric 'Target vs. Current' check:
          1. Wildcard: If self.axis is None, it ignores that axis in 'other'.
          2. Constraint: If self.axis is set, 'other' MUST have a value and be within tolerance.
        """
        if not isinstance(other, StagePosition):
            raise TypeError(f"Cannot compare StagePosition with {type(other)}")

        def chk(target_val, current_val, unit, tol):
            if target_val is None: return True
            if current_val is None: return False
            return abs(target_val.to(unit).magnitude - current_val.to(unit).magnitude) <= tol

        return (chk(self.x, other.x, Units.NM, tol_nm) and
                chk(self.y, other.y, Units.NM, tol_nm) and
                chk(self.z, other.z, Units.NM, tol_nm) and
                chk(self.r, other.r, Units.DEG, tol_deg) and
                chk(self.tilt_x, other.tilt_x, Units.DEG, tol_deg) and
                chk(self.tilt_y, other.tilt_y, Units.DEG, tol_deg))

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> 'StagePosition':
        return _auto_from_dict(StagePosition, d, mode, alias_map={
            "x": "x_nm", "y": "y_nm", "z": "z_nm",
            "r": "r_deg", "tilt_x": "tilt_x_deg", "tilt_y": "tilt_y_deg"
        })

@dataclass
class StageSystemSettings:
    """
    Safety limits and step sizes for physical stage motion.

    Role:     Gatekeeper (System Limits)
    Context:  Control-plane
    Category: A (Strict Runtime Config)

    Attributes:
        enabled (Optional[bool]): Master toggle for stage interaction.
            None Behavior: Defaulted to True.
        can_x, can_y, can_z (Optional[bool]): Individual axis toggles.
            None Behavior: Defaulted to True.
        can_r, can_tilt_x, can_tilt_y (Optional[bool]): Individual axis toggles.
            None Behavior: Defaulted to False.
        x_limits, y_limits, z_limits (Optional[Tuple[Quantity, Quantity]]): Physical travel bounds.
            None Behavior: Preserved as None (Axis disabled).
        r_limits, tilt_x_limits, tilt_y_limits (Optional[Tuple[Quantity, Quantity]]): Physical travel bounds.
            None Behavior: Preserved as None (Axis disabled).
        max_step_distance (Optional[Quantity]): Safety cap for XY/Z travel per move. Units: nm.
            None Behavior: Defaulted to 50,000 nm.
        max_step_deg (Optional[Quantity]): Safety cap for rotation per move. Units: deg.
            None Behavior: Defaulted to 1.0 deg.
        eucentric_z (Optional[Quantity]): Calibrated sample height. Units: nm.
            None Behavior: Preserved as None.
        settle_time (Optional[Quantity]): Time to wait for vibration damping. Units: s.
            None Behavior: Defaulted to 0.2 s.
        timeout (Optional[Quantity]): Max duration for move operations. Units: s.
            None Behavior: Defaulted to 10.0 s.
    """
    enabled: Optional[bool] = None
    can_x: Optional[bool] = None
    can_y: Optional[bool] = None
    can_z: Optional[bool] = None
    can_r: Optional[bool] = None
    can_tilt_x: Optional[bool] = None
    can_tilt_y: Optional[bool] = None
    x_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    y_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    z_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    r_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    tilt_x_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    tilt_y_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    max_step_distance: Optional["Quantity"] = None
    max_step_deg: Optional["Quantity"] = None
    eucentric_z: Optional["Quantity"] = None
    settle_time: Optional["Quantity"] = None
    timeout: Optional["Quantity"] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "x_limits": Units.NM, "y_limits": Units.NM, "z_limits": Units.NM,
        "r_limits": Units.DEG, "tilt_x_limits": Units.DEG, "tilt_y_limits": Units.DEG,
        "max_step_distance": Units.NM, "max_step_deg": Units.DEG,
        "eucentric_z": Units.NM, "settle_time": Units.SEC, "timeout": Units.SEC
    }

    _KEYS = {
        "max_step_distance": "max_step_nm",
        "max_step_deg": "max_step_deg"
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "StageSystemSettings")
        # Category A: Apply Defaults (Booleans)
        self.enabled = p.bool(self.enabled, "enabled", default=True)
        self.can_x = p.bool(self.can_x, "can_x", default=True)
        self.can_y = p.bool(self.can_y, "can_y", default=True)
        self.can_z = p.bool(self.can_z, "can_z", default=True)
        self.can_r = p.bool(self.can_r, "can_r", default=False)
        self.can_tilt_x = p.bool(self.can_tilt_x, "can_tilt_x", default=False)
        self.can_tilt_y = p.bool(self.can_tilt_y, "can_tilt_y", default=False)

        # Category C: Limits (If None, axis will be disabled during validation)
        self.x_limits = p.pair_qty(self.x_limits, "x_limits", Units.NM)
        self.y_limits = p.pair_qty(self.y_limits, "y_limits", Units.NM)
        self.z_limits = p.pair_qty(self.z_limits, "z_limits", Units.NM)
        self.r_limits = p.pair_qty(self.r_limits, "r_limits", Units.DEG)
        self.tilt_x_limits = p.pair_qty(self.tilt_x_limits, "tilt_x_limits", Units.DEG)
        self.tilt_y_limits = p.pair_qty(self.tilt_y_limits, "tilt_y_limits", Units.DEG)

        # Category A: Apply Defaults (Safety Settings)
        self.max_step_distance = p.qty(self.max_step_distance, "max_step_distance", Units.NM,
                                       default=Q_(50000.0, Units.NM))
        self.max_step_deg = p.qty(self.max_step_deg, "max_step_deg", Units.DEG, default=Q_(1.0, Units.DEG))
        self.eucentric_z = p.qty(self.eucentric_z, "eucentric_z", Units.NM)
        self.settle_time = p.qty(self.settle_time, "settle_time", Units.SEC, default=Q_(0.2, Units.SEC))
        self.timeout = p.qty(self.timeout, "timeout", Units.SEC, default=Q_(10.0, Units.SEC))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)

        # 1. Limit Logic (Min < Max)
        def _validate_range(lims, name):
            if not lims: return
            v.check(lims[0] <= lims[1], f"{name}_limits", "min > max", raw=lims,
                    heal=lambda: setattr(self, f"{name}_limits", (lims[1], lims[0])))

        for axis in ["x", "y", "z", "r", "tilt_x", "tilt_y"]:
            _validate_range(getattr(self, f"{axis}_limits"), axis)

            # 2. Consistency (Enabled -> Limits must exist)
            is_enabled = getattr(self, f"can_{axis}")
            has_limits = getattr(self, f"{axis}_limits") is not None
            v.check(not (is_enabled and not has_limits), f"{axis}_safety",
                    f"Axis {axis} enabled without limits",
                    heal=lambda: setattr(self, f"can_{axis}", False))

        # 3. Value Checks
        v.check_gt_zero(self.max_step_distance, "max_step_distance", unit_aware=True, reset_to=Q_(50000.0, Units.NM))
        v.check_ge_zero(self.settle_time, "settle_time", unit_aware=True, reset_to=Q_(0.2, Units.SEC))

        # 4. Eucentric Check
        if self.eucentric_z is not None and self.z_limits:
            z_min, z_max = self.z_limits
            v.check(z_min <= self.eucentric_z <= z_max, "eucentric_z",
                    f"eucentric_z {self.eucentric_z} outside limits",
                    heal=lambda: setattr(self, 'eucentric_z', None))

        return v.valid

    def is_safe_move(self, target: StagePosition, current: Optional[StagePosition] = None,
                     relative: bool = False) -> SafetyCheck:
        """
        RUNTIME CHECK: External Safety.
        Returns a SafetyCheck object (True/False + reasons) without modifying self.extra.

        Checks:
        1. Is the axis enabled? (can_x, can_tilt_x, etc.)
        2. Is the destination within absolute limits? (x_limits, etc.)
        3. Is the step size within dynamic limits? (max_step_distance, etc.)
        """
        reasons = []

        # --- 1. Validate Axis Availability (Intents vs Capabilities) ---
        # If the user intends to move an axis (value is not None), that axis MUST be enabled.
        axes_map = {
            "x": self.can_x, "y": self.can_y, "z": self.can_z,
            "r": self.can_r, "tilt_x": self.can_tilt_x, "tilt_y": self.can_tilt_y
        }

        for axis, is_enabled in axes_map.items():
            if getattr(target, axis) is not None and not is_enabled:
                reasons.append(f"Movement requested on disabled axis: '{axis}'")

        # If we already failed basic capability checks, return early to avoid math errors
        if reasons:
            return SafetyCheck(allowed=False, reasons=reasons)

        # --- 2. Resolve Absolute Target & Step Vector ---
        abs_target = target
        step_vector = None

        if relative:
            if current is None:
                return SafetyCheck.failure("Cannot perform relative move without current position")

            # Check if we have a valid starting point for all requested deltas
            for axis in ["x", "y", "z", "r", "tilt_x", "tilt_y"]:
                if getattr(target, axis) is not None and getattr(current, axis) is None:
                    return SafetyCheck.failure(f"Relative move on '{axis}' impossible: current position unknown.")

            abs_target = current + target
            step_vector = target
        else:
            # Absolute move
            if current:
                step_vector = target - current
            # If current is None, we can still check absolute limits, but skip step checks

        # --- 3. Check Static Limits (Boundaries) ---
        def check_bound(val, lims, name):
            if val is not None and lims:
                if not (lims[0] <= val <= lims[1]):
                    reasons.append(f"{name} target {val} outside limits {lims}")

        check_bound(abs_target.x, self.x_limits, "x")
        check_bound(abs_target.y, self.y_limits, "y")
        check_bound(abs_target.z, self.z_limits, "z")
        check_bound(abs_target.r, self.r_limits, "r")
        check_bound(abs_target.tilt_x, self.tilt_x_limits, "tilt_x")
        check_bound(abs_target.tilt_y, self.tilt_y_limits, "tilt_y")

        # --- 4. Check Dynamic Limits (Step Size) ---
        if step_vector is not None:
            # XY Euclidian Distance
            if step_vector.x is not None or step_vector.y is not None:
                dx = step_vector.x if step_vector.x is not None else Q_(0, Units.NM)
                dy = step_vector.y if step_vector.y is not None else Q_(0, Units.NM)
                distance = (dx.to(Units.NM).magnitude ** 2 + dy.to(Units.NM).magnitude ** 2) ** 0.5
                max_dist_nm = self.max_step_distance.to(Units.NM).magnitude

                if distance > max_dist_nm:
                    reasons.append(f"XY step {distance:.1f}nm exceeds limit {max_dist_nm:.1f}nm")

            # Z Step Limit
            if step_vector.z is not None:
                d_z = abs(step_vector.z.to(Units.NM).magnitude)
                max_dist_nm = self.max_step_distance.to(Units.NM).magnitude

                if d_z > max_dist_nm:
                    reasons.append(f"Z step {d_z:.1f}nm exceeds limit {max_dist_nm:.1f}nm")

            # Tilt Step Limits
            if step_vector.tilt_x is not None:
                d_tilt = abs(step_vector.tilt_x)
                if d_tilt > self.max_step_deg:
                    reasons.append(f"Tilt X step {d_tilt} exceeds limit {self.max_step_deg}")

            if step_vector.tilt_y is not None:
                d_tilt = abs(step_vector.tilt_y)
                if d_tilt > self.max_step_deg:
                    reasons.append(f"Tilt Y step {d_tilt} exceeds limit {self.max_step_deg}")

        return SafetyCheck(allowed=(len(reasons) == 0), reasons=reasons)

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS, key_map=self._KEYS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageSystemSettings":
        return _auto_from_dict(StageSystemSettings, d, mode, alias_map={
            "x_limits": "x_limits_nm", "y_limits": "y_limits_nm", "z_limits": "z_limits_nm",
            "r_limits": "r_limits_deg", "tilt_x_limits": "tilt_x_limits_deg", "tilt_y_limits": "tilt_y_limits_deg",
            "max_step_distance": "max_step_nm", "max_step_deg": "max_step_angle",
            "eucentric_z": "eucentric_z_nm", "settle_time": "settle_time_s", "timeout": "timeout_s"
        })

@dataclass
class BeamSettings:
    """
    Optical parameters of the electron beam (Gun + Condensers).

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Both (Data-plane / Control-plane)
    Category: C / D (Measured State / Intent)

    Attributes:
        voltage (Optional[Quantity]): Accelerating voltage. Units: kV.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        beam_current (Optional[Quantity]): Measured current. Units: nA.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        spot_size (Optional[int]): Beam focus/condenser index.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        convergence_angle (Optional[Quantity]): Alpha angle. Units: mrad.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        beam_shift (Optional[Point]): Beam alignment shift.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        condenser_stigmation (Optional[Point]): Stigmator coil values.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        gun_tilt (Optional[Point]): Gun alignment tilt.
            None Behavior: Snapshot (Unknown) | Intent (No Change).

    Behavior:
        1. Snapshot (Telemetry): Represents the active state of the column.
        2. Intent (Control): Represents a partial configuration change.
           Example: BeamSettings(spot_size=3) will change ONLY the spot size,
           leaving voltage and alignments untouched.
    """
    voltage: Optional["Quantity"] = None
    beam_current: Optional["Quantity"] = None
    spot_size: Optional[int] = None
    convergence_angle: Optional["Quantity"] = None

    # Gun/Condenser alignments
    beam_shift: Optional[Point] = None
    condenser_stigmation: Optional[Point] = None  # Renamed from 'stigmation'
    gun_tilt: Optional[Point] = None  # Added missing alignment

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "voltage": Units.KV,
        "beam_current": Units.NA,
        "convergence_angle": Units.MRAD
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "BeamSettings")
        self.voltage = p.qty(self.voltage, "voltage", Units.KV)
        self.beam_current = p.qty(self.beam_current, "beam_current", Units.NA)
        self.spot_size = p.int(self.spot_size, "spot_size")
        self.convergence_angle = p.qty(self.convergence_angle, "convergence_angle", Units.MRAD)

        self.beam_shift = p.model(Point, self.beam_shift, "beam_shift")
        self.condenser_stigmation = p.model(Point, self.condenser_stigmation, "condenser_stigmation")
        self.gun_tilt = p.model(Point, self.gun_tilt, "gun_tilt")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.beam_shift)

        v.check_ge_zero(self.convergence_angle, "convergence_angle", unit_aware=True, reset_to=None)
        v.check_ge_zero(self.voltage, "voltage", unit_aware=True, reset_to=None)
        v.check_ge_zero(self.beam_current, "beam_current", unit_aware=True, reset_to=None)
        v.check_ge_zero(self.spot_size, "spot_size", reset_to=None)

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSettings":
        return _auto_from_dict(BeamSettings, d, mode, alias_map={
            "voltage": "voltage_kv", "beam_current": "beam_current_na",
            "convergence_angle": "convergence_angle_mrad"
        })

# Alias for semantic clarity in Read-Only contexts
BeamState = BeamSettings

@dataclass
class BeamSystemSettings:
    """
    Safety limits and capabilities for the Illumination system.

    Role:     Gatekeeper (System Limits)
    Context:  Control-plane
    Category: A (Strict Runtime Config)

    Attributes:
        enabled (Optional[bool]): Master toggle for beam control.
            None Behavior: Defaulted to True.
        default_beam (Optional[BeamSettings]): Baseline settings for reset.
            None Behavior: Structural Default (Empty object).
        voltage_limits (Optional[Tuple[Quantity, Quantity]]): Min/Max HT. Units: kV.
            None Behavior: Preserved as None (Unlimited).
        beam_current_limits (Optional[Tuple[Quantity, Quantity]]): Min/Max Current. Units: nA.
            None Behavior: Preserved as None (Unlimited).
        spot_size_limits (Optional[Tuple[int, int]]): Valid range for spot indices.
            None Behavior: Preserved as None (Unlimited).
        convergence_angle_limits (Optional[Tuple[Quantity, Quantity]]): Valid range. Units: mrad.
            None Behavior: Preserved as None (Unlimited).
    """
    enabled: Optional[bool] = None
    default_beam: Optional[BeamSettings] = None
    voltage_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    beam_current_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    spot_size_limits: Optional[Tuple[int, int]] = None
    convergence_angle_limits: Optional[Tuple["Quantity", "Quantity"]] = None

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "voltage_limits": Units.KV,
        "beam_current_limits": Units.NA,
        "convergence_angle_limits": Units.MRAD,
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "BeamSystemSettings")
        self.enabled = p.bool(self.enabled, "enabled", default=True)
        self.spot_size_limits = p.pair_int(self.spot_size_limits, "spot_size_limits")
        self.voltage_limits = p.pair_qty(self.voltage_limits, "voltage_limits", Units.KV)
        self.beam_current_limits = p.pair_qty(self.beam_current_limits, "beam_current_limits", Units.NA)
        self.convergence_angle_limits = p.pair_qty(self.convergence_angle_limits, "convergence_angle_limits",
                                                   Units.MRAD)
        # Category B: Structural Default
        self.default_beam = p.model(BeamSettings, self.default_beam, "default_beam", default=BeamSettings(_mode=p.mode))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.default_beam)

        def _check_range(rng, name):
            if not rng: return
            mn, mx = rng
            # 1. Range Integrity
            v.check(mn <= mx, name, "min > max",
                    heal=lambda: setattr(self, name, (mx, mn)))

            # 2. Non-Negativity (use updated values if healed)
            # Re-fetch in case strict=False and we swapped
            curr_rng = getattr(self, name)
            curr_min = curr_rng[0]

            is_neg = curr_min.magnitude < 0 if hasattr(curr_min, "magnitude") else curr_min < 0
            if is_neg:
                zero = Q_(0, curr_min.units) if hasattr(curr_min, "units") else 0
                v.check(False, name, "min < 0",
                        heal=lambda: setattr(self, name, (zero, curr_rng[1])))

        _check_range(self.voltage_limits, "voltage_limits")
        _check_range(self.beam_current_limits, "beam_current_limits")
        _check_range(self.convergence_angle_limits, "convergence_angle_limits")
        _check_range(self.spot_size_limits, "spot_size_limits")

        return v.valid

    def is_safe_beam(self, target: BeamSettings) -> SafetyCheck:
        """
        Runtime Gatekeeper: Checks if a target beam configuration respects system limits.
        """
        reasons = []

        def _check(val, limit_tuple, name):
            if val is not None and limit_tuple is not None:
                min_lim, max_lim = limit_tuple
                if not (min_lim <= val <= max_lim):
                    reasons.append(f"Beam {name} {val} outside limits {limit_tuple}")

        _check(target.voltage, self.voltage_limits, "voltage")
        _check(target.beam_current, self.beam_current_limits, "current")
        _check(target.convergence_angle, self.convergence_angle_limits, "convergence")

        if target.spot_size is not None and self.spot_size_limits:
            min_s, max_s = self.spot_size_limits
            if not (min_s <= target.spot_size <= max_s):
                reasons.append(f"Spot size {target.spot_size} outside {self.spot_size_limits}")

        return SafetyCheck(allowed=(len(reasons) == 0), reasons=reasons)

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamSystemSettings":
        return _auto_from_dict(BeamSystemSettings, d, mode, alias_map={
            "voltage_limits": "voltage_limits_kv", "beam_current_limits": "beam_current_limits_na",
            "convergence_angle_limits": "convergence_angle_limits_mrad"
        })

@dataclass
class ProjectionSettings:
    """
    Imaging system controls (Objective + Projectors).

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Both (Data-plane / Control-plane)
    Category: C / D (Measured State / Intent)

    Attributes:
        optical_mode (Optional[str]): Active lens program (e.g. "IMAGING", "DIFFRACTION").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        magnification (Optional[int]): Magnification.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        defocus (Optional[Quantity]): Deviation from focus. Units: nm.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        camera_length (Optional[Quantity]): Effective camera length (Diffraction). Units: mm.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        objective_stigmation (Optional[Point]): Objective stigmator coils.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        image_shift (Optional[Point]): Image shift coils.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        diffraction_shift (Optional[Point]): Diffraction shift coils.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        screen_position (Optional[str]): Mechanical screen state ("UP", "DOWN").
            None Behavior: Snapshot (Unknown) | Intent (No Change).

    Behavior:
        1. Snapshot (Telemetry): Represents the active imaging state.
        2. Intent (Control): Represents a partial configuration change.
           - Mode Switching: If `optical_mode` changes (e.g. to DIFFRACTION),
             associated fields (e.g. `camera_length`) become mandatory.
    """
    optical_mode: Optional[str] = None  # "IMAGING", "DIFFRACTION", "LAD"

    # Imaging Parameters (IMAGING mode)
    magnification: Optional[int] = None
    defocus: Optional["Quantity"] = None
    objective_stigmation: Optional[Point] = None
    image_shift: Optional[Point] = None

    # Diffraction Parameters (DIFFRACTION mode)
    camera_length: Optional["Quantity"] = None
    diffraction_shift: Optional[Point] = None

    screen_position: Optional[str] = None  # "UP", "DOWN"

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "defocus": Units.NM,
        "camera_length": Units.MM
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ProjectionSettings")
        self.optical_mode = p.str(self.optical_mode, "optical_mode")
        self.magnification = p.int(self.magnification, "magnification")
        self.defocus = p.qty(self.defocus, "defocus", Units.NM)
        self.camera_length = p.qty(self.camera_length, "camera_length", Units.MM)
        self.screen_position = p.str(self.screen_position, "screen_position")

        self.objective_stigmation = p.model(Point, self.objective_stigmation, "objective_stigmation")
        self.image_shift = p.model(Point, self.image_shift, "image_shift")
        self.diffraction_shift = p.model(Point, self.diffraction_shift, "diffraction_shift")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)

        v.check_nested(self.objective_stigmation)
        v.check_nested(self.image_shift)

        # --- NEW VALIDATION LOGIC ---
        v.check_ge_zero(self.camera_length, "camera_length", unit_aware=True, reset_to=None)
        v.check_ge_zero(self.magnification, "magnification", reset_to=None)

        if self.screen_position:
            v.check(self.screen_position in {"UP", "DOWN"}, "screen_position",
                    "Must be UP or DOWN", heal=lambda: setattr(self, 'screen_position', None))

        # Consistency Check: Mode vs Value
        if self.optical_mode == "DIFFRACTION" and self.camera_length is None:
            # In strict mode, if you switch to diffraction, you must know the length
            v.check(False, "missing_cam_len", "Diffraction mode requires camera_length")

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ProjectionSettings":
        return _auto_from_dict(ProjectionSettings, d, mode, alias_map={
            "defocus": "defocus_nm", "camera_length": "camera_length_mm"
        })

@dataclass
class ProjectionSystemSettings:
    """
    Safety limits for the Projection system.

    Role:     Gatekeeper (System Limits)
    Context:  Control-plane
    Category: A (Strict Runtime Config)

    Attributes:
        enabled (Optional[bool]): Master toggle for projection controls.
            None Behavior: Defaulted to True.
        default_projection (Optional[ProjectionSettings]): Baseline settings.
            None Behavior: Structural Default (Empty object).
        camera_length_limits (Optional[Tuple[Quantity, Quantity]]): Valid range for diffraction.
            None Behavior: Preserved as None (Unlimited).
        magnification_limits (Optional[Tuple[int, int]]): Valid range for mag indices.
            None Behavior: Preserved as None (Unlimited).
        defocus_limits (Optional[Tuple[Quantity, Quantity]]): Safety limits for defocus.
            None Behavior: Preserved as None (Unlimited).
    """
    enabled: Optional[bool] = None
    default_projection: Optional[ProjectionSettings] = None

    # Limits
    camera_length_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    magnification_limits: Optional[Tuple[int, int]] = None
    defocus_limits: Optional[Tuple["Quantity", "Quantity"]] = None

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "camera_length_limits": Units.MM,
        "defocus_limits": Units.NM
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ProjectionSystemSettings")
        self.enabled = p.bool(self.enabled, "enabled", default=True)
        self.default_projection = p.model(ProjectionSettings, self.default_projection, "default_projection",
                                          default=ProjectionSettings(_mode=p.mode))

        self.camera_length_limits = p.pair_qty(self.camera_length_limits, "camera_length_limits", Units.MM)
        self.magnification_limits = p.pair_int(self.magnification_limits, "magnification_limits")
        self.defocus_limits = p.pair_qty(self.defocus_limits, "defocus_limits", Units.NM)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        # Standard range logic (simplified for brevity)
        if self.camera_length_limits:
            mn, mx = self.camera_length_limits
            v.check(mn <= mx, "camera_length_limits", "min > max")
        return v.valid

    def is_safe_projection(self, target: ProjectionSettings) -> SafetyCheck:
        reasons = []

        # Check Diffraction Limits
        if target.optical_mode == "DIFFRACTION" and target.camera_length is not None:
            if self.camera_length_limits:
                mn, mx = self.camera_length_limits
                if not (mn <= target.camera_length <= mx):
                    reasons.append(f"Camera length {target.camera_length} outside limits {self.camera_length_limits}")

        # Check Defocus Limits
        if target.defocus is not None and self.defocus_limits:
            mn, mx = self.defocus_limits
            if not (mn <= target.defocus <= mx):
                reasons.append(f"Defocus {target.defocus} outside limits {self.defocus_limits}")

        # Check Mag Limits
        if target.magnification is not None and self.magnification_limits:
            mn, mx = self.magnification_limits
            if not (mn <= target.magnification <= mx):
                reasons.append(f"Magnification {target.magnification} outside limits {self.magnification_limits}")

        return SafetyCheck(allowed=(len(reasons) == 0), reasons=reasons)

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ProjectionSystemSettings":
        return _auto_from_dict(ProjectionSystemSettings, d, mode, alias_map={
            "camera_length_limits": "camera_length_limits_mm",
            "defocus_limits": "defocus_limits_nm"
        })

@dataclass
class DetectorSettings:
    """
    Detector parameters for state reporting and acquisition requests.

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Both (Data-plane / Control-plane)
    Category: C / D (Measured State / Intent)

    Attributes:
        detector_id (Optional[str]): The identifier of the camera.
            None Behavior: Preserved as None.
        exposure (Optional[Quantity]): Integration time. Units: ms.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        binning_index (Optional[int]): Discrete binning level.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        binning_xy (Optional[Tuple[int, int]]): Explicit X/Y binning.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        roi (Optional[ROI]): Sub-region readout window.
            None Behavior: Snapshot (Full Sensor) | Intent (No Change).
        frame_integration (Optional[int]): Number of internal hardware averages.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        gain_index, offset_index (Optional[int]): Sensor amplifier settings.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        digital_rotation (Optional[Quantity]): Post-processing rotation. Units: deg.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        frame_rate (Optional[Quantity]): Continuous acquisition speed. Units: Hz.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        total_frames (Optional[int]): Number of frames to capture (Movie mode).
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        readout_mode (Optional[str]): Sensor readout strategy (e.g. "LINEAR").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        shutter_mode (Optional[str]): Shutter timing (e.g. "PRE_SPECIMEN").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        save_frames (Optional[bool]): Whether to save individual movie frames.
            None Behavior: Snapshot (Unknown) | Intent (No Change).

    Behavior:
        1. Snapshot (Telemetry): Represents the idle state of the camera.
        2. Intent (Control): Used to configure the camera permanently.
        3. Intent (Acquisition): Used ephemerally for a single 'Take Picture' event.
           Values defined here override the persistent camera state for that one shot.
    """
    detector_id: Optional[str] = None
    exposure: Optional["Quantity"] = None
    binning_index: Optional[int] = None
    binning_xy: Optional[Tuple[int, int]] = None
    roi: Optional[ROI] = None
    frame_integration: Optional[int] = None
    gain_index: Optional[int] = None
    offset_index: Optional[int] = None
    digital_rotation: Optional["Quantity"] = None
    frame_rate: Optional["Quantity"] = None  # e.g. 40 Hz
    total_frames: Optional[int] = None  # e.g. 40 frames

    # --- NEW ADVANCED PROPERTIES ---
    readout_mode: Optional[str] = None  # "LINEAR", "COUNTING", "SUPER_RESOLUTION"
    shutter_mode: Optional[str] = None  # "PRE_SPECIMEN", "POST_SPECIMEN"
    save_frames: Optional[bool] = None  # True = save movie stack

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "exposure": Units.MS,
        "digital_rotation": Units.DEG,
        "frame_rate": Units.HZ  # <--- NEW UNIT
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "DetectorSettings")
        self.detector_id = p.id(self.detector_id, "detector_id")
        self.binning_index = p.int(self.binning_index, "binning_index")
        self.binning_xy = p.pair_int(self.binning_xy, "binning_xy")
        self.frame_rate = p.qty(self.frame_rate, "frame_rate", Units.HZ)
        self.total_frames = p.int(self.total_frames, "total_frames")
        self.frame_integration = p.int(self.frame_integration, "frame_integration")
        self.gain_index = p.int(self.gain_index, "gain_index")
        self.offset_index = p.int(self.offset_index, "offset_index")
        self.digital_rotation = p.qty(self.digital_rotation, "digital_rotation", Units.DEG)
        self.exposure = p.qty(self.exposure, "exposure", Units.MS)
        # ROI is preserved as None if missing (tristate)
        self.roi = p.model(ROI, self.roi, "roi", default=None)

        # --- NEW PARSERS ---
        self.readout_mode = p.str(self.readout_mode, "readout_mode")
        self.shutter_mode = p.str(self.shutter_mode, "shutter_mode")
        self.save_frames = p.bool(self.save_frames, "save_frames")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_gt_zero(self.exposure, "exposure", unit_aware=True, reset_to=None)

        if self.binning_xy:
            v.check(self.binning_xy[0] > 0 and self.binning_xy[1] > 0, "binning_xy", "must be > 0",
                    raw=self.binning_xy, heal=lambda: setattr(self, 'binning_xy', None))

        v.check(self.frame_integration is None or self.frame_integration >= 1,
                "frame_integration", "must be >= 1",
                heal=lambda: setattr(self, 'frame_integration', 1))

        v.check(self.gain_index is None or self.gain_index >= 0,
                "gain_index", "must be >= 0",
                heal=lambda: setattr(self, 'gain_index', 0))

        if self.exposure and self.frame_rate and self.total_frames:
            exp_sec = self.exposure.to(Units.SEC).magnitude
            rate_hz = self.frame_rate.to(Units.HZ).magnitude
            # Logic: Exposure * Rate ~= Frames
            if rate_hz > 0:
                calc_frames = int(exp_sec * rate_hz)
                # Tolerate +/- 1 frame rounding error
                v.check(abs(calc_frames - self.total_frames) <= 1, "dose_logic",
                        f"Mismatch: {exp_sec}s * {rate_hz}Hz != {self.total_frames}")

        v.check_nested(self.roi)
        if self.shutter_mode:
            v.check(self.shutter_mode in {"PRE_SPECIMEN", "POST_SPECIMEN", "BOTH"}, "shutter_mode",
                    "Invalid shutter mode")
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSettings":
        return _auto_from_dict(DetectorSettings, d, mode, alias_map={
            "exposure": "exposure_ms",
            "digital_rotation": "digital_rotation_deg",
            "frame_rate": "frame_rate_hz"
        })

# Alias for semantic clarity in Read-Only contexts
DetectorState = DetectorSettings

@dataclass
class DetectorCapabilities:
    """
    Hardware-specific profile of detector features and limits.

    Role:     Gatekeeper (Capabilities)
    Context:  Control-plane
    Category: C (Measured State)

    Attributes:
        can_binning (Optional[bool]): Supports pixel binning.
        binning_index_min, binning_index_max (Optional[int]): Range of bin indices.
        binning_xy_min, binning_xy_max (Optional[Tuple[int, int]]): Range of bin factors.
        exposure_min, exposure_max (Optional[Quantity]): Exposure limits. Units: ms.
        frame_integration_min, frame_integration_max (Optional[int]): Averaging limits.
        roi_size_min, roi_size_max (Optional[Tuple[int, int]]): ROI dimension limits.
        can_gain (Optional[bool]): Supports gain adjustment.
        gain_index_min, gain_index_max (Optional[int]): Gain limits.
        can_offset (Optional[bool]): Supports offset adjustment.
        offset_index_min, offset_index_max (Optional[int]): Offset limits.
        can_digital_rotation (Optional[bool]): Supports hardware rotation.
        digital_rotation_min, digital_rotation_max (Optional[Quantity]): Rotation limits.
    """
    can_binning: Optional[bool] = None
    binning_index_min: Optional[int] = None
    binning_index_max: Optional[int] = None
    binning_xy_min: Optional[Tuple[int, int]] = None
    binning_xy_max: Optional[Tuple[int, int]] = None
    exposure_min: Optional["Quantity"] = None
    exposure_max: Optional["Quantity"] = None
    frame_integration_min: Optional[int] = None
    frame_integration_max: Optional[int] = None
    roi_size_min: Optional[Tuple[int, int]] = None
    roi_size_max: Optional[Tuple[int, int]] = None
    can_gain: Optional[bool] = None
    gain_index_min: Optional[int] = None
    gain_index_max: Optional[int] = None
    can_offset: Optional[bool] = None
    offset_index_min: Optional[int] = None
    offset_index_max: Optional[int] = None
    can_digital_rotation: Optional[bool] = None
    digital_rotation_min: Optional["Quantity"] = None
    digital_rotation_max: Optional["Quantity"] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False, compare=False)

    # Capabilities have unconventional naming (infix units like exposure_ms_min)
    # We map them explicitly to preserve your API contract.
    _UNITS = {
        "exposure_min": Units.MS, "exposure_max": Units.MS,
        "digital_rotation_min": Units.DEG, "digital_rotation_max": Units.DEG
    }
    _KEYS = {
        "exposure_min": "exposure_ms_min", "exposure_max": "exposure_ms_max",
        "digital_rotation_min": "digital_rotation_deg_min",
        "digital_rotation_max": "digital_rotation_deg_max",
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "DetectorCapabilities")
        self.can_binning = p.bool(self.can_binning, "can_binning")
        self.can_gain = p.bool(self.can_gain, "can_gain")
        self.can_offset = p.bool(self.can_offset, "can_offset")
        self.can_digital_rotation = p.bool(self.can_digital_rotation, "can_digital_rotation")
        self.binning_index_min = p.int(self.binning_index_min, "binning_index_min")
        self.binning_index_max = p.int(self.binning_index_max, "binning_index_max")
        self.binning_xy_min = p.pair_int(self.binning_xy_min, "binning_xy_min")
        self.binning_xy_max = p.pair_int(self.binning_xy_max, "binning_xy_max")
        self.exposure_min = p.qty(self.exposure_min, "exposure_min", Units.MS)
        self.exposure_max = p.qty(self.exposure_max, "exposure_max", Units.MS)
        self.frame_integration_min = p.int(self.frame_integration_min, "frame_integration_min")
        self.frame_integration_max = p.int(self.frame_integration_max, "frame_integration_max")
        self.roi_size_min = p.pair_int(self.roi_size_min, "roi_size_min")
        self.roi_size_max = p.pair_int(self.roi_size_max, "roi_size_max")
        self.gain_index_min = p.int(self.gain_index_min, "gain_index_min")
        self.gain_index_max = p.int(self.gain_index_max, "gain_index_max")
        self.offset_index_min = p.int(self.offset_index_min, "offset_index_min")
        self.offset_index_max = p.int(self.offset_index_max, "offset_index_max")
        self.digital_rotation_min = p.qty(self.digital_rotation_min, "digital_rotation_min", Units.DEG)
        self.digital_rotation_max = p.qty(self.digital_rotation_max, "digital_rotation_max", Units.DEG)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)

        def _check_range(min_attr, max_attr, name):
            mn, mx = getattr(self, min_attr), getattr(self, max_attr)
            if mn is not None and mx is not None:
                v.check(mn <= mx, name, f"min > max ({mn} > {mx})",
                        heal=lambda: (setattr(self, min_attr, mx), setattr(self, max_attr, mn)))

        # 1. Range Consistency
        _check_range("binning_index_min", "binning_index_max", "binning_index")
        _check_range("frame_integration_min", "frame_integration_max", "frame_integration")
        _check_range("gain_index_min", "gain_index_max", "gain_index")
        _check_range("offset_index_min", "offset_index_max", "offset_index")
        _check_range("exposure_min", "exposure_max", "exposure")
        _check_range("digital_rotation_min", "digital_rotation_max", "digital_rotation")

        # 2. Physical Non-negativity
        v.check_ge_zero(self.exposure_min, "exposure_min", unit_aware=True,
                        reset_to=Q_(0.0, Units.MS))

        # 3. Tuple consistency (ROI)
        if self.roi_size_min and self.roi_size_max:
            min_w, min_h = self.roi_size_min
            max_w, max_h = self.roi_size_max
            if min_w > max_w or min_h > max_h:
                def _heal_roi():
                    self.roi_size_min = (min(min_w, max_w), min(min_h, max_h))
                    self.roi_size_max = (max(min_w, max_w), max(min_h, max_h))

                v.check(False, "roi_size", "ROI size min > max", heal=_heal_roi)

        return v.valid

    def supports(self, settings: DetectorSettings) -> SafetyCheck:
        """
        Validates if the requested settings are supported by this specific detector hardware.
        Returns a SafetyCheck object containing all rejection reasons.
        """
        reasons = []

        # --- 1. Binning Checks ---
        if settings.binning_xy is not None:
            bx, by = settings.binning_xy

            # Feature Flag
            if self.can_binning is False and (bx > 1 or by > 1):
                reasons.append(f"Binning XY ({bx}, {by}) requested, but 'can_binning' is False.")

            # Ranges
            if self.binning_xy_min:
                min_x, min_y = self.binning_xy_min
                if bx < min_x or by < min_y:
                    reasons.append(f"Binning XY ({bx}, {by}) below limit {self.binning_xy_min}.")

            if self.binning_xy_max:
                max_x, max_y = self.binning_xy_max
                if bx > max_x or by > max_y:
                    reasons.append(f"Binning XY ({bx}, {by}) exceeds limit {self.binning_xy_max}.")

        if settings.binning_index is not None:
            bi = settings.binning_index

            if self.can_binning is False and bi > 1:
                reasons.append(f"Binning Index {bi} requested, but 'can_binning' is False.")

            if self.binning_index_min is not None and bi < self.binning_index_min:
                reasons.append(f"Binning Index {bi} below limit {self.binning_index_min}.")

            if self.binning_index_max is not None and bi > self.binning_index_max:
                reasons.append(f"Binning Index {bi} exceeds limit {self.binning_index_max}.")

        # --- 2. Exposure Checks ---
        if settings.exposure is not None:
            # Note: settings.exposure is a Quantity (Units.MS)
            if self.exposure_min is not None and settings.exposure < self.exposure_min:
                reasons.append(f"Exposure {settings.exposure} below limit {self.exposure_min}.")

            if self.exposure_max is not None and settings.exposure > self.exposure_max:
                reasons.append(f"Exposure {settings.exposure} exceeds limit {self.exposure_max}.")

        # --- 3. Frame Integration ---
        if settings.frame_integration is not None:
            fi = settings.frame_integration
            if self.frame_integration_min is not None and fi < self.frame_integration_min:
                reasons.append(f"Frame Integration {fi} below limit {self.frame_integration_min}.")

            if self.frame_integration_max is not None and fi > self.frame_integration_max:
                reasons.append(f"Frame Integration {fi} exceeds limit {self.frame_integration_max}.")

        # --- 4. Gain & Offset ---
        if settings.gain_index is not None:
            gi = settings.gain_index
            if self.can_gain is False and gi != 0:
                reasons.append(f"Gain Index {gi} requested, but 'can_gain' is False.")

            if self.gain_index_min is not None and gi < self.gain_index_min:
                reasons.append(f"Gain Index {gi} below limit {self.gain_index_min}.")

            if self.gain_index_max is not None and gi > self.gain_index_max:
                reasons.append(f"Gain Index {gi} exceeds limit {self.gain_index_max}.")

        if settings.offset_index is not None:
            oi = settings.offset_index
            if self.can_offset is False and oi != 0:
                reasons.append(f"Offset Index {oi} requested, but 'can_offset' is False.")

            if self.offset_index_min is not None and oi < self.offset_index_min:
                reasons.append(f"Offset Index {oi} below limit {self.offset_index_min}.")

            if self.offset_index_max is not None and oi > self.offset_index_max:
                reasons.append(f"Offset Index {oi} exceeds limit {self.offset_index_max}.")

        # --- 5. Digital Rotation (Cleaned Up) ---
        if settings.digital_rotation is not None:
            rot_val = settings.digital_rotation

            if self.can_digital_rotation is False:
                # Check if rotation is non-zero (using magnitude for safety against -0.0)
                if abs(rot_val.magnitude) > 1e-6:
                    reasons.append(
                        f"Digital Rotation {rot_val} requested, but 'can_digital_rotation' is False.")

            if self.digital_rotation_min is not None and rot_val < self.digital_rotation_min:
                reasons.append(f"Digital Rotation {rot_val} below limit {self.digital_rotation_min}.")

            if self.digital_rotation_max is not None and rot_val > self.digital_rotation_max:
                reasons.append(f"Digital Rotation {rot_val} exceeds limit {self.digital_rotation_max}.")

        # --- 6. ROI Checks ---
        if settings.roi is not None:
            w, h = settings.roi.width, settings.roi.height

            if self.roi_size_min:
                min_w, min_h = self.roi_size_min
                if w < min_w or h < min_h:
                    reasons.append(f"ROI size {w}x{h} below limit {self.roi_size_min}.")

            if self.roi_size_max:
                max_w, max_h = self.roi_size_max
                if w > max_w or h > max_h:
                    reasons.append(f"ROI size {w}x{h} exceeds limit {self.roi_size_max}.")

        return SafetyCheck(allowed=(len(reasons) == 0), reasons=reasons)

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS, key_map=self._KEYS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "DetectorCapabilities":
        return _auto_from_dict(DetectorCapabilities, d, mode, alias_map={
            "roi_size_min": "roi_min", "roi_size_max": "roi_max",
            "exposure_min": "exposure_ms_min", "exposure_max": "exposure_ms_max",
            "digital_rotation_min": "digital_rotation_deg_min",
            "digital_rotation_max": "digital_rotation_deg_max"
        })

@dataclass
class DetectorSystemSettings:
    """
    Registry for available detectors and their capability profiles.

    Role:     Gatekeeper (System Registry)
    Context:  Control-plane
    Category: B (Structural Container)

    Attributes:
        enabled (Optional[bool]): Master toggle for detector system.
            None Behavior: Defaulted to True.
        available_detector_ids (Optional[List[str]]): List of known device names.
            None Behavior: Structural Default (Empty List).
        default_detector_id (Optional[str]): Primary detector name.
            None Behavior: Preserved as None.
        defaults_by_id (Optional[Dict[str, DetectorSettings]]): Base settings per device.
            None Behavior: Structural Default (Empty Dict).
        capabilities_by_id (Optional[Dict[str, DetectorCapabilities]]): Hardware limits per device.
            None Behavior: Structural Default (Empty Dict).
    """
    enabled: Optional[bool] = None
    defaults_by_id: Optional[Dict[str, DetectorSettings]] = None
    default_detector_id: Optional[str] = None
    capabilities_by_id: Optional[Dict[str, DetectorCapabilities]] = None
    available_detector_ids: Optional[List[str]] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "DetectorSystemSettings")
        self.enabled = p.bool(self.enabled, "enabled", default=True)
        self.default_detector_id = p.id(self.default_detector_id, "default_detector_id")
        self.available_detector_ids = list(
            dict.fromkeys(p.list_str(self.available_detector_ids, "available_detector_ids")))
        self.defaults_by_id = p.map_model(DetectorSettings, self.defaults_by_id, "defaults_by_id", "detector_id")
        self.capabilities_by_id = p.map_model(DetectorCapabilities, self.capabilities_by_id, "capabilities_by_id")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)

        # 1. Default Existence
        ids_available = set(self.available_detector_ids)
        if self.default_detector_id:
            known = ids_available | set(self.defaults_by_id.keys()) | set(self.capabilities_by_id.keys())
            v.check(self.default_detector_id in known, "default_detector_id",
                    f"Default '{self.default_detector_id}' is unknown",
                    heal=lambda: setattr(self, 'default_detector_id', None))

            # 2. Completeness (Heal by removing the broken ID from availability)
            def _heal_missing(missing_set):
                # Remove the IDs that have no config from the available list
                self.available_detector_ids = [x for x in self.available_detector_ids if x not in missing_set]

            missing_defaults = ids_available - set(self.defaults_by_id.keys())
            v.check(not missing_defaults, "completeness.defaults",
                    f"Missing defaults for: {missing_defaults}",
                    heal=lambda: _heal_missing(missing_defaults))

            missing_caps = ids_available - set(self.capabilities_by_id.keys())
            v.check(not missing_caps, "completeness.capabilities",
                    f"Missing capabilities for: {missing_caps}",
                    heal=lambda: _heal_missing(missing_caps))

        # 3. Recursive Checks
        v.check_nested_map(self.defaults_by_id)
        v.check_nested_map(self.capabilities_by_id)
        return v.valid

    def is_supported(self, settings: DetectorSettings) -> SafetyCheck:
        if not settings.detector_id:
            return SafetyCheck.failure("No detector_id specified in settings")

        caps = self.capabilities_by_id.get(settings.detector_id)
        if not caps:
            return SafetyCheck.failure(f"Unknown capabilities for detector '{settings.detector_id}'")

        return caps.supports(settings)

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorSystemSettings":
        return _auto_from_dict(DetectorSystemSettings, d, mode, alias_map={
            "available_detector_ids": "available_detectors"
        })

@dataclass
class ScanSettings:
    """
    Configuration for the STEM raster scan engine.

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Both (Data-plane / Control-plane)
    Category: C / D (Measured State / Intent)

    Attributes:
        scan_mode (Optional[str]): The scanning strategy (e.g. "Full Frame").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        width_px (Optional[int]): Scan grid width.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        height_px (Optional[int]): Scan grid height.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        pixel_dwell_time (Optional[Quantity]): Time per pixel. Units: us.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        flyback_time (Optional[Quantity]): Retrace time between lines. Units: us.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        scan_rotation (Optional[Quantity]): Rotation of scan axes. Units: deg.
            None Behavior: Snapshot (Unknown) | Intent (No Change).

    Behavior:
        1. Snapshot (Telemetry): Represents the currently active scan parameters.
        2. Intent (Control): Used in `ScanControlRequest` to configure the engine.
           Any field set to None retains the previous value.
    """
    scan_mode: Optional[str] = None

    # --- ADDED: Grid Dimensions ---
    width_px: Optional[int] = None
    height_px: Optional[int] = None

    pixel_dwell_time: Optional["Quantity"] = None
    flyback_time: Optional["Quantity"] = None

    # Moved here from BeamSettings
    scan_rotation: Optional["Quantity"] = None

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "pixel_dwell_time": Units.US,
        "flyback_time": Units.US,
        "scan_rotation": Units.DEG
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ScanSettings")
        self.scan_mode = p.str(self.scan_mode, "scan_mode")

        # --- ADDED: Parsers ---
        self.width_px = p.int(self.width_px, "width_px")
        self.height_px = p.int(self.height_px, "height_px")

        self.pixel_dwell_time = p.qty(self.pixel_dwell_time, "pixel_dwell_time", Units.US)
        self.flyback_time = p.qty(self.flyback_time, "flyback_time", Units.US)
        self.scan_rotation = p.qty(self.scan_rotation, "scan_rotation", Units.DEG)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_gt_zero(self.pixel_dwell_time, "pixel_dwell_time", unit_aware=True, reset_to=None)
        v.check_ge_zero(self.flyback_time, "flyback_time", unit_aware=True, reset_to=None)
        if self.width_px:
            v.check(self.width_px > 0, "width_px", ">0", heal=lambda: setattr(self, 'width_px', 512))
        if self.height_px:
            v.check(self.height_px > 0, "height_px", ">0", heal=lambda: setattr(self, 'height_px', 512))
        return v.valid

    @property
    def estimated_duration(self) -> Optional["Quantity"]:
        """Helper to calculate total scan time."""
        if None in (self.width_px, self.height_px, self.pixel_dwell_time): return None

        # (W * H * Dwell) + (H * Flyback)
        t_dwell = self.width_px * self.height_px * self.pixel_dwell_time.to(Units.US).magnitude
        t_fly = 0
        if self.flyback_time:
            t_fly = self.height_px * self.flyback_time.to(Units.US).magnitude

        return Q_((t_dwell + t_fly) / 1e6, Units.SEC)

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ScanSettings":
        return _auto_from_dict(ScanSettings, d, mode, alias_map={
            "pixel_dwell_time": "pixel_dwell_time_us",
            "flyback_time": "flyback_time_us",
            "scan_rotation": "scan_rotation_deg"
        })

@dataclass
class ScanSystemSettings:
    """
    Hardware capabilities and limits for the STEM scan engine.

    Role:     Gatekeeper (System Limits)
    Context:  Control-plane
    Category: A (Strict Runtime Config)

    Attributes:
        enabled (Optional[bool]): Master toggle for scan engine.
            None Behavior: Defaulted to True.
        available_scan_modes (Optional[List[str]]): Supported strategies.
            None Behavior: Preserved as None.
        pixel_dwell_time_limits (Optional[Tuple[Quantity, Quantity]]): Min/Max dwell.
            None Behavior: Preserved as None (Unlimited).
        flyback_time_limits (Optional[Tuple[Quantity, Quantity]]): Min/Max flyback.
            None Behavior: Preserved as None (Unlimited).
        scan_rotation_limits (Optional[Tuple[Quantity, Quantity]]): Min/Max rotation.
            None Behavior: Preserved as None (Unlimited).
    """
    enabled: Optional[bool] = None

    # --- Capabilities ---
    available_scan_modes: Optional[List[str]] = None  # e.g. ["SPOT", "FULL_FRAME"]

    # --- Limits ---
    pixel_dwell_time_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    flyback_time_limits: Optional[Tuple["Quantity", "Quantity"]] = None
    scan_rotation_limits: Optional[Tuple["Quantity", "Quantity"]] = None

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {
        "pixel_dwell_time_limits": Units.US,
        "flyback_time_limits": Units.US,
        "scan_rotation_limits": Units.DEG
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ScanSystemSettings")
        self.enabled = p.bool(self.enabled, "enabled", default=True)
        self.available_scan_modes = p.list_str(self.available_scan_modes, "available_scan_modes")

        self.pixel_dwell_time_limits = p.pair_qty(self.pixel_dwell_time_limits, "pixel_dwell_time_limits", Units.US)
        self.flyback_time_limits = p.pair_qty(self.flyback_time_limits, "flyback_time_limits", Units.US)
        self.scan_rotation_limits = p.pair_qty(self.scan_rotation_limits, "scan_rotation_limits", Units.DEG)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)

        def _check_range(rng, name):
            if not rng: return
            mn, mx = rng
            v.check(mn <= mx, name, "min > max", heal=lambda: setattr(self, name, (mx, mn)))
            # Timing cannot be negative
            if "rotation" not in name:
                v.check_ge_zero(mn, f"{name}.min", unit_aware=True)

        _check_range(self.pixel_dwell_time_limits, "pixel_dwell_time_limits")
        _check_range(self.flyback_time_limits, "flyback_time_limits")
        _check_range(self.scan_rotation_limits, "scan_rotation_limits")

        return v.valid

    def is_safe_scan(self, settings: "ScanSettings") -> SafetyCheck:
        """
        Runtime check: Is the requested scan within hardware timing limits?
        """
        reasons = []

        # 1. Check Mode
        if settings.scan_mode and self.available_scan_modes:
            if settings.scan_mode not in self.available_scan_modes:
                reasons.append(f"Scan mode '{settings.scan_mode}' not supported {self.available_scan_modes}")

        # 2. Check Timing Limits
        def _chk(val, lims, name):
            if val is not None and lims:
                if not (lims[0] <= val <= lims[1]):
                    reasons.append(f"{name} {val} outside limits {lims}")

        _chk(settings.pixel_dwell_time, self.pixel_dwell_time_limits, "dwell_time")
        _chk(settings.flyback_time, self.flyback_time_limits, "flyback_time")
        _chk(settings.scan_rotation, self.scan_rotation_limits, "scan_rotation")

        return SafetyCheck(allowed=(len(reasons) == 0), reasons=reasons)

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ScanSystemSettings":
        return _auto_from_dict(ScanSystemSettings, d, mode, alias_map={
            "pixel_dwell_time_limits": "pixel_dwell_time_limits_us",
            "flyback_time_limits": "flyback_time_limits_us",
            "scan_rotation_limits": "scan_rotation_limits_deg"
        })

@dataclass
class VacuumSettings:
    """
    Status and control of the vacuum system (Valves, Pumps, Pressures).

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Control-plane
    Category: C / D (Measured State / Intent)

    Attributes:
        column_valve_state (Optional[str]): V7/V4 state ("OPEN", "CLOSED").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        gun_valve_state (Optional[str]): V1 state ("OPEN", "CLOSED").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        turbo_pump_state (Optional[str]): Turbo status ("ON", "OFF").
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        column_pressure (Optional[Quantity]): Column gauge. Units: Pa.
            None Behavior: Snapshot (Unknown).
        gun_pressure (Optional[Quantity]): Gun/FEG gauge. Units: Pa.
            None Behavior: Snapshot (Unknown).
        buffer_tank_pressure (Optional[Quantity]): Buffer gauge. Units: Pa.
            None Behavior: Snapshot (Unknown).

    Behavior:
        1. Snapshot (Telemetry): Returns all valve states and gauge readings.
        2. Intent (Control): Only 'state' fields (valves/pumps) are actionable.
           Pressure fields are Read-Only; setting them in a request is ignored.
    """
    column_valve_state: Optional[str] = None  # "OPEN", "CLOSED" (aka V7/V4)
    gun_valve_state: Optional[str] = None  # "OPEN", "CLOSED" (aka V1)
    turbo_pump_state: Optional[str] = None  # "ON", "OFF"

    # Telemetry Only (Usually)
    column_pressure: Optional["Quantity"] = None
    gun_pressure: Optional["Quantity"] = None
    buffer_tank_pressure: Optional["Quantity"] = None

    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    _UNITS = {
        "column_pressure": Units.PA,
        "gun_pressure": Units.PA,
        "buffer_tank_pressure": Units.PA
    }

    def __post_init__(self):
        p = FieldParser(self, self._mode, "VacuumSettings")
        self.column_valve_state = p.str(self.column_valve_state, "column_valve_state")
        self.gun_valve_state = p.str(self.gun_valve_state, "gun_valve_state")
        self.turbo_pump_state = p.str(self.turbo_pump_state, "turbo_pump_state")

        self.column_pressure = p.qty(self.column_pressure, "column_pressure", Units.PA)
        self.gun_pressure = p.qty(self.gun_pressure, "gun_pressure", Units.PA)
        self.buffer_tank_pressure = p.qty(self.buffer_tank_pressure, "buffer_tank_pressure", Units.PA)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        # Pressure can't be negative
        v.check_ge_zero(self.column_pressure, "column_pressure", unit_aware=True)
        v.check_ge_zero(self.gun_pressure, "gun_pressure", unit_aware=True)
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "VacuumSettings":
        return _auto_from_dict(VacuumSettings, d, mode, alias_map={
            "column_valve_state": "column_valve",
            "column_pressure": "column_pressure_pa",
            "gun_pressure": "gun_pressure_pa",
            "buffer_tank_pressure": "buffer_tank_pressure_pa"
        })

@dataclass
class Aperture:
    """
    Physical aperture status or a request to modify insertion/size.

    Role:     Dual-Use (Snapshot and Intent)
    Context:  Both (Data-plane / Control-plane)
    Category: C / D (Measured State / Intent)

    Attributes:
        aperture_id (Optional[str]): The mechanism ID (e.g. "condenser", "objective").
            None Behavior: Preserved as None.
        inserted (Optional[bool]): Whether the aperture is in the beam path.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        size_index (Optional[int]): The currently selected hole size index.
            None Behavior: Snapshot (Unknown) | Intent (No Change).
        position (Optional[Point]): Mechanical micro-position.
            None Behavior: Snapshot (Unknown) | Intent (No Change).

    Behavior:
        1. Snapshot (Telemetry): Represents the current mechanism state.
        2. Intent (Control): Represents a desire to change state.
           - Insert/Retract: Set `inserted` to True/False.
           - Change Size: Set `size_index`.
           - Align: Set `position`.
    """
    aperture_id: Optional[str] = None
    inserted: Optional[bool] = None
    size_index: Optional[int] = None
    position: Optional[Point] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "Aperture")
        self.aperture_id = p.id(self.aperture_id, "aperture_id")
        self.inserted = p.bool(self.inserted, "inserted")
        self.size_index = p.int(self.size_index, "size_index")
        self.position = p.model(Point, self.position, "position")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check(self.size_index is None or self.size_index >= 0, "size_index", "must be >= 0",
                raw=self.size_index, heal=lambda: setattr(self, 'size_index', None))

        if self.position:
            if not self.position.validate(mode=v.mode):
                v.check(False, "position", "Invalid aperture position",
                        heal=lambda: setattr(self, 'position', None))
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "Aperture":
        return _auto_from_dict(Aperture, d, mode, alias_map={
            "aperture_id": "id"
        })

@dataclass
class MicroscopeState:
    """
    Comprehensive snapshot of the microscope telemetry at a specific timestamp.

    Role:     Snapshot (Measured State)
    Context:  Data-plane
    Category: C (Nullable Runtime)

    Attributes:
        timestamp (Optional[str]): ISO8601 time of capture.
            None Behavior: Defaulted to "Now".
        mode (Optional[str]): Global instrument mode (e.g. "STEM", "TEM").
            None Behavior: Preserved as None.
        stage_position (Optional[StagePosition]): Current physical coordinates.
        beam (Optional[BeamSettings]): Current gun/optics state.
        projection (Optional[ProjectionSettings]): Current imaging state.
        scan (Optional[ScanSettings]): Current raster state.
        vacuum (Optional[VacuumSettings]): Current vacuum state.
        apertures (Optional[Dict[str, Aperture]]): State of all apertures.
        detectors (Optional[Dict[str, DetectorSettings]]): State of all cameras.
        active_detector_ids (Optional[List[str]]): List of currently active cameras.
        primary_detector_id (Optional[str]): The main camera in use.
    """
    timestamp: Optional[str] = None
    mode: Optional[str] = None
    stage_position: Optional[StagePosition] = None
    beam: Optional[BeamSettings] = None
    projection: Optional[ProjectionSettings] = None  # --- ADDED ---
    scan: Optional[ScanSettings] = None
    vacuum: Optional[VacuumSettings] = None
    apertures: Optional[Dict[str, Aperture]] = None
    detectors: Optional[Dict[str, DetectorSettings]] = None
    active_detector_ids: Optional[List[str]] = None
    primary_detector_id: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "MicroscopeState")
        self.timestamp = p.str(self.timestamp, "timestamp",
                               default=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.mode = p.str(self.mode, "mode")
        self.primary_detector_id = p.id(self.primary_detector_id, "primary_detector_id")
        self.active_detector_ids = p.list_str(self.active_detector_ids, "active_detector_ids")
        # Category B: Structural Defaults
        self.stage_position = p.model(StagePosition, self.stage_position, "stage_position",
                                      default=StagePosition(_mode=p.mode))
        self.beam = p.model(BeamState, self.beam, "beam", default=BeamState(_mode=p.mode))
        self.projection = p.model(ProjectionSettings, self.projection, "projection",
                                  default=ProjectionSettings(_mode=p.mode))
        self.scan = p.model(ScanSettings, self.scan, "scan", default=ScanSettings(_mode=p.mode))
        self.vacuum = p.model(VacuumSettings, self.vacuum, "vacuum", default=VacuumSettings(_mode=p.mode))  # <--- NEW

        self.apertures = p.map_model(Aperture, self.apertures, "apertures", "aperture_id")
        self.detectors = p.map_model(DetectorState, self.detectors, "detectors", "detector_id")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.stage_position)
        v.check_nested(self.beam)
        v.check_nested(self.projection)
        v.check_nested(self.scan)
        v.check_nested(self.vacuum)  # <--- NEW
        v.check_nested_map(self.apertures)
        v.check_nested_map(self.detectors)

        valid_ids = []
        for det_id in self.active_detector_ids:
            if not v.check(det_id in self.detectors, f"active_detector_ids.{det_id}",
                           f"Active detector '{det_id}' not found"):
                continue  # Skip adding to valid_ids if check failed
            valid_ids.append(det_id)

        if not v.strict: self.active_detector_ids = valid_ids
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeState":
        return _auto_from_dict(MicroscopeState, d, mode)

@dataclass
class MicroscopeImageMetadata:
    """
    Scientific metadata describing the context of an image acquisition.

    Role:     Snapshot (Scientific Context)
    Context:  Data-plane
    Category: C (Measured State)

    Attributes:
        version (Optional[str]): Metadata schema version.
        created_at (Optional[str]): Acquisition timestamp.
        magnification (Optional[float]): Indicated mag.
        camera_length_mm (Optional[float]): Indicated camera length.
        pixel_size_nm (Optional[Tuple[float, float]]): Calibrated pixel scale (X, Y).
        image_size_px (Optional[Tuple[int, int]]): Image dimensions (W, H).
        accelerating_voltage_kv (Optional[float]): Beam voltage.
        beam_current_na (Optional[float]): Beam current.
        exposure_ms (Optional[float]): Exposure time.
        microscope_state (Optional[MicroscopeState]): Full telemetry snapshot.
    """
    version: Optional[str] = None
    created_at: Optional[str] = None
    magnification: Optional[float] = None
    camera_length_mm: Optional[float] = None
    pixel_size_nm: Optional[Tuple[float, float]] = None
    image_size_px: Optional[Tuple[int, int]] = None
    accelerating_voltage_kv: Optional[float] = None
    beam_current_na: Optional[float] = None
    exposure_ms: Optional[float] = None
    microscope_state: Optional[MicroscopeState] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "MicroscopeImageMetadata")
        self.version = p.str(self.version, "version", default=str(METADATA_VERSION))
        self.created_at = p.str(self.created_at, "created_at",
                                default=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.magnification = p.float(self.magnification, "magnification")
        self.camera_length_mm = p.float(self.camera_length_mm, "camera_length_mm", Units.MM)
        self.accelerating_voltage_kv = p.float(self.accelerating_voltage_kv, "accelerating_voltage_kv", Units.KV)
        self.beam_current_na = p.float(self.beam_current_na, "beam_current_na", Units.NA)
        self.exposure_ms = p.float(self.exposure_ms, "exposure_ms", Units.MS)
        self.pixel_size_nm = p.pair_float(self.pixel_size_nm, "pixel_size_nm")
        self.image_size_px = p.pair_int(self.image_size_px, "image_size_px")
        self.microscope_state = p.model(MicroscopeState, self.microscope_state, "microscope_state")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.microscope_state)
        v.check_ge_zero(self.magnification, "magnification", reset_to=None)
        v.check_ge_zero(self.exposure_ms, "exposure_ms", reset_to=None)
        v.check_ge_zero(self.accelerating_voltage_kv, "accelerating_voltage_kv", reset_to=None)
        return v.valid

    def to_dict(self) -> dict:
        # Since fields like 'exposure_ms' are already floats (parsed via p.float)
        # and named correctly, we can just use default export.
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeImageMetadata":
        return _auto_from_dict(MicroscopeImageMetadata, d, mode)

class MicroscopeImage:
    """
    Container for image data and associated metadata.

    Handles I/O operations (Load/Save) and manages the "sidecar" metadata relationship.

    Attributes:
        data: The raw 2D image data (numpy array).
        metadata: The associated acquisition parameters and state.

    Notes:
        Supports TIFF (with embedded metadata), PNG, and JPEG formats.
    """

    def __init__(self, data: np.ndarray, metadata: Optional[MicroscopeImageMetadata] = None):
        # 1. Normalize: Squeeze trivial dimensions if 3D (e.g. [1, H, W] -> [H, W])
        if data.ndim == 3:
            if data.shape[0] == 1:
                data = data[0]
            elif data.shape[-1] == 1:
                data = data[..., 0]
        if not _check_data_format(data):
            raise ValueError(
                f"Invalid data format for MicroscopeImage. Expected 2D uint8/uint16, got shape={data.shape} dtype={data.dtype}")
        self.data = data
        self.metadata = MicroscopeImageMetadata.from_dict(metadata) if isinstance(metadata, dict) else metadata

    @staticmethod
    def _decode_description(desc: Any) -> Optional[Dict[str, Any]]:
        if desc is None: return None
        if isinstance(desc, bytes):
            try: desc = desc.decode("utf-8", errors="replace")
            except Exception: return None
        if not isinstance(desc, str): return None
        desc = desc.strip()
        if not desc: return None
        try:
            obj = json.loads(desc)
            return obj if isinstance(obj, dict) else None
        except Exception: return None

    @staticmethod
    def _encode_description(md: Optional[MicroscopeImageMetadata]) -> str:
        if md is None: return ""
        try:
            safe = _jsonable(md.to_dict())
            return json.dumps(safe, ensure_ascii=False)
        except Exception:
            minimal = {"created_at": getattr(md, "created_at", None), "version": getattr(md, "version", None)}
            return json.dumps(_jsonable(minimal), ensure_ascii=False)

    @staticmethod
    def _to_uint8_preview(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
        a = np.asarray(arr)
        if a.size == 0: return a.astype(np.uint8, copy=False)
        af = a.astype(np.float32, copy=False)
        finite = af[np.isfinite(af)]
        if finite.size == 0: return np.zeros_like(a, dtype=np.uint8)
        lo, hi = np.percentile(finite, [p_low, p_high])
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 1e-9:
            mean_val = np.mean(finite)
            clipped = np.clip(mean_val, 0.0, 255.0).astype(np.uint8)
            return np.full_like(a, clipped, dtype=np.uint8)
        scaled = (af - lo) * (255.0 / rng)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MicroscopeImage":
        path = Path(path)
        ext = path.suffix.lower().lstrip(".")
        if ext in ("tif", "tiff"):
            with tff.TiffFile(str(path)) as tif:
                data = tif.asarray()
                if data.ndim == 3 and data.shape[0] == 1: data = data[0]
                if data.ndim == 3 and data.shape[-1] == 1: data = data[..., 0]
                if data.ndim != 2: raise ValueError(f"Expected single-frame 2D grayscale TIFF, got shape={data.shape}")
                if data.dtype == np.int16:
                    if data.min() >= 0: data = data.astype(np.uint16)
                    else: raise ValueError("Loaded TIFF is int16 with negative values.")
                if data.dtype in (np.int32, np.uint32):
                    if data.min() >= 0 and data.max() <= 65535: data = data.astype(np.uint16)
                    else: raise ValueError("TIFF is 32-bit with values outside uint16 range.")
                metadata = None
                try:
                    desc = tif.pages[0].tags["ImageDescription"].value
                    d = cls._decode_description(desc)
                    if d is not None: metadata = MicroscopeImageMetadata.from_dict(d)
                except Exception: metadata = None
            return cls(data=data, metadata=metadata)

        with Image.open(path) as img:
            if img.mode not in ("L", "I;16", "I;16B", "I;16L"): img = img.convert("L")
            data = np.array(img)
            if data.ndim == 3 and data.shape[-1] == 1: data = data[..., 0]
            if data.dtype == np.int32 and img.mode in ("I;16", "I;16B", "I;16L"): data = data.astype(np.uint16)
            if data.dtype not in (np.uint8, np.uint16):
                if np.issubdtype(data.dtype, np.number): data = np.clip(data, 0, 255).astype(np.uint8)
                else: data = data.astype(np.uint8)
        metadata = None
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.exists():
            try:
                d = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(d, dict): metadata = MicroscopeImageMetadata.from_dict(d)
            except Exception: metadata = None
        return cls(data=data, metadata=metadata)

    def save(self, path: Union[str, Path], file_format: Optional[str] = None) -> Path:
        supported = {"tiff", "tif", "png", "jpg", "jpeg", "bmp"}
        path = Path(path)
        requested_ext = (file_format or path.suffix.lstrip(".") or "tiff").lower().strip()
        if requested_ext not in supported: raise ValueError(f"Unsupported file_format: {requested_ext!r}")
        fmt = "tiff" if requested_ext in ("tif", "tiff") else requested_ext
        suffix = ".tif" if requested_ext == "tif" else ".tiff" if requested_ext == "tiff" else f".{requested_ext}"
        path = path.with_suffix(suffix)
        os.makedirs(path.parent, exist_ok=True)
        desc = self._encode_description(self.metadata)
        if fmt == "tiff":
            tff.imwrite(str(path), self.data, description=desc)
            return path
        data_to_save = self.data
        if fmt in ("jpg", "jpeg", "bmp"):
            if data_to_save.dtype != np.uint8: data_to_save = self._to_uint8_preview(data_to_save)
        else:
            if data_to_save.dtype not in (np.uint8, np.uint16): data_to_save = np.clip(data_to_save, 0, 255).astype(np.uint8)
        img = Image.fromarray(data_to_save)
        pil_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "bmp": "BMP"}[fmt]
        img.save(path, format=pil_format)
        sidecar = path.with_suffix(path.suffix + ".json")
        try:
            if self.metadata is not None:
                sidecar.write_text(json.dumps(self.metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception: pass
        return path

@dataclass
class ImageOutputSettings:
    """
    Configuration for naming conventions and storage formats.

    Role:     Gatekeeper (Output Config)
    Context:  Control-plane
    Category: A (Config)

    Attributes:
        file_format (Optional[str]): "tiff", "png", "jpg", or "bmp".
            None Behavior: Defaulted to "tiff".
        path (Optional[str]): Destination directory or path template.
            None Behavior: Preserved as None.
    """
    file_format: Optional[str] = None
    path: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ImageOutputSettings")
        _fmt = p.str(self.file_format, "file_format")
        self.file_format = (_fmt.lower() if _fmt else "tiff")
        self.path = p.str(self.path, "path")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check(self.file_format in {"tiff", "tif", "png", "jpg", "jpeg", "bmp"},
                "file_format", f"Unsupported format: {self.file_format}",
                heal=lambda: setattr(self, 'file_format', "tiff"))
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ImageOutputSettings":
        return _auto_from_dict(ImageOutputSettings, d, mode)

@dataclass
class SystemInfo:
    """
    Static identifying information about the microscope hardware and software.

    Role:     Structure / Identity
    Context:  Data-plane
    Category: A (Config/Identity)

    Attributes:
        name (Optional[str]): Human-readable system name.
        ip_address (Optional[str]): Control PC network address.
        manufacturer, model, serial_number (Optional[str]): Hardware IDs.
        hardware_version, software_version (Optional[str]): Vendor versions.
        supertem_version (Optional[str]): Library version.
        application, application_version (Optional[str]): Client app info.
    """
    name: Optional[str] = None
    ip_address: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    hardware_version: Optional[str] = None
    software_version: Optional[str] = None
    supertem_version: Optional[str] = None
    application: Optional[str] = None
    application_version: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.LENIENT, repr=False, compare=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "SystemInfo")
        self.name = p.str(self.name, "name", default="Unknown")
        self.ip_address = p.str(self.ip_address, "ip_address", default="Unknown")
        self.manufacturer = p.str(self.manufacturer, "manufacturer", default="Unknown")
        self.model = p.str(self.model, "model", default="Unknown")
        self.serial_number = p.str(self.serial_number, "serial_number", default="Unknown")
        self.hardware_version = p.str(self.hardware_version, "hardware_version", default="Unknown")
        self.software_version = p.str(self.software_version, "software_version", default="Unknown")
        self.supertem_version = p.str(self.supertem_version, "supertem_version", default=__version__)
        self.application = p.str(self.application, "application", default="Unknown")
        self.application_version = p.str(self.application_version, "application_version", default="Unknown")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        if self.ip_address and self.ip_address != "Unknown":
            try:
                ipaddress.ip_address(self.ip_address.strip())
            except Exception:
                v.check(False, "ip_address", f"Invalid IP: {self.ip_address}",
                        raw=self.ip_address, heal=lambda: setattr(self, 'ip_address', "Unknown"))
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemInfo":
        return _auto_from_dict(SystemInfo, d, mode)

@dataclass
class SystemSettings:
    """
    Root container for system-wide settings and limit registries.

    Role:     Structure / Gateway
    Context:  Control-plane
    Category: B (Structural Container)

    Attributes:
        stage_system (Optional[StageSystemSettings]): Stage limits.
        beam_system (Optional[BeamSystemSettings]): Beam limits.
        projection_system (Optional[ProjectionSystemSettings]): Projection limits.
        scan_system (Optional[ScanSystemSettings]): Scan limits.
        detector_system (Optional[DetectorSystemSettings]): Detector registry.
        info (Optional[SystemInfo]): Static identity.
    """
    stage_system: Optional[StageSystemSettings] = None
    beam_system: Optional[BeamSystemSettings] = None
    projection_system: Optional[ProjectionSystemSettings] = None
    scan_system: Optional[ScanSystemSettings] = None  # <--- NEW
    detector_system: Optional[DetectorSystemSettings] = None
    info: Optional[SystemInfo] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "SystemSettings")
        self.stage_system = p.model(StageSystemSettings, self.stage_system, "stage_system",
                                    default=StageSystemSettings(_mode=p.mode))
        self.beam_system = p.model(BeamSystemSettings, self.beam_system, "beam_system",
                                   default=BeamSystemSettings(_mode=p.mode))
        self.projection_system = p.model(ProjectionSystemSettings, self.projection_system, "projection_system",
                                         default=ProjectionSystemSettings(_mode=p.mode))

        # --- NEW PARSER ---
        self.scan_system = p.model(ScanSystemSettings, self.scan_system, "scan_system",
                                   default=ScanSystemSettings(_mode=p.mode))

        self.detector_system = p.model(DetectorSystemSettings, self.detector_system, "detector_system",
                                       default=DetectorSystemSettings(_mode=p.mode))
        self.info = p.model(SystemInfo, self.info, "info", default=SystemInfo(_mode=p.mode))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.stage_system)
        v.check_nested(self.beam_system)
        v.check_nested(self.projection_system)
        v.check_nested(self.scan_system)  # <--- NEW CHECK
        v.check_nested(self.detector_system)
        v.check_nested(self.info)
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "SystemSettings":
        return _auto_from_dict(SystemSettings, d, mode, alias_map={
            "stage_system": "stage", "beam_system": "beam", "detector_system": "detector"
        })

@dataclass
class MicroscopeSettings:
    """
    Root configuration for a microscope instance.

    Role:     Structure / Gateway
    Context:  Control-plane
    Category: B (Structural Container)

    Attributes:
        system (Optional[SystemSettings]): Hardware limits and registries.
        image (Optional[ImageOutputSettings]): Default output settings.
        protocol (Optional[dict]): Automation scripts configuration.
    """
    system: Optional[SystemSettings] = None
    image: Optional[ImageOutputSettings] = None
    protocol: Optional[dict] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False, compare=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "MicroscopeSettings")
        self.system = p.model(SystemSettings, self.system, "system", default=SystemSettings(_mode=p.mode))
        self.image = p.model(ImageOutputSettings, self.image, "image", default=ImageOutputSettings(_mode=p.mode))
        self.protocol = p.dict(self.protocol, "protocol", default={"name": "demo"})

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.system)
        v.check_nested(self.image)
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.LENIENT) -> "MicroscopeSettings":
        return _auto_from_dict(MicroscopeSettings, d, mode)

# =============================================================================
# Requests
# =============================================================================

@dataclass
class StageMoveRequest:
    """
    Command to move the sample stage to a specific coordinate.

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Tristate Intent)

    Attributes:
        target (Optional[StagePosition]): Destination coordinates.
            None Behavior: Structural Default (Empty).
        relative (Optional[bool]): If True, target is a delta.
            None Behavior: Defaulted to False.
        drive_type (Optional[str]): Mechanism ("mechanical", "piezo", "default").
            None Behavior: Defaulted to "default".
        backlash_correction (Optional[bool]): Apply anti-hysteresis.
            None Behavior: Defaulted to True.
        wait_for_settle (Optional[bool]): Block until vibration stops.
            None Behavior: Defaulted to True.
        settle_time (Optional[Quantity]): Override default wait. Units: s.
            None Behavior: Preserved as None.
    """
    target: Optional[StagePosition] = None
    relative: Optional[bool] = None
    drive_type: Optional[str] = None  # Use StageDriveType values
    backlash_correction: Optional[bool] = None
    wait_for_settle: Optional[bool] = None
    settle_time: Optional["Quantity"] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    _UNITS = {"settle_time": Units.SEC}

    def __post_init__(self):
        p = FieldParser(self, self._mode, "StageMoveRequest")
        self.relative = p.bool(self.relative, "relative", default=False)
        self.drive_type = p.str(self.drive_type, "drive_type", default=StageDriveType.DEFAULT.value)
        self.backlash_correction = p.bool(self.backlash_correction, "backlash_correction", default=True)
        self.wait_for_settle = p.bool(self.wait_for_settle, "wait_for_settle", default=True)
        self.settle_time = p.qty(self.settle_time, "settle_time", Units.SEC)
        self.target = p.model(StagePosition, self.target, "target", default=StagePosition(_mode=p.mode))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.target)

        # Check emptiness of target
        axes = [self.target.x, self.target.y, self.target.z, self.target.r, self.target.tilt_x, self.target.tilt_y]
        v.check(any(a is not None for a in axes), "empty", "StageMoveRequest has no target coordinates")

        if self.drive_type == StageDriveType.PIEZO.value and self.relative:
            # Heuristic: Warn if requesting massive moves (> 5um) on Piezo
            # This prevents accidental "Piezo Saturation"
            limit_nm = 5000.0
            for axis in ['x', 'y', 'z']:
                val = getattr(self.target, axis)
                if val is not None:
                    mag = abs(val.to(Units.NM).magnitude)
                    v.check(mag < limit_nm, f"piezo_limit.{axis}",
                            f"Piezo request {mag}nm exceeds typical range ({limit_nm}nm)")

        v.check_ge_zero(self.settle_time, "settle_time", unit_aware=True, reset_to=None)
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self, unit_map=self._UNITS)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageMoveRequest":
        return _auto_from_dict(StageMoveRequest, d, mode, alias_map={
            "settle_time": "settle_time_s"
        })

@dataclass
class StageControlRequest:
    """
    Command to alter stage motor state (Stop, Home, Reset).

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        action (Optional[str]): "STOP", "ABORT", "HOME", "RESET_ERROR".
            None Behavior: Validation Error (Mandatory).
        axes (Optional[List[str]]): Specific axes to target (e.g. ["x"]).
            None Behavior: Applies to ALL axes.
    """
    action: Optional[str] = None
    axes: Optional[List[str]] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "StageControlRequest")
        self.action = p.str(self.action, "action")
        self.axes = p.list_str(self.axes, "axes")

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)

        # 1. Action is mandatory
        valid_actions = {"STOP", "ABORT", "HOME", "RESET_ERROR", "ZERO_ENCODERS"}
        v.check(self.action in valid_actions, "action",
                f"Action must be one of {valid_actions}")

        # 2. Axes validation (if provided)
        if self.axes:
            valid_axes = {"x", "y", "z", "r", "tilt_x", "tilt_y"}
            for axis in self.axes:
                v.check(axis in valid_axes, f"axes.{axis}", f"Invalid axis '{axis}'")

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "StageControlRequest":
        return _auto_from_dict(StageControlRequest, d, mode)

@dataclass
class DetectorControlRequest:
    """
    Command to configure detector hardware (Temp, Insert) without acquisition.

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        detector_id (Optional[str]): Target device ID.
            None Behavior: Required if not in target.
        target (Optional[DetectorSettings]): Configuration parameters to apply.
            None Behavior: Structural Default (Empty).
        action (Optional[str]): Mechanical cmd ("INSERT", "RETRACT", "COOLDOWN").
            None Behavior: Preserved as None.
    """
    detector_id: Optional[str] = None
    target: Optional[DetectorSettings] = None
    action: Optional[str] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "DetectorControlRequest")
        self.detector_id = p.id(self.detector_id, "detector_id")
        self.target = p.model(DetectorSettings, self.target, "target", default=DetectorSettings(_mode=p.mode))
        self.action = p.str(self.action, "action")

        # Sync: If settings has an ID but request doesn't, bubble it up
        if self.target.detector_id and not self.detector_id:
            self.detector_id = self.target.detector_id
        # Sync: If request has ID, push it down to settings for consistency
        elif self.detector_id and not self.target.detector_id:
            self.target.detector_id = self.detector_id

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check(bool(self.detector_id), "detector_id", "detector_id is required")
        v.check_nested(self.target)

        # Logic Check: Ensure we are doing *something*
        # 1. Check if settings has any non-None fields (Intent)
        has_settings_intent = False
        for f in dataclasses.fields(self.target):
            # Ignore internal fields and the ID itself (which is just identity, not a change)
            if f.name not in ["extra", "_mode", "detector_id"] and getattr(self.target, f.name) is not None:
                has_settings_intent = True
                break

        # 2. Check if action is present
        has_action = self.action is not None

        v.check(has_settings_intent or has_action, "empty_intent",
                "Request must have either settings to apply or an action to execute")

        if self.action:
            valid_actions = {"INSERT", "RETRACT", "COOLDOWN", "WARMUP", "RESET"}
            v.check(self.action in valid_actions, "action",
                    f"Action must be one of {valid_actions}")

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "DetectorControlRequest":
        return _auto_from_dict(DetectorControlRequest, d, mode)

@dataclass
class BeamControlRequest:
    """
    Command to change optical parameters (Voltage, Mode, Spot Size).

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        target (Optional[BeamSettings]): The desired beam configuration.
            None Behavior: Validation Error (Mandatory).
    """
    target: Optional[BeamSettings] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "BeamControlRequest")
        self.target = p.model(BeamSettings, self.target, "target", default=BeamSettings(_mode=p.mode))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.target)

        # Check for empty intent
        has_intent = False
        for f in dataclasses.fields(self.target):
            if f.name not in ["extra", "_mode"] and getattr(self.target, f.name) is not None:
                has_intent = True
                break

        v.check(has_intent, "empty_target", "Beam request has no parameters set")
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "BeamControlRequest":
        return _auto_from_dict(BeamControlRequest, d, mode)

@dataclass
class ProjectionControlRequest:
    """
    Command to change imaging parameters (Defocus, Mag, Shifts).

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        target (Optional[ProjectionSettings]): The desired optical configuration.
            None Behavior: Validation Error (Mandatory).
    """
    target: Optional[ProjectionSettings] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ProjectionControlRequest")
        self.target = p.model(ProjectionSettings, self.target, "target", default=ProjectionSettings(_mode=p.mode))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.target)

        # Check for empty intent
        has_intent = False
        for f in dataclasses.fields(self.target):
            if f.name not in ["extra", "_mode"] and getattr(self.target, f.name) is not None:
                has_intent = True
                break

        v.check(has_intent, "empty_target", "Projection request has no parameters set")

        # Specific Logic: If switching to DIFFRACTION, you must provide a camera length
        if self.target.optical_mode == "DIFFRACTION":
            v.check(self.target.camera_length is not None, "missing_cam_len",
                    "Switching to Diffraction requires a camera_length")

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ProjectionControlRequest":
        return _auto_from_dict(ProjectionControlRequest, d, mode)

@dataclass
class ScanControlRequest:
    """
    Command to control the STEM raster engine state.

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        action (Optional[str]): "START", "STOP", "SINGLE_FRAME".
            None Behavior: Validation Error (Mandatory).
        target (Optional[ScanSettings]): Parameters to apply if starting.
            None Behavior: Validation Error if action is START.
    """
    action: Optional[str] = None  # "START", "STOP", "SINGLE_FRAME"
    target: Optional[ScanSettings] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        # Renamed from ScanRequest to ScanControlRequest
        p = FieldParser(self, self._mode, "ScanControlRequest")
        self.action = p.str(self.action, "action")
        self.target = p.model(ScanSettings, self.target, "target", default=ScanSettings(_mode=p.mode))

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        # 1. Action is mandatory
        v.check(self.action in {"START", "STOP", "SINGLE_FRAME"}, "action",
                "Action must be START, STOP, or SINGLE_FRAME")

        # 2. Settings required if Starting
        if self.action in {"START", "SINGLE_FRAME"}:
            v.check_nested(self.target)

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ScanControlRequest":
        return _auto_from_dict(ScanControlRequest, d, mode)

@dataclass
class VacuumControlRequest:
    """
    Command to change valve or pump states.

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        target (Optional[VacuumSettings]): Desired valve/pump states.
            None Behavior: Validation Error (Mandatory).
        force (Optional[bool]): Bypass software pressure checks.
            None Behavior: Defaulted to False.
    """
    target: Optional[VacuumSettings] = None
    force: Optional[bool] = None  # If True, bypasses some software soft-checks (use with caution)
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "VacuumControlRequest")
        self.target = p.model(VacuumSettings, self.target, "target", default=VacuumSettings(_mode=p.mode))
        self.force = p.bool(self.force, "force", default=False)

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check_nested(self.target)

        # Ensure at least one valve/pump state is being requested
        has_intent = (self.target.column_valve_state is not None or
                      self.target.gun_valve_state is not None or
                      self.target.turbo_pump_state is not None)

        v.check(has_intent, "empty_target", "Request must specify at least one state change")
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "VacuumControlRequest":
        return _auto_from_dict(VacuumControlRequest, d, mode)

@dataclass
class ApertureControlRequest:
    """
    Command to modify an aperture's position or size.

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: D (Intent)

    Attributes:
        aperture_id (Optional[str]): Mechanism to target.
            None Behavior: Required.
        target (Optional[Aperture]): Desired configuration.
            None Behavior: Structural Default (Empty).
        relative (Optional[bool]): If True, position is a delta.
            None Behavior: Defaulted to False.
    """
    aperture_id: Optional[str] = None
    target: Optional[Aperture] = None
    relative: Optional[bool] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "ApertureControlRequest")
        self.aperture_id = p.id(self.aperture_id, "aperture_id")
        self.relative = p.bool(self.relative, "relative", default=False)
        self.target = p.model(Aperture, self.target, "target", default=Aperture(_mode=p.mode))
        if self.aperture_id is None and self.target.aperture_id is not None:
            self.aperture_id = self.target.aperture_id
        elif self.target.aperture_id is None and self.aperture_id is not None:
            self.target.aperture_id = self.aperture_id

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check(bool(self.aperture_id), "aperture_id", "aperture_id is required")
        v.check_nested(self.target)

        # ID Mismatch
        if self.target.aperture_id and self.aperture_id and self.target.aperture_id != self.aperture_id:
            v.check(False, "id_mismatch",
                    f"Ambiguous IDs: '{self.aperture_id}' vs '{self.target.aperture_id}'",
                    heal=lambda: setattr(self.target, 'aperture_id', self.aperture_id))

        # Relative logic check
        if self.relative:
            v.check(self.target.position is not None, "relative_no_pos", "Relative mode requires a position vector")

        # No-Op Check
        has_intent = (
                self.target.inserted is not None or self.target.size_index is not None or self.target.position is not None)
        v.check(has_intent, "empty_payload", "Request contains no changes")

        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "ApertureControlRequest":
        return _auto_from_dict(ApertureControlRequest, d, mode, alias_map={
            "state": "aperture"
        })

@dataclass
class AcquisitionRequest:
    """
    Intent to capture an image using a specific detector.

    Role:     Intent (User Request)
    Context:  Control-plane
    Category: B / D (Structure / Intent)

    Attributes:
        detector_id (Optional[str]): Target hardware device.
            None Behavior: Required (can be inferred from detector).
        detector (Optional[DetectorSettings]): Acquisition parameters.
            None Behavior: Structural Default (Empty).
        image (Optional[ImageOutputSettings]): Output file overrides.
            None Behavior: Structural Default (Empty).
    """
    detector_id: Optional[str] = None
    detector: Optional[DetectorSettings] = None
    image: Optional[ImageOutputSettings] = None
    extra: Extras = field(default_factory=Extras)
    _mode: ParseMode = field(default=ParseMode.STRICT, repr=False)

    def __post_init__(self):
        p = FieldParser(self, self._mode, "AcquisitionRequest")
        self.detector_id = p.id(self.detector_id, "detector_id")
        self.detector = p.model(DetectorSettings, self.detector, "detector", default=DetectorSettings(_mode=p.mode))
        self.image = p.model(ImageOutputSettings, self.image, "image", default=ImageOutputSettings(_mode=p.mode))
        # Sync Logic
        if self.detector_id is None and self.detector.detector_id is not None:
            self.detector_id = self.detector.detector_id
        elif self.detector.detector_id is None and self.detector_id is not None:
            self.detector.detector_id = self.detector_id

    def validate(self, *, mode: Union[ParseMode, str, None] = None) -> bool:
        v = Validator(self, mode)
        v.check(bool(self.detector_id), "detector_id", "detector_id is required")
        v.check_nested(self.detector)
        v.check_nested(self.image)

        # Cross-field consistency
        if self.detector.detector_id and self.detector_id and self.detector.detector_id != self.detector_id:
            v.check(False, "id_mismatch",
                    f"Ambiguous IDs: outer={self.detector_id}, inner={self.detector.detector_id}",
                    heal=lambda: setattr(self.detector, 'detector_id', self.detector_id))
        return v.valid

    def to_dict(self) -> dict:
        return _auto_to_dict(self)

    @staticmethod
    def from_dict(d: Any, *, mode: Union[ParseMode, str, None] = ParseMode.STRICT) -> "AcquisitionRequest":
        return _auto_from_dict(AcquisitionRequest, d, mode)

