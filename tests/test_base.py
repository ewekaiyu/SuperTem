"""Unit tests for :mod:`supertem.structures.base`."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pint
import pytest

from supertem.structures.base import (
    AcquisitionRequest,
    Aperture,
    ApertureControlRequest,
    BeamControlRequest,
    BeamSettings,
    BeamState,
    BeamSystemSettings,
    DetectorCapabilities,
    DetectorControlRequest,
    DetectorSettings,
    DetectorState,
    DetectorSystemSettings,
    Extras,
    FieldParser,
    ImageOutputSettings,
    MicroscopeImage,
    MicroscopeImageMetadata,
    MicroscopeSettings,
    MicroscopeState,
    ParseMode,
    Point,
    ProjectionControlRequest,
    ProjectionSettings,
    ProjectionSystemSettings,
    Q_,
    ROI,
    SafetyCheck,
    ScanControlRequest,
    ScanSettings,
    ScanSystemSettings,
    StageControlRequest,
    StageDriveType,
    StageMoveRequest,
    StagePosition,
    StageSystemSettings,
    SystemInfo,
    SystemSettings,
    Units,
    VacuumControlRequest,
    VacuumSettings,
    ensure_quantity,
)


def _extra_contains(extra: Extras, needle: str) -> bool:
    """Return True if *needle* appears anywhere in Extras.raw/notes."""
    if extra is None:
        return False

    raw = getattr(extra, "raw", None)
    notes = getattr(extra, "notes", None)

    haystacks: list[str] = []
    if raw is not None:
        try:
            haystacks.append(json.dumps(raw, default=str, sort_keys=True))
        except TypeError:
            haystacks.append(str(raw))
    if notes is not None:
        try:
            haystacks.append(json.dumps(notes, default=str, sort_keys=True))
        except TypeError:
            haystacks.append(str(notes))

    return needle in "\n".join(haystacks)


def _assert_recorded(extra: Extras, needle: str) -> None:
    """Assert that *needle* was recorded somewhere in Extras."""
    assert _extra_contains(extra, needle), f"Expected '{needle}' to be recorded in Extras"


def test_extras_lifecycle_complex():
    """Verify that nested 'extra' fields are parsed correctly."""
    raw_input = {
        "x": 10,
        "extra": {
            "vendor": {"fei": {"param": 1}},
            "unknown": {"legacy_field": "foo"},
            "notes": {"warning": "old config"}
        },
        "garbage_key": 999
    }
    p = Point.from_dict(raw_input)
    assert p.x == 10.0
    assert p.extra.vendor["fei"]["param"] == 1
    assert p.extra.unknown["legacy_field"] == "foo"
    assert p.extra.notes["warning"] == "old config"
    assert p.extra.unknown["garbage_key"] == 999


def test_extras_filtering():
    """Verify that Extras.to_dict() automatically removes empty internal dictionaries."""
    ex = Extras()
    ex.vendor = {}
    ex.notes = {"error": "bad"}
    d = ex.to_dict()
    assert "vendor" not in d
    assert "notes" in d
    assert d["notes"] == {"error": "bad"}


def test_extras_identity_branch():
    """Verify Extras.from_any returns the object unchanged if input is already Extras."""
    ex_orig = Extras(notes={"a": 1})
    ex_new = Extras.from_any(ex_orig)
    assert ex_new is ex_orig


def test_extras_nested_update_logic():
    """Verify Extras.from_any handles non-dict vendor/unknown inputs gracefully."""
    data = {"vendor": "just_a_string", "random": 123}
    ex = Extras.from_any(data)
    assert ex.unknown.get("random") == 123


def test_extras_edge_cases():
    """Tests Extras.from_any with weird inputs and Extras.to_dict with empties."""

    ex = Extras.from_any(["some", "list"], owner="Test")
    assert "Test.extra" in ex.raw

    ex2 = Extras(vendor={}, unknown={}, raw={}, notes={})
    assert ex2.to_dict() == {}

    ex3 = Extras(unknown={"key": None})
    assert ex3.to_dict() == {'unknown': {'key': None}}


def test_safety_check_behavior():
    """Verify the boolean behavior of SafetyCheck objects."""
    s = SafetyCheck.success()
    assert s
    assert s.allowed is True
    f = SafetyCheck.failure("Too hot")
    assert not f
    assert f.allowed is False
    assert "Too hot" in f.reasons
    f.add_reason("Too fast")
    assert len(f.reasons) == 2


def test_field_parser_primitives_edge_cases():
    """Verify FieldParser handles type coercion and enforces validation."""

    p = DetectorSettings(binning_index=10.0, frame_integration=10.5, _mode=ParseMode.LENIENT)
    assert p.binning_index == 10
    assert p.frame_integration is None
    assert "DetectorSettings.frame_integration" in p.extra.raw

    with pytest.raises(ValueError):
        DetectorSettings(detector_id="   ", _mode=ParseMode.STRICT)
    d = DetectorSettings(detector_id="", _mode=ParseMode.LENIENT)
    assert d.detector_id is None
    assert "DetectorSettings.detector_id.empty" in d.extra.notes


def test_field_parser_invalid_bool_string_records_and_defaults():
    """invalid bool strings should be recorded and replaced by default."""

    @dataclass
    class BoolTest:
        flag: bool = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = BoolTest(flag="maybe")
    p = FieldParser(obj, ParseMode.LENIENT, "BoolTest")
    obj.flag = p.bool(obj.flag, "flag", default=False)

    assert obj.flag is False

    assert "BoolTest.flag" in obj.extra.raw or "BoolTest.flag" in str(obj.extra.notes)


def test_field_parser_int_string_not_integer_like_records_and_defaults():
    """int parsing should record non-integer-like strings and return default."""

    @dataclass
    class IntTest:
        val: int = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = IntTest(val="1.2")
    p = FieldParser(obj, ParseMode.LENIENT, "IntTest")
    obj.val = p.int(obj.val, "val", default=7)

    assert obj.val == 7
    assert "IntTest.val" in obj.extra.raw or "IntTest.val" in str(obj.extra.notes)


def test_field_parser_float_unit_fallback_records_error():
    """unit-based float parsing should fall back cleanly when quantity parsing fails."""

    @dataclass
    class FloatUnitTest:
        val: float = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = FloatUnitTest(val="10 blargh")
    p = FieldParser(obj, ParseMode.LENIENT, "FloatUnitTest")
    obj.val = p.float(obj.val, "val", unit=Units.NM, default=None)

    assert obj.val is None
    assert "FloatUnitTest.val" in obj.extra.raw or "FloatUnitTest.val" in str(obj.extra.notes)


def test_field_parser_float_strictness():
    """Test Float parsing failure modes."""

    @dataclass
    class FloatTest:
        val: float = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = FloatTest()
    p = FieldParser(obj, ParseMode.LENIENT, "FloatTest")

    obj.val = p.float(Q_(10, 'nm'), "val")
    assert obj.val is None
    assert "FloatTest.val" in str(obj.extra.raw) or str(obj.extra.notes)


def test_field_parser_bool_guard():
    """Verify that the parser rejects boolean values passed to numeric fields."""

    @dataclass
    class GuardTest:
        f_val: float = None
        i_val: int = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    gt = GuardTest(f_val=True, i_val=False)
    p = FieldParser(gt, ParseMode.LENIENT, "GuardTest")
    gt.f_val = p.float(gt.f_val, "f_val")
    gt.i_val = p.int(gt.i_val, "i_val")

    assert gt.f_val is None
    assert gt.i_val is None
    assert "GuardTest.f_val" in gt.extra.raw or "GuardTest.f_val" in str(gt.extra.notes)


def test_parser_pair_structure_failures():
    """Verify behavior when parsing tuple/list fields with invalid structures."""

    d = DetectorSettings(binning_xy="NotAList", _mode=ParseMode.LENIENT)
    assert d.binning_xy is None

    d2 = DetectorSettings(binning_xy=[1], _mode=ParseMode.LENIENT)
    assert d2.binning_xy is None


def test_fuzzy_boolean_parsing():
    """Verify that string inputs like 'yes', 'on', '1' are correctly parsed as Boolean."""
    cases = [
        ("yes", True), ("YES", True), ("y", True),
        ("on", True), ("1", True), ("true", True),
        ("no", False), ("off", False), ("0", False), ("false", False)
    ]
    for input_val, expected in cases:
        s = StageSystemSettings(enabled=input_val, _mode=ParseMode.LENIENT)
        assert s.enabled is expected, f"Failed parsing {input_val}"


def test_parser_garbage_inputs():
    """Tests FieldParser handling of structurally invalid inputs for lists/tuples/models."""

    class MockObj:
        extra = Extras()

    p = FieldParser(MockObj(), ParseMode.LENIENT, "Mock")

    assert p.pair_int([1], "bad_pair") is None
    assert p.pair_int("not_list", "bad_pair") is None
    assert p.pair_int([None, 1], "bad_pair") is None

    assert p.pair_float([1.0], "bad_pair") is None
    assert p.pair_float("not_list", "bad_pair") is None

    assert p.list_str("not_a_list", "bad_list") == []

    assert p.model(Point, "not_a_dict", "bad_model") is None

    assert p.dict("not_a_dict", "bad_dict") is None


def test_field_parser_qty_empty_string_records_error_branch():
    """FieldParser.qty hits the 'val not None but ensure_quantity returned None' record branch."""

    @dataclass
    class QtyTest:
        q: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = QtyTest(q="")
    p = FieldParser(obj, ParseMode.LENIENT, "QtyTest")
    res = p.qty(obj.q, "q", Units.MS)
    assert res is None
    assert "QtyTest.q" in obj.extra.notes


def test_field_parser_pair_qty_type_error_records_branch():
    """FieldParser.pair_qty wrong container triggers TypeError record branch."""

    @dataclass
    class PairQtyTest:
        pair: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = PairQtyTest(pair=[1, 2, 3])
    p = FieldParser(obj, ParseMode.LENIENT, "PairQtyTest")
    res = p.pair_qty(obj.pair, "pair", Units.MS)
    assert res is None
    assert "PairQtyTest.pair" in obj.extra.notes


def test_field_parser_pair_float_success_path():
    """FieldParser.pair_float success path (a,b) computed without exceptions."""

    @dataclass
    class PairFloatTest:
        pair: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = PairFloatTest(pair=("1.25", 2))
    p = FieldParser(obj, ParseMode.LENIENT, "PairFloatTest")
    res = p.pair_float(obj.pair, "pair")
    assert res == (1.25, 2.0)


def test_field_parser_model_identity_non_dataclass_and_typeerror_guard():
    """FieldParser.model identity return for non-dataclass, plus TypeError guard path."""
    import supertem.structures.base as base_mod
    from typing import List as TList

    class Plain:
        pass

    @dataclass
    class Wrap:
        v: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = Wrap(v=Plain())
    p = FieldParser(obj, ParseMode.LENIENT, "Wrap")

    out = p.model(Plain, obj.v, "v")
    assert isinstance(out, Plain)

    out2 = p.model(TList[int], {"a": 1}, "v2", default=None)
    assert out2 is None

    assert "Wrap.v2" not in obj.extra.notes


def test_field_parser_model_from_dict_exception_records_and_defaults():
    """FieldParser.model catches exceptions from from_dict and records error."""

    @dataclass
    class BadModel:
        x: int = 0

        @staticmethod
        def from_dict(d, *, mode=None):
            raise ValueError("nope")

    @dataclass
    class Wrap:
        m: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    w = Wrap(m={"x": 1})
    p = FieldParser(w, ParseMode.LENIENT, "Wrap")
    out = p.model(BadModel, w.m, "m", default=None)
    assert out is None
    assert "Wrap.m" in w.extra.notes


def test_field_parser_map_model_non_dict_and_invalid_object_records():
    """FieldParser.map_model non-dict branch + invalid object per-key branch."""

    @dataclass
    class Tiny:
        a: int = 1
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

        @staticmethod
        def from_dict(d, *, mode=None):

            if "a" not in d:
                raise ValueError("missing a")
            return Tiny(a=int(d["a"]))

    @dataclass
    class Wrap:
        mp: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    w = Wrap(mp=[1, 2, 3])
    p = FieldParser(w, ParseMode.LENIENT, "Wrap")
    out = p.map_model(Tiny, w.mp, "mp")
    assert out == {}
    assert "Wrap.mp" in w.extra.notes

    w2 = Wrap(mp={"ok": {"a": 2}, "bad": {"zzz": 9}})
    p2 = FieldParser(w2, ParseMode.LENIENT, "Wrap")
    out2 = p2.map_model(Tiny, w2.mp, "mp")
    assert "ok" in out2

    assert any("Wrap.mp.bad" in k for k in w2.extra.notes.keys())


@pytest.mark.parametrize(
    "mode, should_raise",
    [
        (ParseMode.STRICT, True),
        (ParseMode.LENIENT, False),
    ],
)

def test_field_parser_id_empty_string_strict_vs_lenient(mode, should_raise):
    """FieldParser.id empty-string behavior differs by mode."""

    @dataclass
    class IdTest:
        ident: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = mode

    obj = IdTest(ident="   ")
    p = FieldParser(obj, mode, "IdTest")

    if should_raise:
        with pytest.raises(ValueError):
            p.id(obj.ident, "ident")
    else:
        out = p.id(obj.ident, "ident")
        assert out is None

        assert "IdTest.ident" in obj.extra.raw
        assert "IdTest.ident.empty" in obj.extra.notes


def test_field_parser_float_unit_success_path_parses_quantity():
    """FieldParser.float(unit=...) succeeds via ensure_quantity and returns magnitude."""

    @dataclass
    class FloatUnit:
        v: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = FloatUnit(v="12.5 nm")
    p = FieldParser(obj, ParseMode.LENIENT, "FloatUnit")
    out = p.float(obj.v, "v", unit=Units.NM, default=None)
    assert out == 12.5


def test_field_parser_int_and_float_typeerror_paths_record_lenient():
    """int/float TypeError paths for unparseable types are recorded in LENIENT."""

    @dataclass
    class NumTest:
        i: object = None
        f: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = NumTest(i=object(), f=object())
    p = FieldParser(obj, ParseMode.LENIENT, "NumTest")

    assert p.int(obj.i, "i", default=None) is None
    assert p.float(obj.f, "f", default=None) is None

    assert "NumTest.i" in obj.extra.notes
    assert "NumTest.f" in obj.extra.notes


def test_field_parser_str_bool_guard_lenient_records_and_defaults():
    """FieldParser.str rejects bool values and records error in LENIENT."""

    @dataclass
    class StrBool:
        s: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = StrBool(s=True)
    p = FieldParser(obj, ParseMode.LENIENT, "StrBool")
    out = p.str(obj.s, "s", default=None)
    assert out is None
    assert "StrBool.s" in obj.extra.notes


def test_field_parser_model_dataclass_cls_starstar_branch():
    """FieldParser.model instantiates dataclass via cls(**val) when no from_dict."""

    @dataclass
    class Dummy:
        x: int
        y: int = 2

    @dataclass
    class Wrap:
        v: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = Wrap(v={"x": 5, "y": 7})
    p = FieldParser(obj, ParseMode.LENIENT, "Wrap")
    out = p.model(Dummy, obj.v, "v", default=None)
    assert isinstance(out, Dummy)
    assert out.x == 5 and out.y == 7


def test_field_parser_str_lenient_coerces_and_records_note():
    """FieldParser.str in LENIENT mode coerces non-str via str(...)."""

    @dataclass
    class StrTest:
        val: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = StrTest(val=123)
    p = FieldParser(obj, ParseMode.LENIENT, "StrTest")
    out = p.str(obj.val, "val", default=None)
    assert out == "123"

    assert obj.extra.notes == {}


def test_field_parser_str_lenient_bool_guard_records_note_and_returns_default():
    """FieldParser.str rejects bool even in LENIENT mode and records a note."""

    @dataclass
    class StrBoolTest:
        val: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = StrBoolTest(val=True)
    p = FieldParser(obj, ParseMode.LENIENT, "StrBoolTest")
    out = p.str(obj.val, "val", default=None)
    assert out is None
    assert "StrBoolTest.val" in obj.extra.notes


def test_field_parser_pair_qty_returns_none_if_any_side_is_missing():
    """FieldParser.pair_qty returns None if any side fails to parse."""

    @dataclass
    class PairQtyTest:
        pair: object = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    obj = PairQtyTest(pair=[Q_(1, Units.MS), ""])
    p = FieldParser(obj, ParseMode.LENIENT, "PairQtyTest")
    out = p.pair_qty(obj.pair, "pair", Units.MS)
    assert out is None


def test_validator_check_return_value_and_healing():
    """Verify Validator.check logic when 'heal' returns success."""
    from supertem.structures.base import Validator
    class Dummy:
        extra = Extras()
        _mode = ParseMode.LENIENT
        val = -1

    obj = Dummy()
    v = Validator(obj, mode_override=ParseMode.LENIENT)

    def heal(): obj.val = 1

    result = v.check(obj.val > 0, "test_key", "error", heal=heal)
    assert result is True
    assert obj.val == 1

    def crashing_heal(): raise RuntimeError("Heal crashed")

    v.check(False, "crash_key", "error", heal=crashing_heal)
    assert v.valid is False
    assert "Dummy.crash_key" in obj.extra.notes


def test_validator_check_nested_and_check_nested_map_sets_valid_false():
    """Validator.check_nested + check_nested_map failure paths."""

    from supertem.structures.base import Validator

    @dataclass
    class Child:
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

        def validate(self, *, mode=None):
            return False

    @dataclass
    class Parent:
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    v = Validator(Parent(), ParseMode.LENIENT)
    assert v.check_nested(Child()) is False
    assert v.valid is False

    v2 = Validator(Parent(), ParseMode.LENIENT)
    assert v2.check_nested_map({"a": Child()}) is False
    assert v2.valid is False


def test_validator_check_has_intent_none_object_lenient():
    """Validator.check_has_intent when obj is None."""

    from supertem.structures.base import Validator

    @dataclass
    class Parent:
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT

    v = Validator(Parent(), ParseMode.LENIENT)
    ok = v.check_has_intent(None, "thing", "missing")
    assert ok is False
    assert v.valid is False


def test_unit_conversion_on_ingest():
    """Verify that compatible units are automatically converted to the base unit."""
    p1 = StagePosition.from_dict({"x": "1 um"})
    assert p1.x.magnitude == pytest.approx(1000.0)
    assert str(p1.x.units) == "nanometer"

    p2 = StagePosition.from_dict({"x": "500"})
    assert p2.x.magnitude == 500.0

    p3 = StagePosition(x=Q_(1e-6, 'm'))
    assert p3.x.magnitude == pytest.approx(1000.0)


def test_quantity_dimensionality_error():
    """Verify that physically incompatible units result in a parsing error."""
    p = StagePosition(x=Q_(10, 's'), _mode=ParseMode.LENIENT)
    assert p.x is None
    assert "StagePosition.x" in p.extra.raw or "StagePosition.x" in str(p.extra.notes)


def test_ensure_quantity_edge_cases():
    """Verify ensure_quantity logic for specific edge cases."""

    with pytest.raises(ValueError):
        ensure_quantity({"unit": "nm"}, "nm")

    with pytest.raises(TypeError):
        ensure_quantity(True, "nm")

    q = ensure_quantity("10 nm", "nm")
    assert q.magnitude == 10.0


def test_missing_unit_definition_strictness():
    """Verify that if a class defines a Quantity field but forgets to add it to _UNITS,."""

    @dataclass
    class DefectiveClass:
        val: Q_
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.STRICT

    obj = DefectiveClass(val=Q_(10, 'nm'))
    from supertem.structures.base import _auto_to_dict

    with pytest.raises(ValueError) as exc:
        _auto_to_dict(obj)
    assert "missing from _UNITS" in str(exc.value)

    obj._mode = ParseMode.LENIENT
    d = _auto_to_dict(obj)
    assert d['val']['magnitude'] == 10
    assert "serialization_warnings" in obj.extra.notes


def test_serialization_strips_units():
    """Verify that to_dict() converts complex Quantity objects into simple floats."""
    pos = StagePosition(x=Q_(1.5, "um"), y=Q_(200, "nm"), _mode="strict")
    data = pos.to_dict()

    assert isinstance(data.get("x_nm"), (float, int, type(None)))
    assert not isinstance(data.get("x_nm"), pint.Quantity)
    assert data["x_nm"] == pytest.approx(1500.0)


def test_ensure_quantity_fallback_floatable_object():
    """ensure_quantity fallback path uses float(val) for unexpected types."""

    class Floatable:
        def __float__(self):
            return 3.0

    q = ensure_quantity(Floatable(), Units.MS)
    assert q is not None
    assert float(q.to(Units.MS).magnitude) == 3.0


def test_auto_to_dict_quantity_list_unit_conversion_path():
    """_auto_to_dict list-of-Quantity branch for fields present in _UNITS."""
    import supertem.structures.base as base_mod
    from typing import List

    @dataclass
    class Foo:
        times: List[pint.Quantity] = field(default_factory=list)
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT
        _UNITS = {"times": Units.MS}

        def to_dict(self):
            return base_mod._auto_to_dict(self, unit_map=self._UNITS)

    f = Foo(times=[Q_(1, Units.MS), Q_(2, Units.MS)])
    d = f.to_dict()
    assert d["times_ms"] == [1.0, 2.0]


def test_ensure_quantity_dict_magnitude_unit_success():
    """ensure_quantity supports dict form {magnitude, unit}."""
    q = ensure_quantity({"magnitude": 10, "unit": "ms"}, Units.MS)
    assert q is not None
    assert float(q.to(Units.MS).magnitude) == 10.0


def test_auto_to_dict_conversion_failure_handling():
    """Trigger a Unit Conversion error during serialization (e.g. incompatible units)."""

    @dataclass
    class BrokenUnitClass:
        val: Q_
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.LENIENT
        _UNITS = {"val": "sec"}

    obj = BrokenUnitClass(val=Q_(10, 'nm'))

    from supertem.structures.base import _auto_to_dict

    data = _auto_to_dict(obj, unit_map=BrokenUnitClass._UNITS)

    assert "val_sec" in data
    assert data["val_sec"]["magnitude"] == 10
    assert data["val_sec"]["unit"] == "nanometer"

    assert "serialization_errors" in obj.extra.notes

    obj._mode = ParseMode.STRICT
    with pytest.raises(ValueError):
        _auto_to_dict(obj, unit_map=BrokenUnitClass._UNITS)


def test_map_model_parsing_logic():
    """Verify the logic for parsing dictionaries of objects (map_model)."""
    data = {
        "detectors": {
            "CamA": {"exposure_ms": 10},
            "CamB": None,
            "CamC": "InvalidString"
        }
    }
    state = MicroscopeState.from_dict(data, mode=ParseMode.LENIENT)
    assert "CamA" in state.detectors
    assert state.detectors["CamA"].exposure.magnitude == 10.0
    assert "CamB" not in state.detectors
    assert "CamC" not in state.detectors


def test_map_model_id_healing():
    """Verify map_model detects and heals mismatches between map key and object ID."""
    data = {
        "detectors": {
            "CamA": {"detector_id": "CamB"}
        }
    }

    with pytest.raises(ValueError):
        MicroscopeState.from_dict(data, mode=ParseMode.STRICT)

    state = MicroscopeState.from_dict(data, mode=ParseMode.LENIENT)
    assert state.detectors["CamA"].detector_id == "CamA"
    assert "id_mismatch" in str(state.extra.notes) or "id_mismatch" in str(state.extra.raw)


def test_jsonable_exception_path_falls_back_to_str():
    """_jsonable handles objects whose tolist/item throw and falls back to str."""
    import supertem.structures.base as base_mod

    class BadToList:
        def tolist(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "BAD_TOLIST"

    out = base_mod._jsonable(BadToList())
    assert out == "BAD_TOLIST"


def test_note_or_raise_lenient_with_no_extra_is_noop():
    """note_or_raise returns early when extra is None in LENIENT."""
    import supertem.structures.base as base_mod

    base_mod.note_or_raise(None, "k", ValueError("x"), mode=ParseMode.LENIENT, raw={"a": 1})


def test_finish_to_dict_and_add_extra_if_any_behaviors():
    """drop_none_keys/add_extra_if_any/_finish_to_dict integrate extra only when non-empty."""
    import supertem.structures.base as base_mod

    payload = {"a": 1, "b": None}

    out1 = base_mod._finish_to_dict(dict(payload), Extras())
    assert "b" not in out1
    assert "extra" not in out1

    ex = Extras(notes={"k": [{"error": "x", "type": "ValueError"}]})
    out2 = base_mod._finish_to_dict(dict(payload), ex)
    assert "b" not in out2
    assert "extra" in out2
    assert "notes" in out2["extra"]

    out3 = base_mod.add_extra_if_any({"a": 1}, {"notes": {"n": "v"}})
    assert out3["extra"]["notes"] == {"n": "v"}


def test_note_or_raise_falls_back_when_jsonable_raises(monkeypatch):
    """note_or_raise raw-serialization fallback branch when _jsonable crashes."""
    import supertem.structures.base as base_mod

    ex = Extras()

    def boom(_):
        raise RuntimeError("nope")

    monkeypatch.setattr(base_mod, "_jsonable", boom)
    base_mod.note_or_raise(ex, "K", ValueError("x"), mode=ParseMode.LENIENT, raw={"a": 1})
    assert "K" in ex.raw
    assert "K" in ex.notes


def test_extra_put_raw_supports_plain_dict_container():
    """_extra_put_raw can write into a plain dict (not just Extras)."""
    import supertem.structures.base as base_mod

    raw = {}
    base_mod._extra_put_raw(raw, "k", 123)

    assert raw["k_raw"] == 123


def test_normalize_extra_lenient_creates_raw_entry_on_bad_type():
    """normalize_extra_lenient exception path writes owner.extra into raw."""
    import supertem.structures.base as base_mod

    ex = base_mod.normalize_extra_lenient(42, owner="Owner")
    assert isinstance(ex, Extras)
    assert ex.raw["Owner.extra"] == repr(42)


def test_setup_from_dict_unknown_deepcopy_failure_goes_to_raw():
    """_setup_from_dict unknown-key deepcopy failure falls back to extra.raw."""
    import supertem.structures.base as base_mod

    class BadDeepcopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("no")

    d = {"x": 1.0, "weird": BadDeepcopy()}
    pt = base_mod.Point.from_dict(d, mode=ParseMode.LENIENT)
    assert "Point.unknown.weird" in pt.extra.raw


def test_auto_from_dict_alias_list_and_instance_replace_paths():
    """_auto_from_dict alias list branch + instance replacement branch."""
    import supertem.structures.base as base_mod

    @dataclass
    class AliasDemo:
        a: int | None = None
        extra: Extras = field(default_factory=Extras)
        _mode: ParseMode = ParseMode.STRICT

        @staticmethod
        def from_dict(d, *, mode=None):
            return base_mod._auto_from_dict(AliasDemo, d, mode, alias_map={"a": ["a_alt", "a2"]})

    obj = AliasDemo.from_dict({"a2": 5}, mode=ParseMode.LENIENT)
    assert obj.a == 5

    pt = base_mod.Point(x=1.0, _mode=ParseMode.STRICT)
    pt2 = base_mod.Point.from_dict(pt, mode=ParseMode.LENIENT)
    assert pt2._mode == ParseMode.LENIENT


def test_has_actionable_extras_detects_intent_in_nested_structures():
    """_has_actionable_extras nested dict/list/str/number branches."""
    import supertem.structures.base as base_mod

    assert base_mod._has_actionable_extras(None) is False

    ex = Extras()
    assert base_mod._has_actionable_extras(ex) is False

    ex.vendor["jeol"] = {"a": None, "b": "   ", "c": []}
    assert base_mod._has_actionable_extras(ex) is False

    ex.vendor["jeol"]["gain"] = 0
    assert base_mod._has_actionable_extras(ex) is True

    ex2 = Extras(unknown={"x": [None, ""]})
    assert base_mod._has_actionable_extras(ex2) is False
    ex2.unknown["x"] = [0]
    assert base_mod._has_actionable_extras(ex2) is True


def test_setup_from_dict_lenient_non_dict_input_creates_extras_raw():
    """_setup_from_dict lenient non-dict input returns empty data and Extras from input."""
    import supertem.structures.base as base_mod

    data, mode, extra = base_mod._setup_from_dict(Point, "not-a-dict", ParseMode.LENIENT)
    assert data == {}
    assert mode == ParseMode.LENIENT
    assert isinstance(extra, Extras)

    assert "Point.extra" in extra.raw

    with pytest.raises(TypeError):
        base_mod._setup_from_dict(Point, "not-a-dict", ParseMode.STRICT)


def test_jsonable_numpy_scalar_item_branch_explicit():
    """_jsonable uses .item() for numpy scalar types."""
    import supertem.structures.base as base_mod

    val = np.int32(7)
    out = base_mod._jsonable(val)
    assert isinstance(out, int)
    assert out == 7


def test_as_parse_mode_unknown_string_defaults_to_strict():
    """as_parse_mode falls back to STRICT on unknown values."""
    import supertem.structures.base as base_mod
    assert base_mod.as_parse_mode("definitely-not-a-real-mode") == ParseMode.STRICT


def test_point_scenarios():
    """Verify Point parsing robustness and default behaviors."""
    with pytest.raises(ValueError):
        Point(x="not_a_number", _mode=ParseMode.STRICT)
    p = Point(x="10.5", y=None, z=5, name=123, _mode=ParseMode.LENIENT)
    assert p.x == 10.5
    assert p.y == None
    assert p.name == "123"

    p_bad = Point(x=float('inf'), _mode=ParseMode.LENIENT)
    p_bad.validate()
    assert p_bad.x == 0.0


def test_point_tuple_input():
    """Verify Point.from_dict correctly handles list/tuple inputs."""
    p = Point.from_dict([10, 20])
    assert p.x == 10.0 and p.y == 20.0 and p.z == None
    p2 = Point.from_dict((1, 2, 3))
    assert p2.z == 3.0


def test_roi_scenarios():
    """Verify ROI validation logic."""
    roi = ROI.from_dict({"x": 10, "y": 10, "w": 100, "h": 200})
    assert roi.width == 100

    roi_inv = ROI(x=-5, width=-100, _mode=ParseMode.LENIENT)
    roi_inv.validate()
    assert roi_inv.x == 0
    assert roi_inv.width == 512

    roi_tup = ROI.from_dict((0, 0, 1024, 1024))
    assert roi_tup.width == 1024


def test_image_output_settings():
    """Verify ImageOutputSettings defaults and format validation logic."""
    ios = ImageOutputSettings(file_format="JPEG", path="/tmp")
    assert ios.file_format == "jpeg"

    ios_bad = ImageOutputSettings(file_format="gif", _mode=ParseMode.LENIENT)
    ios_bad.validate()
    assert ios_bad.file_format == "tiff"


def test_stage_position_math_and_logic():
    """Verify StagePosition arithmetic (add/sub) and proximity checks."""
    p1 = StagePosition(x=Q_(10, 'nm'), y=Q_(20, 'nm'))
    p2 = StagePosition(x=Q_(5, 'nm'), z=Q_(10, 'nm'))
    p3 = p1 + p2
    assert p3.x.magnitude == 15.0
    assert p3.y is None

    p4 = p1 - p2
    assert p4.x.magnitude == 5.0

    target = StagePosition(x=Q_(10, 'nm'))
    current = StagePosition(x=Q_(10.5, 'nm'), y=Q_(500, 'nm'))
    assert target.is_close(current, tol_nm=1.0) is True
    assert target.is_close(current, tol_nm=0.1) is False


def test_stage_position_wildcard_logic():
    """Verify 'wildcard' behavior in StagePosition.is_close."""
    p_req = StagePosition(x=None, y=Q_(10, 'nm'))
    p_curr = StagePosition(x=Q_(999, 'nm'), y=Q_(10, 'nm'))

    assert p_req.is_close(p_curr) is True

    p_req2 = StagePosition(z=Q_(10, 'nm'))
    p_curr2 = StagePosition(x=Q_(0, 'nm'))
    assert p_req2.is_close(p_curr2) is False


def test_stage_position_round_trip():
    """Verify that StagePosition objects can be serialized to JSON and reconstructed exactly."""
    original = StagePosition(x=Q_(1.5, 'um'), y=Q_(0, 'nm'), r=Q_(45, 'deg'), name="Target A")
    payload = original.to_dict()
    assert payload["x_nm"] == pytest.approx(1500.0)
    assert payload["r_deg"] == pytest.approx(45.0)

    reconstructed = StagePosition.from_dict(payload)
    assert reconstructed.x.magnitude == pytest.approx(1500.0)
    assert reconstructed.name == "Target A"


def test_stage_system_settings_complex():
    """Verify StageSystemSettings defaults, boolean logic, and limits configuration."""
    sys = StageSystemSettings(can_x=True, x_limits=None, _mode=ParseMode.LENIENT)
    sys.validate()

    assert sys.can_x is False

    sys = StageSystemSettings(z_limits=(Q_(-10, 'um'), Q_(10, 'um')), eucentric_z=Q_(50, 'um'), _mode=ParseMode.LENIENT)
    sys.validate()

    assert sys.eucentric_z is None

    sys = StageSystemSettings(max_step_distance=Q_(1, 'um'))
    d = sys.to_dict()
    assert d["max_step_nm"] == pytest.approx(1000.0)


def test_stage_move_request_piezo_safety():
    """Verify the specific warning logic for large Piezo moves."""
    req = StageMoveRequest(drive_type="piezo", relative=True, target=StagePosition(x=Q_(10, 'um')),
                           _mode=ParseMode.LENIENT)
    req.validate()
    assert "StageMoveRequest.piezo_limit.x" in req.extra.notes


def test_complex_stage_safety_relative():
    """Verify complex safety logic for relative moves (Unknown current pos vs Limit breach)."""
    limits = StageSystemSettings(x_limits=(Q_(-100, 'um'), Q_(100, 'um')), enabled=True, can_x=True)
    req_move = StagePosition(x=Q_(10, 'um'))

    check = limits.is_safe_move(req_move, current=None, relative=True)
    assert not check.allowed
    assert "without current position" in check.reasons[0]

    current = StagePosition(x=Q_(50, 'um'))
    check = limits.is_safe_move(req_move, current=current, relative=True)
    assert check.allowed

    big_move = StagePosition(x=Q_(60, 'um'))
    check = limits.is_safe_move(big_move, current=current, relative=True)
    assert not check.allowed


def test_stage_safety_ignore_step_limit():
    """Verify that passing `ignore_step_limit=True` bypasses the velocity checks."""
    settings = StageSystemSettings(
        max_step_distance=Q_(10, 'nm'),
        can_x=True, x_limits=(Q_(-1000, 'nm'), Q_(1000, 'nm'))
    )
    current = StagePosition(x=Q_(0, 'nm'))
    target = StagePosition(x=Q_(100, 'nm'))

    assert not settings.is_safe_move(target, current).allowed

    assert settings.is_safe_move(target, current, ignore_step_limit=True).allowed


def test_stage_tilt_step_safety():
    """Specifically target the 'tilt' branch in is_safe_move velocity checks."""
    settings = StageSystemSettings(
        max_step_deg=Q_(1.0, 'deg'),
        can_tilt_x=True, tilt_x_limits=(Q_(-10, 'deg'), Q_(10, 'deg'))
    )

    current = StagePosition(tilt_x=Q_(0, 'deg'))
    target = StagePosition(tilt_x=Q_(2.0, 'deg'))

    check = settings.is_safe_move(target, current, relative=True)
    assert not check.allowed
    assert "Tilt X step" in check.reasons[0]


def test_stage_control_request_validation():
    """Verify validation logic for StageControlRequest, including Action strings and Axes."""

    with pytest.raises(ValueError):
        StageControlRequest(action="DANCE", _mode=ParseMode.STRICT).validate()

    valid = StageControlRequest(action="HOME", axes=["x", "y"])
    assert valid.validate()

    s = StageControlRequest(action="INVALID_ACTION", _mode=ParseMode.STRICT)
    with pytest.raises(ValueError) as exc:
        s.validate()
    assert "Action must be one of" in str(exc.value)


def test_stage_system_settings_disabled_axis_and_relative_missing_current_axis_and_step_limits():
    """StageSystemSettings.is_safe_move early exits + Z/TiltY step limit branches."""

    sys = StageSystemSettings(can_z=False, z_limits=(Q_(-100, 'nm'), Q_(100, 'nm')))
    check = sys.is_safe_move(StagePosition(z=Q_(1, 'nm')))
    assert not check.allowed
    assert "disabled axis" in check.reasons[0]

    sys2 = StageSystemSettings(can_z=True, z_limits=(Q_(-1000, 'nm'), Q_(1000, 'nm')))
    check2 = sys2.is_safe_move(StagePosition(z=Q_(10, 'nm')), current=StagePosition(x=Q_(0, 'nm')), relative=True)
    assert not check2.allowed
    assert "current position unknown" in check2.reasons[0]

    sys3 = StageSystemSettings(
        can_z=True, can_tilt_y=True,
        z_limits=(Q_(-10000, 'nm'), Q_(10000, 'nm')),
        tilt_y_limits=(Q_(-10, 'deg'), Q_(10, 'deg')),
        max_step_distance=Q_(10, 'nm'),
        max_step_deg=Q_(1.0, 'deg'),
    )
    cur = StagePosition(z=Q_(0, 'nm'), tilt_y=Q_(0, 'deg'))
    tgt = StagePosition(z=Q_(100, 'nm'), tilt_y=Q_(5.0, 'deg'))
    check3 = sys3.is_safe_move(tgt, current=cur, relative=False)
    assert not check3.allowed
    assert any("Z step" in r for r in check3.reasons)
    assert any("Tilt Y step" in r for r in check3.reasons)


def test_stage_position_is_close_raises_on_wrong_type():
    """StagePosition.is_close raises TypeError when comparing to non-StagePosition."""
    pos = StagePosition(x=Q_(0, Units.NM))
    with pytest.raises(TypeError):
        pos.is_close("not-a-stage-position")


def test_beam_settings_parsing():
    """Verify BeamSettings parsing logic for voltages, currents, and integers."""
    b = BeamSettings.from_dict({"voltage": "300 kV", "beam_current": "100 pA", "spot_size": "1"})
    assert b.voltage.magnitude == 300.0
    assert b.beam_current.to("nA").magnitude == pytest.approx(0.1)

    b_bad = BeamSettings(spot_size=-1, _mode=ParseMode.LENIENT)
    b_bad.validate()
    assert b_bad.spot_size is None


def test_beam_system_settings_limits():
    """Verify BeamSystemSettings safely rejects BeamSettings outside defined limits."""
    sys = BeamSystemSettings(voltage_limits=(Q_(100, 'kV'), Q_(200, 'kV')), _mode=ParseMode.STRICT)
    res = sys.is_safe_beam(BeamSettings(voltage=Q_(300, 'kV')))
    assert not res.allowed


def test_beam_system_settings_range_integrity_and_non_negativity_healing():
    """Validate range swapping (min>max) and clamping negative minimums."""
    sys = BeamSystemSettings(
        voltage_limits=(Q_(200, 'kV'), Q_(100, 'kV')),
        beam_current_limits=(Q_(-1, 'nA'), Q_(10, 'nA')),
        spot_size_limits=(-1, 10),
        _mode=ParseMode.LENIENT
    )

    sys.validate()

    vmin, vmax = sys.voltage_limits
    assert vmin.to('kV').magnitude == pytest.approx(100.0)
    assert vmax.to('kV').magnitude == pytest.approx(200.0)

    cmin, cmax = sys.beam_current_limits
    assert cmin.to('nA').magnitude == pytest.approx(0.0)
    assert cmax.to('nA').magnitude == pytest.approx(10.0)

    smin, smax = sys.spot_size_limits
    assert smin == 0
    assert smax == 10


def test_beam_request_extras_only_validation():
    """Verify that a request containing ONLY vendor extras is considered a valid intent."""
    req = BeamControlRequest(
        target=BeamSettings(extra={"vendor": {"JEOL": {"alpha_index": 1}}})
    )
    assert req.validate() is True


def test_beam_system_settings_is_safe_reasons_spot_size_limit():
    """BeamSystemSettings.is_safe reports spot_size outside allowed limits."""
    bsys = BeamSystemSettings(spot_size_limits=(1, 3), _mode=ParseMode.LENIENT)
    bs = BeamSettings(spot_size=5, _mode=ParseMode.LENIENT)

    chk = bsys.is_safe_beam(bs)
    assert not chk.allowed
    assert any("Spot size" in r for r in chk.reasons)


def test_projection_mode_dependencies():
    """Verify logical dependencies in ProjectionSettings."""
    p1 = ProjectionSettings(optical_mode="IMAGING", magnification=50000, _mode=ParseMode.STRICT)
    assert p1.validate()

    p2 = ProjectionSettings(optical_mode="DIFFRACTION", _mode=ParseMode.LENIENT)
    assert p2.validate() is False
    assert "ProjectionSettings.missing_cam_len" in p2.extra.notes


def test_projection_system_limits():
    """Verify ProjectionSystemSettings limit checking logic."""
    sys = ProjectionSystemSettings(magnification_limits=(1000, 100000))
    check = sys.is_safe_projection(ProjectionSettings(magnification=200000))
    assert not check.allowed


def test_projection_control_request_validation():
    """Targets ProjectionControlRequest validation logic."""

    req = ProjectionControlRequest(
        target=ProjectionSettings(optical_mode="IMAGING"),
        _mode=ParseMode.STRICT
    )
    assert req.validate()

    bad_req = ProjectionControlRequest(
        target=ProjectionSettings(optical_mode="DIFFRACTION"),
        _mode=ParseMode.STRICT
    )
    with pytest.raises(ValueError) as exc:
        bad_req.validate()
    assert "Diffraction mode requires camera_length" in str(exc.value)


def test_projection_settings_screen_position_heals_and_records_note_lenient():
    """ProjectionSettings.validate heals invalid screen_position but remains valid in LENIENT."""
    ps = ProjectionSettings(screen_position="SIDEWAYS", _mode=ParseMode.LENIENT)
    ok = ps.validate(mode=ParseMode.LENIENT)
    assert ok is True
    assert ps.screen_position is None

    assert "ProjectionSettings.screen_position" in ps.extra.notes


def test_projection_system_settings_range_and_safety_branches():
    """ProjectionSystemSettings.validate range + is_safe_projection limit checks."""
    pss = ProjectionSystemSettings(
        camera_length_limits=(Q_(10, Units.MM), Q_(5, Units.MM)),
        defocus_limits=(Q_(-1, Units.NM), Q_(1, Units.NM)),
        _mode=ParseMode.LENIENT,
    )
    assert pss.validate(mode=ParseMode.LENIENT) is False

    pss2 = ProjectionSystemSettings(
        camera_length_limits=(Q_(5, Units.MM), Q_(10, Units.MM)),
        defocus_limits=(Q_(-1, Units.NM), Q_(1, Units.NM)),
        magnification_limits=(10, 20),
        _mode=ParseMode.LENIENT,
    )
    target = ProjectionSettings(optical_mode="DIFFRACTION", camera_length=Q_(50, Units.MM), defocus=Q_(2, Units.NM),
                               magnification=100, _mode=ParseMode.LENIENT)
    check = pss2.is_safe_projection(target)
    assert not check.allowed
    assert len(check.reasons) >= 2


def test_detector_settings_logic():
    """Verify DetectorSettings logic: non-negative exposure/integration."""
    d = DetectorSettings(exposure=Q_(-1, 's'), _mode=ParseMode.LENIENT)
    d.validate()
    assert d.exposure is None

    d = DetectorSettings(frame_integration=0, _mode=ParseMode.LENIENT)
    d.validate()
    assert d.frame_integration == 1


def test_detector_dose_consistency_logic():
    """Verify the logic that checks if Exposure, Frame Rate, and Total Frames are mathematically consistent."""

    d = DetectorSettings(exposure=Q_(1.0, 's'), frame_rate=Q_(10, 'Hz'), total_frames=10, _mode=ParseMode.STRICT)
    assert d.validate()

    d_bad = DetectorSettings(exposure=Q_(1.0, 's'), frame_rate=Q_(10, 'Hz'), total_frames=20, _mode=ParseMode.LENIENT)
    d_bad.validate()
    assert "DetectorSettings.dose_logic" in str(d_bad.extra.notes)


def test_detector_shutter_mode_validation():
    """Verify validation logic for shutter_mode strings."""
    d = DetectorSettings(shutter_mode="INVALID_MODE", _mode=ParseMode.LENIENT)
    d.validate()
    assert "DetectorSettings.shutter_mode" in str(d.extra.notes)


def test_detector_capabilities_comprehensive():
    """Verify DetectorCapabilities.supports() logic against various settings combinations."""
    caps = DetectorCapabilities(can_binning=False, exposure_max=Q_(1000, 'ms'), roi_size_max=(1024, 1024))
    assert not caps.supports(DetectorSettings(binning_index=2))
    assert not caps.supports(DetectorSettings(exposure=Q_(2000, 'ms')))
    assert not caps.supports(DetectorSettings(roi=ROI(width=2048)))


def test_detector_capabilities_supports_binning_xy_offset_and_digital_rotation():
    """Exercise supports() branches for binning_xy, offset_index, and digital_rotation."""
    caps = DetectorCapabilities(
        can_binning=False,
        binning_xy_min=(1, 1),
        binning_xy_max=(2, 2),
        can_offset=False,
        offset_index_min=0,
        offset_index_max=3,
        can_digital_rotation=False,
        digital_rotation_min=Q_(-5, 'deg'),
        digital_rotation_max=Q_(5, 'deg'),
    )

    check = caps.supports(DetectorSettings(binning_xy=(2, 1)))
    assert not check.allowed
    assert any("Binning XY" in r for r in check.reasons)

    check2 = caps.supports(DetectorSettings(binning_xy=(0, 1)))
    assert not check2.allowed
    assert any("below limit" in r for r in check2.reasons)

    check3 = caps.supports(DetectorSettings(binning_xy=(3, 1)))
    assert not check3.allowed
    assert any("exceeds limit" in r for r in check3.reasons)

    check4 = caps.supports(DetectorSettings(offset_index=1))
    assert not check4.allowed
    assert any("Offset Index" in r for r in check4.reasons)

    check5 = caps.supports(DetectorSettings(digital_rotation=Q_(1, 'deg')))
    assert not check5.allowed
    assert any("Digital Rotation" in r for r in check5.reasons)

    check6 = caps.supports(DetectorSettings(digital_rotation=Q_(10, 'deg')))
    assert not check6.allowed
    assert any("exceeds limit" in r for r in check6.reasons)


def test_detector_capabilities_validate_heals_ranges_and_roi_tuple_consistency():
    """validate() should swap min/max and heal ROI tuple min>max in LENIENT mode."""
    caps = DetectorCapabilities(
        exposure_min=Q_(10, 'ms'),
        exposure_max=Q_(1, 'ms'),
        digital_rotation_min=Q_(10, 'deg'),
        digital_rotation_max=Q_(0, 'deg'),
        roi_size_min=(2048, 1024),
        roi_size_max=(1024, 512),
        _mode=ParseMode.LENIENT
    )

    caps.validate()

    assert caps.exposure_min.to('ms').magnitude == pytest.approx(1.0)
    assert caps.exposure_max.to('ms').magnitude == pytest.approx(10.0)

    assert caps.digital_rotation_min.to('deg').magnitude == pytest.approx(0.0)
    assert caps.digital_rotation_max.to('deg').magnitude == pytest.approx(10.0)

    assert caps.roi_size_min == (1024, 512)
    assert caps.roi_size_max == (2048, 1024)


def test_detector_system_registry():
    """Verify DetectorSystemSettings registry logic."""
    sys = DetectorSystemSettings(
        available_detector_ids=["CamA"],
        capabilities_by_id={"CamA": DetectorCapabilities(exposure_max=Q_(10, 'ms'))},
        defaults_by_id={"CamA": DetectorSettings(exposure=Q_(5, 'ms'))},
        _mode=ParseMode.STRICT
    )
    assert sys.validate()
    assert sys.is_supported(DetectorSettings(detector_id="CamA", exposure=Q_(5, 'ms')))
    assert not sys.is_supported(DetectorSettings(detector_id="CamA", exposure=Q_(20, 'ms')))


def test_detector_system_consistency_garbage_data():
    """Verify that invalid entries in the detector registry are sanitized in Lenient mode."""
    raw_data = {
        "available_detectors": ["CamA", "CamB"],
        "defaults_by_id": {"CamA": {"exposure_ms": 100}, "CamB": "GARBAGE_STRING"},
        "capabilities_by_id": {"CamA": {}, "CamB": {}}
    }
    sys = DetectorSystemSettings.from_dict(raw_data, mode=ParseMode.LENIENT)
    sys.validate()
    assert "CamB" not in sys.defaults_by_id
    assert "CamB" not in sys.available_detector_ids


def test_detector_control_request_sync():
    """Targets DetectorControlRequest ID sync logic."""

    req = DetectorControlRequest(
        target=DetectorSettings(detector_id="CamA", exposure=Q_(10, 'ms')),
        _mode=ParseMode.LENIENT
    )
    assert req.detector_id == "CamA"
    assert req.validate()

    empty_req = DetectorControlRequest(detector_id="CamA", _mode=ParseMode.STRICT)
    with pytest.raises(ValueError):
        empty_req.validate()


def test_detector_control_action_enum():
    """Targets the valid_set check for DetectorControlRequest actions."""
    req = DetectorControlRequest(
        detector_id="CamA",
        action="DANCE_PARTY",
        _mode=ParseMode.STRICT
    )
    with pytest.raises(ValueError) as exc:
        req.validate()
    assert "Action must be one of" in str(exc.value)


def test_detector_control_empty_intent_logic():
    """Targets the dual-path logic for DetectorControlRequest."""

    req_action = DetectorControlRequest(detector_id="CamA", action="INSERT", _mode=ParseMode.STRICT)
    assert req_action.validate()

    req_settings = DetectorControlRequest(
        detector_id="CamA",
        target=DetectorSettings(exposure=Q_(1, 's')),
        _mode=ParseMode.STRICT
    )
    assert req_settings.validate()

    req_empty = DetectorControlRequest(detector_id="CamA", _mode=ParseMode.STRICT)
    with pytest.raises(ValueError) as exc:
        req_empty.validate()
    assert "Request must have either settings to apply or an action" in str(exc.value)


def test_acquisition_request_validation():
    """Verify AcquisitionRequest validation logic, including ID mismatch detection."""
    req = AcquisitionRequest(detector_id="CamA", detector=DetectorSettings(detector_id="CamB"), _mode=ParseMode.LENIENT)
    req.validate()
    assert req.detector.detector_id == "CamA"
    assert "AcquisitionRequest.id_mismatch" in req.extra.notes


def test_acquisition_request_id_mismatch_strict():
    """Strict ID mismatch in AcquisitionRequest should raise ValueError."""
    req = AcquisitionRequest(
        detector_id="OuterID",
        detector=DetectorSettings(detector_id="InnerID"),
        _mode=ParseMode.STRICT
    )
    with pytest.raises(ValueError) as exc:
        req.validate()
    assert "Ambiguous IDs" in str(exc.value)


def test_acquisition_request_empty_intent():
    """Targets AcquisitionRequest empty intent logic."""

    req_blind = AcquisitionRequest(detector_id="CamA", _mode=ParseMode.STRICT)
    with pytest.raises(ValueError) as exc:
        req_blind.validate()
    assert "Acquisition requires explicit detector settings" in str(exc.value)

    req_valid = AcquisitionRequest(
        detector_id="CamA",
        detector=DetectorSettings(exposure=Q_(0.5, 's')),
        _mode=ParseMode.STRICT
    )
    assert req_valid.validate()


def test_detector_capabilities_supports_hits_more_rejection_branches():
    """DetectorCapabilities.supports for exposure/frame_integration/gain/roi range checks."""
    caps = DetectorCapabilities(
        can_gain=True,
        can_offset=True,
        can_binning=True,
        exposure_min=Q_(5, Units.MS),
        exposure_max=Q_(50, Units.MS),
        frame_integration_min=2,
        frame_integration_max=4,
        gain_index_min=1,
        gain_index_max=3,
        roi_size_min=(64, 64),
        roi_size_max=(128, 128),
        _mode=ParseMode.LENIENT,
    )

    bad = DetectorSettings(
        exposure=Q_(1, Units.MS),
        frame_integration=1,
        gain_index=99,
        roi=ROI(width=32, height=512, _mode=ParseMode.LENIENT),
        _mode=ParseMode.LENIENT,
    )
    check = caps.supports(bad)
    assert bool(check) is False

    msg = "\n".join(check.reasons)
    assert "Exposure" in msg
    assert "Frame Integration" in msg
    assert "Gain Index" in msg
    assert "ROI size" in msg


def test_detector_settings_binning_xy_heals_to_none_lenient():
    """DetectorSettings.validate binning_xy >0 check + heal branch."""
    ds = DetectorSettings(
        binning_xy=(0, 2),
        exposure=Q_(10, Units.MS),
        _mode=ParseMode.LENIENT,
    )
    ok = ds.validate(mode=ParseMode.LENIENT)
    assert ok is True
    assert ds.binning_xy is None


def test_detector_capabilities_supports_binning_index_limits_reasons():
    """DetectorCapabilities.supports reports binning_index min/max violations."""
    caps = DetectorCapabilities(
        can_binning=True,
        binning_index_min=2,
        binning_index_max=4,
        _mode=ParseMode.LENIENT,
    )

    low = DetectorSettings(binning_index=1, exposure=Q_(10, Units.MS), _mode=ParseMode.LENIENT)
    high = DetectorSettings(binning_index=5, exposure=Q_(10, Units.MS), _mode=ParseMode.LENIENT)

    c1 = caps.supports(low)
    assert not c1.allowed
    assert any("below limit" in r for r in c1.reasons)

    c2 = caps.supports(high)
    assert not c2.allowed
    assert any("exceeds limit" in r for r in c2.reasons)


def test_detector_capabilities_supports_gain_offset_rotation_and_frame_integration_reasons():
    """DetectorCapabilities.supports returns detailed reasons for multiple violations."""
    caps = DetectorCapabilities(
        frame_integration_max=1,
        can_gain=False,
        gain_index_min=2,
        gain_index_max=3,
        can_offset=True,
        offset_index_min=0,
        offset_index_max=5,
        can_digital_rotation=True,
        digital_rotation_min=Q_(0, Units.DEG),
        digital_rotation_max=Q_(90, Units.DEG),
        _mode=ParseMode.LENIENT,
    )

    s = DetectorSettings(
        frame_integration=2,
        gain_index=1,
        offset_index=-1,
        digital_rotation=Q_(-1, Units.DEG),
        exposure=Q_(10, Units.MS),
        _mode=ParseMode.LENIENT,
    )
    chk = caps.supports(s)
    assert not chk.allowed

    rs = [r.lower() for r in chk.reasons]
    assert any("frame integration" in r for r in rs)
    assert any("gain" in r for r in rs)
    assert any("offset" in r for r in rs)
    assert any("digital rotation" in r for r in rs)
    assert any("below limit" in r for r in rs)

    s2 = DetectorSettings(offset_index=999, exposure=Q_(10, Units.MS), _mode=ParseMode.LENIENT)
    chk2 = caps.supports(s2)
    assert not chk2.allowed
    assert any("offset" in r.lower() and "exceeds" in r.lower() for r in chk2.reasons)


def test_detector_system_settings_default_detector_id_heals_and_is_supported_guards():
    """DetectorSystemSettings.validate default ID healing + is_supported early returns."""
    dsys = DetectorSystemSettings(
        available_detector_ids=["DET-A"],
        default_detector_id="DET-UNKNOWN",
        defaults_by_id={
            "DET-A": DetectorSettings(detector_id="DET-A", exposure=Q_(1, Units.MS), _mode=ParseMode.LENIENT)
        },
        capabilities_by_id={"DET-A": DetectorCapabilities(_mode=ParseMode.LENIENT)},
        _mode=ParseMode.LENIENT,
    )
    ok = dsys.validate(mode=ParseMode.LENIENT)
    assert ok is True
    assert dsys.default_detector_id is None

    chk_no_id = dsys.is_supported(DetectorSettings(detector_id=None, _mode=ParseMode.LENIENT))
    assert chk_no_id.allowed is False

    chk_unknown = dsys.is_supported(DetectorSettings(detector_id="DET-UNKNOWN", _mode=ParseMode.LENIENT))
    assert chk_unknown.allowed is False


def test_scan_settings_estimation_and_healing():
    """Verify ScanSettings duration estimation + LENIENT healing in one place."""

    s = ScanSettings(
        width_px=512,
        height_px=512,
        pixel_dwell_time=Q_(10, 'us'),
        flyback_time=Q_(100, 'us'),
    )
    assert s.estimated_duration.to('s').magnitude == pytest.approx(2.67264)

    s_bad = ScanSettings(pixel_dwell_time=Q_(-10, 'us'), width_px=-50, _mode=ParseMode.LENIENT)
    ok = s_bad.validate()
    assert ok is True
    assert s_bad.pixel_dwell_time is None
    assert s_bad.width_px == 512


def test_scan_system_safety():
    """Verify ScanSystemSettings checks for timing limits and supported modes."""
    sys = ScanSystemSettings(pixel_dwell_time_limits=(Q_(1, 'us'), Q_(10, 'us')), available_scan_modes=["Frame"])
    assert not sys.is_safe_scan(ScanSettings(pixel_dwell_time=Q_(100, 'us')))
    assert not sys.is_safe_scan(ScanSettings(scan_mode="Line"))


def test_scan_control_request_logic():
    """Targets ScanControlRequest action dependencies."""

    req_stop = ScanControlRequest(action="STOP", _mode=ParseMode.STRICT)
    assert req_stop.validate()

    req_start_none = ScanControlRequest(action="START", target=None, _mode=ParseMode.STRICT)
    with pytest.raises(ValueError):
        req_start_none.validate()

    req_start_empty = ScanControlRequest(action="START", target=ScanSettings(), _mode=ParseMode.STRICT)
    with pytest.raises(ValueError):
        req_start_empty.validate()

    req_valid = ScanControlRequest(
        action="START",
        target=ScanSettings(pixel_dwell_time=Q_(10, 'us')),
        _mode=ParseMode.STRICT
    )
    assert req_valid.validate()


def test_scan_settings_height_px_validation_heals_to_default():
    """ScanSettings.validate enforces height_px > 0 and heals in LENIENT."""
    s = ScanSettings(width_px=128, height_px=-2, _mode=ParseMode.LENIENT)
    ok = s.validate(mode=ParseMode.LENIENT)
    assert ok is True
    assert s.height_px == 512


def test_scan_system_settings_check_range_swap_and_ge_zero_note():
    """ScanSystemSettings range helper swaps min/max and notes negative mins."""
    ss = ScanSystemSettings(
        pixel_dwell_time_limits=(Q_(5, Units.US), Q_(1, Units.US)),
        flyback_time_limits=(Q_(-1, Units.US), Q_(1, Units.US)),
        _mode=ParseMode.LENIENT,
    )
    ok = ss.validate(mode=ParseMode.LENIENT)
    assert ok is True

    assert ss.pixel_dwell_time_limits[0] == Q_(1, Units.US)
    assert ss.pixel_dwell_time_limits[1] == Q_(5, Units.US)

    assert any(k.startswith("ScanSystemSettings.flyback_time_limits.min") for k in ss.extra.notes.keys())


def test_vacuum_settings_readonly_inputs():
    """Verify validation of VacuumSettings (inputs must be non-negative) and request structure."""
    req = VacuumControlRequest(target=VacuumSettings(turbo_pump_state="ON"), _mode=ParseMode.STRICT)
    assert req.validate()
    v = VacuumSettings(column_pressure=Q_(-5, 'Pa'), _mode=ParseMode.LENIENT)
    v.validate()
    assert "VacuumSettings.column_pressure" in v.extra.notes


@pytest.mark.parametrize(
    "name, req_factory_empty, req_factory_valid, expected_msg",
    [
        (
            "BeamControlRequest",
            lambda: BeamControlRequest(target=BeamSettings(), _mode=ParseMode.STRICT),
            lambda: BeamControlRequest(target=BeamSettings(spot_size=1), _mode=ParseMode.STRICT),
            "Beam request has no parameters set",
        ),
        (
            "ProjectionControlRequest",
            lambda: ProjectionControlRequest(target=ProjectionSettings(), _mode=ParseMode.STRICT),
            lambda: ProjectionControlRequest(
                target=ProjectionSettings(optical_mode="IMAGING"), _mode=ParseMode.STRICT
            ),
            "Projection request has no parameters set",
        ),
        (
            "VacuumControlRequest",
            lambda: VacuumControlRequest(target=VacuumSettings(), _mode=ParseMode.STRICT),
            lambda: VacuumControlRequest(
                target=VacuumSettings(turbo_pump_state="ON"), _mode=ParseMode.STRICT
            ),
            "Request must specify at least one state change",
        ),
    ],
)

def test_control_request_empty_intent_and_valid(name, req_factory_empty, req_factory_valid, expected_msg):
    """Consolidate empty-intent guards for control requests."""

    req_empty = req_factory_empty()
    with pytest.raises(ValueError) as exc:
        req_empty.validate()
    assert expected_msg in str(exc.value), f"{name} did not raise expected message"

    req_valid = req_factory_valid()
    assert req_valid.validate() is True


def test_aperture_control_scenarios():
    """Verify ApertureControlRequest validation logic (Relative moves require position)."""
    req = ApertureControlRequest(aperture_id="obj", relative=True, target=Aperture(), _mode=ParseMode.LENIENT)
    assert req.validate() is False
    req.target.position = Point(x=10)
    assert req.validate() is True


def test_aperture_request_id_mismatch():
    """Targets the ID sync/validation logic in ApertureControlRequest."""
    req = ApertureControlRequest(
        aperture_id="objective",
        target=Aperture(aperture_id="condenser"),
        _mode=ParseMode.STRICT
    )

    with pytest.raises(ValueError):
        req.validate()

    req._mode = ParseMode.LENIENT
    req.validate()

    assert "ApertureControlRequest.id_mismatch" in str(req.extra.notes)


def test_aperture_request_relative_logic_failure():
    """Relative aperture move requires a 'position' field to be set."""
    req = ApertureControlRequest(
        aperture_id="obj",
        relative=True,
        target=Aperture(size_index=1),
        _mode=ParseMode.STRICT
    )
    with pytest.raises(ValueError) as exc:
        req.validate()
    assert "Relative mode requires a position" in str(exc.value)


def test_aperture_request_no_op_logic():
    """Verify logic that detects "Empty Payload" vs "Actionable Extras"."""

    req1 = ApertureControlRequest(aperture_id="obj", _mode=ParseMode.STRICT)
    with pytest.raises(ValueError):
        req1.validate()

    req2 = ApertureControlRequest(
        aperture_id="obj",
        target=Aperture(extra={"vendor": {"cmd": "special_align"}}),
        _mode=ParseMode.STRICT
    )
    assert req2.validate()


def test_aperture_validate_invalid_position_branch_via_monkeypatch(monkeypatch):
    """Aperture.validate can heal invalid position to None (forced via monkeypatch)."""
    a = Aperture(position=Point(x=0.0, y=0.0), _mode=ParseMode.LENIENT)

    monkeypatch.setattr(a.position, "validate", lambda **_: False)
    ok = a.validate(mode=ParseMode.LENIENT)
    assert ok is True
    assert a.position is None
    assert any(k.startswith("Aperture.position") for k in a.extra.notes.keys())


def test_system_settings_structure():
    """Verify parsing of the full SystemSettings hierarchy from a dictionary."""
    sys_dict = {
        "stage": {"max_step_nm": 500},
        "beam": {"voltage_limits_kv": [100, 300]},
        "info": {"ip_address": "127.0.0.1"}
    }
    settings = SystemSettings.from_dict(sys_dict, mode=ParseMode.STRICT)
    assert settings.stage_system.max_step_distance.magnitude == 500.0


def test_system_settings_round_trip():
    """Verify full round-trip serialization of SystemSettings."""
    sys = SystemSettings(stage_system=StageSystemSettings(max_step_distance=Q_(10, 'um')),
                         info=SystemInfo(name="TestScope"))
    payload = sys.to_dict()
    reconstructed = SystemSettings.from_dict(payload)
    assert reconstructed.stage_system.max_step_distance.magnitude == pytest.approx(10000.0)


def test_microscope_settings_root():
    """Verify the root MicroscopeSettings object parsing and structure."""
    ms = MicroscopeSettings(image=ImageOutputSettings(file_format="png"), _mode=ParseMode.STRICT)
    assert ms.validate()
    assert ms.to_dict()["image"]["file_format"] == "png"


def test_mixed_mode_hierarchy():
    """Verify that strictness settings propagate or cause conflicts correctly in the hierarchy."""
    root = SystemSettings(_mode=ParseMode.STRICT)
    root.stage_system = StageSystemSettings(max_step_distance=Q_(-1, 'nm'), _mode=ParseMode.LENIENT)

    with pytest.raises(ValueError):
        root.validate()


def test_system_info_ip_validation():
    """Targets the specific 'ipaddress' validation try/except block in SystemInfo."""

    sys = SystemInfo(ip_address="192.168.1.1", _mode=ParseMode.STRICT)
    assert sys.validate()

    sys_bad = SystemInfo(ip_address="999.999.999.999", _mode=ParseMode.STRICT)
    with pytest.raises(ValueError) as exc:
        sys_bad.validate()
    assert "Invalid IP" in str(exc.value)

    sys_heal = SystemInfo(ip_address="not.an.ip", _mode=ParseMode.LENIENT)
    sys_heal.validate()
    assert sys_heal.ip_address == "Unknown"
    assert "SystemInfo.ip_address" in str(sys_heal.extra.raw)


def test_microscope_state_validate_keeps_only_existing_active_detector_ids():
    """MicroscopeState.validate filters active_detector_ids and appends valid ones."""
    ms = MicroscopeState(
        active_detector_ids=["DET-1", "DET-2"],
        detectors={"DET-1": DetectorSettings(detector_id="DET-1", _mode=ParseMode.LENIENT)},
        _mode=ParseMode.LENIENT,
    )
    ok = ms.validate(mode=ParseMode.LENIENT)

    assert ok is False
    assert ms.active_detector_ids == ["DET-1"]
    assert any("active detector" in n["error"].lower() for notes in ms.extra.notes.values() for n in notes)


def test_microscope_image_encode_description_success_path_and_round_trip():
    """MicroscopeImage._encode_description returns a JSON string for valid metadata."""
    md = MicroscopeImageMetadata(version="1", created_at="2026-01-01T00:00:00Z", _mode=ParseMode.LENIENT)
    s = MicroscopeImage._encode_description(md)
    assert isinstance(s, str) and s.strip()
    d = json.loads(s)
    assert isinstance(d, dict)
    assert d.get("version") == "1"


def test_microscope_image_save_and_load_sidecar_json(tmp_path):
    """MicroscopeImage.save writes sidecar JSON and load reads it for non-TIFF formats."""
    data = (np.arange(16, dtype=np.uint8).reshape(4, 4))
    md = MicroscopeImageMetadata(version="1", created_at="2026-01-01T00:00:00Z", _mode=ParseMode.LENIENT)
    img = MicroscopeImage(data=data, metadata=md)
    out_path = img.save(tmp_path / "x.png")
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    assert sidecar.exists()
    loaded = MicroscopeImage.load(out_path)
    assert loaded.metadata is not None
    assert loaded.metadata.version == "1"


def test_microscope_image_load_non_uint_dtype_numeric_and_non_numeric_cast_branches(monkeypatch, tmp_path):
    """MicroscopeImage.load casts numeric and non-numeric arrays to uint8 (forced)."""
    import PIL.Image

    class DummyImg:
        mode = "L"

        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def convert(self, _):
            return self

        def __array__(self, dtype=None, copy=None):
            return self._arr

    numeric = np.array([[300.0, -5.0]], dtype=np.float32)
    monkeypatch.setattr(PIL.Image, "open", lambda *_a, **_k: DummyImg(numeric))
    p = tmp_path / "num.png"
    p.write_bytes(b"x")
    out = MicroscopeImage.load(p)
    assert out.data.dtype == np.uint8
    assert out.data.max() <= 255 and out.data.min() >= 0

    nonnum = np.array([["a", "b"]], dtype=object)
    monkeypatch.setattr(PIL.Image, "open", lambda *_a, **_k: DummyImg(nonnum))
    p2 = tmp_path / "non.png"
    p2.write_bytes(b"x")

    with pytest.raises(ValueError):
        _ = MicroscopeImage.load(p2)


def test_metadata_defaults():
    """Verify ImageMetadata defaults (timestamps, versions) and validation logic."""
    meta = MicroscopeImageMetadata(_mode=ParseMode.LENIENT)
    assert meta.version is not None
    meta.magnification = -500
    meta.validate()
    assert meta.magnification is None


def test_metadata_json_safety_with_numpy():
    """Verify that metadata containing Numpy types is correctly serialized."""
    meta = MicroscopeImageMetadata(
        extra={"analysis": {"score": np.float32(0.95), "counts": np.array([1, 2, 3], dtype=np.uint8)}})
    loaded = json.loads(json.dumps(meta.to_dict()))
    analysis_data = loaded["extra"]["unknown"]["analysis"]
    assert analysis_data["score"] == pytest.approx(0.95)
    assert analysis_data["counts"] == [1, 2, 3]


def test_image_io_formats(tmp_path):
    """Verify loading and saving of standard formats (16-bit TIFF, 8-bit JPG)."""

    data16 = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
    img = MicroscopeImage(data16)
    loaded = MicroscopeImage.load(img.save(tmp_path / "test16.tif"))
    assert loaded.data.dtype == np.uint16

    data8 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img8 = MicroscopeImage(data8)
    loaded_jpg = MicroscopeImage.load(img8.save(tmp_path / "test8.jpg"))
    assert loaded_jpg.data.dtype == np.uint8


def test_image_io_bmp_branch(tmp_path):
    """Verify the specific code path for saving BMP images."""
    data = np.zeros((10, 10), dtype=np.uint8)
    img = MicroscopeImage(data)

    path = img.save(tmp_path / "test.bmp", file_format="bmp")
    assert path.suffix == ".bmp"
    assert path.exists()


def test_image_load_int32_coercion(tmp_path):
    """Verify that 32-bit integer TIFFs are safely coerced to uint16 if the values fit."""
    import tifffile as tff
    data32 = np.array([[100, 200], [300, 400]], dtype=np.int32)
    path = tmp_path / "test32.tif"
    tff.imwrite(str(path), data32)
    img = MicroscopeImage.load(path)
    assert img.data.dtype == np.uint16
    assert img.data[0, 0] == 100


def test_image_load_3d_squeeze(tmp_path):
    """Verify that 3D image stacks (1, H, W) are automatically squeezed to 2D on load."""
    import tifffile as tff
    data3d = np.zeros((1, 50, 50), dtype=np.uint16)
    path = tmp_path / "stack.tif"
    tff.imwrite(str(path), data3d)
    img = MicroscopeImage.load(path)
    assert img.data.ndim == 2


def test_image_init_squeezing():
    """Verify that the constructor __init__ also squeezes 3D arrays."""

    d1 = np.zeros((1, 10, 10), dtype=np.uint8)
    img1 = MicroscopeImage(d1)
    assert img1.data.shape == (10, 10)

    d2 = np.zeros((10, 10, 1), dtype=np.uint8)
    img2 = MicroscopeImage(d2)
    assert img2.data.shape == (10, 10)


def test_image_description_parsing_failure(tmp_path):
    """Verify that invalid JSON in the image description tag is ignored safely."""
    import tifffile as tff
    data = np.zeros((10, 10), dtype=np.uint8)
    path = tmp_path / "bad_meta.tif"
    tff.imwrite(str(path), data, description="{this is not valid json}")
    img = MicroscopeImage.load(path)
    assert img.metadata is None


def test_image_description_encode_decode_helpers():
    """Exercise MicroscopeImage._decode_description and _encode_description fallbacks."""

    assert MicroscopeImage._decode_description(b'{"a": 1}') == {"a": 1}

    assert MicroscopeImage._decode_description(b"\xff\xfe\xff") is None

    assert MicroscopeImage._decode_description(123) is None

    class BadMD:
        created_at = "2020-01-01T00:00:00Z"
        version = "x"

        def to_dict(self):
            raise RuntimeError("boom")

    payload = MicroscopeImage._encode_description(BadMD())
    decoded = json.loads(payload)
    assert decoded.get("created_at") == "2020-01-01T00:00:00Z"
    assert decoded.get("version") == "x"


def test_image_path_handling():
    """Verify that load/save methods handle both string paths and Path objects."""
    data = np.zeros((10, 10), dtype=np.uint8)
    img = MicroscopeImage(data)
    path_obj = Path("test_img_path.tif")
    try:
        img.save(path_obj)
        assert path_obj.exists()
        img2 = MicroscopeImage.load(str(path_obj))
        assert img2.data.shape == (10, 10)
    finally:
        if path_obj.exists(): path_obj.unlink()
        if Path("test_img_path.json").exists(): Path("test_img_path.json").unlink()


def test_image_unsupported_format_save():
    """Verify that saving with an unsupported format raises a ValueError."""
    img = MicroscopeImage(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        img.save("test.txt", file_format="txt")


def test_image_preview_logic_edge_cases():
    """Test the specific branches of _to_uint8_preview logic."""

    empty = np.array([])
    assert MicroscopeImage._to_uint8_preview(empty).size == 0

    flat = np.full((10, 10), 100, dtype=np.float32)
    res_flat = MicroscopeImage._to_uint8_preview(flat)
    assert np.all(res_flat == 100)
    assert res_flat.dtype == np.uint8

    messy = np.array([[np.inf, np.nan], [10, 20]], dtype=np.float32)
    res_messy = MicroscopeImage._to_uint8_preview(messy)
    assert res_messy.shape == (2, 2)
    assert res_messy.dtype == np.uint8

    all_inf = np.full((5, 5), np.inf)
    res_inf = MicroscopeImage._to_uint8_preview(all_inf)
    assert np.all(res_inf == 0)


def test_image_save_auto_convert_float_to_jpg(tmp_path):
    """Save a uint16 array as JPEG to trigger auto-contrast/preview conversion."""

    data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    img = MicroscopeImage(data)

    path = img.save(tmp_path / "int16_to.jpg")

    loaded = MicroscopeImage.load(path)
    assert loaded.data.dtype == np.uint8
    assert loaded.data.max() > 0


def test_image_save_clip_int_to_uint8(tmp_path):
    """Verify that saving as PNG preserves 16-bit depth (does NOT clip), or forces clip."""
    data = np.array([[100, 500]], dtype=np.uint16)
    img = MicroscopeImage(data)

    img.data = img.data.astype(np.int32)

    path = img.save(tmp_path / "clipped.png", file_format="png")

    loaded = MicroscopeImage.load(path)

    assert loaded.data.dtype == np.uint8
    assert loaded.data[0, 1] == 255


def test_image_init_invalid_shape():
    """Reject non-2D image arrays during initialization."""

    bad_data = np.zeros((2, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as exc:
        MicroscopeImage(bad_data)
    assert "Invalid data format" in str(exc.value)


def test_image_preview_flat_variance():
    """_to_uint8_preview edge cases (all zeros, etc)."""

    zeros = np.zeros((10, 10), dtype=np.float32)
    preview = MicroscopeImage._to_uint8_preview(zeros)
    assert np.all(preview == 0)

    nans = np.full((10, 10), np.nan, dtype=np.float32)
    preview2 = MicroscopeImage._to_uint8_preview(nans)
    assert np.all(preview2 == 0)


def test_tiff_load_edge_cases(tmp_path):
    """Specific TIFF loading branches for int16 and uint32 handling."""
    import tifffile as tff

    path_int16 = tmp_path / "valid_int16.tif"
    data_int16 = np.array([[0, 100], [200, 32000]], dtype=np.int16)
    tff.imwrite(str(path_int16), data_int16)
    img = MicroscopeImage.load(path_int16)
    assert img.data.dtype == np.uint16

    path_neg = tmp_path / "neg_int16.tif"
    data_neg = np.array([[-5, 100]], dtype=np.int16)
    tff.imwrite(str(path_neg), data_neg)
    with pytest.raises(ValueError):
        MicroscopeImage.load(path_neg)

    path_u32 = tmp_path / "valid_u32.tif"
    data_u32 = np.array([[0, 65000]], dtype=np.uint32)
    tff.imwrite(str(path_u32), data_u32)
    img3 = MicroscopeImage.load(path_u32)
    assert img3.data.dtype == np.uint16

    path_u32_big = tmp_path / "big_u32.tif"
    data_u32_big = np.array([[0, 70000]], dtype=np.uint32)
    tff.imwrite(str(path_u32_big), data_u32_big)
    with pytest.raises(ValueError):
        MicroscopeImage.load(path_u32_big)


def test_parse_mode_strict_vs_lenient():
    """Verify STRICT mode raises errors immediately."""
    bad_payload = {"voltage": "invalid_string"}

    with pytest.raises((ValueError, pint.errors.UndefinedUnitError)):
        BeamSettings.from_dict(bad_payload, mode=ParseMode.STRICT)

    obj = BeamSettings.from_dict(bad_payload, mode=ParseMode.LENIENT)
    assert obj.voltage is None


def test_misc_wrappers_to_dict_from_dict_and_sync_lines():
    """Hit various thin wrappers and __post_init__ sync lines."""

    ios = ImageOutputSettings.from_dict({"file_format": "png"}, mode=ParseMode.LENIENT)
    assert isinstance(ios, ImageOutputSettings)

    ms = MicroscopeSettings.from_dict({"protocol": {"name": "demo"}}, mode=ParseMode.LENIENT)
    assert isinstance(ms, MicroscopeSettings)

    ds = DetectorSettings.from_dict({"exposure": "5 ms"}, mode=ParseMode.LENIENT)
    assert isinstance(ds, DetectorSettings)

    caps = DetectorCapabilities(can_gain=True, _mode=ParseMode.LENIENT)
    assert isinstance(caps.to_dict(), dict)

    smr = StageMoveRequest(target=StagePosition(x=Q_(1, Units.NM), _mode=ParseMode.LENIENT), _mode=ParseMode.LENIENT)
    d = smr.to_dict()
    smr2 = StageMoveRequest.from_dict(d, mode=ParseMode.LENIENT)
    assert isinstance(smr2, StageMoveRequest)

    scr = StageControlRequest(action="STOP", axes=["x"], _mode=ParseMode.LENIENT)
    scr2 = StageControlRequest.from_dict(scr.to_dict(), mode=ParseMode.LENIENT)
    assert isinstance(scr2, StageControlRequest)

    dcr = DetectorControlRequest(detector_id="DET", _mode=ParseMode.LENIENT)
    dcr2 = DetectorControlRequest.from_dict(dcr.to_dict(), mode=ParseMode.LENIENT)
    assert isinstance(dcr2, DetectorControlRequest)

    bcr = BeamControlRequest(target=BeamSettings(spot_size=1, _mode=ParseMode.LENIENT), _mode=ParseMode.LENIENT)
    bcr2 = BeamControlRequest.from_dict(bcr.to_dict(), mode=ParseMode.LENIENT)
    assert isinstance(bcr2, BeamControlRequest)

    pcr = ProjectionControlRequest(
        target=ProjectionSettings(optical_mode="DIFFRACTION", camera_length=None, _mode=ParseMode.LENIENT),
        _mode=ParseMode.LENIENT,
    )
    ok = pcr.validate(mode=ParseMode.LENIENT)
    assert ok is False

    _ = ProjectionControlRequest.from_dict(pcr.to_dict(), mode=ParseMode.LENIENT)

    sc = ScanControlRequest(action="STOP", _mode=ParseMode.LENIENT)
    _ = ScanControlRequest.from_dict(sc.to_dict(), mode=ParseMode.LENIENT)

    ss = ScanSettings(
        width_px=2,
        height_px=3,
        pixel_dwell_time=Q_(1, Units.US),
        scan_mode="Raster",
        _mode=ParseMode.LENIENT,
    )
    assert ss.estimated_duration is not None
    ss2 = ScanSettings(
        width_px=2,
        height_px=3,
        pixel_dwell_time=Q_(1, Units.US),
        flyback_time=Q_(2, Units.US),
        scan_mode="Raster",
        _mode=ParseMode.LENIENT,
    )
    assert ss2.estimated_duration is not None

    vc = VacuumControlRequest(target=VacuumSettings(turbo_pump_state="ON", _mode=ParseMode.LENIENT), _mode=ParseMode.LENIENT)
    _ = VacuumControlRequest.from_dict(vc.to_dict(), mode=ParseMode.LENIENT)

    vs = VacuumSettings.from_dict({"column_valve": "OPEN", "column_pressure_pa": 1}, mode=ParseMode.LENIENT)
    assert isinstance(vs.to_dict(), dict)

    st = MicroscopeState(
        active_detector_ids=["DET"],
        detectors={
            "DET": DetectorSettings(
                detector_id="DET",
                exposure=Q_(1, Units.MS),
                _mode=ParseMode.LENIENT,
            )
        },
        _mode=ParseMode.LENIENT,
    )
    assert isinstance(st.to_dict(), dict)

    st = MicroscopeState(
        detectors={},
        stage_position=StagePosition(_mode=ParseMode.LENIENT),
        beam=BeamSettings(_mode=ParseMode.LENIENT),
        projection=ProjectionSettings(_mode=ParseMode.LENIENT),
        scan=ScanSettings(_mode=ParseMode.LENIENT),
        vacuum=VacuumSettings(_mode=ParseMode.LENIENT),
        _mode=ParseMode.LENIENT,
    )
    assert isinstance(st.to_dict(), dict)

    acr = ApertureControlRequest(target=Aperture(aperture_id="A1", _mode=ParseMode.LENIENT), _mode=ParseMode.LENIENT)
    assert acr.aperture_id == "A1"
    _ = ApertureControlRequest.from_dict(acr.to_dict(), mode=ParseMode.LENIENT)

    ar = AcquisitionRequest(
        detector=DetectorSettings(detector_id="D1", exposure=Q_(1, Units.MS), _mode=ParseMode.LENIENT),
        _mode=ParseMode.LENIENT,
    )
    assert ar.detector_id == "D1"
    _ = AcquisitionRequest.from_dict(ar.to_dict(), mode=ParseMode.LENIENT)

    st = MicroscopeState(active_detector_ids=["D1"], detectors={"D1": DetectorSettings(detector_id="D1", exposure=Q_(1, Units.MS), _mode=ParseMode.LENIENT)}, _mode=ParseMode.LENIENT)
    assert isinstance(st.to_dict(), dict)
