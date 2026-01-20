"""JEOL microscope driver tests (PyJEM mocked).

This suite is intentionally contract-focused:
- converts and validates units correctly
- routes commands to the right PyJEM APIs for TEM/STEM and imaging/diffraction modes
- fails loudly on unsafe / disconnected / unsupported operations
- preserves vendor-specific extras (e.g., alpha_index, defocus DAC)

The tests use a single connected `jeol_scope` fixture with mocked TEM3 submodules and a
mocked detector module, so each test stays fast and deterministic.
"""

import pytest
import sys
import numpy as np
import itertools
import re
from unittest.mock import MagicMock, patch, call
from types import ModuleType, SimpleNamespace
from typing import Dict, Any, List

from supertem.microscope import MicroscopeSettings
from supertem.structures.base import (
    SystemSettings,
    ParseMode,
    StageMoveRequest,
    StageControlRequest,
    BeamControlRequest,
    ProjectionControlRequest,
    VacuumControlRequest,
    ApertureControlRequest,
    ScanControlRequest,
    StagePosition,
    BeamSettings,
    ProjectionSettings,
    AcquisitionRequest,
    DetectorSettings,
    DetectorControlRequest,
    Aperture,
    VacuumSettings,
    Q_,
    Units,
    Extras,
    Point,
    ROI
)

from supertem.microscopes.jeol_microscope import JeolMicroscope
import supertem.microscopes.jeol_microscope as jeol_module


# ============================================================================
# FIXTURES & MOCKS
# ============================================================================

@pytest.fixture
def mock_pyjem():
    """
    Patches PyJEM imports. Returns mocks for TEM3 and detector.
    """
    with patch("supertem.microscopes.jeol_microscope.TEM3") as mock_tem3,\
            patch("supertem.microscopes.jeol_microscope.detector") as mock_det:

        mock_tem3.Stage3.return_value = MagicMock()
        mock_tem3.EOS3.return_value = MagicMock()
        mock_tem3.HT3.return_value = MagicMock()
        mock_tem3.Lens3.return_value = MagicMock()
        mock_tem3.Def3.return_value = MagicMock()
        mock_tem3.Apt3.return_value = MagicMock()
        mock_tem3.Scan3.return_value = MagicMock()
        mock_tem3.VACUUM3.return_value = MagicMock()
        mock_tem3.GUN3.return_value = MagicMock()
        mock_tem3.FEG3.return_value = MagicMock()
        mock_tem3.Detector3.return_value = MagicMock()


        mock_det.get_attached_detector.return_value = ["Camera1"]
        mock_det.Detector.return_value = MagicMock()


        mock_det.function = MagicMock()
        mock_det.function.get_attached_detector.return_value = ["Camera1"]

        yield mock_tem3, mock_det

@pytest.fixture
def jeol_scope(mock_pyjem):
    """
    Returns an instantiated and connected JeolMicroscope.
    """
    mock_tem3, mock_det = mock_pyjem


    settings = MicroscopeSettings(system=SystemSettings())

    settings.defocus_scale = 0.5

    scope = JeolMicroscope(settings)
    scope.connect("localhost")


    if getattr(scope, "system_settings", None) and getattr(scope.system_settings, "detector_system", None):
        scope.system_settings.detector_system.is_supported = MagicMock(return_value=True)

    return scope

@pytest.fixture
def mock_eos_list(monkeypatch):
    """Helper to mock JEOL EOS table lookups via `jeol_module.get_list`.

    Usage:
        mock_eos_list(key, name, data)

    It returns `data` only when both key and name match; otherwise returns an empty list.
    """

    def _apply(key_match, list_name_match, data):
        def _fake_get_list(key, name):
            if key == key_match and name == list_name_match:
                return data
            return []

        monkeypatch.setattr(jeol_module, "get_list", _fake_get_list, raising=True)

    return _apply


# ============================================================================
# CONNECTION & LIFECYCLE
# ============================================================================

def test_connect_with_valid_host_sets_connected_state(mock_pyjem):
    """Verify that `connect` with valid host sets connected state."""
    mock_tem3, _ = mock_pyjem
    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    scope.connect("1.2.3.4")

    mock_tem3.connect.assert_called_once()
    assert scope.is_connected() is True

def test_disconnect_when_connected_sets_disconnected_state(jeol_scope):
    """Verify that `disconnect` when connected sets disconnected state."""
    jeol_scope.disconnect()
    assert jeol_scope.is_connected() is False

def test_get_instrument_info_when_connected_returns_jeol_metadata(jeol_scope):
    """Verify that `get_instrument_info` when connected returns jeol metadata."""
    info = jeol_scope.get_instrument_info()
    assert info.manufacturer == "JEOL"
    assert "PyJEM" in info.software_version

def test_init_when_pyjem_missing_does_not_crash():
    """Verify JeolMicroscope initialization when PyJEM missing does not crash."""
    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    assert scope is not None

def test_init_when_tem3_is_none_attempts_lazy_import(monkeypatch):
    """Verify JeolMicroscope initialization when TEM3 is None attempts lazy import."""
    monkeypatch.setattr(jeol_module, 'TEM3', None, raising=False)
    monkeypatch.setattr(jeol_module, 'detector', None, raising=False)

    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    assert scope is not None

def test_connect_when_tem3_unavailable_raises_runtime_error(monkeypatch):
    """Verify that `connect` when TEM3 unavailable raises runtime error."""
    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))

    monkeypatch.setattr(jeol_module, "TEM3", None, raising=False)
    with pytest.raises(RuntimeError):
        scope.connect("localhost")

def test_require_connected_guard_raises_before_connect_called(jeol_scope):
    """Verify that `_require_connected` raises before connect called."""
    jeol_scope._connected = False
    with pytest.raises(RuntimeError, match=r"not connected"):
        jeol_scope._require_connected()

def test_init_with_pyjem_in_sys_modules_loads_successfully(monkeypatch):
    """Verify JeolMicroscope initialization with PyJEM in sys modules loads successfully."""
    monkeypatch.setattr(jeol_module, 'TEM3', None, raising=False)
    monkeypatch.setattr(jeol_module, 'detector', None, raising=False)

    pyjem = ModuleType('PyJEM')
    pyjem.TEM3 = SimpleNamespace()
    pyjem.detector = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'PyJEM', pyjem)

    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    assert scope is not None
    assert jeol_module.TEM3 is not None
    assert jeol_module.detector is not None

def test_init_with_pyjem_offline_in_sys_modules_loads_successfully(monkeypatch):
    """Verify JeolMicroscope initialization with PyJEM offline in sys modules loads successfully."""
    monkeypatch.setattr(jeol_module, 'TEM3', None, raising=False)
    monkeypatch.setattr(jeol_module, 'detector', None, raising=False)

    pyjem = ModuleType('PyJEM')
    offline = ModuleType('PyJEM.offline')
    offline.TEM3 = SimpleNamespace()
    offline.detector = SimpleNamespace()

    monkeypatch.setitem(sys.modules, 'PyJEM', pyjem)
    monkeypatch.setitem(sys.modules, 'PyJEM.offline', offline)

    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    assert scope is not None
    assert jeol_module.TEM3 is not None
    assert jeol_module.detector is not None

def test_connect_when_submodule_init_fails_logs_and_raises(mock_pyjem):
    """Verify that `connect` when submodule init fails logs and raises."""
    mock_tem3, _ = mock_pyjem
    mock_tem3.Stage3.side_effect = RuntimeError('boom')

    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    with pytest.raises(RuntimeError):
        scope.connect('localhost')

def test_lazy_loading_import_failures_do_not_crash(monkeypatch):
    """Verify that `lazy_loading_import` failures do not crash."""
    monkeypatch.setattr(jeol_module, "TEM3", None, raising=False)
    monkeypatch.setattr(jeol_module, "detector", None, raising=False)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("PyJEM"):
            raise ImportError("no pyjem")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    scope = JeolMicroscope(MicroscopeSettings(system=SystemSettings()))
    assert scope is not None
    assert getattr(jeol_module, "TEM3") is None
    assert getattr(jeol_module, "detector") is None


# =============================================================================
# STAGE CONTROL
# =============================================================================

def test_get_stage_position_when_connected_returns_converted_coordinates(jeol_scope):
    """Verify that `get_stage_position` when connected returns converted coordinates."""
    jeol_scope.stage.GetPos.return_value = [1000.0, 2000.0, 500.0, 1.5, -0.5]
    pos = jeol_scope.get_stage_position()

    assert pos.x.magnitude == pytest.approx(1000.0)
    assert pos.y.magnitude == pytest.approx(2000.0)
    assert pos.tilt_x.magnitude == pytest.approx(1.5)

def test_get_stage_position_via_adapter_returns_correct_object(monkeypatch, jeol_scope):
    """Verify that `get_stage_position` via adapter returns correct object."""
    jeol_scope.stage.GetPos.return_value = [1, 2, 3, 4, 5]
    sp = StagePosition(x=Q_(1, Units.NM), y=Q_(2, Units.NM), z=Q_(3, Units.NM),
                       tilt_x=Q_(4, Units.DEG), tilt_y=Q_(5, Units.DEG), _mode=ParseMode.LENIENT)

    def _from(raw):
        assert raw == [1, 2, 3, 4, 5]
        return sp

    monkeypatch.setattr(jeol_module.jeol_adapter, "from_jeol_stage_position", _from)
    out = jeol_scope.get_stage_position()
    assert out is sp

def test_get_stage_position_returns_none_when_disconnected(jeol_scope):
    """Verify that `get_stage_position` returns None when disconnected."""
    jeol_scope.stage = None
    assert jeol_scope.get_stage_position() is None

def test_move_stage_absolute_motor_sets_all_axes(jeol_scope):
    """Verify that `move_stage_absolute_motor` sets all axes."""
    tgt = StagePosition(
        x=Q_(1, 'um'), y=Q_(2, 'um'), z=Q_(3, 'um'),
        tilt_x=Q_(1, 'deg'), tilt_y=Q_(2, 'deg'),
    )
    jeol_scope.stage.GetStatus.return_value = [0, 0, 0, 0, 0]
    jeol_scope.stage.GetPos.return_value = [1000.0, 2000.0, 3000.0, 1.0, 2.0]

    jeol_scope.move_stage_absolute(tgt, drive_type='motor', tolerance_nm=1e6, tolerance_deg=10)

    jeol_scope.stage.SetX.assert_called()
    jeol_scope.stage.SetY.assert_called()
    jeol_scope.stage.SetZ.assert_called()
    jeol_scope.stage.SetTiltXAngle.assert_called()
    jeol_scope.stage.SetTiltYAngle.assert_called()

def test_move_stage_absolute_motor_calls_sel_drv_mode_zero(monkeypatch, jeol_scope):
    """Verify that `move_stage_absolute_motor` calls sel drv mode zero."""
    st = SimpleNamespace(SelDrvMode=MagicMock(), SetX=MagicMock())
    jeol_scope.stage = st

    tgt = StagePosition(x=Q_(5, Units.NM), _mode="lenient")
    jeol_scope.move_stage_absolute(tgt, drive_type="motor", wait=False)

    st.SelDrvMode.assert_called_once_with(0)
    st.SetX.assert_called_once_with(5.0)

def test_move_stage_absolute_piezo_sets_xy_and_restores_drive_mode(monkeypatch, jeol_scope):
    """Verify that `move_stage_absolute_piezo` sets xy and restores drive mode."""
    st = SimpleNamespace(SelDrvMode=MagicMock(), SetX=MagicMock(), SetY=MagicMock())
    jeol_scope.stage = st

    tgt = StagePosition(x=Q_(1, Units.NM), y=Q_(2, Units.NM), _mode="lenient")
    jeol_scope.move_stage_absolute(tgt, drive_type="piezo", wait=False)


    st.SelDrvMode.assert_has_calls([call(1), call(0)])
    st.SetX.assert_called_once_with(1.0)
    st.SetY.assert_called_once_with(2.0)

def test_move_stage_absolute_piezo_restores_drive_mode_even_on_error(jeol_scope):
    """Verify that `move_stage_absolute_piezo_restores_drive_mode_even` on error."""
    jeol_scope.stage.SelDrvMode = MagicMock()
    jeol_scope.stage.SetX = MagicMock(side_effect=RuntimeError("boom"))

    tgt = StagePosition(x=Q_(100, Units.NM))

    with pytest.raises(RuntimeError, match="boom"):
        jeol_scope.move_stage_absolute(tgt, drive_type="piezo", wait=False)

    assert jeol_scope.stage.SelDrvMode.call_args_list[0] == call(1)
    assert jeol_scope.stage.SelDrvMode.call_args_list[-1] == call(0)

def test_move_stage_absolute_piezo_raises_if_sel_drv_mode_missing(monkeypatch, jeol_scope):
    """Verify that `move_stage_absolute_piezo` raises if sel drv mode missing."""
    jeol_scope.stage = SimpleNamespace(SetX=MagicMock(), SetY=MagicMock())
    tgt = StagePosition(x=Q_(100, Units.NM), _mode="lenient")

    with pytest.raises(RuntimeError, match=r"Piezo control not supported"):
        jeol_scope.move_stage_absolute(tgt, drive_type="piezo", wait=False)

def test_move_stage_absolute_piezo_warns_on_unsupported_axes(caplog, jeol_scope):
    """Verify that `move_stage_absolute_piezo_warns` on unsupported axes."""
    tgt = StagePosition(x=Q_(50, 'nm'), z=Q_(10, 'nm'))
    caplog.set_level('WARNING')
    jeol_scope.move_stage_absolute(tgt, drive_type='piezo', wait=False)
    assert any('Piezo mode supports X/Y only' in r.message for r in caplog.records)

def test_move_stage_absolute_retries_until_tolerance_met(mock_pyjem):
    """Verify that `move_stage_absolute` retries until tolerance met."""
    scope = JeolMicroscope(MicroscopeSettings())
    scope.connect("localhost")
    scope.stage.SetX = MagicMock()
    scope.stage.GetStatus.return_value = [0, 0, 0, 0, 0]


    scope.stage.GetPos.side_effect = [
        {"x": 900, "y": 0},
        {"x": 950, "y": 0},
        {"x": 1000, "y": 0}
    ]

    with patch("time.sleep", return_value=None):
        target = StagePosition(x=Q_(1000, "nm"), y=Q_(0, "nm"))
        scope.move_stage_absolute(target, tolerance_nm=10.0, max_retries=3)

    assert scope.stage.SetX.call_count >= 2

def test_move_stage_absolute_dispatch_error_is_raised_immediately(jeol_scope):
    """Verify that `move_stage_absolute` dispatch error is raised immediately."""
    jeol_scope.stage.GetStatus.return_value = [0, 0, 0, 0, 0]
    jeol_scope.stage.SetX.side_effect = RuntimeError("setx failed")

    with pytest.raises(RuntimeError, match="setx failed"):
        jeol_scope.move_stage_absolute(StagePosition(x=Q_(10, "um")), wait=False)

def test_move_stage_absolute_retry_correction_error_is_logged_but_continues(caplog, monkeypatch, jeol_scope):
    """Verify that `move_stage_absolute` retry correction error is logged but continues."""
    jeol_scope.stage.SetX = MagicMock(side_effect=[None, RuntimeError('correction fail')])
    jeol_scope.stage.GetStatus.return_value = [0] * 5


    monkeypatch.setattr(jeol_scope, 'get_stage_position', lambda: object())
    target = SimpleNamespace(is_close=lambda *a, **k: False)

    monkeypatch.setattr(jeol_module.jeol_adapter, 'to_jeol_stage_args', lambda _t: {'x': 1.0})
    monkeypatch.setattr(jeol_scope, '_wait_for_stage', lambda *a, **k: None)
    monkeypatch.setattr(jeol_module.time, 'sleep', lambda *_a, **_k: None)

    caplog.set_level('ERROR')
    jeol_scope.move_stage_absolute(target, drive_type='motor', wait=True, max_retries=1)

    assert jeol_scope.stage.SetX.call_count == 2
    assert any('Correction IO Error' in r.message and 'correction fail' in r.message for r in caplog.records)

def test_stop_stage_no_stop_method_is_noop(caplog, jeol_scope):
    """Verify that `stop_stage_no` stop method is noop."""
    jeol_scope.stage = SimpleNamespace()

    caplog.set_level('ERROR')
    jeol_scope.stop_stage()

    assert not any('Stop failed' in r.message for r in caplog.records)

@pytest.mark.parametrize(
    "api_method, hw_attr",
    [
        ("stop_stage", "Stop"),
        ("home_stage", "SetOrg"),
    ],
)
def test_stage_simple_commands_call_hardware(api_method, hw_attr, jeol_scope):
    """Verify that `stage_simple_commands` call hardware."""
    fn = getattr(jeol_scope, api_method)
    fn()
    getattr(jeol_scope.stage, hw_attr).assert_called_once()

@pytest.mark.parametrize("api_method", ["stop_stage", "home_stage"])
def test_stage_simple_commands_raise_when_disconnected(api_method, jeol_scope):
    """Verify that `stage_simple_commands_raise` when disconnected."""
    jeol_scope.stage = None
    fn = getattr(jeol_scope, api_method)
    with pytest.raises(RuntimeError):
        fn()

@pytest.mark.parametrize(
    "api_method, hw_attr",
    [
        ("stop_stage", "Stop"),
        ("home_stage", "SetOrg"),
    ],
)
def test_stage_simple_commands_raise_when_hardware_fails(api_method, hw_attr, jeol_scope):
    """Verify that `stage_simple_commands_raise` when hardware fails."""
    getattr(jeol_scope.stage, hw_attr).side_effect = RuntimeError("boom")
    fn = getattr(jeol_scope, api_method)
    with pytest.raises(RuntimeError):
        fn()

def test_home_stage_logs_warning_if_setorg_missing(caplog, jeol_scope):
    """Verify that `home_stage` logs warning if setorg missing."""
    jeol_scope.stage = SimpleNamespace()

    caplog.set_level('WARNING')
    jeol_scope.home_stage()

    assert any('SetOrg' in r.message for r in caplog.records)

def test_wait_for_stage_returns_immediately_when_idle(jeol_scope):
    """Verify that `wait` for stage returns immediately when idle."""
    jeol_scope.stage.GetStatus.return_value = [0, 0, 0, 0, 0]
    with patch("time.sleep") as mock_sleep:
        jeol_scope._wait_for_stage(timeout=0.5)
    mock_sleep.assert_not_called()

def test_wait_for_stage_times_out_if_always_busy(jeol_scope):
    """Verify that `wait` for stage times out if always busy."""
    jeol_scope.stage.GetStatus.return_value = [1, 0, 0, 0, 0]

    with patch("time.time", side_effect=itertools.count(start=0, step=10)):
        with patch("time.sleep"):
            jeol_scope.stage.GetPos.return_value = [0.0] * 5
            jeol_scope.move_stage_absolute(StagePosition(x=Q_(10, "um")), wait=True)

    assert jeol_scope.stage.GetStatus.called

def test_wait_for_stage_swallows_exception_and_continues(monkeypatch, jeol_scope):
    """Verify that `wait` for stage swallows exception and continues."""
    jeol_scope.stage.GetStatus.side_effect = RuntimeError("comms error")

    t = [0.0]
    monkeypatch.setattr("time.time", lambda: (t.__setitem__(0, t[0] + 0.1) or t[0]))
    monkeypatch.setattr("time.sleep", lambda _: None)

    jeol_scope._wait_for_stage(timeout=0.5)
    assert jeol_scope.stage.GetStatus.called

def test_wait_for_stage_sleeps_once_if_getstatus_missing(monkeypatch, jeol_scope):
    """Verify that `wait` for stage sleeps once if getstatus missing."""
    class NoStatusStage: pass
    jeol_scope.stage = NoStatusStage()

    sleeps = []
    monkeypatch.setattr(jeol_module.time, 'sleep', lambda t: sleeps.append(t))

    jeol_scope._wait_for_stage(timeout=1.0)
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    "setup, api_getter, expected_magnitude, expected_units",
    [
        (lambda s: setattr(s.ht.GetHtValue, "return_value", 200000.0), "get_acceleration_voltage", 200.0, "kV"),
        (lambda s: setattr(s.gun.GetEmissionCurrent, "return_value", 150.0), "get_beam_current", 150000.0, "nA"),
        (lambda s: setattr(s.eos.GetSpotSize, "return_value", 1), "get_spot_size", 1, None),
    ],
    ids=["acceleration_voltage", "beam_current", "spot_size"],
)
# =============================================================================
# BEAM CONTROL (ILLUMINATION)
# =============================================================================

def test_beam_property_getters_convert_units_and_return_values(jeol_scope, setup, api_getter, expected_magnitude, expected_units):
    """Verify that `beam_property_getters_convert_units` and return values."""
    setup(jeol_scope)
    out = getattr(jeol_scope, api_getter)()
    if expected_units is None:
        assert out == expected_magnitude
    else:
        assert float(out.to(expected_units).magnitude) == pytest.approx(float(expected_magnitude))

@pytest.mark.parametrize(
    "api_method, hw_module, hw_method, input_val, expected_hw_arg",
    [
        ("set_acceleration_voltage", "ht", "SetHtValue", Q_(300, "kV"), 300000.0),
        ("set_spot_size", "eos", "SelectSpotSize", 3, 3),
    ],
    ids=["set_acceleration_voltage", "set_spot_size"],
)
def test_beam_property_setters_call_hardware_with_base_units(jeol_scope, api_method, hw_module, hw_method, input_val, expected_hw_arg):
    """Verify that `beam_property_setters_call_hardware` with base units."""
    getattr(jeol_scope, api_method)(input_val)
    module = getattr(jeol_scope, hw_module)
    getattr(module, hw_method).assert_called_with(pytest.approx(expected_hw_arg) if isinstance(expected_hw_arg, float) else expected_hw_arg)

def test_set_beam_current_raises_not_implemented_error(jeol_scope):
    """Verify that `set_beam_current` raises not implemented error."""
    with pytest.raises(NotImplementedError):
        jeol_scope.set_beam_current(Q_(1, "nA"))


@pytest.mark.parametrize(
    "api_getter, hw_method, hw_return, expected",
    [
        ("get_beam_shift", "GetCLA1", [100.1, 200.2], (100.1, 200.2)),
        ("get_condenser_stigmation", "GetCLs", [5.5, -5.5], (5.5, -5.5)),
        ("get_gun_tilt", "GetAngBal", [10.0, 10.0], (10.0, 10.0)),
    ],
    ids=["beam_shift", "condenser_stigmation", "gun_tilt"],
)
def test_beam_alignment_getters_return_xy_tuples(jeol_scope, api_getter, hw_method, hw_return, expected):
    """Verify that `beam_alignment_getters` return xy tuples."""
    getattr(jeol_scope.def_, hw_method).return_value = hw_return
    result = getattr(jeol_scope, api_getter)()
    assert result == pytest.approx(expected)

@pytest.mark.parametrize(
    "api_setter, hw_method, args",
    [
        ("set_beam_shift", "SetCLA1", (50, 60)),
        ("set_condenser_stigmation", "SetCLs", (1, 2)),
        ("set_gun_tilt", "SetAngBal", (3, 4)),
    ],
    ids=["beam_shift", "condenser_stigmation", "gun_tilt"],
)
def test_beam_alignment_setters_call_deflector_hardware(jeol_scope, api_setter, hw_method, args):
    """Verify that `beam_alignment_setters` call deflector hardware."""
    getattr(jeol_scope, api_setter)(*args)
    getattr(jeol_scope.def_, hw_method).assert_called_with(*args)

def test_get_beam_shift_malformed_returns_none_tuple(jeol_scope):
    """Verify that `get_beam_shift_malformed` returns None tuple."""
    jeol_scope.def_.GetCLA1.return_value = ['bad']
    assert jeol_scope.get_beam_shift() == (None, None)

def test_get_condenser_stigmation_returns_tuple_from_mock_list(jeol_scope):
    """Verify that `get_condenser_stigmation` returns tuple from mock list."""
    jeol_scope.def_.GetCLs.return_value = [1, 2]
    assert jeol_scope.get_condenser_stigmation() == (1.0, 2.0)

def test_get_gun_tilt_success_path_returns_tuple_explicit(jeol_scope):
    """Verify that `get_gun_tilt_success_path` returns tuple explicit."""
    jeol_scope.def_.GetAngBal.return_value = [-1, 2]
    assert jeol_scope.get_gun_tilt() == (-1.0, 2.0)

def test_set_beam_blank_calls_hardware(jeol_scope):
    """Verify that `set_beam_blank` calls hardware."""
    jeol_scope.set_beam_blank(True)
    jeol_scope.def_.SetBeamBlank.assert_called_with(1)

    jeol_scope.def_.GetBeamBlank.return_value = 1
    assert jeol_scope.get_beam_blank() is True

def test_get_beam_blank_returns_false_on_exception(jeol_scope):
    """Verify that `get_beam_blank` returns false on exception."""
    jeol_scope.def_ = MagicMock()
    jeol_scope.def_.GetBeamBlank = MagicMock(side_effect=RuntimeError("boom"))
    assert jeol_scope.get_beam_blank() is False


def test_get_alpha_index_returns_int(jeol_scope):
    """Verify that `get_alpha_index` returns int."""
    jeol_scope.eos.GetAlpha.return_value = 3
    assert jeol_scope.get_alpha_index() == 3

@pytest.mark.parametrize(
    'eos, expect_calls',
    [
        (None, 0),
        (SimpleNamespace(GetAlpha=MagicMock(side_effect=RuntimeError('boom'))), 1),
    ],
)
def test_get_alpha_index_returns_none_on_missing_or_exception(eos, expect_calls, jeol_scope):
    """Verify that `get_alpha_index` returns None on missing or exception."""
    jeol_scope.eos = eos
    assert jeol_scope.get_alpha_index() is None
    if expect_calls:
        eos.GetAlpha.assert_called_once()

def test_set_alpha_index_calls_hardware(jeol_scope):
    """Verify that `set_alpha_index` calls hardware."""
    jeol_scope.set_alpha_index(3)
    jeol_scope.eos.SetAlphaSelector.assert_called_once_with(3)


def test_get_beam_settings_includes_vendor_extras(jeol_scope):
    """Verify that `get_beam_settings` includes vendor extras."""
    jeol_scope.eos.GetAlpha.return_value = 3
    jeol_scope.ht.GetHtValue.return_value = 200000.0
    beam = jeol_scope.get_beam_settings()
    assert beam.extra.vendor["JEOL"]["alpha_index"] == 3

def test_get_beam_settings_records_flags_on_conversion_failures(monkeypatch, jeol_scope):
    """Verify that `get_beam_settings_records_flags` on conversion failures."""
    class BadQty:
        def to(self, *_args, **_kwargs):
            raise RuntimeError("bad convert")

    monkeypatch.setattr(jeol_scope, "get_acceleration_voltage", lambda: BadQty())
    monkeypatch.setattr(jeol_scope, "get_beam_current", lambda: BadQty())
    monkeypatch.setattr(jeol_scope, "get_spot_size", lambda: None)
    monkeypatch.setattr(jeol_scope, "get_alpha_index", lambda: None)
    monkeypatch.setattr(jeol_scope, "get_beam_shift", lambda: (None, None))

    bs = jeol_scope.get_beam_settings()
    v = bs.extra.vendor["JEOL"]

    assert v["ht_unavailable"] is True
    assert v["beam_current_unavailable"] is True
    assert v["spot_size_unavailable"] is True
    assert v["alpha_unavailable"] is True
    assert v["beam_shift_unavailable"] is True

def test_get_beam_settings_sets_beam_current_unavailable_flag_when_none(monkeypatch, jeol_scope):
    """Verify that `get_beam_settings` sets beam current unavailable flag when none."""
    monkeypatch.setattr(jeol_scope, "get_acceleration_voltage", lambda: Q_(200, Units.KV))
    monkeypatch.setattr(jeol_scope, "get_beam_current", lambda: None)
    monkeypatch.setattr(jeol_scope, "get_spot_size", lambda: 1)
    monkeypatch.setattr(jeol_scope, "get_alpha_index", lambda: 2)
    monkeypatch.setattr(jeol_scope, "get_beam_shift", lambda: (10.0, 20.0))

    bs = jeol_scope.get_beam_settings()
    assert bs.extra.vendor["JEOL"]["beam_current_unavailable"] is True

def test_get_beam_settings_rounds_beam_shift_into_dac(monkeypatch, jeol_scope):
    """Verify that `get_beam_settings` rounds beam shift into DAC."""
    monkeypatch.setattr(jeol_scope, "get_acceleration_voltage", lambda: Q_(200, Units.KV))
    monkeypatch.setattr(jeol_scope, "get_beam_current", lambda: Q_(1, Units.NA))
    monkeypatch.setattr(jeol_scope, "get_spot_size", lambda: 1)
    monkeypatch.setattr(jeol_scope, "get_alpha_index", lambda: 2)
    monkeypatch.setattr(jeol_scope, "get_beam_shift", lambda: (1.6, 2.4))

    bs = jeol_scope.get_beam_settings()
    assert bs.extra.vendor["JEOL"]["beam_shift_dac"] == (2, 2)
    assert bs.beam_shift.x == 2.0 and bs.beam_shift.y == 2.0

def test_apply_beam_settings_applies_canonical_fields(jeol_scope):
    """Verify that `apply_beam_settings` applies canonical fields."""
    jeol_scope.set_acceleration_voltage = MagicMock()
    jeol_scope.set_beam_current = MagicMock()
    jeol_scope.set_spot_size = MagicMock()

    settings = BeamSettings(voltage=Q_(200, "kV"), beam_current=Q_(1, "nA"), spot_size=4)
    jeol_scope.apply_beam_settings(settings)

    jeol_scope.set_acceleration_voltage.assert_called_once()
    jeol_scope.set_beam_current.assert_called_once()
    jeol_scope.set_spot_size.assert_called_once_with(4)

def test_apply_beam_settings_applies_full_2d_coil_points(jeol_scope):
    """Verify that `apply_beam_settings` applies full 2d coil points."""
    jeol_scope.set_beam_shift = MagicMock()
    jeol_scope.set_condenser_stigmation = MagicMock()
    jeol_scope.set_gun_tilt = MagicMock()

    s = BeamSettings(
        beam_shift=Point(x=10.0, y=20.0),
        condenser_stigmation=Point(x=-1.0, y=2.0),
        gun_tilt=Point(x=3.0, y=4.0),
        _mode="lenient",
    )
    jeol_scope.apply_beam_settings(s)

    jeol_scope.set_beam_shift.assert_called_once_with(10.0, 20.0)
    jeol_scope.set_condenser_stigmation.assert_called_once_with(-1.0, 2.0)
    jeol_scope.set_gun_tilt.assert_called_once_with(3.0, 4.0)

def test_apply_beam_settings_applies_individual_xy_fields(monkeypatch, jeol_scope):
    """Verify that `apply_beam_settings` applies individual xy fields."""
    jeol_scope.set_beam_shift = MagicMock()
    s = BeamSettings(beam_shift=Point(x=1.0, y=2.0), _mode="lenient")
    jeol_scope.apply_beam_settings(s)
    jeol_scope.set_beam_shift.assert_called_once_with(1.0, 2.0)

    jeol_scope.set_condenser_stigmation = MagicMock()
    s2 = BeamSettings(condenser_stigmation=Point(x=3.0, y=4.0), _mode="lenient")
    jeol_scope.apply_beam_settings(s2)
    jeol_scope.set_condenser_stigmation.assert_called_once_with(3.0, 4.0)

    jeol_scope.set_gun_tilt = MagicMock()
    s3 = BeamSettings(gun_tilt=Point(x=5.0, y=6.0), _mode="lenient")
    jeol_scope.apply_beam_settings(s3)
    jeol_scope.set_gun_tilt.assert_called_once_with(5.0, 6.0)

def test_apply_beam_settings_sets_alpha_index(jeol_scope):
    """Verify that `apply_beam_settings` sets alpha index."""
    jeol_scope.apply_beam_settings(BeamSettings(
        extra=Extras(vendor={"JEOL": {"alpha_index": 5}})
    ))
    jeol_scope.eos.SetAlphaSelector.assert_called_with(5)

def test_apply_beam_settings_alpha_index_none_is_noop(jeol_scope):
    """Verify that `apply_beam_settings` alpha index None is noop."""
    jeol_scope.eos.SetAlphaSelector.reset_mock()
    jeol_scope.apply_beam_settings(BeamSettings(extra=Extras(vendor={"JEOL": {"alpha_index": None}})))
    jeol_scope.eos.SetAlphaSelector.assert_not_called()

@pytest.mark.parametrize(
    'value, match',
    [
        ('abc', r'alpha_index must be an int-like'),
        (99, r'out of bounds'),
    ],
)
def test_apply_beam_settings_alpha_index_invalid_values_raise(value, match, jeol_scope):
    """Verify that `apply_beam_settings` alpha index invalid values raise."""
    with pytest.raises(ValueError, match=match):
        jeol_scope.apply_beam_settings(
            BeamSettings(extra=Extras(vendor={"JEOL": {"alpha_index": value}}))
        )

def test_apply_beam_settings_with_convergence_angle_raises_value_error(jeol_scope):
    """Verify that `apply_beam_settings` with convergence angle raises value error."""
    with pytest.raises(ValueError, match="alpha_index"):
        jeol_scope.apply_beam_settings(BeamSettings(
            convergence_angle=Q_(10, "mrad")
        ))

def test_apply_beam_settings_rejects_partial_2d_coil_points(jeol_scope):
    """Verify that `apply_beam_settings` rejects partial 2d coil points."""
    with pytest.raises(ValueError, match=r"beam_shift requires both x and y"):
        jeol_scope.apply_beam_settings(BeamSettings(beam_shift=Point(x=1.0, y=None)))

    with pytest.raises(ValueError, match=r"condenser_stigmation requires both x and y"):
        jeol_scope.apply_beam_settings(BeamSettings(condenser_stigmation=Point(x=None, y=2.0)))

    with pytest.raises(ValueError, match=r"gun_tilt requires both x and y"):
        jeol_scope.apply_beam_settings(BeamSettings(gun_tilt=Point(x=0.0, y=None)))


def test_beam_getters_missing_hardware_and_exceptions(monkeypatch, jeol_scope):
    """Verify that `beam_getters_missing_hardware` and exceptions."""
    jeol_scope.ht = MagicMock()
    jeol_scope.ht.GetHtValue = MagicMock(side_effect=RuntimeError("boom"))
    assert jeol_scope.get_acceleration_voltage() is None

    jeol_scope.gun = MagicMock()
    jeol_scope.gun.GetEmissionCurrent = MagicMock(side_effect=RuntimeError("boom"))
    assert jeol_scope.get_beam_current() is None

    jeol_scope.eos = MagicMock()
    jeol_scope.eos.GetSpotSize = MagicMock(side_effect=RuntimeError("boom"))
    assert jeol_scope.get_spot_size() is None

    jeol_scope.def_ = MagicMock()
    jeol_scope.def_.GetCLA1 = MagicMock(side_effect=RuntimeError("boom"))
    assert jeol_scope.get_beam_shift() == (None, None)

def test_beam_getters_disconnected_branches(jeol_scope):
    """Verify that `beam_getters_disconnected` branches."""
    jeol_scope.gun = None
    assert jeol_scope.get_beam_current() is None
    jeol_scope.ht = None
    assert jeol_scope.get_acceleration_voltage() is None
    jeol_scope.eos = None
    assert jeol_scope.get_spot_size() is None
    jeol_scope.def_ = None
    assert jeol_scope.get_beam_shift() == (None, None)

@pytest.mark.parametrize(
    "getter, hw_method",
    [
        ("get_condenser_stigmation", "GetCLs"),
        ("get_gun_tilt", "GetAngBal"),
        ("get_objective_stigmation", "GetOLs"),
        ("get_diffraction_shift", "GetPLA"),
    ],
)
def test_deflector_xy_getters_return_none_tuple_on_exception(getter, hw_method, jeol_scope):
    """Verify that `deflector_xy_getters_return_none_tuple` on exception."""
    jeol_scope.def_ = MagicMock()
    getattr(jeol_scope.def_, hw_method).side_effect = RuntimeError("boom")
    fn = getattr(jeol_scope, getter)
    assert fn() == (None, None)

@pytest.mark.parametrize(
    "deflector, expected_method",
    [
        (SimpleNamespace(GetIS1=MagicMock(side_effect=RuntimeError("boom"))), "GetIS1"),
        (SimpleNamespace(GetIS=MagicMock(side_effect=RuntimeError("boom"))), "GetIS"),
    ],
)
def test_get_image_shift_returns_none_tuple_on_exception(deflector, expected_method, jeol_scope):
    """Verify that `get_image_shift` returns None tuple on exception."""
    jeol_scope.def_ = deflector

    assert jeol_scope.get_image_shift() == (None, None)
    getattr(deflector, expected_method).assert_called_once()

@pytest.mark.parametrize(
    "method, args",
    [
        ("set_condenser_stigmation", (1, 2)),
        ("set_gun_tilt", (1, 2)),
    ],
)
def test_set_deflector_xy_raises_when_deflector_missing(method, args, jeol_scope):
    """Verify that `set_deflector_xy` raises when deflector missing."""
    jeol_scope.def_ = None
    fn = getattr(jeol_scope, method)
    with pytest.raises(RuntimeError, match=r"Deflector hardware not connected"):
        fn(*args)

@pytest.mark.parametrize(
    "method, setter_attr, args",
    [
        ("set_condenser_stigmation", "SetCLs", (1, 2)),
        ("set_gun_tilt", "SetAngBal", (1, 2)),
    ],
)
def test_set_deflector_xy_raises_when_hardware_fails(method, setter_attr, args, jeol_scope):
    """Verify that `set_deflector_xy` raises when hardware fails."""
    jeol_scope.def_ = MagicMock()
    setattr(jeol_scope.def_, setter_attr, MagicMock(side_effect=RuntimeError("fail")))
    fn = getattr(jeol_scope, method)
    with pytest.raises(RuntimeError, match=r"fail"):
        fn(*args)

@pytest.mark.parametrize(
    "label, missing_attr, call",
    [
        (
            "acceleration_voltage",
            "ht",
            lambda s: s.set_acceleration_voltage(Q_(200, "kV")),
        ),
        (
            "spot_size",
            "eos",
            lambda s: s.set_spot_size(3),
        ),
        (
            "beam_shift",
            "def_",
            lambda s: s.set_beam_shift(1, 2),
        ),
    ],
    ids=["acceleration_voltage", "spot_size", "beam_shift"],
)
def test_simple_setters_raise_when_required_hardware_missing(label, missing_attr, call, jeol_scope):
    """Verify that `simple_setters_raise` when required hardware missing."""
    setattr(jeol_scope, missing_attr, None)
    with pytest.raises(RuntimeError):
        call(jeol_scope)

@pytest.mark.parametrize(
    "label, hw_attr, failing_method, call, exc_msg",
    [
        (
            "acceleration_voltage",
            "ht",
            "SetHtValue",
            lambda s: s.set_acceleration_voltage(Q_(200, "kV")),
            "ht fail",
        ),
        (
            "spot_size",
            "eos",
            "SelectSpotSize",
            lambda s: s.set_spot_size(3),
            "ss fail",
        ),
        (
            "beam_shift",
            "def_",
            "SetCLA1",
            lambda s: s.set_beam_shift(1, 2),
            "cla1 fail",
        ),
    ],
    ids=["acceleration_voltage", "spot_size", "beam_shift"],
)
def test_simple_setters_propagate_hardware_exception(
    label, hw_attr, failing_method, call, exc_msg, jeol_scope
):
    """Verify that `simple_setters_propagate` hardware exception."""
    hw = MagicMock()
    setattr(hw, failing_method, MagicMock(side_effect=RuntimeError(exc_msg)))
    setattr(jeol_scope, hw_attr, hw)

    with pytest.raises(RuntimeError, match=re.escape(exc_msg)):
        call(jeol_scope)

@pytest.mark.parametrize(
    "label, hw_attr, call, expected_msg",
    [
        ("beam_blank", "def_", lambda s: s.set_beam_blank(True), r"Deflector hardware not connected"),
        ("alpha_index", "eos", lambda s: s.set_alpha_index(1), r"EOS hardware not connected"),
        ("set_mode", "eos", lambda s: s.set_mode("TEM"), r"EOS hardware not connected"),
        ("select_eos_mode_key", "eos", lambda s: s._select_eos_mode_key("TEM:DIFF"), r"EOS hardware not connected"),
        ("screen_position", "det3", lambda s: s.set_screen_position("UP"), r"Detector3 hardware not connected"),
    ],
    ids=["beam_blank", "alpha_index", "set_mode", "select_eos_mode_key", "screen_position"],
)
def test_entrypoints_raise_when_required_hardware_missing(label, hw_attr, call, expected_msg, jeol_scope):
    """Verify that `entrypoints_raise` when required hardware missing."""
    setattr(jeol_scope, hw_attr, None)
    with pytest.raises(RuntimeError, match=expected_msg):
        call(jeol_scope)

@pytest.mark.parametrize(
    "label, hw_attr, failing_method, call, exc_msg",
    [
        ("beam_blank", "def_", "SetBeamBlank", lambda s: s.set_beam_blank(True), "blank fail"),
        ("alpha_index", "eos", "SetAlphaSelector", lambda s: s.set_alpha_index(2), "alpha fail"),
    ],
    ids=["beam_blank", "alpha_index"],
)
def test_entrypoints_propagate_hardware_exception(label, hw_attr, failing_method, call, exc_msg, jeol_scope):
    """Verify that `entrypoints_propagate_hardware` exception."""
    hw = MagicMock()
    setattr(hw, failing_method, MagicMock(side_effect=RuntimeError(exc_msg)))
    setattr(jeol_scope, hw_attr, hw)

    with pytest.raises(RuntimeError, match=re.escape(exc_msg)):
        call(jeol_scope)


# =============================================================================
# EOS & MODE CONTROL
# =============================================================================

@pytest.mark.parametrize("mode, expected", [("TEM", 0), ("STEM", 1)])
def test_set_mode_selects_temstem_index(jeol_scope, mode, expected):
    """Verify that `set_mode_selects` temstem index."""
    jeol_scope.set_mode(mode)
    jeol_scope.eos.SelectTemStem.assert_called_with(expected)

def test_set_mode_noops_for_invalid_or_missing_method(jeol_scope):
    """Verify that `set_mode_noops` for invalid or missing method."""
    eos = SimpleNamespace(SelectTemStem=MagicMock())
    jeol_scope.eos = eos

    jeol_scope.set_mode('banana')
    eos.SelectTemStem.assert_not_called()


    jeol_scope.eos = SimpleNamespace()
    jeol_scope.set_mode('TEM')

def test_set_mode_propagates_hardware_exception(jeol_scope):
    """Verify that `set_mode` propagates hardware exception."""
    jeol_scope.eos = SimpleNamespace(SelectTemStem=MagicMock(side_effect=RuntimeError('mode fail')))
    with pytest.raises(RuntimeError, match='mode fail'):
        jeol_scope.set_mode('STEM')

def test_get_mode_maps_tem_stem_and_unknown(jeol_scope):
    """Verify that `get_mode_maps_tem_stem` and unknown."""
    jeol_scope.eos.GetTemStemMode.return_value = 0
    assert jeol_scope.get_mode() == "TEM"

    jeol_scope.eos.GetTemStemMode.return_value = 1
    assert jeol_scope.get_mode() == "STEM"


    jeol_scope.eos = None
    assert jeol_scope.get_mode() == "UNKNOWN"

def test_get_mode_returns_unknown_on_exception(jeol_scope):
    """Verify that `get_mode` returns unknown on exception."""
    jeol_scope.eos.GetTemStemMode.side_effect = RuntimeError("nope")
    assert jeol_scope.get_mode() == "UNKNOWN"


def test_set_projection_mode_diffraction_calls_function_mode(jeol_scope):
    """Verify that `set_projection_mode_diffraction` calls function mode."""
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.set_projection_mode("DIFFRACTION")

    jeol_scope.eos.SelectFunctionMode.assert_called_with(4)

def test_set_projection_mode_with_colon_calls_select_directly(monkeypatch, jeol_scope):
    """Verify that `set_projection_mode` with colon calls select directly."""
    spy = MagicMock()
    monkeypatch.setattr(jeol_scope, "_select_eos_mode_key", spy)
    jeol_scope.set_projection_mode("TEM:DIFF")
    spy.assert_called_once_with("TEM:DIFF")

def test_set_projection_mode_mapping_and_explicit_key(jeol_scope):
    """Verify that `set_projection_mode_mapping` and explicit key."""
    jeol_scope._select_eos_mode_key = MagicMock()

    with pytest.raises(ValueError):
        jeol_scope.set_projection_mode("")


    jeol_scope.set_projection_mode("TEM:MAG")
    jeol_scope._select_eos_mode_key.assert_called_with("TEM:MAG")


    jeol_scope._select_eos_mode_key.reset_mock()
    jeol_scope.get_mode = MagicMock(return_value="STEM")
    jeol_scope.set_projection_mode("diffraction")
    jeol_scope._select_eos_mode_key.assert_called_with("STEM:UUDIFF")

    jeol_scope._select_eos_mode_key.reset_mock()
    jeol_scope.get_mode = MagicMock(return_value="TEM")
    jeol_scope.set_projection_mode("imaging")
    jeol_scope._select_eos_mode_key.assert_called_with("TEM:MAG")

def test_select_eos_mode_key_tem_and_stem_switch_calls_expected_indices(jeol_scope):
    """Verify that `select_eos_mode_key_tem` and stem switch calls expected indices."""
    jeol_scope.eos.SelectTemStem = MagicMock()
    jeol_scope.eos.SelectFunctionMode = MagicMock()

    jeol_scope._select_eos_mode_key("TEM:DIFF")
    jeol_scope.eos.SelectTemStem.assert_called_with(0)
    jeol_scope.eos.SelectFunctionMode.assert_called_with(4)

    jeol_scope.eos.SelectTemStem.reset_mock()
    jeol_scope.eos.SelectFunctionMode.reset_mock()

    jeol_scope._select_eos_mode_key("STEM:AMAG")
    jeol_scope.eos.SelectTemStem.assert_called_with(1)
    jeol_scope.eos.SelectFunctionMode.assert_called_with(3)

def test_helpers_resolve_eos_table_info(monkeypatch, jeol_scope):
    """Verify that `helpers_resolve_eos` table info."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", MagicMock(return_value="TEM:DIFF"))
    key, list_name = jeol_scope._resolve_eos_table_info()
    assert key == "TEM:DIFF"
    assert list_name == "MagList"

    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", MagicMock(return_value="STEM:SM-MAG"))
    key2, list_name2 = jeol_scope._resolve_eos_table_info()
    assert key2 == "STEM:SM-MAG"
    assert list_name2 == "StemCamList"

def test_get_eos_mode_key_fallback_builds_prefix(monkeypatch, jeol_scope):
    """Verify that `get_eos_mode` key fallback builds prefix."""

    jeol_scope.eos.GetFunctionMode.return_value = [7]

    jeol_scope.eos.GetTemStemMode.return_value = 0
    assert jeol_scope._get_eos_mode_key() == "TEM:7"

    jeol_scope.eos.GetTemStemMode.return_value = 1
    assert jeol_scope._get_eos_mode_key() == "STEM:7"

def test_normalize_eos_key_is_case_insensitive(monkeypatch, jeol_scope):
    """Verify that `normalize_eos_key` is case insensitive."""
    monkeypatch.setattr(jeol_module, "EOS_MODE_TABLES", {"TEM:MAG": {}, "Tem:Diff": {}})


    assert jeol_scope._normalize_eos_key("tem:mag") == "TEM:MAG"

    assert jeol_scope._normalize_eos_key("TEM:DIFF") == "Tem:Diff"

def test_normalize_eos_key_returns_none_on_invalid_input(jeol_scope):
    """Verify that `normalize_eos_key` returns None on invalid input."""
    assert jeol_scope._normalize_eos_key(None) is None
    assert jeol_scope._normalize_eos_key("GARBAGE_MODE") is None


def test_select_eos_mode_key_with_invalid_format_raises_value_error(jeol_scope):
    """Verify that `select_eos_mode_key` with invalid format raises value error."""
    with pytest.raises(ValueError, match="Invalid EOS mode key"):
        jeol_scope._select_eos_mode_key("TEMMAG")

def test_select_eos_mode_key_with_unknown_table_raises_value_error(jeol_scope):
    """Verify that `select_eos_mode_key` with unknown table raises value error."""
    with pytest.raises(ValueError, match="Unknown EOS"):
        jeol_scope._select_eos_mode_key("BANANA:MAG")

def test_select_eos_mode_key_propagates_exceptions(jeol_scope):
    """Verify that `select_eos_mode_key` propagates exceptions."""
    jeol_scope.eos.SelectTemStem.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        jeol_scope._select_eos_mode_key("TEM:MAG")

def test_select_eos_mode_key_no_select_methods_is_noop(caplog, jeol_scope):
    """Verify that `select_eos_mode` key no select methods is noop."""
    jeol_scope.eos = SimpleNamespace()

    caplog.set_level('ERROR')
    jeol_scope._select_eos_mode_key('TEM:DIFF')
    jeol_scope._select_eos_mode_key('STEM:AMAG')

    assert not any(r.levelname == 'ERROR' for r in caplog.records)

def test_get_eos_mode_key_returns_none_when_missing_or_exception(jeol_scope):
    """Verify that `get_eos_mode_key` returns None when missing or exception."""
    jeol_scope.eos = None
    assert jeol_scope._get_eos_mode_key() is None


    jeol_scope.eos = SimpleNamespace()
    assert jeol_scope._get_eos_mode_key() is None


    jeol_scope.eos = SimpleNamespace(
        GetFunctionMode=MagicMock(side_effect=RuntimeError('boom')),
        GetTemStemMode=MagicMock(return_value=0),
    )
    assert jeol_scope._get_eos_mode_key() is None

def test_resolve_eos_table_info_returns_none_when_no_mode_key(monkeypatch, jeol_scope):
    """Verify that `resolve_eos_table_info` returns None when no mode key."""
    monkeypatch.setattr(jeol_scope, "_resolve_eos_table_info", lambda: (None, None))
    monkeypatch.setattr(jeol_scope, "get_mode", lambda: "TEM")
    with pytest.raises(RuntimeError, match=r"Camera length control unavailable"):
        jeol_scope.set_camera_length(Q_(10, Units.MM))


# ============================================================================
# PROJECTION (MAGNIFICATION & FOCUS)
# ============================================================================

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_set_magnification_logic(mock_get_list, jeol_scope):
    """Verify set_magnification_logic behavior."""

    mock_get_list.return_value = [("2000.0", "x", "2k"), ("5000.0", "x", "5k")]
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]


    jeol_scope.eos.GetMagValue.side_effect = Exception("HW Fail")
    jeol_scope.eos.GetCurrentMagSelectorID.side_effect = Exception("Not supported")


    jeol_scope.set_magnification(5000)
    jeol_scope.eos.SetSelector.assert_called_with(2)


    jeol_scope.eos.GetSelector.return_value = 2
    mag = jeol_scope.get_magnification()
    assert mag == 5000

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_get_magnification_fallback_uses_getselector_and_checks_neighbors(mock_get_list, jeol_scope):
    """Verify that `get_magnification_fallback` uses getselector and checks neighbors."""
    mock_get_list.return_value = [
        ("1000", "X", "1k"),
        ("2000", "X", "2k"),
        ("3000", "X", "3k"),
    ]


    jeol_scope.eos.GetMagValue.side_effect = RuntimeError("no direct mag")


    jeol_scope.eos.GetCurrentMagSelectorID.side_effect = RuntimeError("no selector id")
    jeol_scope.eos.GetSelector.return_value = 2

    assert jeol_scope.get_magnification() == 2000

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_set_magnification_falls_back_when_setselector_one_based_fails(mock_get_list, jeol_scope):
    """Verify that `set_magnification` falls back when setselector one based fails."""
    mock_get_list.return_value = [
        ("1000", "X", "1k"),
        ("5000", "X", "5k"),
    ]


    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]


    jeol_scope.eos.SetSelector.side_effect = [RuntimeError("no 1-based"), None]

    jeol_scope.set_magnification(5000)

    assert jeol_scope.eos.SetSelector.call_args_list[0] == call(2)
    assert jeol_scope.eos.SetSelector.call_args_list[1] == call(1)

def test_set_magnification_wrong_mode(jeol_scope):
    """Verify that `set_magnification_wrong` mode."""

    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [4]


    with pytest.raises(RuntimeError) as exc:
        jeol_scope.set_magnification(5000)

    assert "Magnification table not found for mode TEM:DIFF" in str(exc.value)

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_get_magnification_falls_back_to_selector_and_table_using_getselector(mock_get_list, jeol_scope):
    """Verify that `get_magnification` falls back to selector and table using getselector."""

    mock_get_list.return_value = [
        ("1000", "X", "1k"),
        ("2000", "X", "2k"),
        ("5000", "X", "5k"),
        ("10000", "X", "10k"),
    ]

    jeol_scope.eos.GetMagValue.side_effect = RuntimeError("no direct mag")
    jeol_scope.eos.GetFunctionMode.return_value = [0]
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetCurrentMagSelectorID.side_effect = RuntimeError("not supported")
    jeol_scope.eos.GetSelector.return_value = 3

    assert jeol_scope.get_magnification() == 5000

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_get_magnification_selector_off_by_one_robustness(mock_get_list, jeol_scope):
    """Verify that `get_magnification_selector_off` by one robustness."""
    mock_get_list.return_value = [
        ("1000", "X", "1k"),
        ("2000", "X", "2k"),
        ("BAD", "X", "bad"),
        ("7000", "X", "7k"),
        ("9000", "X", "9k"),
    ]

    jeol_scope.eos.GetMagValue.side_effect = RuntimeError("no direct mag")
    jeol_scope.eos.GetFunctionMode.return_value = [0]
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetCurrentMagSelectorID.return_value = 3


    assert jeol_scope.get_magnification() == 7000

def test_get_magnification_direct_and_failure_paths(jeol_scope):

    """Verify that `get_magnification_direct` and failure paths."""
    jeol_scope.eos.GetMagValue.return_value = [100000.0, 'x', 'MAG']
    assert jeol_scope.get_magnification() == 100000


    jeol_scope.eos.GetMagValue.return_value = 50000.0
    assert jeol_scope.get_magnification() == 50000


    jeol_scope.eos = None
    assert jeol_scope.get_magnification() is None

def test_get_magnification_falls_back_to_selector_and_table(monkeypatch, jeol_scope):
    """Verify that `get_magnification` falls back to selector and table."""

    jeol_scope.eos.GetMagValue.side_effect = RuntimeError('no direct mag')


    jeol_scope.eos.GetSelector.side_effect = None
    jeol_scope.eos.GetSelector.return_value = 3


    jeol_scope.eos.GetCurrentMagSelectorID.side_effect = RuntimeError('not supported')


    monkeypatch.setattr(jeol_scope, '_get_eos_mode_key', MagicMock(return_value='TEM:MAG'))


    def _fake_get_list(key, name):
        if name != 'MagList': return []
        return [(1000, 'X', ''), (2000, 'X', ''), (3000, 'X', ''), (4000, 'X', '')]

    monkeypatch.setattr(jeol_module, 'get_list', _fake_get_list, raising=True)


    assert jeol_scope.get_magnification() == 3000

def test_set_magnification_requires_mag_table(monkeypatch, jeol_scope):
    """Verify that `set_magnification` requires mag table."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")

    monkeypatch.setattr(jeol_module, "get_list", lambda key, name: [])
    with pytest.raises(RuntimeError, match=r"Magnification table"):
        jeol_scope.set_magnification(1000)

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_get_magnification_direct_non_x_falls_back_to_table(mock_get_list, jeol_scope):
    """Verify that `get_magnification_direct_non_x` falls back to table."""

    jeol_scope.eos.GetMagValue.side_effect = None
    jeol_scope.eos.GetMagValue.return_value = [10.0, "mm", "L"]


    mock_get_list.return_value = [("1000", "X", "1k"), ("5000", "X", "5k")]
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]
    jeol_scope.eos.GetCurrentMagSelectorID.side_effect = RuntimeError("no selector id")
    jeol_scope.eos.GetSelector.return_value = 2

    assert jeol_scope.get_magnification() == 5000

@patch("supertem.microscopes.jeol_microscope.get_list")
def test_set_magnification_selects_floor_and_skips_non_numeric(mock_get_list, jeol_scope):
    """Verify that `set_magnification_selects_floor` and skips non numeric."""
    mock_get_list.return_value = [
        ("1000", "X", "1k"),
        ("bad", "X", "bad"),
        ("2000", "X", "2k"),
        ("9000", "X", "9k"),
    ]
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]


    jeol_scope.set_magnification(3000)
    jeol_scope.eos.SetSelector.assert_called_with(3)

def test_set_magnification_selects_best_and_fallbacks_on_selector_call(monkeypatch, jeol_scope):

    """Verify that `set_magnification_selects_best` and fallbacks on selector call."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda _k: "TEM:MAG")
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [(10, "X", "10x")])

    jeol_scope.eos.SetSelector = MagicMock(side_effect=[RuntimeError("fail1"), None])
    jeol_scope.set_magnification(10)
    assert jeol_scope.eos.SetSelector.call_count == 2

def test_get_magnification_returns_none_when_table_lookup_raises(monkeypatch, jeol_scope):

    """Verify that `get_magnification` returns None when table lookup raises."""
    jeol_scope.eos.GetMagValue = MagicMock(side_effect=RuntimeError("no direct"))
    jeol_scope.eos.GetCurrentMagSelectorID = MagicMock(return_value=0)

    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda k: "TEM:MAG")
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [("bad", "X", "badx")])

    assert jeol_scope.get_magnification() is None

def test_set_magnification_raises_when_eos_mode_key_unresolved(monkeypatch, jeol_scope):

    """Verify that `set_magnification` raises when EOS mode key unresolved."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda _k: "")
    with pytest.raises(RuntimeError, match=r"Cannot resolve EOS mode"):
        jeol_scope.set_magnification(1000)

def test_get_magnification_table_falls_back_to_get_selector(monkeypatch, jeol_scope):

    """Verify that `get_magnification_table` falls back to get selector."""
    jeol_scope.eos.GetMagValue = MagicMock(side_effect=RuntimeError("no direct"))
    jeol_scope.eos.GetCurrentMagSelectorID = MagicMock(side_effect=RuntimeError("no cur"))
    jeol_scope.eos.GetSelector = MagicMock(return_value=1)

    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda k: "TEM:MAG")
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [(50, "X", "50x"), (500, "X", "500x")])

    assert jeol_scope.get_magnification() == 50

def test_get_magnification_returns_none_when_no_selector_available(monkeypatch, jeol_scope):

    """Verify that `get_magnification` returns None when no selector available."""
    jeol_scope.eos.GetMagValue = MagicMock(side_effect=RuntimeError("no direct"))
    jeol_scope.eos.GetCurrentMagSelectorID = MagicMock(side_effect=RuntimeError("no cur"))
    jeol_scope.eos.GetSelector = MagicMock(side_effect=RuntimeError("no sel"))

    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda k: "TEM:MAG")
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [(1, "X", "1x")])

    assert jeol_scope.get_magnification() is None

def test_set_magnification_raises_when_maglist_empty(monkeypatch, jeol_scope):
    """Verify that `set_magnification` raises when maglist empty."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda _k: "TEM:MAG")
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [])
    with pytest.raises(RuntimeError, match=r"Magnification table not found"):
        jeol_scope.set_magnification(1000)

def test_get_magnification_returns_none_when_maglist_unit_not_X(monkeypatch, jeol_scope):

    """Verify that `get_magnification` returns None when maglist unit not X."""
    jeol_scope.eos.GetMagValue = MagicMock(side_effect=RuntimeError("no direct"))
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda s: "TEM:MAG")


    monkeypatch.setattr(jeol_module, "get_list", lambda key, name: [(10, "mm", "10mm")], raising=True)

    assert jeol_scope.get_magnification() is None

def test_set_magnification_raises_when_maglist_unit_not_x(monkeypatch, jeol_scope):

    """Verify that `set_magnification` raises when maglist unit not x."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda _k: "TEM:MAG")
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [(10, "mm", "10mm")])
    with pytest.raises(RuntimeError, match=r"Magnification table not found|MagList unavailable"):
        jeol_scope.set_magnification(1000)

def test_set_magnification_handles_get_list_exception_then_raises(monkeypatch, jeol_scope):

    """Verify that `set_magnification` handles get list exception then raises."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda s: "TEM:MAG")

    def _boom(*args, **kwargs):
        raise RuntimeError("table fetch failed")

    monkeypatch.setattr(jeol_module, "get_list", _boom, raising=True)

    with pytest.raises(RuntimeError, match=r"Magnification table not found"):
        jeol_scope.set_magnification(100000)

def test_set_magnification_raises_when_both_selector_calls_fail(monkeypatch, jeol_scope):

    """Verify that `set_magnification` raises when both selector calls fail."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda s: "TEM:MAG")


    monkeypatch.setattr(
        jeol_module,
        "get_list",
        lambda key, name: [(10, "X", "10x"), (50, "X", "50x"), (100, "X", "100x")],
        raising=True,
    )


    jeol_scope.eos.SetSelector = MagicMock(side_effect=[RuntimeError("first"), RuntimeError("second")])

    with pytest.raises(RuntimeError, match=r"second"):
        jeol_scope.set_magnification(100)

def test_get_magnification_table_uses_current_selector_id(monkeypatch, jeol_scope):

    """Verify that `get_magnification_table` uses current selector id."""
    jeol_scope.eos.GetMagValue = MagicMock(side_effect=RuntimeError("no direct"))
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:MAG")
    monkeypatch.setattr(jeol_scope, "_normalize_eos_key", lambda k: k)
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [(10, "X", "10x"), (100, "X", "100x")])


    if hasattr(jeol_scope.eos, "GetCurrentMagSelectorID"):
        delattr(jeol_scope.eos, "GetCurrentMagSelectorID")
    jeol_scope.eos.GetSelector = MagicMock(return_value=2)

    assert jeol_scope.get_magnification() == 100


@pytest.mark.parametrize("temstem,function_mode,expected_mm", [(0, 4, 1500.0), (1, 0, 80.0)])
def test_get_camera_length_tem_and_stem(jeol_scope, temstem, function_mode, expected_mm):
    """Verify that `get_camera_length_tem` and stem."""
    jeol_scope.eos.GetTemStemMode.return_value = temstem
    jeol_scope.eos.GetFunctionMode.return_value = [function_mode]

    if temstem == 0:

        jeol_scope.eos.GetMagValue.return_value = [150.0, "cm", "L"]
    else:

        jeol_scope.eos.GetStemCamValue.return_value = [80.0, "mm", "L"]

    cl = jeol_scope.get_camera_length()
    assert cl.to("mm").magnitude == pytest.approx(expected_mm)

def test_set_camera_length_wrong_mode(jeol_scope):
    """Verify that `set_camera_length` wrong mode."""

    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]


    with pytest.raises(RuntimeError) as exc:
        jeol_scope.set_camera_length(Q_(100, "cm"))


    assert "Cannot set Camera Length in mode TEM" in str(exc.value)

def test_get_camera_length_returns_none_when_not_diff_or_stem(jeol_scope):

    """Verify that `get_camera_length` returns None when not diff or stem."""
    jeol_scope.eos.GetTemStemMode.return_value = 0
    jeol_scope.eos.GetFunctionMode.return_value = [0]
    assert jeol_scope.get_camera_length() is None

def test_set_camera_length_selects_best_selector_index(monkeypatch, jeol_scope):
    """Verify that `set_camera_length` selects best selector index."""
    monkeypatch.setattr(jeol_scope, '_get_eos_mode_key', MagicMock(return_value='TEM:DIFF'))

    def _fake_get_list(key, name):
        assert key == 'TEM:DIFF'
        assert name == 'MagList'
        return [(100.0, 'mm', ''), (200.0, 'mm', ''), (500.0, 'mm', '')]

    monkeypatch.setattr(jeol_module, 'get_list', _fake_get_list, raising=True)

    jeol_scope.eos.SetSelector = MagicMock()
    jeol_scope.set_camera_length(Q_(210, Units.MM))


    jeol_scope.eos.SetSelector.assert_called_once_with(2)

def test_get_camera_length_tem_diff_and_stem_paths(monkeypatch, jeol_scope):
    """Verify that `get_camera_length_tem_diff` and stem paths."""


    monkeypatch.setattr(jeol_scope, '_get_eos_mode_key', MagicMock(return_value='TEM:DIFF'))
    jeol_scope.eos.GetMagValue.return_value = (800.0, 'mm', 'DIFF')
    cl = jeol_scope.get_camera_length()
    assert cl is not None
    assert float(cl.to(Units.MM).magnitude) == pytest.approx(800.0)


    monkeypatch.setattr(jeol_scope, '_get_eos_mode_key', MagicMock(return_value='STEM:SM-MAG'))
    jeol_scope.eos.GetStemCamValue = MagicMock(return_value=(1200.0, 'mm', ''))
    cl2 = jeol_scope.get_camera_length()
    assert cl2 is not None
    assert float(cl2.to(Units.MM).magnitude) == pytest.approx(1200.0)

def test_set_camera_length_selects_best_table_entry(monkeypatch, jeol_scope):
    """Verify that `set_camera_length` selects best table entry."""

    monkeypatch.setattr(jeol_scope, "_resolve_eos_table_info", lambda: ("TEM:DIFF", "MagList"))


    table = [
        (100, "mm", "100"),
        (200, "mm", "200"),
        (0.3, "m", "300"),
    ]
    monkeypatch.setattr(jeol_module, "get_list", lambda key, name: table)

    jeol_scope.eos.SetSelector = MagicMock()


    jeol_scope.set_camera_length(Q_(250, Units.MM))
    jeol_scope.eos.SetSelector.assert_called_with(2)

def test_set_camera_length_rejects_non_length_units(monkeypatch, jeol_scope):
    """Verify that `set_camera_length` rejects non length units."""
    monkeypatch.setattr(jeol_scope, "_resolve_eos_table_info", lambda: ("TEM:DIFF", "MagList"))
    monkeypatch.setattr(jeol_module, "get_list", lambda key, name: [(10, "X", "10x")])
    with pytest.raises(RuntimeError, match=r"Cannot set Camera Length"):
        jeol_scope.set_camera_length(Q_(10, Units.MM))

def test_set_camera_length_raises_when_no_valid_entries(monkeypatch, jeol_scope):
    """Verify that `set_camera_length` raises when no valid entries."""
    monkeypatch.setattr(jeol_scope, "_resolve_eos_table_info", lambda: ("TEM:DIFF", "MagList"))


    monkeypatch.setattr(jeol_module, "get_list", lambda key, name: [("nan", "mm", "x"), ("nan", "mm", "y")])

    with pytest.raises(RuntimeError, match=r"No valid camera length"):
        jeol_scope.set_camera_length(Q_(10, Units.MM))

def test_set_camera_length_rejects_none(jeol_scope):

    """Verify that `set_camera_length` rejects none."""
    with pytest.raises(ValueError, match=r"Camera length cannot be None"):
        jeol_scope.set_camera_length(None)

@patch('supertem.microscopes.jeol_microscope.get_list')
def test_set_camera_length_raises_when_no_selector_method(mock_get_list, jeol_scope):
    """Verify that `set_camera_length` raises when no selector method."""
    jeol_scope._resolve_eos_table_info = MagicMock(return_value=("TEM:DIFF", "MagList"))
    mock_get_list.return_value = [(10.0, "mm", "10mm"), (20.0, "mm", "20mm")]


    class _EOS:  # noqa: N801
        pass

    jeol_scope.eos = _EOS()

    with pytest.raises(AttributeError, match=r"No suitable selector method"):
        jeol_scope.set_camera_length(Q_(10, Units.MM))

def test_get_camera_length_returns_none_when_eos_missing(jeol_scope):

    """Verify that `get_camera_length` returns None when EOS missing."""
    jeol_scope.eos = None
    assert jeol_scope.get_camera_length() is None

def test_get_camera_length_returns_none_when_stem_cam_method_missing(monkeypatch, jeol_scope):
    """Verify that `get_camera_length` returns None when stem cam method missing."""
    monkeypatch.setattr(jeol_scope, '_get_eos_mode_key', lambda: 'STEM:AMAG')
    jeol_scope.eos = SimpleNamespace()

    assert jeol_scope.get_camera_length() is None

def test_get_camera_length_exception_returns_none(monkeypatch, jeol_scope):

    """Verify that `get_camera_length_exception` returns none."""
    monkeypatch.setattr(jeol_scope, "_get_eos_mode_key", lambda: "TEM:DIFF")
    jeol_scope.eos.GetMagValue = MagicMock(side_effect=RuntimeError("boom"))
    assert jeol_scope.get_camera_length() is None

def test_set_camera_length_raises_when_eos_missing(monkeypatch, jeol_scope):

    """Verify that `set_camera_length` raises when EOS missing."""
    jeol_scope.eos = None
    with pytest.raises(RuntimeError, match=r"EOS hardware not connected"):
        jeol_scope.set_camera_length(Q_(10, Units.MM))

def test_set_camera_length_raises_when_targets_empty(monkeypatch, jeol_scope):

    """Verify that `set_camera_length` raises when targets empty."""
    monkeypatch.setattr(jeol_scope, "_resolve_eos_table_info", lambda: ("TEM:DIFF", "MagList"))
    monkeypatch.setattr(jeol_module, "get_list", lambda *_a, **_k: [])
    with pytest.raises(RuntimeError, match=r"Camera length table empty"):
        jeol_scope.set_camera_length(Q_(10, Units.MM))

def test_set_camera_length_stem_prefers_set_stem_cam_selector(monkeypatch, jeol_scope):


    """Verify that `set_camera_length` stem prefers set stem cam selector."""
    monkeypatch.setattr(jeol_scope, "_resolve_eos_table_info", lambda: ("STEM:DIFF", "MagList"))

    def _fake_get_list(key, name):
        assert key == "STEM:DIFF"
        assert name == "MagList"
        return [(100.0, "mm", "100"), (200.0, "mm", "200"), (500.0, "mm", "500")]

    monkeypatch.setattr(jeol_module, "get_list", _fake_get_list, raising=True)


    jeol_scope.eos = SimpleNamespace(SetStemCamSelector=MagicMock(), SetSelector=MagicMock())

    jeol_scope.set_camera_length(Q_(210, Units.MM))


    jeol_scope.eos.SetStemCamSelector.assert_called_once_with(2)
    jeol_scope.eos.SetSelector.assert_not_called()

def test_projection_camera_length_stem_and_diff_paths(jeol_scope):

    """Verify that `projection_camera_length_stem` and diff paths."""
    jeol_scope.eos.GetFunctionMode.return_value = [0]
    jeol_scope.eos.GetTemStemMode.return_value = 1
    jeol_scope.eos.GetStemCamValue.return_value = (15.0, "mm", "CL")
    cl = jeol_scope.get_camera_length()
    assert cl is not None
    assert float(cl.to(Units.MM).magnitude) == pytest.approx(15.0)


    jeol_scope.eos.GetTemStemMode.return_value = 0

    jeol_scope.eos.GetFunctionMode.return_value = [99]
    jeol_scope.eos.GetMagValue.return_value = (200.0, "mm", "CL")

    jeol_scope._get_eos_mode_key = MagicMock(return_value="TEM:DIFF")
    cl2 = jeol_scope.get_camera_length()
    assert cl2 is not None
    assert float(cl2.to(Units.MM).magnitude) == pytest.approx(200.0)


def test_set_defocus_uses_calibration_scale(jeol_scope):
    """Verify that `set_defocus` uses calibration scale."""
    target_defocus = Q_(100, "nm")
    jeol_scope.set_defocus(target_defocus)

    jeol_scope.lens.SetOLc.assert_called_with(50)

def test_get_defocus_paths(jeol_scope):
    """Verify get_defocus_paths behavior."""
    jeol_scope._has_defocus_calibration = False
    assert jeol_scope.get_defocus() is None

    jeol_scope._has_defocus_calibration = True
    jeol_scope.defocus_scale = 2.0
    jeol_scope.lens.GetOLc.return_value = 10.0
    df = jeol_scope.get_defocus()
    assert df.to('nm').magnitude == pytest.approx(5.0)

def test_defocus_set_and_dac_helpers(monkeypatch, jeol_scope):

    """Verify that `defocus_set` and DAC helpers."""
    jeol_scope._has_defocus_calibration = True
    jeol_scope.defocus_scale = 0.5
    jeol_scope.lens.SetOLc = MagicMock()
    jeol_scope.set_defocus(Q_(10, Units.NM))

    jeol_scope.lens.SetOLc.assert_called_with(5)


    jeol_scope._has_defocus_calibration = False
    with pytest.raises(ValueError):
        jeol_scope.set_defocus(Q_(1, Units.NM))


    jeol_scope.lens = MagicMock()
    jeol_scope.lens.GetOLc = MagicMock(return_value=123)
    assert jeol_scope.get_defocus_dac() == 123

    jeol_scope.lens.GetOLc = MagicMock(side_effect=RuntimeError('boom'))
    assert jeol_scope.get_defocus_dac() is None

    jeol_scope.lens = None
    assert jeol_scope.get_defocus_dac() is None


    jeol_scope.lens = None
    with pytest.raises(RuntimeError):
        jeol_scope.set_defocus_dac(1)

    jeol_scope.lens = MagicMock()
    jeol_scope.lens.SetOLc = MagicMock()
    jeol_scope.set_defocus_dac(77)
    jeol_scope.lens.SetOLc.assert_called_with(77)

def test_defocus_and_vendor_dac_paths(jeol_scope):
    """Verify that `defocus` and vendor DAC paths."""

    jeol_scope._has_defocus_calibration = False
    with pytest.raises(ValueError):
        jeol_scope.set_defocus(Q_(1, Units.NM))


    jeol_scope.lens.GetOLc = MagicMock(return_value=123)
    assert jeol_scope.get_defocus_dac() == 123
    jeol_scope.lens.SetOLc = MagicMock()
    jeol_scope.set_defocus_dac(456)
    jeol_scope.lens.SetOLc.assert_called_with(456)

@pytest.mark.parametrize(
    'lens, expected_calls',
    [
        (None, None),
        (SimpleNamespace(GetOLc=MagicMock(side_effect=RuntimeError('boom'))), 1),
    ],
)
def test_get_defocus_returns_none_when_lens_missing_or_raises(lens, expected_calls, jeol_scope):
    """Verify that `get_defocus` returns None when lens missing or raises."""
    jeol_scope._has_defocus_calibration = True
    jeol_scope.defocus_scale = 2.0
    jeol_scope.lens = lens

    assert jeol_scope.get_defocus() is None

    if expected_calls:
        lens.GetOLc.assert_called_once()

def test_set_defocus_raises_when_lens_missing(monkeypatch, jeol_scope):

    """Verify that `set_defocus` raises when lens missing."""
    jeol_scope._has_defocus_calibration = True
    jeol_scope.lens = None
    with pytest.raises(RuntimeError, match=r"Lens hardware not connected"):
        jeol_scope.set_defocus(Q_(1, Units.NM))

def test_set_defocus_raises_when_lens_fails(jeol_scope):

    """Verify that `set_defocus` raises when lens fails."""
    jeol_scope._has_defocus_calibration = True
    jeol_scope.defocus_scale = 1.0

    jeol_scope.lens.SetOLc = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        jeol_scope.set_defocus(Q_(10, Units.NM))

def test_set_defocus_dac_raises_when_lens_fails(jeol_scope):

    """Verify that `set_defocus_dac` raises when lens fails."""
    jeol_scope.lens.SetOLc = MagicMock(side_effect=RuntimeError("dac write fail"))
    with pytest.raises(RuntimeError, match="dac write fail"):
        jeol_scope.set_defocus_dac(123)


def test_get_screen_position_paths(jeol_scope):
    """Verify that `get_screen_position` paths."""
    jeol_scope.det3.GetScreen.return_value = 2
    assert jeol_scope.get_screen_position() == 'DOWN'
    jeol_scope.det3.GetScreen.return_value = 1
    assert jeol_scope.get_screen_position() == 'UP'

    jeol_scope.det3.GetScreen.side_effect = RuntimeError('screen fail')
    assert jeol_scope.get_screen_position() == 'UNKNOWN'

    jeol_scope.det3 = None
    assert jeol_scope.get_screen_position() == 'UNKNOWN'

def test_set_screen_position_raises_when_det3_fails(jeol_scope):

    """Verify that `set_screen_position` raises when det3 fails."""
    jeol_scope.det3.SetScreen = MagicMock(side_effect=RuntimeError("screen jam"))
    with pytest.raises(RuntimeError, match="screen jam"):
        jeol_scope.set_screen_position("UP")

def test_set_screen_position_calls_hardware(jeol_scope):
    """Verify that `set_screen_position` calls hardware."""
    jeol_scope.det3.SetScreen = MagicMock()

    jeol_scope.set_screen_position("DOWN")
    jeol_scope.det3.SetScreen.assert_called_with(2)

    jeol_scope.set_screen_position("UP")
    jeol_scope.det3.SetScreen.assert_called_with(0)

def test_get_projection_settings_uncalibrated_returns_vendor_dac(jeol_scope):
    """Verify that `get_projection_settings_uncalibrated` returns vendor DAC."""
    jeol_scope._has_defocus_calibration = False

    jeol_scope.lens.GetOLc.return_value = 32768
    ps = jeol_scope.get_projection_settings()

    assert ps.defocus is None
    assert ps.extra.vendor["JEOL"]["defocus_olc_dac"] == 32768

    req = ProjectionSettings(extra=Extras(vendor={"JEOL": {"defocus_olc_dac": 12345}}))
    jeol_scope.apply_projection_settings(req)
    jeol_scope.lens.SetOLc.assert_called_with(12345)

def test_get_projection_settings_uncalibrated_defocus_adds_dac(monkeypatch, jeol_scope):
    """Verify that `get_projection_settings` uncalibrated defocus adds DAC."""

    jeol_scope._has_defocus_calibration = False
    jeol_scope.get_defocus_dac = MagicMock(return_value=1234)

    ps = jeol_scope.get_projection_settings()
    assert ps is not None
    assert ps.extra.vendor.get('JEOL', {}).get('defocus_olc_dac') == 1234
    assert 'ProjectionSettings.defocus_uncalibrated' in ps.extra.notes

def test_apply_projection_settings_uncalibrated_physical_defocus_rejected(jeol_scope):
    """Verify that `apply_projection_settings` uncalibrated physical defocus rejected."""
    jeol_scope._has_defocus_calibration = False
    with pytest.raises(ValueError, match=r"cannot apply ProjectionSettings\.defocus"):
        jeol_scope.apply_projection_settings(ProjectionSettings(defocus=Q_(1, Units.NM)))

def test_apply_projection_settings_vendor_dac_none_is_noop(jeol_scope):
    """Verify that `apply_projection_settings` vendor DAC None is noop."""
    jeol_scope.set_defocus_dac = MagicMock()
    jeol_scope._has_defocus_calibration = False

    req = ProjectionSettings(extra=Extras(vendor={"JEOL": {"defocus_olc_dac": None}}))
    jeol_scope.apply_projection_settings(req)

    jeol_scope.set_defocus_dac.assert_not_called()

def test_get_projection_settings_adds_vendor_defocus_dac_when_uncalibrated(monkeypatch, jeol_scope):

    """Verify that `get_projection_settings_adds_vendor_defocus_dac` when uncalibrated."""
    parent = JeolMicroscope.__mro__[1]
    monkeypatch.setattr(parent, "get_projection_settings", lambda self: ProjectionSettings(_mode="lenient"))

    jeol_scope._has_defocus_calibration = False
    jeol_scope.get_defocus_dac = MagicMock(return_value=None)

    ps = jeol_scope.get_projection_settings()
    assert "JEOL" not in (ps.extra.vendor or {})
    assert "ProjectionSettings.defocus_uncalibrated" not in ps.extra.notes

def test_apply_projection_settings_raises_if_defocus_requires_calibration(jeol_scope):

    """Verify that `apply_projection_settings` raises if defocus requires calibration."""
    jeol_scope._has_defocus_calibration = False
    ps = ProjectionSettings(defocus=Q_(5, Units.NM), _mode="lenient")


    with pytest.raises(ValueError, match=r"cannot apply ProjectionSettings\.defocus.*defocus_scale calibration"):
        jeol_scope.apply_projection_settings(ps)

def test_apply_projection_settings_uses_vendor_defocus_olc_dac(monkeypatch, jeol_scope):

    """Verify that `apply_projection_settings` uses vendor defocus olc DAC."""
    parent = JeolMicroscope.__mro__[1]
    monkeypatch.setattr(parent, "apply_projection_settings", lambda self, settings: None)

    jeol_scope.set_defocus_dac = MagicMock()
    ps = ProjectionSettings(extra=Extras(vendor={"JEOL": {"defocus_olc_dac": None}}), _mode="lenient")
    jeol_scope.apply_projection_settings(ps)

    jeol_scope.set_defocus_dac.assert_not_called()


@pytest.mark.parametrize(
    "public_getter, hw_getter, hw_value, expected",
    [
        ("get_objective_stigmation", "GetOLs", [1.1, 2.2], (1.1, 2.2)),
        ("get_diffraction_shift", "GetPLA", [5.1, 6.1], (5.1, 6.1)),
    ],
    ids=["objective_stigmation", "diffraction_shift"],
)
# ============================================================================
# PROJECTION (LENSES & SHIFTS)
# ============================================================================

def test_projection_getters_return_tuples_from_hardware(public_getter, hw_getter, hw_value, expected, jeol_scope):
    """Verify that `projection_getters_return_tuples` from hardware."""
    setattr(jeol_scope.def_, hw_getter, MagicMock(return_value=hw_value))
    assert getattr(jeol_scope, public_getter)() == pytest.approx(expected)

@pytest.mark.parametrize(
    "public_setter, hw_setter, args, expected_ints",
    [
        ("set_objective_stigmation", "SetOLs", (1.2, 2.8), (1, 2)),
        ("set_diffraction_shift", "SetPLA", (7.9, 8.2), (7, 8)),
    ],
    ids=["objective_stigmation", "diffraction_shift"],
)
def test_projection_setters_round_and_call_hardware(public_setter, hw_setter, args, expected_ints, jeol_scope):
    """Verify that `projection_setters_round` and call hardware."""
    setattr(jeol_scope.def_, hw_setter, MagicMock())
    getattr(jeol_scope, public_setter)(*args)
    getattr(jeol_scope.def_, hw_setter).assert_called_with(*expected_ints)

@pytest.mark.parametrize(
    "public_setter, hw_setter, exc_msg",
    [
        ("set_objective_stigmation", "SetOLs", "OLS fail"),
        ("set_diffraction_shift", "SetPLA", "PLA fail"),
    ],
    ids=["objective_stigmation", "diffraction_shift"],
)
def test_projection_setters_propagate_hardware_failure(public_setter, hw_setter, exc_msg, jeol_scope):
    """Verify that `projection_setters_propagate` hardware failure."""
    setattr(jeol_scope.def_, hw_setter, MagicMock(side_effect=RuntimeError(exc_msg)))
    with pytest.raises(RuntimeError, match=re.escape(exc_msg)):
        getattr(jeol_scope, public_setter)(1, 2)

def test_set_objective_stigmation_raises_when_disconnected(jeol_scope):
    """Verify that `set_objective_stigmation` raises when disconnected."""
    jeol_scope.def_ = None
    with pytest.raises(RuntimeError):
        jeol_scope.set_objective_stigmation(0, 0)


def test_get_image_shift_prefers_is1_then_is(jeol_scope):

    """Verify that `get_image_shift` prefers is1 then is."""
    jeol_scope.def_.GetIS1 = MagicMock(return_value=[11, 22])
    assert jeol_scope.get_image_shift() == (11.0, 22.0)


    delattr(jeol_scope.def_, "GetIS1")
    jeol_scope.def_.GetIS = MagicMock(return_value=(33, 44))
    assert jeol_scope.get_image_shift() == (33.0, 44.0)

def test_get_image_shift_uses_getis_when_is1_absent(monkeypatch, jeol_scope):

    """Verify that `get_image_shift` uses getis when is1 absent."""
    jeol_scope.def_ = SimpleNamespace(GetIS=MagicMock(return_value=[7, 8]))
    assert jeol_scope.get_image_shift() == (7.0, 8.0)

def test_get_image_shift_returns_none_tuple_when_hardware_missing(monkeypatch, jeol_scope):

    """Verify that `get_image_shift` returns None tuple when hardware missing."""
    jeol_scope.def_ = None
    assert jeol_scope.get_image_shift() == (None, None)

def test_set_image_shift_prefers_setis1_then_setis(jeol_scope):

    """Verify that `set_image_shift` prefers setis1 then setis."""
    jeol_scope.def_ = MagicMock()
    jeol_scope.def_.SetIS1 = MagicMock()
    jeol_scope.set_image_shift(1.2, 3.8)
    jeol_scope.def_.SetIS1.assert_called_with(1, 3)


    delattr(jeol_scope.def_, 'SetIS1')
    jeol_scope.def_.SetIS = MagicMock()
    jeol_scope.set_image_shift(9.9, 8.1)
    jeol_scope.def_.SetIS.assert_called_with(9, 8)

@pytest.mark.parametrize(
    "method,args",
    [
        ("set_image_shift", (0, 0)),
        ("set_diffraction_shift", (0, 0)),
    ],
)
def test_set_shift_raises_when_disconnected(method, args, jeol_scope):
    """Verify that `set_shift` raises when disconnected."""
    jeol_scope.def_ = None
    fn = getattr(jeol_scope, method)
    with pytest.raises(RuntimeError):
        fn(*args)


# ============================================================================
# SCAN CONTROL
# ============================================================================

def test_get_scan_mode_uses_scanmodestr(monkeypatch, jeol_scope):
    """Verify that `get_scan_mode` uses scanmodestr."""
    d = MagicMock()
    d.get_detectorsetting.return_value = {"ScanModeStr": "Area"}
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: d)
    assert jeol_scope.get_scan_mode() == "Area"

@pytest.mark.parametrize(
    "getter_name, cache_key, cache_value",
    [
        ("get_scan_mode", "mode", "Area"),
        ("get_scan_active", "active", True),
    ],
    ids=["scan_mode", "scan_active"],
)
def test_get_scan_falls_back_to_cache_on_exception(getter_name, cache_key, cache_value, monkeypatch, jeol_scope):
    """Verify that `get_scan` falls back to cache on exception."""
    jeol_scope._scan_cfg[cache_key] = cache_value

    if getter_name == "get_scan_mode":
        d = MagicMock()
        d.get_detectorsetting.side_effect = RuntimeError("no setting")
        monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: d)
    else:
        jeol_scope.scan = SimpleNamespace(GetExtScanMode=MagicMock(side_effect=RuntimeError("nope")))

    assert getattr(jeol_scope, getter_name)() == cache_value

@pytest.mark.parametrize(
    "mode, expected_call, expected_cache",
    [
        ("Area", 3, "Area"),
        ("2", 2, "2"),
    ],
)
def test_set_scan_mode_calls_detector_api_and_updates_cache(
    mode, expected_call, expected_cache, jeol_scope
):
    """Verify that `set_scan_mode` calls detector api and updates cache."""
    jeol_scope.scan = None
    jeol_scope._primary_detector_id = "Camera1"

    d = jeol_scope._get_detector("Camera1")
    d.set_scanmode = MagicMock()

    jeol_scope.set_scan_mode(mode)

    d.set_scanmode.assert_called_once_with(expected_call)
    assert jeol_scope._scan_cfg.get("mode") == expected_cache

def test_set_scan_mode_updates_cache_only_when_method_missing(monkeypatch, jeol_scope):
    """Verify that `set_scan_mode_updates_cache_only` when method missing."""
    d = SimpleNamespace()
    monkeypatch.setattr(jeol_scope, '_get_scan_controller_detector', lambda: d)

    jeol_scope.set_scan_mode('Area')
    assert jeol_scope._scan_cfg.get('mode') == 'Area'

def test_set_scan_mode_unsupported_value_raises(jeol_scope):
    """Verify that `set_scan_mode_unsupported_value` raises."""
    with pytest.raises(ValueError, match=r"Unsupported scan mode"):
        jeol_scope.set_scan_mode("banana")

def test_set_scan_mode_raises_when_no_scan_detector(jeol_scope):

    """Verify that `set_scan_mode` raises when no scan detector."""
    with patch.object(jeol_scope, "_get_scan_controller_detector", return_value=None):
        with pytest.raises(RuntimeError, match=r"Scan detector hardware not connected"):
            jeol_scope.set_scan_mode("Area")

def test_get_scan_mode_width_height_from_detector_settings(jeol_scope):
    """Verify that `get_scan_mode_width_height` from detector settings."""
    jeol_scope.scan = None
    jeol_scope._primary_detector_id = "Camera1"

    d = jeol_scope._get_detector("Camera1")
    d.get_detectorsetting.return_value = {
        "ScanMode": 3,
        "Width": 256,
        "Height": 128,
        "ScanRotation": 12.5,
    }

    assert jeol_scope.get_scan_mode() == "Area"
    assert jeol_scope.get_scan_width() == 256
    assert jeol_scope.get_scan_height() == 128

    ang = jeol_scope.get_scan_rotation()
    assert ang is not None
    assert float(ang.to(Units.DEG).magnitude) == pytest.approx(12.5)


def test_get_scan_width_returns_none_on_exception(monkeypatch, jeol_scope):
    """Verify that `get_scan_width` returns None on exception."""
    jeol_scope._scan_cfg["width_px"] = 123

    d = MagicMock()
    d.get_detectorsetting.side_effect = RuntimeError("no setting")
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: d)

    assert jeol_scope.get_scan_width() is None
    assert jeol_scope._scan_cfg["width_px"] == 123

def test_set_scan_dimensions_calls_imaging_area(jeol_scope):
    """Verify that `set_scan_dimensions` calls imaging area."""
    jeol_scope.scan = None
    jeol_scope._primary_detector_id = "Camera1"
    d = jeol_scope._get_detector("Camera1")
    d.set_imaging_area = MagicMock()

    jeol_scope.set_scan_width(800)
    d.set_imaging_area.assert_called_once()
    assert jeol_scope._scan_cfg.get("width_px") == 800

    d.set_imaging_area.reset_mock()
    jeol_scope.set_scan_height(600)
    d.set_imaging_area.assert_called_once()
    assert jeol_scope._scan_cfg.get("height_px") == 600

def test_set_imaging_area_calls_detector_and_updates_cache(monkeypatch, jeol_scope):
    """Verify that `set_imaging_area` calls detector and updates cache."""
    d = MagicMock()
    d.set_imaging_area = MagicMock()
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: d)

    jeol_scope._scan_cfg.clear()
    jeol_scope._set_imaging_area(width=128, height=96, x=3, y=4)
    d.set_imaging_area.assert_called_once_with(128, 96, 3, 4)
    assert jeol_scope._scan_cfg["width_px"] == 128
    assert jeol_scope._scan_cfg["height_px"] == 96
    assert jeol_scope._scan_cfg["x_px"] == 3
    assert jeol_scope._scan_cfg["y_px"] == 4

def test_set_imaging_area_noops_when_method_missing(monkeypatch, jeol_scope):
    """Verify that `set_imaging_area_noops` when method missing."""
    d = SimpleNamespace()
    monkeypatch.setattr(jeol_scope, '_get_scan_controller_detector', lambda: d)

    jeol_scope._scan_cfg.update({'width_px': 123, 'height_px': 456, 'x_px': 7, 'y_px': 8})

    jeol_scope.set_scan_width(999)
    jeol_scope.set_scan_height(888)

    assert jeol_scope._scan_cfg['width_px'] == 123
    assert jeol_scope._scan_cfg['height_px'] == 456

def test_set_imaging_area_raises_runtime_error_when_no_detector(monkeypatch, jeol_scope):
    """Verify that `set_imaging_area` raises runtime error when no detector."""
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: None)
    with pytest.raises(RuntimeError, match=r"Scan detector hardware not connected"):
        jeol_scope._set_imaging_area(width=64, height=64, x=0, y=0)


def test_get_scan_rotation_prefers_hardware_ex(monkeypatch, jeol_scope):

    """Verify that `get_scan_rotation` prefers hardware ex."""
    jeol_scope.scan = SimpleNamespace(GetRotationAngleEx=MagicMock(return_value=12.25))
    ang = jeol_scope.get_scan_rotation()
    assert float(ang.to(Units.DEG).magnitude) == 12.25

def test_get_scan_rotation_falls_back_to_hardware_angle(monkeypatch, jeol_scope):

    """Verify that `get_scan_rotation` falls back to hardware angle."""
    jeol_scope.scan = SimpleNamespace(GetRotationAngle=MagicMock(return_value=44.0))
    ang = jeol_scope.get_scan_rotation()
    assert float(ang.to(Units.DEG).magnitude) == 44.0

def test_get_scan_rotation_falls_back_to_detector_setting(monkeypatch, jeol_scope):

    """Verify that `get_scan_rotation` falls back to detector setting."""
    jeol_scope.scan = None
    d = SimpleNamespace(get_detectorsetting=MagicMock(return_value={"ScanRotationValue": "45.5"}))
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: d)

    ang = jeol_scope.get_scan_rotation()
    assert float(ang.to(Units.DEG).magnitude) == 45.5

def test_set_scan_rotation_prefers_detector_then_falls_back_to_scan_coils(jeol_scope):
    """Verify that `set_scan_rotation_prefers_detector_then` falls back to scan coils."""
    jeol_scope._primary_detector_id = "Camera1"
    d = jeol_scope._get_detector("Camera1")
    d.set_scanrotation = MagicMock()


    jeol_scope.set_scan_rotation(Q_(12.5, Units.DEG))
    d.set_scanrotation.assert_called_once_with(12.5)


    d.set_scanrotation = MagicMock(side_effect=RuntimeError("det fail"))
    jeol_scope.scan = MagicMock()
    jeol_scope.scan.SetRotationAngleEx = MagicMock()
    jeol_scope.set_scan_rotation(Q_(33.0, Units.DEG))
    jeol_scope.scan.SetRotationAngleEx.assert_called_once_with(33.0)

def test_set_scan_rotation_detector_fails_then_scan_angle_int_mod360(monkeypatch, jeol_scope):
    """Verify that `set_scan_rotation` detector fails then scan angle int mod360."""
    monkeypatch.setattr(jeol_scope, '_get_scan_controller_detector', lambda: None)

    scan = SimpleNamespace(SetRotationAngle=MagicMock())
    jeol_scope.scan = scan

    jeol_scope.set_scan_rotation(Q_(90.6, Units.DEG))
    scan.SetRotationAngle.assert_called_once_with(91)

def test_set_scan_rotation_raises_exception_when_hardware_fails(jeol_scope):
    """Verify that `set_scan_rotation` raises exception when hardware fails."""
    det = jeol_scope._get_detector("Camera1")
    det.set_scanrotation.side_effect = Exception("Det Not Supported")


    jeol_scope.scan.SetRotationAngleEx.side_effect = Exception("HW Critical Fail")

    with pytest.raises(Exception) as exc:
        jeol_scope.set_scan_rotation(Q_(90, "deg"))

    assert "HW Critical Fail" in str(exc.value)

@pytest.mark.parametrize(
    'scan_obj',
    [
        SimpleNamespace(),
        None,
    ],
    ids=['scan_present_no_methods', 'scan_missing'],
)
def test_set_scan_rotation_raises_runtime_error_when_no_capability(monkeypatch, jeol_scope, scan_obj):
    """Verify that `set_scan_rotation` raises runtime error when no capability."""
    d = SimpleNamespace(set_scanrotation=MagicMock(side_effect=RuntimeError('det fail')))
    monkeypatch.setattr(jeol_scope, '_get_scan_controller_detector', lambda: d)

    jeol_scope.scan = scan_obj
    with pytest.raises(RuntimeError, match=r'SetRotation failed'):
        jeol_scope.set_scan_rotation(Q_(90, Units.DEG))

def test_get_scan_active_uses_hardware_getextscanmode(monkeypatch, jeol_scope):

    """Verify that `get_scan_active` uses hardware getextscanmode."""
    jeol_scope.scan = SimpleNamespace(GetExtScanMode=MagicMock(return_value=1))
    assert jeol_scope.get_scan_active() is True

    jeol_scope.scan.GetExtScanMode = MagicMock(return_value=0)
    assert jeol_scope.get_scan_active() is False

def test_set_scan_active_prefers_detector_live_then_falls_back_to_ext_scan(monkeypatch, jeol_scope):
    """Verify that `set_scan_active_prefers_detector_live_then` falls back to ext scan."""
    d = MagicMock()
    d.livestart = MagicMock()
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: d)

    jeol_scope.scan = None
    jeol_scope.set_scan_active(True)
    d.livestart.assert_called_once()

@pytest.mark.parametrize(
    "detector_obj",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(livestart=MagicMock(side_effect=RuntimeError("busy"))),
    ],
    ids=["no_detector", "detector_no_live_methods", "detector_live_raises"],
)
def test_set_scan_active_raises_when_no_detector_live_and_no_scan_fallback(monkeypatch, jeol_scope, detector_obj):
    """Verify that `set_scan_active` raises when no detector live and no scan fallback."""
    monkeypatch.setattr(jeol_scope, "_get_scan_controller_detector", lambda: detector_obj)
    jeol_scope.scan = None

    with pytest.raises(RuntimeError, match=r"Scan control failed"):
        jeol_scope.set_scan_active(True)

@pytest.mark.parametrize(
    "setup",
    [
        "empty_list",
        "get_primary_raises",
    ],
    ids=["no_detectors", "primary_raises"],
)
def test_get_scan_controller_detector_returns_none_for_empty_or_error(setup, monkeypatch, jeol_scope):
    """Verify that `get_scan_controller_detector` returns None for empty or error."""
    if setup == "empty_list":
        monkeypatch.setattr(jeol_scope, "get_primary_detector_id", lambda: None)
        monkeypatch.setattr(jeol_scope, "list_detectors", lambda: [])
    else:
        monkeypatch.setattr(
            jeol_scope,
            "get_primary_detector_id",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert jeol_scope._get_scan_controller_detector() is None

def test_scan_controller_detector_falls_back_to_first_listed_detector(monkeypatch, jeol_scope):
    """Verify that `scan_controller_detector` falls back to first listed detector."""
    sentinel = object()
    monkeypatch.setattr(jeol_scope, "get_primary_detector_id", lambda: None)
    monkeypatch.setattr(jeol_scope, "list_detectors", lambda: ["Camera1"])
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: sentinel)
    assert jeol_scope._get_scan_controller_detector() is sentinel


def test_set_scan_pixel_dwell_raises_not_implemented(jeol_scope):
    """Verify that `set_scan_pixel_dwell` raises not implemented."""
    with pytest.raises(NotImplementedError):
        jeol_scope.set_scan_pixel_dwell(Q_(1, "us"))

def test_set_scan_flyback_raises_not_implemented_error(jeol_scope):
    """Verify that `set_scan_flyback` raises not implemented error."""
    with pytest.raises(NotImplementedError, match=r"flyback"):
        jeol_scope.set_scan_flyback(Q_(10, Units.US))


# ============================================================================
# VACUUM & APERTURE
# ============================================================================

def test_get_pressure_prefers_pig_info(jeol_scope):
    """Verify that `get_pressure_prefers` pig info."""
    jeol_scope.vac.GetPigInfo.return_value = [1.23]
    jeol_scope.vac.GetPegInfo.return_value = [9.99]
    p = jeol_scope.get_pressure("P1")
    assert p is not None
    assert float(p.to(Units.PA).magnitude) == pytest.approx(1.23)

def test_get_pressure_falls_back_to_peg_info(jeol_scope):

    """Verify that `get_pressure` falls back to peg info."""
    jeol_scope.vac.GetPigInfo.return_value = []
    jeol_scope.vac.GetPegInfo.return_value = [4.56]
    p = jeol_scope.get_pressure("P4")
    assert p is not None
    assert float(p.to(Units.PA).magnitude) == pytest.approx(4.56)

def test_get_pressure_uses_peg_when_pig_missing(jeol_scope):
    """Verify that `get_pressure` uses peg when pig missing."""
    jeol_scope.vac = SimpleNamespace(GetPegInfo=MagicMock(return_value=[123.0]))

    p = jeol_scope.get_pressure('P4')
    assert p is not None
    assert p.to(Units.PA).magnitude == pytest.approx(123.0)


@pytest.mark.parametrize(
    "hw_return, valve_name, expected_state",
    [
        ([10, 0b01], "column", "OPEN"),
        ([10, 0b01], "turbo", "CLOSED"),
        ([0, 0b01], "column", "OPEN"),
        ([0, 0b11], "turbo", "OPEN"),
        ([10, 0b00], "column", "CLOSED"),
    ],
    ids=["count10_col_open", "count10_turbo_closed", "count0_col_open", "count0_turbo_open", "count10_col_closed"],
)
def test_get_valve_state_decodes_bitfield(jeol_scope, hw_return, valve_name, expected_state):
    """Verify that `get_valve_state` decodes bitfield."""
    jeol_scope.vac.GetValveStatus.return_value = hw_return
    assert jeol_scope.get_valve_state(valve_name) == expected_state

def test_get_valve_state_returns_unknown_when_status_malformed(jeol_scope):
    """Verify that `get_valve_state` returns unknown when status malformed."""
    jeol_scope.vac.GetValveStatus.return_value = [10]
    assert jeol_scope.get_valve_state("column") == "UNKNOWN"

def test_get_valve_state_returns_unknown_on_exception(jeol_scope):
    """Verify that `get_valve_state` returns unknown on exception."""
    jeol_scope.vac.GetValveStatus.side_effect = RuntimeError("vac fail")
    assert jeol_scope.get_valve_state("column") == "UNKNOWN"

def test_get_valve_state_reads_gun_hardware(jeol_scope):
    """Verify that `get_valve_state` reads gun hardware."""
    jeol_scope.gun.GetBeamValve.return_value = 1
    assert jeol_scope.get_valve_state("gun") == "OPEN"

    jeol_scope.gun.GetBeamValve.return_value = 0
    assert jeol_scope.get_valve_state("gun") == "CLOSED"

def test_set_valve_state_gun_calls_beam_valve(jeol_scope):
    """Verify that `set_valve_state_gun` calls beam valve."""
    jeol_scope.set_valve_state("gun", "OPEN")
    jeol_scope.gun.SetBeamValve.assert_called_with(1)

def test_set_valve_state_gun_only(jeol_scope):
    """Verify that `set_valve_state` gun only."""
    jeol_scope.gun.SetBeamValve = MagicMock()
    jeol_scope.set_valve_state("gun", "OPEN")
    jeol_scope.gun.SetBeamValve.assert_called_with(1)
    jeol_scope.set_valve_state("gun", "CLOSED")
    jeol_scope.gun.SetBeamValve.assert_called_with(0)

    with pytest.raises(NotImplementedError):
        jeol_scope.set_valve_state("turbo", "OPEN")

def test_set_valve_state_raises_not_implemented_for_unsupported_valves(jeol_scope):
    """Verify that `set_valve_state` raises not implemented for unsupported valves."""
    with pytest.raises(NotImplementedError, match=r"not supported"):
        jeol_scope.set_valve_state("column", "OPEN")

def test_set_valve_state_raises_runtime_error_when_hardware_fails(jeol_scope):
    """Verify that `set_valve_state` raises runtime error when hardware fails."""
    jeol_scope.gun.SetBeamValve.side_effect = RuntimeError("gun valve fail")
    with pytest.raises(RuntimeError, match=r"gun valve fail"):
        jeol_scope.set_valve_state("gun", "OPEN")

def test_set_valve_state_raises_when_gun_hardware_missing(jeol_scope):
    """Verify that `set_valve_state` raises when gun hardware missing."""
    jeol_scope.gun = None
    with pytest.raises(RuntimeError, match=r"Gun hardware not connected"):
        jeol_scope.set_valve_state("gun", "OPEN")

def test_get_valve_state_returns_unknown_when_hardware_missing(jeol_scope):
    """Verify that `get_valve_state` returns unknown when hardware missing."""

    jeol_scope.vac.GetValveStatus = MagicMock(return_value=None)
    assert jeol_scope.get_valve_state("column") == "UNKNOWN"


    jeol_scope.vac = None
    assert jeol_scope.get_pressure("P1") is None


def test_aperture_get_set_and_invalid_id(monkeypatch, jeol_scope):
    """Verify that `aperture_get_set` and invalid id."""

    fake_ap = Aperture(size_index=2, position=Point(10, 20))
    monkeypatch.setattr(jeol_module.jeol_adapter, "from_jeol_aperture", MagicMock(return_value=fake_ap))

    jeol_scope.apt.SelectExpKind = MagicMock()
    jeol_scope.apt.GetExpSize = MagicMock(return_value=2)
    jeol_scope.apt.GetPosition = MagicMock(return_value=[10, 20])

    ap = jeol_scope.get_aperture("CLA")
    assert ap is not None
    assert ap.size_index == 2
    assert ap.position.x == 10


    jeol_scope.apt.SetExpSize = MagicMock()
    jeol_scope.apt.SetPosition = MagicMock()
    jeol_scope.set_aperture("CLA", Aperture(size_index=3, position=Point(1, 2)))
    jeol_scope.apt.SetExpSize.assert_called()
    jeol_scope.apt.SetPosition.assert_called_with(1, 2)

    with pytest.raises(ValueError):
        jeol_scope.set_aperture("NOT_REAL", Aperture(size_index=1))

def test_get_aperture_returns_position_and_size(jeol_scope):
    """Verify that `get_aperture` returns position and size."""
    jeol_scope.apt.GetExpSize.return_value = 2
    jeol_scope.apt.GetPosition.return_value = [1000, -1000]

    apt = jeol_scope.get_aperture("CLA")
    assert apt.size_index == 2
    assert apt.position.x == pytest.approx(1000.0)

    jeol_scope.set_aperture("CLA", Aperture(size_index=1, position=Point(x=0, y=0)))
    jeol_scope.apt.SetExpSize.assert_called_with(1, 1)
    jeol_scope.apt.SetPosition.assert_called_with(0, 0)

def test_set_aperture_invalid_id_raises_value_error(jeol_scope):
    """Verify that `set_aperture_invalid_id` raises value error."""
    with pytest.raises(ValueError):
        jeol_scope.set_aperture("BAD_APT", Aperture(size_index=1))

def test_set_aperture_reraises_on_hardware_failure(jeol_scope):
    """Verify that `set_aperture_reraises` on hardware failure."""
    jeol_scope.apt.SelectExpKind.side_effect = RuntimeError("apt fail")
    with pytest.raises(RuntimeError, match=r"apt fail"):
        jeol_scope.set_aperture("CLA", Aperture(size_index=1, position=Point(10, 20)))

def test_get_aperture_returns_none_when_hardware_missing(jeol_scope):
    """Verify that `get_aperture` returns None when hardware missing."""
    jeol_scope.apt = None
    assert jeol_scope.get_aperture("CLA") is None


    with pytest.raises(RuntimeError):
        jeol_scope.set_aperture("CLA", Aperture(size_index=1))


# ============================================================================
# DETECTOR & ACQUISITION
# ============================================================================

def test_get_detector_binning_returns_value(jeol_scope):
    """Verify that `get_detector_binning` returns value."""
    det = jeol_scope._get_detector("Camera1")
    det.get_detectorsetting.return_value = {
        "BinningIndex": 2,
        "frameIntegration": 4,
        "ImagingArea": {"Width": 256, "Height": 256, "X": 128, "Y": 128}
    }
    assert jeol_scope.get_detector_binning("Camera1") == 2
    assert jeol_scope.get_detector_integration("Camera1") == 4

def test_get_detector_inserted_returns_true(jeol_scope):
    """Verify that `get_detector_inserted` returns true."""
    det = jeol_scope._get_detector("Camera1")
    det.get_insert_state.return_value = {"Status": "IN"}
    assert jeol_scope.get_detector_inserted("Camera1") is True

@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"InsertState": "IN"}, True),
        ({"Status": "IN"}, True),
        ({"InsertState": "OUT"}, False),
        ({"Status": "OUT"}, False),
        ({"state": False}, False),
    ],
)
def test_get_detector_inserted_parses_common_payload_shapes(monkeypatch, jeol_scope, payload, expected):
    """Verify that `get_detector_inserted` parses common payload shapes."""
    monkeypatch.setattr(jeol_module, "detector", object(), raising=False)

    d = SimpleNamespace(get_insert_state=MagicMock(return_value=payload))
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda det_id: d)

    assert jeol_scope.get_detector_inserted("Camera1") is expected

def test_detector_getters_use_adapter_and_insert_state_parsing(monkeypatch, jeol_scope):
    """Verify that `detector_getters_use_adapter` and insert state parsing."""
    d = jeol_scope._get_detector("Camera1")
    jeol_scope._primary_detector_id = "Camera1"
    d.get_detectorsetting.return_value = {"dummy": 1}

    ds = DetectorSettings(
        detector_id="Camera1",
        exposure=Q_(5, Units.MS),
        binning_index=2,
        frame_integration=3,
        roi=ROI(x=1, y=2, width=3, height=4),
        _mode=ParseMode.LENIENT,
    )

    monkeypatch.setattr(
        jeol_module.jeol_adapter,
        "from_jeol_detector_response",
        lambda payload, detector_id: (ds, MagicMock()),
    )

    exp = jeol_scope.get_detector_exposure("Camera1")
    assert exp is not None
    assert float(exp.to(Units.MS).magnitude) == pytest.approx(5.0)

    assert jeol_scope.get_detector_binning("Camera1") == 2

    roi = jeol_scope.get_detector_roi("Camera1")
    assert roi is not None
    assert (roi.x, roi.y, roi.width, roi.height) == (1, 2, 3, 4)

    assert jeol_scope.get_detector_integration("Camera1") == 3


    d.get_insert_state = MagicMock(return_value={"InsertState": "OUT"})
    assert jeol_scope.get_detector_inserted("Camera1") is False

    d.get_insert_state = MagicMock(return_value={"InsertState": "IN"})
    assert jeol_scope.get_detector_inserted("Camera1") is True

    d.get_insert_state = MagicMock(return_value={"InsertState": 0})
    assert jeol_scope.get_detector_inserted("Camera1") is False

def test_detector_getters_parse_adapter_response(monkeypatch, jeol_scope):
    """Verify that `detector_getters_parse` adapter response."""
    det = jeol_scope._get_detector("Camera1")
    det.get_detectorsetting.return_value = {"Anything": "ok"}

    fake = DetectorSettings(
        exposure=Q_(0.2, Units.SEC),
        binning_index=2,
        roi=ROI(x=1, y=2, width=3, height=4),
        frame_integration=7,
    )

    def _fake_from_resp(payload, detector_id):
        assert detector_id == "Camera1"
        return fake, Extras()

    monkeypatch.setattr(jeol_module.jeol_adapter, "from_jeol_detector_response", _fake_from_resp)

    exp = jeol_scope.get_detector_exposure("Camera1")
    assert exp is not None
    assert float(exp.to(Units.SEC).magnitude) == pytest.approx(0.2)
    assert jeol_scope.get_detector_binning("Camera1") == 2
    assert jeol_scope.get_detector_roi("Camera1") == fake.roi
    assert jeol_scope.get_detector_integration("Camera1") == 7


def test_set_detector_exposure_calls_hardware(jeol_scope):
    """Verify that `set_detector_exposure` calls hardware."""
    det = jeol_scope._get_detector("Camera1")
    jeol_scope.set_detector_exposure("Camera1", Q_(100, "ms"))
    det.set_exposuretime_value.assert_called_with(100000)

def test_set_detector_roi_casts_to_int(jeol_scope):
    """Verify that `set_detector_roi_casts` to int."""
    det = jeol_scope._get_detector('Camera1')
    det.set_areamode_imagingarea = MagicMock()


    roi = ROI(x=0, y=1, width=32, height=16)
    jeol_scope.set_detector_roi('Camera1', roi)

    args, _ = det.set_areamode_imagingarea.call_args
    assert all(isinstance(v, int) for v in args[:4])

def test_set_detector_exposure_prefers_value_method(jeol_scope):
    """Verify that `set_detector_exposure` prefers value method."""
    d = jeol_scope._get_detector("Camera1")
    d.set_exposuretime_value = MagicMock()

    d.set_exposuretime_index = MagicMock()

    jeol_scope.set_detector_exposure("Camera1", Q_(2, Units.MS))
    d.set_exposuretime_value.assert_called_with(2000)
    d.set_exposuretime_index.assert_not_called()


def test_set_detector_exposure_falls_back_to_index_method_when_value_missing(jeol_scope):
    """Verify that `set_detector_exposure` falls back to index method when value missing."""
    d = jeol_scope._get_detector("Camera1")

    if hasattr(d, "set_exposuretime_value"):
        delattr(d, "set_exposuretime_value")
    d.set_exposuretime_index = MagicMock()

    jeol_scope.set_detector_exposure("Camera1", Q_(3, Units.MS))
    d.set_exposuretime_index.assert_called_with(3000)


@pytest.mark.parametrize(
    "api_method, hw_method, args, expected_call",
    [
        ("set_detector_binning", "set_binningindex", (4,), 4),
        ("set_detector_integration", "set_frameintegration", (7,), 7),
    ],
    ids=["binning", "integration"],
)
def test_detector_simple_setters_call_supported_methods(jeol_scope, api_method, hw_method, args, expected_call):
    """Verify that `detector_simple_setters` call supported methods."""
    d = jeol_scope._get_detector("Camera1")
    setattr(d, hw_method, MagicMock())

    getattr(jeol_scope, api_method)("Camera1", *args)
    getattr(d, hw_method).assert_called_with(expected_call)


def test_set_detector_roi_calls_imaging_area_and_skips_when_none(jeol_scope):
    """Verify that `set_detector_roi` calls imaging area and skips when none."""
    d = jeol_scope._get_detector("Camera1")
    d.set_areamode_imagingarea = MagicMock()

    jeol_scope.set_detector_roi("Camera1", ROI(x=1, y=2, width=10, height=20))
    d.set_areamode_imagingarea.assert_called_with(10, 20, 1, 2)

    d.set_areamode_imagingarea.reset_mock()
    jeol_scope.set_detector_roi("Camera1", None)
    d.set_areamode_imagingarea.assert_not_called()


@pytest.mark.parametrize(
    "inserted, expected_method",
    [(True, "insert"), (False, "retract")],
    ids=["insert", "retract"],
)
def test_set_detector_insertion_uses_insert_or_retract(jeol_scope, inserted, expected_method):
    """Verify that `set_detector_insertion` uses insert or retract."""
    d = jeol_scope._get_detector("Camera1")
    d.insert = MagicMock()
    d.retract = MagicMock()

    jeol_scope.set_detector_insertion("Camera1", inserted)
    getattr(d, expected_method).assert_called_once()

def test_execute_detector_control_runs_without_capability_check(monkeypatch, jeol_scope):
    """Verify that `execute_detector_control_runs` without capability check."""

    sys_settings = getattr(jeol_scope, 'system_settings', None)
    if sys_settings and sys_settings.detector_system:
        sys_settings.detector_system.is_supported = MagicMock(return_value=True)

    det = jeol_scope._get_detector('Camera1')
    det.insert = MagicMock()

    req = DetectorControlRequest(
        detector_id='Camera1',
        action='INSERT',
        target=DetectorSettings(detector_id='Camera1', exposure=Q_(2, Units.MS)),
    )
    jeol_scope.execute_detector_control(req)
    det.insert.assert_called_once()


def test_acquire_image_flow_returns_image_object(jeol_scope):
    """Verify that `acquire_image_flow` returns image object."""
    det = jeol_scope._get_detector("Camera1")


    det.get_detectorsetting.return_value = {
        "BinningIndex": 1,
        "ExposureTimeValue": 100.0,
        "ImagingArea": {"Width": 512, "Height": 512}
    }


    det.snapshot_rawdata.return_value = np.zeros((512, 512), dtype=np.uint16).tolist()

    req = AcquisitionRequest(detector_id="Camera1")
    img = jeol_scope.acquire_image(req)

    assert img.data.shape == (512, 512)
    assert img.metadata.exposure_ms == pytest.approx(100.0)

def test_acquire_image_reshaping(jeol_scope):
    """Verify acquire_image_reshaping behavior."""
    det = jeol_scope._get_detector("Camera1")


    det.snapshot_rawdata.return_value = np.zeros(100, dtype=np.uint16).tolist()


    roi = ROI(x=0, y=0, width=10, height=10)
    det.get_detectorsetting.return_value = {"ImagingArea": {"Width": 10, "Height": 10}}

    req = AcquisitionRequest(detector_id="Camera1", detector=DetectorSettings(roi=roi))
    img = jeol_scope.acquire_image(req)

    assert img.data.shape == (10, 10)

def test_acquire_image_with_roi_sets_imaging_area(jeol_scope):
    """Verify that `acquire_image` with ROI sets imaging area."""
    det = jeol_scope._get_detector("Camera1")
    det.snapshot_rawdata.return_value = np.zeros((10, 10)).tolist()

    req = AcquisitionRequest(
        detector_id="Camera1",
        detector=DetectorSettings(roi=ROI(x=0, y=0, width=10, height=10))
    )
    img = jeol_scope.acquire_image(req)
    assert img.data.shape == (10, 10)
    det.set_areamode_imagingarea.assert_called()

def test_acquire_image_metadata_failure_recovery_populates_notes(jeol_scope):
    """Verify that `acquire_image_metadata` failure recovery populates notes."""
    det = jeol_scope._get_detector("Camera1")
    det.snapshot_rawdata.return_value = np.zeros((128, 128), dtype=np.uint16).tolist()
    det.get_detectorsetting.return_value = {}


    with patch.object(jeol_scope, 'get_full_state', side_effect=RuntimeError("Stage Timeout")):
        req = AcquisitionRequest(detector_id="Camera1")
        img = jeol_scope.acquire_image(req)


    assert img.data.shape == (128, 128)


    notes = img.metadata.extra.notes
    assert "MicroscopeImageMetadata.state_capture_failed" in notes
    assert "Stage Timeout" in notes["MicroscopeImageMetadata.state_capture_failed"]

def test_acquire_image_state_capture_failure_is_recorded(monkeypatch, jeol_scope):
    """Verify that `acquire_image_state` capture failure is recorded."""

    assert jeol_module.detector is not None


    d = jeol_scope._get_detector('Camera1')
    d.snapshot_rawdata = MagicMock(return_value=[[1, 2], [3, 4]])


    monkeypatch.setattr(jeol_scope, 'get_full_state', MagicMock(side_effect=RuntimeError('boom')))

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id='Camera1'))
    assert img.data.shape == (2, 2)

    notes = getattr(img.metadata.extra, 'notes', {})
    assert 'MicroscopeImageMetadata.state_capture_failed' in notes

def test_acquire_image_raw_none_returns_1x1(jeol_scope):
    """Verify that `acquire_image_raw_none` returns 1x1."""
    det = jeol_scope._get_detector("Camera1")
    det.snapshot_rawdata.return_value = None
    det.get_detectorsetting.return_value = {}

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id="Camera1"))
    assert img.data.shape == (1, 1)

def test_acquire_image_raw_bytes_converts(jeol_scope):
    """Verify that `acquire_image_raw_bytes` converts."""
    det = jeol_scope._get_detector("Camera1")

    raw = (np.array([1, 2, 3, 4], dtype=np.uint16)).tobytes()
    det.snapshot_rawdata.return_value = raw
    det.get_detectorsetting.return_value = {"ImagingArea": {"Width": 2, "Height": 2}}

    req = AcquisitionRequest(detector_id="Camera1", detector=DetectorSettings(roi=ROI(x=0, y=0, width=2, height=2)))
    img = jeol_scope.acquire_image(req)
    assert img.data.shape == (2, 2)
    assert img.data.dtype == np.uint16

def test_acquire_image_squeezes_singleton_3d_arrays(jeol_scope):
    """Verify that `acquire_image_squeezes` singleton 3d arrays."""
    det = jeol_scope._get_detector("Camera1")
    det.snapshot_rawdata.return_value = np.zeros((1, 8, 8), dtype=np.uint16).tolist()
    det.get_detectorsetting.return_value = {}

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id="Camera1"))
    assert img.data.shape == (8, 8)

def test_acquire_image_squeezes_last_dimension(monkeypatch, jeol_scope):
    """Verify that `acquire_image_squeezes` last dimension."""
    monkeypatch.setattr(jeol_module, "detector", object())

    raw = {"data": np.zeros((5, 4, 1), dtype=np.uint16)}
    d = MagicMock()
    d.snapshot_rawdata.return_value = raw
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)

    req = AcquisitionRequest(detector_id="Camera1", _mode="lenient")
    img = jeol_scope.acquire_image(req)
    assert img.data.shape == (5, 4)
    assert img.data.ndim == 2

def test_acquire_image_bytes_uint16_to_uint8_fallback(monkeypatch, jeol_scope):
    """Verify that `acquire_image_bytes_uint16` to uint8 fallback."""
    monkeypatch.setattr(jeol_module, "detector", object())

    class BadAstype(np.ndarray):
        def astype(self, *args, **kwargs):
            raise TypeError("astype blocked")

    class Raw:
        def __array__(self, *_a, **_k):
            return np.array([1, 2, 3, 4], dtype=np.int32).view(BadAstype)

    d = MagicMock()
    d.snapshot_rawdata.return_value = Raw()
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)
    monkeypatch.setattr(jeol_scope, "get_detector_roi", lambda _id: None)

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id="Cam1", _mode="lenient"))
    assert img.data.dtype == np.uint16

def test_acquire_image_raw_bytes_uint8_fallback(monkeypatch, jeol_scope):
    """Verify that `acquire_image_raw` bytes uint8 fallback."""
    monkeypatch.setattr(jeol_module, "detector", object())

    class BadArray:
        def __array__(self, *_a, **_k):
            raise ValueError("boom")

    d = MagicMock()
    d.snapshot_rawdata.return_value = BadArray()
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)
    monkeypatch.setattr(jeol_scope, "get_detector_roi", lambda _id: None)

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id="Cam1", _mode="lenient"))
    assert img.data.shape == (1, 1)
    assert img.data.dtype == np.uint16

def test_acquire_image_falls_back_to_uint8_on_buffer_error(monkeypatch, jeol_scope):
    """Verify that `acquire_image` falls back to uint8 on buffer error."""
    monkeypatch.setattr(jeol_module, "detector", object(), raising=True)

    raw = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    d = SimpleNamespace(snapshot_rawdata=MagicMock(return_value=raw))
    jeol_scope._get_detector = MagicMock(return_value=d)

    orig_frombuffer = np.frombuffer

    def _frombuffer(buf, dtype=np.uint8, *a, **k):
        if dtype is np.uint16:
            raise ValueError("force uint16 fail")
        return orig_frombuffer(buf, dtype=dtype, *a, **k)

    monkeypatch.setattr(np, "frombuffer", _frombuffer, raising=True)

    req = AcquisitionRequest(detector_id="Camera1", detector=DetectorSettings(_mode="lenient"), _mode="lenient")
    img = jeol_scope.acquire_image(req)

    assert img.data.dtype == np.uint8
    assert img.data.shape[0] >= 1

def test_acquire_image_normalizes_data_formats(monkeypatch, jeol_scope):
    """Verify that `acquire_image` normalizes data formats."""
    assert jeol_module.detector is not None

    roi = ROI(x=0, y=0, width=4, height=3)
    monkeypatch.setattr(jeol_scope, 'get_detector_roi', MagicMock(return_value=roi))

    d = jeol_scope._get_detector('Camera1')


    flat = (np.arange(12, dtype=np.uint16)).tobytes()
    d.snapshot_rawdata = MagicMock(return_value=flat)

    req = AcquisitionRequest(detector_id='Camera1')
    img = jeol_scope.acquire_image(req)
    assert img.data.shape == (3, 4)
    assert img.data.dtype == np.uint16


    d.snapshot_rawdata = MagicMock(return_value={'data': [[1, 2], [3, 4]]})
    img2 = jeol_scope.acquire_image(AcquisitionRequest(detector_id='Camera1'))
    assert img2.data.shape == (2, 2)


    d.snapshot_rawdata = MagicMock(return_value=np.zeros((1, 5, 6), dtype=np.uint16))
    img3 = jeol_scope.acquire_image(AcquisitionRequest(detector_id='Camera1'))
    assert img3.data.shape == (5, 6)

def test_acquire_image_reshapes_flat_array_using_roi(monkeypatch, jeol_scope):
    """Verify that `acquire_image_reshapes` flat array using ROI."""
    monkeypatch.setattr(jeol_module, "detector", object())


    raw_u16 = (np.arange(12, dtype=np.uint16)).tobytes()
    d = MagicMock()
    d.snapshot_rawdata.return_value = raw_u16
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)

    req = AcquisitionRequest(
        detector_id="Camera1",
        detector=DetectorSettings(roi=ROI(x=0, y=0, width=4, height=3), _mode="lenient"),
        _mode="lenient",
    )
    img = jeol_scope.acquire_image(req)
    assert img.data.shape == (3, 4)
    assert img.data.dtype == np.uint16

def test_acquire_image_reshapes_using_request_roi(monkeypatch, jeol_scope):

    """Verify that `acquire_image_reshapes` using request ROI."""
    monkeypatch.setattr(jeol_module, "detector", object(), raising=True)

    raw = np.array([1, 2, 3, 4], dtype=np.uint16).tobytes()
    d = SimpleNamespace(snapshot_rawdata=MagicMock(return_value=raw))
    jeol_scope._get_detector = MagicMock(return_value=d)


    jeol_scope.get_full_state = MagicMock(side_effect=RuntimeError("state fail"))

    roi = ROI(x=0, y=0, width=2, height=2, _mode="lenient")
    req = AcquisitionRequest(detector_id="Camera1", detector=DetectorSettings(roi=roi, _mode="lenient"), _mode="lenient")

    img = jeol_scope.acquire_image(req)

    assert img.data.shape == (2, 2)
    assert "MicroscopeImageMetadata.state_capture_failed" in img.metadata.extra.notes

def test_acquire_image_reshapes_using_hardware_roi(monkeypatch, jeol_scope):
    """Verify that `acquire_image_reshapes` using hardware ROI."""
    monkeypatch.setattr(jeol_module, "detector", object(), raising=True)

    raw = np.array([9, 8, 7, 6], dtype=np.uint16).tobytes()
    d = SimpleNamespace(snapshot_rawdata=MagicMock(return_value=raw))
    jeol_scope._get_detector = MagicMock(return_value=d)

    jeol_scope.get_detector_roi = MagicMock(return_value=ROI(x=0, y=0, width=2, height=2, _mode="lenient"))

    req = AcquisitionRequest(
        detector_id="Camera1",
        detector=DetectorSettings(roi=None, _mode="lenient"),
        _mode="lenient",
    )
    img = jeol_scope.acquire_image(req)

    assert img.data.shape == (2, 2)
    assert jeol_scope.get_detector_roi.call_count >= 1

def test_acquire_image_uses_detector_id_from_settings(monkeypatch, jeol_scope):
    """Verify that `acquire_image` uses detector id from settings."""
    monkeypatch.setattr(jeol_module, "detector", object(), raising=True)

    d = MagicMock()
    d.snapshot_rawdata.return_value = np.zeros((2, 2), dtype=np.uint16)
    jeol_scope._get_detector = MagicMock(return_value=d)

    req = AcquisitionRequest(
        detector_id=None,
        detector=DetectorSettings(detector_id="CamX", _mode="lenient"),
        _mode="lenient",
    )
    img = jeol_scope.acquire_image(req)

    calls = [c.args[0] for c in jeol_scope._get_detector.call_args_list]
    assert "CamX" in calls
    assert img.metadata.extra.vendor["JEOL"]["detector_id"] == "CamX"

def test_acquire_image_uses_request_detector_settings(monkeypatch, jeol_scope):
    """Verify that `acquire_image` uses request detector settings."""
    det = jeol_scope._get_detector("Camera1")


    det.snapshot_rawdata.return_value = np.zeros((8, 8), dtype=np.uint16).tolist()
    det.get_detectorsetting.return_value = {"ImagingArea": {"Width": 8, "Height": 8}}

    jeol_scope.set_detector_exposure = MagicMock()
    jeol_scope.set_detector_binning = MagicMock()
    jeol_scope.set_detector_integration = MagicMock()
    jeol_scope.set_detector_roi = MagicMock()

    req = AcquisitionRequest(
        detector=DetectorSettings(
            detector_id="Camera1",
            exposure=Q_(12, Units.MS),
            binning_index=2,
            frame_integration=3,
            roi=ROI(x=0, y=0, width=8, height=8),
        )
    )

    jeol_scope.acquire_image(req)

    jeol_scope.set_detector_exposure.assert_called_once()
    jeol_scope.set_detector_binning.assert_called_once_with("Camera1", 2)
    jeol_scope.set_detector_integration.assert_called_once_with("Camera1", 3)
    jeol_scope.set_detector_roi.assert_called_once()


def test_acquire_image_falls_back_to_get_image_cache_when_snapshot_missing(monkeypatch, jeol_scope):
    """Verify that `acquire_image` falls back to get image cache when snapshot missing."""
    d = SimpleNamespace(
        get_image_cache=MagicMock(return_value=np.zeros((4, 4), dtype=np.uint16).tolist()),
        get_detectorsetting=MagicMock(return_value={'ImagingArea': {'Width': 4, 'Height': 4}}),
    )
    monkeypatch.setattr(jeol_scope, '_get_detector', lambda _id: d)

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id='Camera1'))

    d.get_image_cache.assert_called_once()
    assert img.data.shape == (4, 4)

def test_acquire_image_falls_back_to_livesnapshot_when_cache_missing(monkeypatch, jeol_scope):
    """Verify that `acquire_image` falls back to livesnapshot when cache missing."""
    d = SimpleNamespace(
        livesnapshot=MagicMock(return_value=np.zeros((4, 4), dtype=np.uint16).tolist()),
        get_detectorsetting=MagicMock(return_value={'ImagingArea': {'Width': 4, 'Height': 4}}),
    )
    monkeypatch.setattr(jeol_scope, '_get_detector', lambda _id: d)

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id='Camera1'))

    d.livesnapshot.assert_called_once_with('tif')
    assert img.data.shape == (4, 4)

def test_acquire_image_uses_get_image_cache_when_snapshot_missing(monkeypatch, jeol_scope):

    """Verify that `acquire_image` uses get image cache when snapshot missing."""
    fake_detector_mod = SimpleNamespace(Detector=lambda *_a, **_k: None)
    monkeypatch.setattr(jeol_module, "detector", fake_detector_mod)

    d = MagicMock()
    d.get_image_cache.return_value = [1, 2, 3, 4]
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)


    bad_roi = ROI(x=0, y=0, width=2, height=2)
    bad_roi.to_dict = MagicMock(side_effect=RuntimeError("nope"))
    monkeypatch.setattr(jeol_scope, "get_detector_roi", lambda _id: bad_roi)


    monkeypatch.setattr(jeol_scope, "get_full_state", MagicMock(side_effect=RuntimeError("state fail")))

    req = AcquisitionRequest(detector_id="Cam1", _mode="lenient")
    img = jeol_scope.acquire_image(req)

    jeol_v = img.metadata.extra.vendor["JEOL"]
    assert jeol_v["detector_id"] == "Cam1"
    assert "roi" in jeol_v and jeol_v["roi"] is None
    assert "MicroscopeImageMetadata.state_capture_failed" in img.metadata.extra.notes

def test_acquire_image_falls_back_to_livesnapshot(monkeypatch, jeol_scope):
    """Verify that `acquire_image` falls back to livesnapshot."""
    class D:
        def livesnapshot(self, fmt):
            assert fmt == "tif"
            return {"data": [[10, 11], [12, 13]]}

    monkeypatch.setattr(jeol_module, "detector", SimpleNamespace())
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda det_id: D())

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id="Camera1"))
    assert img.data.shape == (2, 2)
    assert img.data[0, 0] == 10


def test_refresh_detectors_empty(jeol_scope):
    """Verify refresh_detectors_empty behavior."""

    with patch("supertem.microscopes.jeol_microscope.detector") as mock_det_mod:
        mock_det_mod.get_attached_detector.return_value = []
        mock_det_mod.function = None

        jeol_scope._refresh_detectors()

        assert jeol_scope.list_detectors() == []
        assert jeol_scope.get_primary_detector_id() is None

        with pytest.raises(RuntimeError):
            jeol_scope.acquire_image(AcquisitionRequest(detector_id=None))

def test_refresh_detectors_continues_on_constructor_error(monkeypatch, jeol_scope):
    """Verify that `refresh_detectors_continues` on constructor error."""
    ids = ["BadCam", "GoodCam"]

    def _mk(det_id):
        if det_id == "BadCam":
            raise RuntimeError("nope")
        return MagicMock(name=f"Detector({det_id})")

    fake_detector = SimpleNamespace(
        function=SimpleNamespace(get_attached_detector=MagicMock(return_value=ids)),
        Detector=MagicMock(side_effect=_mk),
    )
    monkeypatch.setattr(jeol_module, "detector", fake_detector, raising=False)

    jeol_scope._refresh_detectors()


    assert "GoodCam" in jeol_scope.list_detectors()
    assert "BadCam" not in jeol_scope.list_detectors()

    assert jeol_scope.get_primary_detector_id() == "BadCam"

def test_refresh_detectors_returns_early_when_module_missing(monkeypatch, jeol_scope):


    """Verify that `refresh_detectors` returns early when module missing."""
    monkeypatch.setattr(jeol_module, "detector", None, raising=True)


    jeol_scope._active_detectors = {"X": object()}
    jeol_scope._primary_detector_id = "X"
    jeol_scope._scan_cfg = {"width_px": 1}

    jeol_scope._refresh_detectors()

    assert jeol_scope._active_detectors == {}
    assert jeol_scope._primary_detector_id is None

    assert jeol_scope._scan_cfg["width_px"] == 512
    assert jeol_scope._scan_cfg["height_px"] == 512

def test_refresh_detectors_ignores_noncallable_getter(monkeypatch, jeol_scope):

    """Verify that `refresh_detectors_ignores` noncallable getter."""
    fake_det = SimpleNamespace(function=SimpleNamespace(get_attached_detector=None), Detector=MagicMock())
    monkeypatch.setattr(jeol_module, 'detector', fake_det, raising=False)
    jeol_scope._refresh_detectors()
    assert jeol_scope.list_detectors() == []

def test_refresh_detectors_continues_on_getter_failure(monkeypatch, jeol_scope):
    """Verify that `refresh_detectors_continues` on getter failure."""
    def _boom():
        raise RuntimeError('nope')

    fake_det = SimpleNamespace(function=SimpleNamespace(get_attached_detector=_boom), Detector=MagicMock())
    monkeypatch.setattr(jeol_module, 'detector', fake_det, raising=False)
    jeol_scope._refresh_detectors()
    assert jeol_scope.list_detectors() == []

def test_get_detector_sets_primary_when_missing(monkeypatch, jeol_scope):

    """Verify that `get_detector` sets primary when missing."""
    jeol_scope._active_detectors.clear()
    jeol_scope._primary_detector_id = None

    d = jeol_scope._get_detector("Camera1")
    assert d is jeol_scope._active_detectors["Camera1"]
    assert jeol_scope._primary_detector_id == "Camera1"

def test_get_detector_function_module_returns_correct_object(monkeypatch, jeol_scope):
    """Verify that `get_detector_function_module` returns correct object."""
    fake_det = SimpleNamespace(function=SimpleNamespace())
    monkeypatch.setattr(jeol_module, "detector", fake_det)
    assert jeol_scope._get_detector_function_module() is fake_det.function

    fake_det2 = SimpleNamespace()
    monkeypatch.setattr(jeol_module, "detector", fake_det2)
    assert jeol_scope._get_detector_function_module() is fake_det2

def test_get_detector_function_module_returns_none_when_missing(monkeypatch, jeol_scope):

    """Verify that `get_detector_function_module` returns None when missing."""
    monkeypatch.setattr(jeol_module, "detector", None)
    assert jeol_scope._get_detector_function_module() is None


def test_detector_module_missing_guards(monkeypatch, jeol_scope):

    """Verify that `detector_module_missing` guards."""
    monkeypatch.setattr(jeol_module, "detector", None, raising=False)

    assert jeol_scope.get_detector_exposure("Camera1") is None
    assert jeol_scope.get_detector_binning("Camera1") is None
    assert jeol_scope.get_detector_roi("Camera1") is None
    assert jeol_scope.get_detector_integration("Camera1") is None


    assert jeol_scope.get_detector_inserted("Camera1") is True

    with pytest.raises(RuntimeError):
        jeol_scope.set_detector_exposure("Camera1", Q_(1, Units.MS))

    with pytest.raises(RuntimeError):
        jeol_scope.set_detector_binning("Camera1", 1)

    with pytest.raises(RuntimeError):
        jeol_scope.set_detector_roi("Camera1", ROI(x=0, y=0, width=1, height=1))

    with pytest.raises(RuntimeError):
        jeol_scope.set_detector_integration("Camera1", 1)

    with pytest.raises(RuntimeError):
        jeol_scope.set_detector_insertion("Camera1", True)

def test_get_detector_inserted_returns_true_on_exception(monkeypatch, jeol_scope):
    """Verify that `get_detector_inserted` returns true on exception."""
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: (_ for _ in ()).throw(RuntimeError("boom")))
    assert jeol_scope.get_detector_inserted("Camera1") is True

def test_acquire_image_raises_when_detector_module_missing(monkeypatch, jeol_scope):
    """Verify that `acquire_image` raises when detector module missing."""
    from supertem.microscopes import jeol_microscope as _jm
    monkeypatch.setattr(_jm, "detector", None)

    with pytest.raises(RuntimeError, match=r"Detector module missing"):
        jeol_scope.acquire_image(AcquisitionRequest(detector_id="Camera1"))

def test_acquire_image_raises_runtime_error_when_no_detector_id(monkeypatch, jeol_scope):
    """Verify that `acquire_image` raises runtime error when no detector id."""
    monkeypatch.setattr(jeol_module, "detector", object())
    monkeypatch.setattr(jeol_scope, "get_primary_detector_id", lambda: None)
    with pytest.raises(RuntimeError, match=r"No detector_id provided"):
        jeol_scope.acquire_image(AcquisitionRequest(detector_id=None))

def test_get_detector_raises_runtime_error_when_module_missing(monkeypatch, jeol_scope):
    """Verify that `get_detector` raises runtime error when module missing."""
    monkeypatch.setattr(jeol_module, 'detector', None, raising=False)
    with pytest.raises(RuntimeError):
        jeol_scope._get_detector('Camera1')

def test_acquire_image_handles_roi_conversion_failure(monkeypatch, jeol_scope):

    """Verify that `acquire_image` handles ROI conversion failure."""
    monkeypatch.setattr(jeol_module, "detector", object())

    d = MagicMock()
    d.snapshot_rawdata.return_value = [1, 2, 3, 4]
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)

    def _boom(_id):
        raise RuntimeError("roi fail")

    monkeypatch.setattr(jeol_scope, "get_detector_roi", _boom)

    img = jeol_scope.acquire_image(AcquisitionRequest(detector_id="Cam1", _mode="lenient"))
    jeol_v = img.metadata.extra.vendor["JEOL"]

    assert "roi" not in jeol_v


@pytest.mark.parametrize(
    "api_method, detector_method, args, msg",
    [
        ("set_detector_exposure", "set_exposuretime_value", ("Camera1", Q_(5, Units.MS)), "set exposure failed"),
        ("set_detector_binning", "set_binningindex", ("Camera1", 2), "set binning failed"),
        ("set_detector_roi", "set_areamode_imagingarea", ("Camera1", ROI(x=0, y=0, width=64, height=64)), "set roi failed"),
        ("set_detector_integration", "set_frameintegration", ("Camera1", 3), "set integration failed"),
        ("set_detector_insertion", "insert", ("Camera1", True), "insert failed"),
    ],
)
def test_detector_setters_reraise_detector_exceptions(monkeypatch, api_method, detector_method, args, msg, jeol_scope):
    """Verify that `detector_setters_reraise` detector exceptions."""
    d = MagicMock()
    getattr(d, detector_method).side_effect = RuntimeError(msg)
    monkeypatch.setattr(jeol_scope, "_get_detector", lambda _id: d)

    fn = getattr(jeol_scope, api_method)
    with pytest.raises(RuntimeError, match=msg):
        fn(*args)


# ============================================================================
# HELPERS & ERROR SAFETY
# ============================================================================

def test_setters_fail_loudly_when_hardware_missing(jeol_scope):
    """Verify that `setters_fail_loudly` when hardware missing."""
    jeol_scope.def_ = None
    with pytest.raises(RuntimeError):
        jeol_scope.set_beam_shift(0, 0)

    jeol_scope.eos = None
    with pytest.raises(RuntimeError):
        jeol_scope.set_magnification(1000)

def test_getters_return_none_when_disconnected(jeol_scope):
    """Verify that `getters_return_none` when disconnected."""
    jeol_scope.def_ = None
    assert jeol_scope.get_beam_shift() == (None, None)

    jeol_scope.vac = None
    assert jeol_scope.get_pressure("P1") is None

def test_helper_coerce_xy_returns_none_on_malformed_input(jeol_scope):
    """Verify that `helper_coerce_xy` returns None on malformed input."""
    assert jeol_scope._coerce_xy(['a', 'b']) is None
    assert jeol_scope._coerce_xy([1]) is None

@pytest.mark.parametrize(
    "helper_name, data, keys, expected",
    [
        ("_first_int", {"Width": "nope", "ImagingAreaWidth": "256"}, ("Width", "ImagingAreaWidth"), 256),
        ("_first_float", {"a": "bad", "b": "1.25"}, ("a", "b"), 1.25),
    ],
)
def test_first_int_float_helpers_skip_invalid_values(helper_name, data, keys, expected):
    """Verify that `first_int_float` helpers skip invalid values."""
    helper = getattr(jeol_module.JeolMicroscope, helper_name)
    assert helper(data, keys) == expected
