import pytest
from unittest.mock import MagicMock
from dataclasses import replace
from typing import Optional, Tuple, List, Dict

from supertem.microscope import TemMicroscope
from supertem.structures.base import (
    MicroscopeSettings, SystemSettings, SystemInfo, SafetyCheck,
    StageSystemSettings, BeamSystemSettings, ScanSystemSettings, DetectorSystemSettings,
    StagePosition, StageMoveRequest, StageControlRequest,
    BeamSettings, BeamControlRequest,
    ProjectionSettings, ProjectionControlRequest,
    ScanSettings, ScanControlRequest,
    DetectorSettings, DetectorControlRequest, DetectorCapabilities, ROI,
    Aperture, ApertureControlRequest, Point,
    VacuumSettings, VacuumControlRequest,
    Q_, Units, ParseMode, StageDriveType
)




# =============================================================================
# 1. MOCK IMPLEMENTATION
#    TemMicroscope is abstract; we use a concrete in-memory implementation to
#    test the base-class orchestration logic.
# =============================================================================


class MockMicroscope(TemMicroscope):
    """Minimal concrete TEM implementation for unit testing."""

    def __init__(self, settings: MicroscopeSettings):
        super().__init__(settings)
        self.connected = False

        # Simulated Hardware State
        self.hw_state = {
            "stage": StagePosition(
                x=Q_(0, 'nm'), y=Q_(0, 'nm'), z=Q_(0, 'nm'),
                r=Q_(0, 'deg'), tilt_x=Q_(0, 'deg'), tilt_y=Q_(0, 'deg')
            ),
            "beam": BeamSettings(voltage=Q_(200, 'kV'), beam_current=Q_(100, 'pA')),
            "scan": {"active": False, "mode": "Spot"},
            "detectors": {"CamA": {"inserted": False, "exposure": Q_(100, 'ms'), "roi": None}},
            "apertures": {"CLA": Aperture(aperture_id="CLA", position=Point(x=0, y=0))},
            "valves": {"column": "CLOSED", "gun": "CLOSED", "turbo": "OFF"},
        }

        # Call tracking
        self.move_history = []

    # --- Connection ---
    def connect(self, host: str, port: Optional[int] = None, **kwargs) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_instrument_info(self) -> SystemInfo:
        return SystemInfo(model="MockTEM")

    def get_mode(self) -> str:
        return "TEM"

    def set_mode(self, mode: str) -> None:
        pass

    # --- Stage Atomic ---
    def get_stage_position(self) -> Optional[StagePosition]:
        return self.hw_state["stage"]

    def move_stage_absolute(self, target: StagePosition, drive_type: str = "default",
                            wait: bool = True, **kwargs) -> None:
        current = self.hw_state["stage"]
        new_pos = StagePosition(
            x=target.x if target.x is not None else current.x,
            y=target.y if target.y is not None else current.y,
            z=target.z if target.z is not None else current.z,
            r=target.r if target.r is not None else current.r,
            tilt_x=target.tilt_x if target.tilt_x is not None else current.tilt_x,
            tilt_y=target.tilt_y if target.tilt_y is not None else current.tilt_y,
        )
        self.hw_state["stage"] = new_pos
        self.move_history.append(new_pos)

    def stop_stage(self) -> None:
        pass

    def home_stage(self) -> None:
        pass

    # --- Beam Atomic ---
    def get_acceleration_voltage(self):
        return self.hw_state["beam"].voltage

    def get_beam_current(self):
        return self.hw_state["beam"].beam_current

    def get_spot_size(self):
        return 1

    def get_convergence_angle(self):
        return Q_(10, 'mrad')

    def get_beam_shift(self):
        return (0.0, 0.0)

    def get_condenser_stigmation(self):
        return (0.0, 0.0)

    def get_gun_tilt(self):
        return (0.0, 0.0)

    def get_beam_blank(self):
        return False

    def set_acceleration_voltage(self, v):
        self.hw_state["beam"].voltage = v

    def set_beam_current(self, c):
        self.hw_state["beam"].beam_current = c

    def set_spot_size(self, i):
        pass

    def set_convergence_angle(self, a):
        pass

    def set_beam_shift(self, x, y):
        pass

    def set_condenser_stigmation(self, x, y):
        pass

    def set_gun_tilt(self, x, y):
        pass

    def set_beam_blank(self, b):
        pass

    # --- Projection Atomic ---
    def get_projection_mode(self):
        return "IMAGING"

    def get_magnification(self):
        return 50000

    def get_camera_length(self):
        return None

    def get_defocus(self):
        return Q_(0, 'nm')

    def get_screen_position(self):
        return "DOWN"

    def get_objective_stigmation(self):
        return (0.0, 0.0)

    def get_image_shift(self):
        return (0.0, 0.0)

    def get_diffraction_shift(self):
        return (0.0, 0.0)

    def set_projection_mode(self, m):
        pass

    def set_magnification(self, i):
        pass

    def set_camera_length(self, l):
        pass

    def set_defocus(self, d):
        pass

    def set_screen_position(self, p):
        pass

    def set_objective_stigmation(self, x, y):
        pass

    def set_image_shift(self, x, y):
        pass

    def set_diffraction_shift(self, x, y):
        pass

    # --- Scan Atomic ---
    def get_scan_mode(self):
        return self.hw_state["scan"]["mode"]

    def get_scan_width(self):
        return 512

    def get_scan_height(self):
        return 512

    def get_scan_pixel_dwell(self):
        return Q_(10, 'us')

    def get_scan_flyback(self):
        return Q_(0, 'us')

    def get_scan_rotation(self):
        return Q_(0, 'deg')

    def get_scan_active(self):
        return self.hw_state["scan"]["active"]

    def set_scan_mode(self, m):
        self.hw_state["scan"]["mode"] = m

    def set_scan_width(self, w):
        pass

    def set_scan_height(self, h):
        pass

    def set_scan_pixel_dwell(self, t):
        pass

    def set_scan_flyback(self, t):
        pass

    def set_scan_rotation(self, r):
        pass

    def set_scan_active(self, a):
        self.hw_state["scan"]["active"] = a

    # --- Detector Atomic ---
    def list_detectors(self):
        return list(self.hw_state["detectors"].keys())

    def get_active_detector_ids(self):
        return []

    def get_primary_detector_id(self):
        return "CamA"

    def get_detector_exposure(self, d_id):
        return self.hw_state["detectors"][d_id]["exposure"]

    def get_detector_binning(self, d_id):
        return 1

    def get_detector_roi(self, d_id):
        return self.hw_state["detectors"][d_id].get("roi")

    def get_detector_integration(self, d_id):
        return 1

    def get_detector_inserted(self, d_id):
        return self.hw_state["detectors"][d_id]["inserted"]

    def get_detector_frame_rate(self, d_id):
        return Q_(10, 'Hz')

    def set_detector_exposure(self, d_id, t):
        self.hw_state["detectors"][d_id]["exposure"] = t

    def set_detector_binning(self, d_id, i):
        pass

    def set_detector_roi(self, d_id, r):
        self.hw_state["detectors"][d_id]["roi"] = r

    def set_detector_integration(self, d_id, i):
        pass

    def set_detector_insertion(self, d_id, b):
        self.hw_state["detectors"][d_id]["inserted"] = b

    def acquire_image(self, req):
        return MagicMock()

    # --- Vacuum/Aperture ---
    def get_valve_state(self, n):
        return self.hw_state["valves"].get(n, "UNKNOWN")

    def set_valve_state(self, n, s):
        self.hw_state["valves"][n] = s

    def get_pressure(self, n):
        return Q_(1e-5, 'Pa')

    def list_apertures(self):
        return list(self.hw_state["apertures"].keys())

    def get_aperture(self, aperture_id: str) -> Optional[Aperture]:
        return self.hw_state["apertures"].get(aperture_id)

    def set_aperture(self, aperture_id: str, target: Aperture) -> None:
        self.hw_state["apertures"][aperture_id] = target

# =============================================================================
# 2. TEST FIXTURES
# =============================================================================

@pytest.fixture
def basic_settings():
    """Returns a permissive configuration."""
    return MicroscopeSettings(
        system=SystemSettings(
            stage_system=StageSystemSettings(
                enabled=True,
                can_x=True, can_y=True, can_z=True,
                x_limits=(Q_(-100, 'um'), Q_(100, 'um')),
                y_limits=(Q_(-100, 'um'), Q_(100, 'um')),
                z_limits=(Q_(-100, 'um'), Q_(100, 'um')),
                max_step_distance=Q_(50, 'um')  # Large step allowed
            ),
            beam_system=BeamSystemSettings(
                voltage_limits=(Q_(80, 'kV'), Q_(300, 'kV'))
            ),
            detector_system=DetectorSystemSettings(
                available_detector_ids=["CamA"],
                capabilities_by_id={
                    "CamA": DetectorCapabilities(
                        exposure_min=Q_(1, 'ms'),
                        exposure_max=Q_(1000, 'ms')
                    )
                }
            )
        )
    )


@pytest.fixture
def mock_scope(basic_settings):
    """Returns an instantiated MockMicroscope."""
    return MockMicroscope(basic_settings)


# =============================================================================
# 2b. BASE CLASS INTERNAL HELPERS
# =============================================================================


def test_init_default_settings_when_none():
    """TemMicroscope should create safe default settings when none are provided."""

    class _InitNoneMicroscope(MockMicroscope):
        def __init__(self):
            # Intentionally bypass MockMicroscope.__init__ to hit TemMicroscope's default-settings branch.
            TemMicroscope.__init__(self, settings=None)

    scope = _InitNoneMicroscope()
    assert scope.system_settings is not None
    assert isinstance(scope.system_settings, SystemSettings)


def test_internal_helpers_summarize_patch_and_require_point_complete():
    """Exercise helper utilities used by control orchestration (intent logging and Point validation)."""
    assert TemMicroscope._summarize_extras(None) == ""

    extra = MagicMock()
    extra.vendor = {
        "JEOL": {"alpha_index": 3, "unused": None},
        "Thermo": 5,
        "Empty": {"x": None},
    }
    extra.unknown = {"mystery": 123, "noop": None}
    s = TemMicroscope._summarize_extras(extra)
    assert "vendor.JEOL(alpha_index)" in s
    assert "vendor.Thermo" in s
    assert "unknown(mystery)" in s

    assert TemMicroscope._summarize_patch(None) == "<none>"
    assert TemMicroscope._summarize_patch(BeamSettings()) == "<empty>"

    patch = BeamSettings(voltage=Q_(200, "kV"))
    patch.extra = extra
    s2 = TemMicroscope._summarize_patch(patch)
    assert "voltage" in s2
    assert "vendor.JEOL(alpha_index)" in s2
    p = Point(x=1.0, y=2.0)
    p.y = None  # simulate a partially-specified Point after construction
    with pytest.raises(ValueError, match="requires both x and y"):
        TemMicroscope._require_point_complete(p, "beam_shift")
    TemMicroscope._require_point_complete(Point(x=1.0, y=2.0), "beam_shift")


def test_require_point_complete_accepts_none():
    TemMicroscope._require_point_complete(None, "probe_shift")



# =============================================================================
# 3. STAGE CONTROL TESTS
# =============================================================================

def test_stage_move_absolute(mock_scope):
    """Test standard absolute movement."""
    target = StagePosition(x=Q_(10, 'um'), y=Q_(20, 'um'))
    req = StageMoveRequest(target=target, relative=False)

    mock_scope.execute_stage_move(req)

    # Check internal state update
    current = mock_scope.get_stage_position()
    assert current.x.magnitude == pytest.approx(10000.0)  # converted to nm
    assert current.y.magnitude == pytest.approx(20000.0)


def test_stage_move_relative(mock_scope):
    """Test relative movement calculation."""
    # 1. Start at 0,0
    mock_scope.hw_state["stage"] = StagePosition(x=Q_(10, 'um'))

    # 2. Move +5 um
    delta = StagePosition(x=Q_(5, 'um'))
    req = StageMoveRequest(target=delta, relative=True)

    mock_scope.execute_stage_move(req)

    # 3. Expect 15 um
    current = mock_scope.get_stage_position()
    assert current.x.magnitude == pytest.approx(15000.0)


def test_stage_safety_bounds_rejection(mock_scope):
    """Test that moves outside configured limits are rejected."""
    # Limit is +/- 100 um
    unsafe_target = StagePosition(x=Q_(200, 'um'))
    req = StageMoveRequest(target=unsafe_target)

    with pytest.raises(RuntimeError) as exc:
        mock_scope.execute_stage_move(req)

    assert "outside limits" in str(exc.value)


def test_stage_relative_move_safety_breach(mock_scope):
    """
    Test that a relative move which WOULD result in an out-of-bounds
    position is rejected before execution.
    """
    # 1. Setup: Limit is +/- 100um. Place stage near the edge (90um).
    mock_scope.hw_state["stage"] = StagePosition(x=Q_(90, 'um'))

    # 2. Request: Move +20um (Target becomes 110um -> Unsafe)
    req = StageMoveRequest(
        target=StagePosition(x=Q_(20, 'um')),
        relative=True
    )

    # 3. Assert Rejection
    with pytest.raises(RuntimeError) as exc:
        mock_scope.execute_stage_move(req)

    assert "outside limits" in str(exc.value)

    # 4. Assert State Unchanged (Atomic Safety)
    current = mock_scope.get_stage_position()
    assert current.x.magnitude == pytest.approx(90000.0)


def test_stage_mixed_axis_atomicity(mock_scope):
    """
    Ensure a move request is atomic: if ANY axis is unsafe,
    NO axes should move.
    """
    # Setup: Disable Z axis explicitly
    mock_scope.system_settings.stage_system.can_z = False

    # Start at 0,0,0
    mock_scope.hw_state["stage"] = StagePosition(x=Q_(0, 'um'), z=Q_(0, 'um'))

    # Request: Move X (Valid) and Z (Invalid)
    target = StagePosition(x=Q_(10, 'um'), z=Q_(10, 'um'))
    req = StageMoveRequest(target=target)

    # Expect Failure
    with pytest.raises(RuntimeError):
        mock_scope.execute_stage_move(req)

    # Verify X did NOT move
    current = mock_scope.get_stage_position()
    assert current.x.magnitude == 0.0


def test_stage_step_interpolation(mock_scope):
    """
    Test the `safe_move_stage` logic.
    Verifies that large moves are broken into EQUAL segments.
    """
    # Configure: Max step = 10 um
    mock_scope.system_settings.stage_system.max_step_distance = Q_(10, 'um')

    # Start: 0 um
    mock_scope.hw_state["stage"] = StagePosition(x=Q_(0, 'um'))

    # Request: 25 um move
    # Logic: 25 // 10 = 2. Steps = 2 + 1 = 3 steps total.
    # Step size = 25 / 3 = 8.333... um
    target = StagePosition(x=Q_(25, 'um'))
    req = StageMoveRequest(target=target)

    mock_scope.execute_stage_move(req)

    # Verify move history
    history = mock_scope.move_history
    assert len(history) == 3

    # Step 1: 1/3 of 25um = 8333.33 nm
    assert history[0].x.to('nm').magnitude == pytest.approx(8333.33, rel=1e-3)

    # Step 2: 2/3 of 25um = 16666.66 nm
    assert history[1].x.to('nm').magnitude == pytest.approx(16666.66, rel=1e-3)

    # Step 3: 3/3 of 25um = 25000.00 nm
    assert history[2].x.to('nm').magnitude == pytest.approx(25000.0, rel=1e-3)


def test_safe_move_stage_interpolation(mock_scope):
    """
    Verify that safe_move_stage breaks a large move into smaller steps.
    """
    # 1. Setup Limits: Max step = 50nm
    mock_scope.system_settings.stage_system.max_step_distance = Q_(50, "nm")

    # 2. Setup Current Position: (0, 0)
    mock_scope.hw_state["stage"] = StagePosition(x=Q_(0, "nm"), y=Q_(0, "nm"), _mode="strict")

    # 3. Request Move: Go to (200, 0) -> Delta is 200nm
    # This requires 200 / 50 = 4 steps + 1 final = 4-5 segments logic
    target = StagePosition(x=Q_(200, "nm"), y=Q_(0, "nm"), _mode="strict")

    # Mock the atomic mover to track calls
    mock_scope.move_stage_absolute = MagicMock()

    # 4. Execute
    mock_scope.safe_move_stage(target, wait=True)

    # 5. Assertions
    # We expect multiple calls.
    # Logic in code: steps = int(200//50) + 1 = 5.
    # It loops range(1, 6).
    # Frac 0.2, 0.4, 0.6, 0.8, 1.0.
    calls = mock_scope.move_stage_absolute.call_args_list
    assert len(calls) == 5

    # Check intermediate steps (approximate)
    # Call 1 target x should be 40.0
    arg1 = calls[0][0][0]
    assert arg1.x.magnitude == pytest.approx(40.0)

    # Last call target x should be 200.0
    arg_last = calls[-1][0][0]
    assert arg_last.x.magnitude == pytest.approx(200.0)


def test_move_stage_relative_calculation(mock_scope):
    """
    Verify relative moves are correctly calculated before execution.
    """
    # 1. Setup Current: (100, 100)
    mock_scope.hw_state["stage"] = StagePosition(x=Q_(100, "nm"), y=Q_(100, "nm"), _mode="strict")

    # 2. Request Relative: +50 x
    delta = StagePosition(x=Q_(50, "nm"), y=Q_(0, "nm"), _mode="strict")

    # Mock safe_move_stage to catch the calculated absolute target
    mock_scope.safe_move_stage = MagicMock()

    # 3. Execute
    mock_scope.move_stage_relative(delta)

    # 4. Assert
    args = mock_scope.safe_move_stage.call_args
    target_sent = args[0][0]

    assert target_sent.x.magnitude == 150.0  # 100 + 50
    assert target_sent.y.magnitude == 100.0  # Unchanged


def test_move_stage_relative_raises_if_current_unknown(mock_scope):
    """move_stage_relative should fail loudly if the stage position cannot be read."""
    mock_scope.get_stage_position = MagicMock(return_value=None)
    with pytest.raises(RuntimeError, match="Stage position is unknown"):
        mock_scope.move_stage_relative(StagePosition(x=Q_(1, "nm")))


def test_execute_stage_move_rejects_invalid_request(mock_scope):
    """execute_stage_move should validate inputs and reject invalid requests."""
    req = StageMoveRequest(target=StagePosition(x=Q_(1, "nm")))
    req.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid StageMoveRequest"):
        mock_scope.execute_stage_move(req)


def test_execute_stage_move_relative_requires_current(mock_scope):
    """Relative stage moves require knowing the current position."""
    mock_scope.get_stage_position = MagicMock(return_value=None)
    req = StageMoveRequest(target=StagePosition(x=Q_(1, "nm")), relative=True)
    with pytest.raises(RuntimeError, match="Current stage position is unknown"):
        mock_scope.execute_stage_move(req)


def test_execute_stage_move_falls_back_when_no_stage_system(mock_scope):
    """When no stage safety system is configured, execute_stage_move should call the atomic mover directly."""
    mock_scope._settings.system.stage_system = None
    mock_scope.move_stage_absolute = MagicMock()

    req = StageMoveRequest(target=StagePosition(x=Q_(10, "nm")), relative=False, drive_type="default")
    mock_scope.execute_stage_move(req)

    mock_scope.move_stage_absolute.assert_called_once()


def test_execute_stage_control_invalid_and_actions(mock_scope):
    """execute_stage_control should validate requests and route STOP/HOME actions."""
    # Invalid
    bad = StageControlRequest(action="STOP")
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid StageControlRequest"):
        mock_scope.execute_stage_control(bad)

    # STOP
    mock_scope.stop_stage = MagicMock()
    mock_scope.execute_stage_control(StageControlRequest(action="STOP"))
    mock_scope.stop_stage.assert_called_once()

    # HOME
    mock_scope.home_stage = MagicMock()
    mock_scope.execute_stage_control(StageControlRequest(action="HOME"))
    mock_scope.home_stage.assert_called_once()


def test_safe_move_stage_pass_through_and_errors(mock_scope):
    """Cover safe_move_stage branches: no step limit, within limit, and missing current position."""
    # 1) No step limit -> pass-through
    mock_scope.system_settings.stage_system.max_step_distance = None
    mock_scope.move_stage_absolute = MagicMock()
    target = StagePosition(x=Q_(123, "nm"))
    mock_scope.safe_move_stage(target)
    mock_scope.move_stage_absolute.assert_called_once()

    # 2) With step limit but current unknown -> error
    mock_scope.system_settings.stage_system.max_step_distance = Q_(10, "nm")
    mock_scope.get_stage_position = MagicMock(return_value=None)
    with pytest.raises(RuntimeError, match="Cannot read current stage position"):
        mock_scope.safe_move_stage(StagePosition(x=Q_(1, "nm")))

    # 3) Within limit -> direct call
    mock_scope.get_stage_position = MagicMock(return_value=StagePosition(x=Q_(0, "nm"), y=Q_(0, "nm")))
    mock_scope.move_stage_absolute = MagicMock()
    mock_scope.safe_move_stage(StagePosition(x=Q_(5, "nm"), y=Q_(0, "nm")))
    mock_scope.move_stage_absolute.assert_called_once()


def test_safe_move_stage_interpolates_multiple_axes_including_z(mock_scope):
    """Interpolation should apply across x/y/z axes (covers the z interpolation branch)."""
    mock_scope.system_settings.stage_system.max_step_distance = Q_(50, "nm")
    mock_scope.get_stage_position = MagicMock(
        return_value=StagePosition(x=Q_(0, "nm"), y=Q_(0, "nm"), z=Q_(0, "nm"))
    )
    mock_scope.move_stage_absolute = MagicMock()

    # Delta 200nm in Z triggers interpolation.
    mock_scope.safe_move_stage(StagePosition(x=Q_(0, "nm"), y=Q_(0, "nm"), z=Q_(200, "nm")))
    calls = mock_scope.move_stage_absolute.call_args_list
    assert len(calls) == 5  # steps=int(200//50)+1
    assert calls[-1][0][0].z.magnitude == pytest.approx(200.0)


# =============================================================================
# 4. BEAM & SCAN TESTS
# =============================================================================

def test_beam_control_orchestration(mock_scope):
    """Test applying beam settings via ControlRequest."""
    target = BeamSettings(voltage=Q_(300, 'kV'))
    req = BeamControlRequest(target=target)

    mock_scope.execute_beam_control(req)

    assert mock_scope.get_acceleration_voltage().magnitude == 300.0


def test_beam_safety_rejection(mock_scope):
    """Test voltage limits."""
    # Limits are 80-300 kV
    target = BeamSettings(voltage=Q_(400, 'kV'))
    req = BeamControlRequest(target=target)

    with pytest.raises(RuntimeError):
        mock_scope.execute_beam_control(req)


def test_apply_beam_settings_calls_setters_and_validates_points(mock_scope):
    """apply_beam_settings should call atomic setters and enforce complete Point values."""
    mock_scope.set_beam_current = MagicMock()
    mock_scope.set_spot_size = MagicMock()
    mock_scope.set_convergence_angle = MagicMock()
    mock_scope.set_beam_shift = MagicMock()
    mock_scope.set_condenser_stigmation = MagicMock()
    mock_scope.set_gun_tilt = MagicMock()

    # Canonical scalar fields
    s = BeamSettings(
        beam_current=Q_(150, "pA"),
        spot_size=2,
        convergence_angle=Q_(5, "mrad"),
    )
    mock_scope.apply_beam_settings(s)
    mock_scope.set_beam_current.assert_called_once()
    mock_scope.set_spot_size.assert_called_once_with(2)
    mock_scope.set_convergence_angle.assert_called_once()

    # Partial Point -> rejected
    bad = BeamSettings(beam_shift=Point(x=1.0, y=2.0))
    bad.beam_shift.y = None
    with pytest.raises(ValueError, match="requires both x and y"):
        mock_scope.apply_beam_settings(bad)

    # Complete Points -> applied
    mock_scope.apply_beam_settings(
        BeamSettings(
            beam_shift=Point(x=1.0, y=2.0),
            condenser_stigmation=Point(x=3.0, y=4.0),
            gun_tilt=Point(x=5.0, y=6.0),
        )
    )
    mock_scope.set_beam_shift.assert_called_once_with(1.0, 2.0)
    mock_scope.set_condenser_stigmation.assert_called_once_with(3.0, 4.0)
    mock_scope.set_gun_tilt.assert_called_once_with(5.0, 6.0)


def test_execute_beam_control_invalid_and_no_target(mock_scope):
    """execute_beam_control should reject invalid requests and tolerate an empty patch."""
    bad = BeamControlRequest(target=BeamSettings(voltage=Q_(200, "kV")))
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid BeamControlRequest"):
        mock_scope.execute_beam_control(bad)

    # No target (defaults to an empty intent) -> validation fails before apply
    mock_scope.apply_beam_settings = MagicMock()
    with pytest.raises(ValueError, match="Beam request has no parameters set"):
        mock_scope.execute_beam_control(BeamControlRequest(target=None))
    mock_scope.apply_beam_settings.assert_not_called()


def test_scan_control_start_stop(mock_scope):
    """Test scan engine state toggling."""
    # Start
    mock_scope.execute_scan_control(ScanControlRequest(action="START", target=ScanSettings(scan_mode="Area")))
    assert mock_scope.get_scan_active() is True

    # Stop
    mock_scope.execute_scan_control(ScanControlRequest(action="STOP"))
    assert mock_scope.get_scan_active() is False


def test_scan_control_invalid_safety_and_single_frame(mock_scope):
    """Cover execute_scan_control: invalid request, safety rejection, and SINGLE_FRAME behavior."""
    bad = ScanControlRequest(action="START", target=ScanSettings(scan_mode="Area"))
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid ScanControlRequest"):
        mock_scope.execute_scan_control(bad)

    # Safety rejection path
    mock_scope._settings.system.scan_system = MagicMock()
    mock_scope._settings.system.scan_system.is_safe_scan = MagicMock(return_value=SafetyCheck.failure("nope"))
    with pytest.raises(RuntimeError, match="Unsafe scan settings rejected"):
        mock_scope.execute_scan_control(ScanControlRequest(action="START", target=ScanSettings(scan_mode="Area")))

    # SINGLE_FRAME should (currently) start scanning
    mock_scope._settings.system.scan_system.is_safe_scan = MagicMock(return_value=SafetyCheck.success())
    mock_scope.set_scan_active = MagicMock()
    mock_scope.execute_scan_control(ScanControlRequest(action="SINGLE_FRAME", target=ScanSettings(scan_mode="Area")))
    mock_scope.set_scan_active.assert_called_once_with(True)


# =============================================================================
# 4b. PROJECTION TESTS
# =============================================================================


def test_apply_projection_settings_calls_setters_and_validates_points(mock_scope):
    """apply_projection_settings should apply scalar fields and enforce complete Point values."""
    mock_scope.set_projection_mode = MagicMock()
    mock_scope.set_magnification = MagicMock()
    mock_scope.set_camera_length = MagicMock()
    mock_scope.set_defocus = MagicMock()
    mock_scope.set_screen_position = MagicMock()
    mock_scope.set_objective_stigmation = MagicMock()
    mock_scope.set_image_shift = MagicMock()
    mock_scope.set_diffraction_shift = MagicMock()

    s = ProjectionSettings(
        optical_mode="IMAGING",
        magnification=50000,
        camera_length=Q_(100, "mm"),
        defocus=Q_(10, "nm"),
        screen_position="UP",
    )
    mock_scope.apply_projection_settings(s)
    mock_scope.set_projection_mode.assert_called_once_with("IMAGING")
    mock_scope.set_magnification.assert_called_once_with(50000)
    mock_scope.set_camera_length.assert_called_once_with(Q_(100, "mm"))
    mock_scope.set_defocus.assert_called_once_with(Q_(10, "nm"))
    mock_scope.set_screen_position.assert_called_once_with("UP")

    # Partial Point -> rejected
    bad = ProjectionSettings(image_shift=Point(x=1.0, y=2.0))
    bad.image_shift.y = None
    with pytest.raises(ValueError, match="requires both x and y"):
        mock_scope.apply_projection_settings(bad)

    mock_scope.apply_projection_settings(
        ProjectionSettings(
            objective_stigmation=Point(x=1.0, y=2.0),
            image_shift=Point(x=3.0, y=4.0),
            diffraction_shift=Point(x=5.0, y=6.0),
        )
    )
    mock_scope.set_objective_stigmation.assert_called_once_with(1.0, 2.0)
    mock_scope.set_image_shift.assert_called_once_with(3.0, 4.0)
    mock_scope.set_diffraction_shift.assert_called_once_with(5.0, 6.0)


def test_execute_projection_control_invalid_and_safety(mock_scope):
    """execute_projection_control should validate and enforce projection safety checks."""
    bad = ProjectionControlRequest(target=ProjectionSettings(optical_mode="IMAGING"))
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid ProjectionControlRequest"):
        mock_scope.execute_projection_control(bad)

    # Safety rejection
    mock_scope._settings.system.projection_system = MagicMock()
    mock_scope._settings.system.projection_system.is_safe_projection = MagicMock(return_value=SafetyCheck.failure("no"))
    with pytest.raises(RuntimeError, match="Unsafe projection settings rejected"):
        mock_scope.execute_projection_control(ProjectionControlRequest(target=ProjectionSettings(optical_mode="IMAGING")))

    # Safety success -> applies
    mock_scope._settings.system.projection_system.is_safe_projection = MagicMock(return_value=SafetyCheck.success())
    mock_scope.apply_projection_settings = MagicMock()
    mock_scope.execute_projection_control(ProjectionControlRequest(target=ProjectionSettings(optical_mode="IMAGING")))
    mock_scope.apply_projection_settings.assert_called_once()


# =============================================================================
# 5. DETECTOR TESTS
# =============================================================================

def test_detector_control_capabilities(mock_scope):
    """Test that detector settings are checked against hardware capabilities."""
    # Cap limit: Max exposure 1000ms
    # Request: 5000ms
    target = DetectorSettings(exposure=Q_(5000, 'ms'))
    req = DetectorControlRequest(detector_id="CamA", target=target)

    with pytest.raises(RuntimeError) as exc:
        mock_scope.execute_detector_control(req)

    assert "exceeds limit" in str(exc.value)


def test_detector_control_actions(mock_scope):
    """Test mechanical insertion/retraction."""
    req = DetectorControlRequest(detector_id="CamA", action="INSERT")
    mock_scope.execute_detector_control(req)
    assert mock_scope.hw_state["detectors"]["CamA"]["inserted"] is True

    req = DetectorControlRequest(detector_id="CamA", action="RETRACT")
    mock_scope.execute_detector_control(req)
    assert mock_scope.hw_state["detectors"]["CamA"]["inserted"] is False


def test_apply_detector_settings_and_execute_detector_control_variants(mock_scope):
    """Cover apply_detector_settings branches and execute_detector_control validation/no-action config."""
    # apply_detector_settings should route to atomic setters for each non-None field
    mock_scope.set_detector_exposure = MagicMock()
    mock_scope.set_detector_binning = MagicMock()
    mock_scope.set_detector_integration = MagicMock()
    mock_scope.set_detector_roi = MagicMock()

    roi = ROI(x=0, y=0, width=128, height=128)
    mock_scope.apply_detector_settings(
        "CamA",
        DetectorSettings(exposure=Q_(10, "ms"), binning_index=2, frame_integration=3, roi=roi),
    )
    mock_scope.set_detector_exposure.assert_called_once()
    mock_scope.set_detector_binning.assert_called_once_with("CamA", 2)
    mock_scope.set_detector_integration.assert_called_once_with("CamA", 3)
    mock_scope.set_detector_roi.assert_called_once_with("CamA", roi)

    # execute_detector_control invalid request
    bad = DetectorControlRequest(detector_id="CamA", target=DetectorSettings(exposure=Q_(10, "ms")))
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid DetectorControlRequest"):
        mock_scope.execute_detector_control(bad)

    # No action + no detector system -> should still apply settings (no capability check)
    mock_scope._settings.system.detector_system = None
    mock_scope.apply_detector_settings = MagicMock()
    mock_scope.execute_detector_control(
        DetectorControlRequest(detector_id="CamA", action=None, target=DetectorSettings(exposure=Q_(11, "ms")))
    )
    mock_scope.apply_detector_settings.assert_called_once()


# =============================================================================
# 6. VACUUM & APERTURE TESTS
# =============================================================================

def test_vacuum_control_routing(mock_scope):
    """Test that vacuum requests map to atomic valve calls."""
    # Mock the atomic setter
    mock_scope.set_valve_state = MagicMock()

    # Request: Open Column Valve
    target = VacuumSettings(column_valve_state="OPEN")
    req = VacuumControlRequest(target=target)

    mock_scope.execute_vacuum_control(req)

    mock_scope.set_valve_state.assert_called_with("column", "OPEN")


def test_vacuum_control_invalid_and_multi_valve_apply(mock_scope):
    """Cover execute_vacuum_control validation and apply_vacuum_settings for gun/turbo valves."""
    bad = VacuumControlRequest(target=VacuumSettings(column_valve_state="OPEN"))
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid VacuumControlRequest"):
        mock_scope.execute_vacuum_control(bad)

    mock_scope.set_valve_state = MagicMock()
    mock_scope.apply_vacuum_settings(VacuumSettings(gun_valve_state="OPEN", turbo_pump_state="ON"))
    mock_scope.set_valve_state.assert_any_call("gun", "OPEN")
    mock_scope.set_valve_state.assert_any_call("turbo", "ON")


def test_aperture_relative_movement(mock_scope):
    """
    Verify the Orchestrator logic for calculating relative aperture positions.
    """
    # 1. Setup Mock Hardware State
    # Aperture 'CLA' is currently at x=1000, y=2000
    mock_scope.hw_state["apertures"] = {
        "CLA": Aperture(aperture_id="CLA", position=Point(x=1000, y=2000))
    }

    # Spy on the setter
    mock_scope.set_aperture = MagicMock(wraps=mock_scope.set_aperture)

    # 2. Request: Shift x+500, y-500
    delta = Aperture(position=Point(x=500, y=-500))
    req = ApertureControlRequest(aperture_id="CLA", target=delta, relative=True)

    # 3. Execute
    mock_scope.execute_aperture_control(req)

    # 4. Verify the *Absolute* value sent to hardware
    # Expected: x=1500, y=1500
    args = mock_scope.set_aperture.call_args
    assert args is not None

    target_sent = args[0][1]  # Second arg is the target Aperture object
    assert target_sent.position.x == 1500
    assert target_sent.position.y == 1500


def test_aperture_get_all_and_error_branches(mock_scope):
    """Cover get_all_apertures filtering and execute_aperture_control validation/error branches."""
    # get_all_apertures should exclude missing/None values
    mock_scope.list_apertures = MagicMock(return_value=["CLA", "MISSING"])
    mock_scope.get_aperture = MagicMock(side_effect=lambda a_id: Aperture(aperture_id=a_id) if a_id == "CLA" else None)

    all_apts = mock_scope.get_all_apertures()
    assert list(all_apts.keys()) == ["CLA"]

    # execute_aperture_control invalid
    bad = ApertureControlRequest(aperture_id="CLA", target=Aperture(aperture_id="CLA"))
    bad.validate = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="Invalid ApertureControlRequest"):
        mock_scope.execute_aperture_control(bad)

    # relative move but cannot read current state -> RuntimeError
    mock_scope.get_aperture = MagicMock(return_value=None)
    with pytest.raises(RuntimeError, match="Failed to read current state"):
        mock_scope.execute_aperture_control(
            ApertureControlRequest(aperture_id="CLA", target=Aperture(position=Point(x=1, y=1)), relative=True)
        )


# =============================================================================
# 7. FULL STATE AGGREGATION
# =============================================================================

def test_get_full_state(mock_scope):
    """Verify that get_full_state aggregates all subsystems."""
    state = mock_scope.get_full_state()

    assert state.mode == "TEM"
    assert state.stage_position.x is not None
    assert state.beam.voltage.magnitude == 200.0
    assert "CamA" in state.detectors


def test_get_full_state_partial_failure(mock_scope):
    """
    Test that get_full_state survives if a subsystem returns None
    (e.g. Beam is offline).
    """
    # Simulate Beam Offline (getters return None)
    mock_scope.get_acceleration_voltage = MagicMock(return_value=None)
    mock_scope.get_beam_current = MagicMock(return_value=None)

    state = mock_scope.get_full_state()

    # Should successfully return a state object
    assert state is not None
    # Beam part should be empty/None but present structure
    assert state.beam.voltage is None
    # Other parts should still work
    assert state.stage_position.x is not None


def test_base_class_abstract_stubs_are_executable_for_coverage(mock_scope):
    """Execute TemMicroscope's abstract stubs directly so the `pass` lines get counted by coverage.

    This looks slightly silly (because real drivers override these), but it keeps coverage honest
    without changing production code.
    """
    # Connection / identity
    TemMicroscope.connect(mock_scope, host="localhost")
    TemMicroscope.disconnect(mock_scope)
    assert TemMicroscope.is_connected(mock_scope) is None
    assert TemMicroscope.get_instrument_info(mock_scope) is None

    # Mode
    assert TemMicroscope.get_mode(mock_scope) is None
    TemMicroscope.set_mode(mock_scope, "TEM")

    # Stage
    assert TemMicroscope.get_stage_position(mock_scope) is None
    TemMicroscope.move_stage_absolute(mock_scope, StagePosition(x=Q_(0, "nm")))
    TemMicroscope.stop_stage(mock_scope)
    TemMicroscope.home_stage(mock_scope)

    # Beam
    assert TemMicroscope.get_acceleration_voltage(mock_scope) is None
    assert TemMicroscope.get_beam_current(mock_scope) is None
    assert TemMicroscope.get_spot_size(mock_scope) is None
    assert TemMicroscope.get_convergence_angle(mock_scope) is None
    assert TemMicroscope.get_beam_shift(mock_scope) is None
    assert TemMicroscope.get_condenser_stigmation(mock_scope) is None
    assert TemMicroscope.get_gun_tilt(mock_scope) is None
    assert TemMicroscope.get_beam_blank(mock_scope) is None
    TemMicroscope.set_acceleration_voltage(mock_scope, Q_(200, "kV"))
    TemMicroscope.set_beam_current(mock_scope, Q_(100, "pA"))
    TemMicroscope.set_spot_size(mock_scope, 1)
    TemMicroscope.set_convergence_angle(mock_scope, Q_(1, "mrad"))
    TemMicroscope.set_beam_shift(mock_scope, 0.0, 0.0)
    TemMicroscope.set_condenser_stigmation(mock_scope, 0.0, 0.0)
    TemMicroscope.set_gun_tilt(mock_scope, 0.0, 0.0)
    TemMicroscope.set_beam_blank(mock_scope, False)

    # Projection
    assert TemMicroscope.get_projection_mode(mock_scope) is None
    assert TemMicroscope.get_magnification(mock_scope) is None
    assert TemMicroscope.get_camera_length(mock_scope) is None
    assert TemMicroscope.get_defocus(mock_scope) is None
    assert TemMicroscope.get_screen_position(mock_scope) is None
    assert TemMicroscope.get_objective_stigmation(mock_scope) is None
    assert TemMicroscope.get_image_shift(mock_scope) is None
    assert TemMicroscope.get_diffraction_shift(mock_scope) is None
    TemMicroscope.set_projection_mode(mock_scope, "IMAGING")
    TemMicroscope.set_magnification(mock_scope, 1)
    TemMicroscope.set_camera_length(mock_scope, 1)
    TemMicroscope.set_defocus(mock_scope, Q_(0, "nm"))
    TemMicroscope.set_screen_position(mock_scope, "UP")
    TemMicroscope.set_objective_stigmation(mock_scope, 0.0, 0.0)
    TemMicroscope.set_image_shift(mock_scope, 0.0, 0.0)
    TemMicroscope.set_diffraction_shift(mock_scope, 0.0, 0.0)

    # Scan
    assert TemMicroscope.get_scan_mode(mock_scope) is None
    assert TemMicroscope.get_scan_width(mock_scope) is None
    assert TemMicroscope.get_scan_height(mock_scope) is None
    assert TemMicroscope.get_scan_pixel_dwell(mock_scope) is None
    assert TemMicroscope.get_scan_flyback(mock_scope) is None
    assert TemMicroscope.get_scan_rotation(mock_scope) is None
    assert TemMicroscope.get_scan_active(mock_scope) is None
    TemMicroscope.set_scan_mode(mock_scope, "Area")
    TemMicroscope.set_scan_width(mock_scope, 64)
    TemMicroscope.set_scan_height(mock_scope, 64)
    TemMicroscope.set_scan_pixel_dwell(mock_scope, Q_(1, "us"))
    TemMicroscope.set_scan_flyback(mock_scope, Q_(1, "us"))
    TemMicroscope.set_scan_rotation(mock_scope, Q_(0, "deg"))
    TemMicroscope.set_scan_active(mock_scope, False)

    # Detector
    assert TemMicroscope.list_detectors(mock_scope) is None
    assert TemMicroscope.get_active_detector_ids(mock_scope) is None
    assert TemMicroscope.get_primary_detector_id(mock_scope) is None
    assert TemMicroscope.get_detector_exposure(mock_scope, "CamA") is None
    assert TemMicroscope.get_detector_binning(mock_scope, "CamA") is None
    assert TemMicroscope.get_detector_roi(mock_scope, "CamA") is None
    assert TemMicroscope.get_detector_integration(mock_scope, "CamA") is None
    assert TemMicroscope.get_detector_inserted(mock_scope, "CamA") is None
    assert TemMicroscope.get_detector_frame_rate(mock_scope, "CamA") is None
    TemMicroscope.set_detector_exposure(mock_scope, "CamA", Q_(10, "ms"))
    TemMicroscope.set_detector_binning(mock_scope, "CamA", 1)
    TemMicroscope.set_detector_roi(mock_scope, "CamA", None)
    TemMicroscope.set_detector_integration(mock_scope, "CamA", 1)
    TemMicroscope.set_detector_insertion(mock_scope, "CamA", True)
    TemMicroscope.acquire_image(mock_scope, None)

    # Vacuum / aperture
    assert TemMicroscope.get_valve_state(mock_scope, "column") is None
    TemMicroscope.set_valve_state(mock_scope, "column", "OPEN")
    assert TemMicroscope.get_pressure(mock_scope, "IG") is None
    assert TemMicroscope.list_apertures(mock_scope) is None
    assert TemMicroscope.get_aperture(mock_scope, "CLA") is None
    TemMicroscope.set_aperture(mock_scope, "CLA", Aperture(aperture_id="CLA"))