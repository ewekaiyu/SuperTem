from typing import Dict, Any, Optional, List, Tuple

from supertem.microscope import TemMicroscope
from PyJEM import TEM3

from supertem.structures.base import ImageSettings, TemImage, TemStagePosition


class JeolMicroscope(TemMicroscope):

    def __init__(self):
        self.apt = TEM3.Apt3()
        self.deflector = TEM3.Def()
        self.detector = TEM3.Detector3()
        self.eos = TEM3.EOS3()
        self.feg = TEM3.FEG3()
        self.gun = TEM3.GUN3()
        self.ht = TEM3.HT3()
        self.lens = TEM3.Lens3()
        self.stage = TEM3.Stage3()
        self.vac = TEM3.VACUUM3()
        self.magnification_list = [6000, 8000, 10000, 20000, 30000, 40000, 50000, 80000, 100000, 120000, 150000]
        self.camera_length_list = []
        self.aperture_dict = dict(CL1 = 0, CL2 = 1, OL_Upper = 2, Ol_Lower = 3, SA = 4, ENT = 5, HX = 6, BF = 7, AUX1 = 8, AUX2 = 9, AUX3 = 10, AUX4 = 11)

    def connect_to_microscope(self, ip_address: str = "0.0.0.0", port: int = 8000, timeout_s: float = 10.0) -> None:
        TEM3.connect()

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return TEM3.is_connect()

    def get_instrument_info(self) -> Dict[str, Any]:
        pass

    def get_status(self) -> Dict[str, Any]:
        pass

    def get_mode(self) -> str:
        pass

    def set_mode(self, mode: str):
        pass

    def set_acceleration_voltage(self, kv: float) -> None:
        self.gun.SetHtValue(1000*kv)

    def get_acceleration_voltage(self) -> float:
        return self.gun.GetHtValue() / 1000

    def get_emission_current(self) -> Optional[float]:
        return self.gun.GetEmissionCurrentValue()

    def set_beam_blank(self, blank: bool) -> None:
        if blank:
            self.feg.ExecEmissionOn(1)
        else:
            self.feg.SetFEGEmissionOff(1)

    def get_beam_blank(self) -> bool:
        return True if self.feg.GetEmissionOnStatus()[1] == 1 else False

    def set_magnification(self, mag: float) -> None:
        if self.eos.GetTemStemMode() == 0:
            actual_mag = self.magnification_list.index(max(x for x in self.magnification_list if x < mag))
            self.eos.SetSelector(actual_mag)

        else:
            actual_camera_length = self.camera_length_list.index(max(x for x in self.camera_length_list if x < mag))
            self.eos.SetStemCamSelector(actual_camera_length)

    def get_magnification(self) -> float:
        if self.eos.GetTemStemMode() == 0:
            return float(self.eos.GetMagValue()[0])
        return float(self.eos.GetStemCamValue())

    def set_spot_size(self, index: int) -> None:
        self.eos.SelectSpotSize(index)

    def get_spot_size(self) -> int:
        return self.eos.GetSpotSize()

    def list_apertures(self) -> List[str]:
        return list(self.aperture_dict.keys())

    def set_aperture(self, kind: str) -> None:
        pass

    def set_defocus(self, defocus: float) -> None:
        pass

    def get_defocus(self) -> float:
        pass

    def set_stigmation(self, x: float, y: float) -> None:
        pass

    def get_stigmation(self) -> float:
        pass

    def align_beam(self, mode: str = "center") -> Dict[str, Any]:
        pass

    def list_detectors(self) -> List[str]:
        pass

    def select_detector(self, name: str) -> None:
        pass

    def get_detector_settings(self, detector: Optional[str] = None) -> Dict[str, Any]:
        pass

    def set_detector_settings(self, body: Dict[str, Any], detector: Optional[str] = None) -> Dict[str, Any]:
        pass

    def acquire_image(self, settings: ImageSettings, detector: Optional[str] = None) -> TemImage:
        pass

    def start_live(self, settings: Optional[ImageSettings] = None) -> None:
        pass

    def stop_live(self) -> None:
        pass

    def get_live_image(self) -> bytes:
        pass

    def get_raw_image_data(self, detector: Optional[str] = None) -> bytes:
        pass

    def set_imaging_area(self, width: int, height: int, x: int = 0, y: int = 0) -> None:
        pass

    def set_binning(self, binning: Tuple[int, int]) -> None:
        pass

    def set_exposure_time(self, ms: float) -> None:
        pass

    def set_dwell_time(self, us: float) -> None:
        pass

    def get_stage_position(self) -> TemStagePosition:
        pass

    def move_stage_absolute(self, pos: TemStagePosition, wait: bool = True, tolerance_nm: float = 10.0) -> None:
        pass

    def move_stage_relative(self, dx: float, dy: float, dz: float = 0.0, dtx: float = 0.0, dty: float = 0.0,
                            wait: bool = True, tolerance_nm: float = 10.0) -> None:
        pass

    def set_stage_drive_mode(self, mode: str) -> None:
        pass

    def stop_stage(self) -> None:
        pass

    def get_stage_status(self) -> Dict[str, Any]:
        pass

    def insert_holder(self) -> None:
        pass

    def retract(self) -> None:
        pass

    def run_autofunction(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass

    def discover_capabilities(self) -> Dict[str, Any]:
        pass

    def get_log(self, n: int = 100) -> List[str]:
        pass

    def send_raw_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Any:
        pass

    def safe_move_stage(self, pos: TemStagePosition, max_step_nm: float = 50000.0) -> None:
        pass