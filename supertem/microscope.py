from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from supertem.structures.base import SystemSettings, ImageSettings, TemStagePosition, TemImage




class TemMicroscope(ABC):
    """
    抽象基类：尽可能覆盖 TEM 的“原子”操作（最小粒度的控制指令）。
    各厂商实现应继承并实现这些方法；保持通用性与可扩展性。
    """

    # -----------------------
    # 连接 / 会话
    # -----------------------
    @abstractmethod
    def connect_to_microscope(self, ip_address: str, port: int, timeout_s: float = 10.0) -> None:
        """建立与仪器的连接（TCP/REST/SDK/...），抛出异常代表失败。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接，释放资源。"""

    @abstractmethod
    def is_connected(self) -> bool:
        """返回当前连接状态。"""

    # -----------------------
    # 仪器信息与状态
    # -----------------------
    @abstractmethod
    def get_instrument_info(self) -> Dict[str, Any]:
        """返回厂商、型号、固件版本等信息。"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """返回仪器当前状态摘要（vacuum, gun, faults, temperatures...）。"""

    @abstractmethod
    def get_mode(self) -> str:
        """Get the current mode of the microscope."""

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """Set the current mode of the microscope."""

    # -----------------------
    # 高压/枪（Gun）
    # -----------------------
    @abstractmethod
    def set_acceleration_voltage(self, kv: float) -> None:
        """设置加速电压（kV）。"""

    @abstractmethod
    def get_acceleration_voltage(self) -> float:
        """获取当前加速电压（kV）。"""

    @abstractmethod
    def get_emission_current(self) -> Optional[float]:
        """查询发射电流（若可用），单位 nA / µA 由实现说明。"""

    @abstractmethod
    def set_beam_blank(self, blank: bool) -> None:
        """设置束空（blank on/off）。"""

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """返回当前束空状态。"""

    # -----------------------
    # 透镜 / 光学（Lens/Condenser/Objectives）
    # -----------------------
    @abstractmethod
    def set_magnification(self, mag: float) -> None:
        """设置放大倍数（或直接设置 mag index），由实现决定精确语义。"""

    @abstractmethod
    def get_magnification(self) -> float:
        """读取当前放大倍数。"""

    @abstractmethod
    def set_spot_size(self, index: int) -> None:
        """设置 spot size / condenser aperture 索引。"""

    @abstractmethod
    def get_spot_size(self) -> int:
        """读取 spot size 索引。"""

    @abstractmethod
    def list_apertures(self) -> List[str]:
        """Get the aperture list."""

    @abstractmethod
    def set_aperture(self, kind: str) -> None:
        """Insert selected aperture."""

    # -----------------------
    # 像差/调谐（Focus / Stigmator / Alignment）
    # -----------------------
    @abstractmethod
    def set_defocus(self, defocus_nm: float) -> None:
        """设置 defocus，以 nm 为单位（实现须说明单位）。"""

    @abstractmethod
    def get_defocus(self) -> float:
        """读取当前 defocus（nm）。"""

    @abstractmethod
    def set_stigmation(self, x: float, y: float) -> None:
        """设置像散校正（X/Y），单位与范围由实现决定。"""

    @abstractmethod
    def get_stigmation(self) -> None:
        """Get the stigmation of the beam."""

    @abstractmethod
    def align_beam(self, mode: str = "center") -> Dict[str, Any]:
        """做 beam alignment；mode 例如 'center','pivot' 等。"""

    # -----------------------
    # 探测器 / 相机
    # -----------------------
    @abstractmethod
    def list_detectors(self) -> List[str]:
        """列出可用探测器/相机名。"""

    @abstractmethod
    def select_detector(self, name: str) -> None:
        """选择当前探测器。"""

    @abstractmethod
    def get_detector_settings(self, detector: Optional[str] = None) -> Dict[str, Any]:
        """读取探测器的完整设置字典（用于发现可设置项）。"""

    @abstractmethod
    def set_detector_settings(self, body: Dict[str, Any], detector: Optional[str] = None) -> Dict[str, Any]:
        """以原子字段设置探测器参数（wildcard），返回设备回显。"""

    @abstractmethod
    def acquire_image(self, settings: ImageSettings, detector: Optional[str] = None) -> TemImage:
        """按给定图像设置采集单帧，返回TemImage对象。"""

    @abstractmethod
    def start_live(self, settings: Optional[ImageSettings] = None) -> None:
        """开始实时流（live view）。"""

    @abstractmethod
    def stop_live(self) -> None:
        """停止实时流。"""

    @abstractmethod
    def get_live_frame(self) -> bytes:
        """获取当前 live 帧的原始二进制数据（最低延迟）。"""

    @abstractmethod
    def get_raw_image_data(self, detector: Optional[str] = None) -> bytes:
        """获取相机的原始 raw 数据缓冲（如果可用）。"""

    # -----------------------
    # 成像设置（解耦探测器）—— 原子级图像控制
    # -----------------------
    @abstractmethod
    def set_imaging_area(self, width: int, height: int, x: int = 0, y: int = 0) -> None:
        """设置视场 / ROI（像素）。"""

    @abstractmethod
    def set_binning(self, binning: int) -> None:
        """设置 binning（像素合并）。"""

    @abstractmethod
    def set_exposure_time(self, ms: float) -> None:
        """设置曝光时间（毫秒）。"""

    @abstractmethod
    def set_dwell_time(self, us: float) -> None:
        """设置扫描 dwell time（微秒），常用于扫描探测器/扫描模式。"""

    # -----------------------
    # 样台（Stage）原子操作
    # -----------------------
    @abstractmethod
    def get_stage_position(self) -> TemStagePosition:
        """读取样台位置（单位由实现说明，建议 nm / deg）。"""

    @abstractmethod
    def move_stage_absolute(self, pos: TemStagePosition, wait: bool = True, tolerance_nm: float = 10.0) -> None:
        """绝对移动样台到 pos（若不支持某分量可忽略）。"""

    @abstractmethod
    def move_stage_relative(self, dx: float, dy: float, dz: float = 0.0, dtx: float = 0.0, dty: float = 0.0,
                            wait: bool = True, tolerance_nm: float = 10.0) -> None:
        """相对移动样台。"""

    @abstractmethod
    def set_stage_drive_mode(self, mode: str) -> None:
        """设置驱动模式，例如 'motor' 或 'piezo'。"""

    @abstractmethod
    def stop_stage(self) -> None:
        """停止样台运动。"""

    @abstractmethod
    def get_stage_status(self) -> Dict[str, Any]:
        """返回样台状态（每轴状态/错误/limits 等）。"""

    @abstractmethod
    def insert_holder(self) -> None:
        """插入样座或探测器（如有）。"""

    @abstractmethod
    def retract_holder(self) -> None:
        """撤回样座或探测器（如有）。"""

    # -----------------------
    # 自动化 / 对齐 / 校准（复合操作可由原子操作组合）
    # -----------------------
    @abstractmethod
    def run_autofunction(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行厂商/用户定义的自动化函数（如 'autofocus','autostig','autocontrast' 等）。
        返回运行结果/诊断信息。
        """

    # -----------------------
    # 诊断 / 能力发现
    # -----------------------
    @abstractmethod
    def discover_capabilities(self) -> Dict[str, Any]:
        """
        查询并返回设备支持的功能列表与可配置字段（用于动态 UI / 校验）。
        建议返回字段：detectors, stage_axes, imaging_keys, limits, units。
        """

    @abstractmethod
    def get_log(self, n: int = 100) -> List[str]:
        """读取仪器最近日志或通知条目（若可用）。"""

    # -----------------------
    # 低层/原始命令接口（保留）
    # -----------------------
    @abstractmethod
    def send_raw_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """发送厂商原生命令/REST path，直接返回原始响应（仅用于高级/调试）。"""

    # -----------------------
    # 辅助：安全检查/回退（可选实现）
    # -----------------------
    def safe_move_stage(self, pos: TemStagePosition, max_step_nm: float = 50000.0) -> None:
        """
        可由子类复写；默认实现使用 move_stage_relative 分步到达以避免大位移。
        这里只给出默认策略（子类可覆盖更精细策略）。
        """
        # 默认实现基于子类实现的原子方法，不作抽象要求
        cur = self.get_stage_position()
        dx = pos.x - cur.x
        dy = pos.y - cur.y
        dz = pos.z - cur.z
        # 简短分步逻辑（具体数值与单位需子类保证一致）
        steps = int(max(abs(dx), abs(dy), abs(dz)) / max_step_nm) + 1
        for i in range(1, steps + 1):
            frac = i / steps
            self.move_stage_relative(dx * frac - dx * (frac - 1 / steps),
                                     dy * frac - dy * (frac - 1 / steps),
                                     dz * frac - dz * (frac - 1 / steps),
                                     wait=True)