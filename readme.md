# SuperTEM
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
一个为透射电子显微镜设计的、语言无关的抽象控制接口。本项目旨在通过定义一套标准化的、最小粒度的“原子”操作，为不同厂商（如 Thermo Fisher, JEOL, Nion 等）的 TEM 设备提供统一的 Python 控制层，从而简化上层应用（如自动化实验、数据采集、AI 控制）的开发。
## 核心目标
- **统一性**: 为不同品牌的 TEM 提供一致的编程接口。
- **原子化**: 将复杂的显微镜操作分解为最小、最基础的控制指令。
- **可扩展性**: 轻松添加对新厂商或新功能的支持，而不影响现有代码。
- **清晰性**: 基于抽象基类（ABC）强制实现，确保接口的完整性和一致性。
## 安装
```bash
pip install SuperTEM
```
> **注意**: 这是一个接口定义库。要实际控制显微镜，还需要安装对应厂商的具体实现包（例如 `PyJEM`）。
## 快速开始
### 1. 定义厂商实现
首先，厂商或开发者需要创建一个继承自 `TemMicroscope` 的具体实现类。以下是一个连接到虚构 "SuperTem" 品牌显微镜的示例：
```python
from supertem.structures.base import SystemSettings, ImageSettings, TemStagePosition, TemImage
from tem_microscope_interface import TemMicroscope
class SuperTemMicroscope(TemMicroscope):
    """SuperTem 品牌显微镜的具体实现"""
    def __init__(self):
        self._connection = None
    def connect_to_microscope(self, ip_address: str, port: int, timeout_s: float = 10.0) -> None:
        """使用 TCP Socket 连接到 SuperTem 控制器"""
        print(f"正在连接到 {ip_address}:{port}...")
        # 这里是实际的连接逻辑，例如使用 socket 库
        self._connection = f"连接到 {ip_address}:{port}" # 模拟连接对象
        print("连接成功！")
    def disconnect(self) -> None:
        """断开连接"""
        if self._connection:
            print("正在断开连接...")
            self._connection = None
            print("已断开连接。")
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connection is not None
    def get_instrument_info(self) -> Dict[str, Any]:
        """获取仪器信息"""
        if not self.is_connected():
            raise ConnectionError("未连接到显微镜")
        return {
            "vendor": "SuperTem Inc.",
            "model": "ST-3000X",
            "serial_number": "SN123456789",
            "firmware_version": "2.1.5"
        }
    def get_status(self) -> Dict[str, Any]:
        """获取当前仪器状态"""
        if not self.is_connected():
            raise ConnectionError("未连接到显微镜")
        return {
            "vacuum": "OK",
            "high_voltage": "ON",
            "column_valves": "OPEN",
            "stage_position": TemStagePosition(x=0.0, y=0.0, z=0.0, a=0.0, b=0.0)
        }
    # ... 必须实现所有其他抽象方法 ...
```
### 2. 使用统一接口进行控制
上层应用程序现在可以使用这个统一接口，而无需关心底层是哪个品牌的显微镜。
```python
# 假设上面的 SuperTemMicroscope 类已定义
def run_experiment(microscope: TemMicroscope):
    """一个与厂商无关的实验流程"""
    try:
        # 1. 获取仪器信息
        info = microscope.get_instrument_info()
        print(f"当前仪器: {info['vendor']} {info['model']}")
        # 2. 检查状态
        status = microscope.get_status()
        print(f"真空状态: {status['vacuum']}")
        # 3. 执行其他控制操作...
        # microscope.move_stage(...)
        # microscope.acquire_image(...)
    except Exception as e:
        print(f"实验出错: {e}")
    finally:
        # 确保断开连接
        microscope.disconnect()
# --- 主程序 ---
if __name__ == "__main__":
    # 初始化具体的显微镜实例
    my_tem = SuperTemMicroscope()
    # 连接
    my_tem.connect_to_microscope(ip_address="192.168.0.10", port=8000)
    # 运行与厂商无关的实验
    run_experiment(my_tem)
```
## 核心组件
### `TemMicroscope` 抽象基类
这是项目的核心，位于 `tem_microscope_interface/tem_microscope.py`。它定义了所有 TEM 控制器必须实现的方法。
#### 已定义的接口模块：
1.  **连接与会话管理**
    - `connect_to_microscope()`: 建立与仪器的连接。
    - `disconnect()`: 断开连接并释放资源。
    - `is_connected()`: 检查当前连接状态。
2.  **仪器信息与状态**
    - `get_instrument_info()`: 获取厂商、型号、固件版本等静态信息。
    - `get_status()`: 获取真空、高压、样品台位置等动态状态。
3.  **仪器控制** (规划中)
    - 样品台控制
    - 束流控制
    - 透镜系统控制
    - 图像采集
    - ... 更多模块
### `supertem.structures.base`
定义了接口中使用的标准数据结构，确保了数据传递的一致性。
- `SystemSettings`: 系统级设置。
- `ImageSettings`: 图像采集相关设置。
- `TemStagePosition`: 样品台位置信息。
- `TemImage`: 采集到的图像数据。
## 🛠️ 开发路线图
- [x] **核心接口定义**: 完成连接、状态查询等基础接口。
- [ ] **样品台控制**: 实现样品台移动、位置获取等原子操作。
- [ ] **束流与透镜控制**: 实现束流对中、聚焦、消像散等控制。
- [ ] **图像采集**: 实现相机参数设置、图像采集与获取。
- [ ] **自动化与脚本**: 提供更高级别的组合操作接口。
- [ ] **官方厂商驱动**: 推动或协助主流厂商提供官方实现。
## 🤝 贡献
我们欢迎社区贡献！如果您想为项目添砖加瓦，请遵循以下步骤：
1.  Fork 本仓库。
2.  创建您的特性分支 (`git checkout -b feature/AmazingFeature`)。
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4.  推送到分支 (`git push origin feature/AmazingFeature`)。
5.  开启一个 Pull Request。
在贡献代码前，请确保：
- 代码符合 PEP 8 规范。
- 添加了必要的单元测试。
- 更新了相关文档。
## 📄 许可证
本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
---
**免责声明**: 本项目仅提供抽象接口定义，不包含任何特定厂商设备的实际驱动代码。用户需自行获取或开发对应的具体实现。
