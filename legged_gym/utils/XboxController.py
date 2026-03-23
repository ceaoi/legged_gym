import threading
import time
from typing import Optional, Callable, Dict, Any

try:
    import pygame
except Exception as e:
    pygame = None

import numpy as np


class XboxController:
    """
    Xbox 手柄控制器（基于 pygame），接口与 `RemoteController` 保持一致，方便部署时替换。

    说明：在 Linux 上需要先安装 `pygame`：
        pip install pygame

    默认轴映射（常见 Linux 映射）：
        axis 0: left stick horizontal
        axis 1: left stick vertical
        axis 3: right stick horizontal
        axis 4: right stick vertical

    通过 `axis_mapping` 可以自定义将手柄轴映射到虚拟的 16 通道之一（与 RemoteController 的 `channels` 对齐）。
    """

    def __init__(
        self,
        device_index: int = 0,
        axis_mapping: Optional[Dict[int, int]] = None,
        poll_hz: float = 50.0,
        stick_min: int = 174,
        stick_max: int = 1811,
        stick_deadband: int = 50,
    ):
        if pygame is None:
            raise RuntimeError("pygame not available, install with `pip install pygame` to use XboxController")

        self.device_index = device_index
        self.axis_mapping = axis_mapping or {
            1: 0,  # channel 1 -> left stick vertical
            2: 1,  # channel 2 -> left stick horizontal
            4: 3,  # channel 4 -> right stick horizontal
        }
        # channels mirror RemoteController: 16 channels
        self.channels = [ (stick_min + stick_max)//2 ] * 16
        self.failsafe_status = 0
        self.link_stats = { 'vtx_voltage': 0.0 }

        self.STICK_MIN = stick_min
        self.STICK_MAX = stick_max
        self.STICK_CENTER = (self.STICK_MIN + self.STICK_MAX) // 2
        self.STICK_DEADBAND = stick_deadband

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._joystick = None
        self._data_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_update_time = 0.0
        self.poll_hz = poll_hz
        self.alpha = 0.5
        self.last_cmd: Optional[np.ndarray] = None
        self._data_lock = threading.Lock()
        self.buttons = []

    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self._data_callback = callback

    def _axis_to_channel_value(self, axis_val: float) -> int:
        # axis_val in [-1, 1] -> map to [STICK_MIN, STICK_MAX]
        v = float(axis_val)
        # flip vertical axis sign convention if needed (user can adjust mapping externally)
        mid = self.STICK_CENTER
        half_range = (self.STICK_MAX - self.STICK_MIN) / 2.0
        scaled = int(round(mid + v * half_range))
        return max(self.STICK_MIN, min(self.STICK_MAX, scaled))

    def _poll_loop(self):
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= 0:
                print("XboxController: no joystick detected")
            else:
                idx = min(self.device_index, pygame.joystick.get_count()-1)
                self._joystick = pygame.joystick.Joystick(idx)
                self._joystick.init()
                # initialize buttons state
                self.buttons = [0] * self._joystick.get_numbuttons()

            period = 1.0 / max(1.0, self.poll_hz)
            while self._running:
                start = time.time()
                pygame.event.pump()

                with self._data_lock:
                    # update mapped axes
                    for ch, axis in self.axis_mapping.items():
                        val = 0.0
                        if self._joystick is not None and axis < self._joystick.get_numaxes():
                            val = self._joystick.get_axis(axis)
                        # map to RC-like integer range
                        self.channels[ch-1] = self._axis_to_channel_value(val)
                    self._last_update_time = time.time()
                    # update buttons if joystick present
                    if self._joystick is not None:
                        nbuttons = self._joystick.get_numbuttons()
                        # ensure list length
                        if len(self.buttons) != nbuttons:
                            self.buttons = [0] * nbuttons
                        for bi in range(nbuttons):
                            try:
                                self.buttons[bi] = int(self._joystick.get_button(bi))
                            except Exception:
                                self.buttons[bi] = 0

                if self._data_callback is not None:
                    try:
                        self._data_callback(self.get_all_data())
                    except Exception:
                        pass

                elapsed = time.time() - start
                to_sleep = period - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        except Exception as e:
            print(f"XboxController poll error: {e}")
        finally:
            pygame.joystick.quit()
            pygame.quit()

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_channel(self, channel: int) -> int:
        if 1 <= channel <= 16:
            with self._data_lock:
                return int(self.channels[channel-1])
        raise ValueError("Channel must be between 1 and 16")

    def normalize_stick(self, channel: int, apply_deadband: bool = True) -> float:
        raw = self.get_channel(channel)
        # convert back to -1..1
        centered = raw - self.STICK_CENTER
        if apply_deadband and abs(centered) <= self.STICK_DEADBAND:
            return 0.0
        if centered > 0:
            if apply_deadband:
                return (centered - self.STICK_DEADBAND) / (self.STICK_MAX - self.STICK_CENTER - self.STICK_DEADBAND)
            else:
                return centered / (self.STICK_MAX - self.STICK_CENTER)
        else:
            if apply_deadband:
                return (centered + self.STICK_DEADBAND) / (self.STICK_CENTER - self.STICK_MIN - self.STICK_DEADBAND)
            else:
                return centered / (self.STICK_CENTER - self.STICK_MIN)

    def get_cmd(self) -> np.ndarray:
        # follow RemoteController mapping
        v_y = -self.normalize_stick(1)
        v_x = -self.normalize_stick(2)
        w_z = -self.normalize_stick(4)
        raw_cmd = np.array([v_x, v_y, w_z], dtype=np.float32)
        if self.last_cmd is None:
            self.last_cmd = raw_cmd.copy()
        filtered_cmd = self.alpha * raw_cmd + (1 - self.alpha) * self.last_cmd
        self.last_cmd = filtered_cmd.copy()
        return filtered_cmd

    def get_raw_sticks(self) -> dict:
        return {
            'channel_1': {'raw': self.get_channel(1), 'normalized': self.normalize_stick(1), 'description': 'right stick horizontal'},
            'channel_2': {'raw': self.get_channel(2), 'normalized': self.normalize_stick(2), 'description': 'right stick vertical'},
            'channel_3': {'raw': self.get_channel(3), 'normalized': self.normalize_stick(3), 'description': 'left stick vertical'},
            'channel_4': {'raw': self.get_channel(4), 'normalized': self.normalize_stick(4), 'description': 'left stick horizontal'},
        }

    def get_button(self, index: int) -> int:
        with self._data_lock:
            if 0 <= index < len(self.buttons):
                return int(self.buttons[index])
            return 0

    def is_connected(self, max_age: float = 2.0) -> bool:
        return (time.time() - self._last_update_time) < max_age if self._last_update_time > 0 else False

    def get_failsafe_status(self) -> int:
        return self.failsafe_status

    def get_link_stats(self) -> dict:
        return self.link_stats.copy()

    def get_all_data(self) -> dict:
        with self._data_lock:
            return {
                'channels': self.channels.copy(),
                'buttons': list(self.buttons),
                'failsafe_status': self.failsafe_status,
                'link_stats': self.link_stats.copy(),
                'last_update_time': self._last_update_time,
            }

    def print_stick_status(self):
        sticks = self.get_raw_sticks()
        cmd = self.get_cmd()
        print(f"\n=== Xbox 手柄状态 ===")
        for ch, data in sticks.items():
            print(f"{ch}: {data['raw']:4d} -> {data['normalized']:+6.3f} | {data['description']}")
        print(f"\n控制命令: [v_x={cmd[0]:+6.3f}, v_y={cmd[1]:+6.3f}, w_z={cmd[2]:+6.3f}]")
        print(f"连接状态: {'✅ 已连接' if self.is_connected() else '❌ 断开'}")

    def print_all_channels(self):
        with self._data_lock:
            print("\n=== 所有通道原始值 ===")
            for i, val in enumerate(self.channels, 1):
                print(f"Channel {i}: {val}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == '__main__':
    # 简单测试脚本
    try:
        xb = XboxController()
    except RuntimeError as e:
        print(e)
        print("请先安装 pygame: pip install pygame")
        raise

    def cb(data):
        pass

    xb.set_data_callback(cb)
    xb.start()
    try:
        for _ in range(100):
            time.sleep(0.1)
            if xb.is_connected():
                xb.print_stick_status()
    except KeyboardInterrupt:
        pass
    finally:
        xb.stop()
