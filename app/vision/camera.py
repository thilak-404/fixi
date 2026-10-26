"""
Camera input module for Fixi.
Supports local webcams (/dev/video*), threaded buffered capture,
and synthetic/virtual cameras for headless testing and simulation.
"""

import cv2
import numpy as np
import threading
import time
import os
import subprocess
import select
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from app.config.settings import settings


class BaseCamera(ABC):
    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def is_active(self) -> bool:
        pass

    def get_jpeg(self) -> Optional[bytes]:
        success, frame = self.read_frame()
        if not success or frame is None:
            return None
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ret else None


class AdbMotorolaCamera(BaseCamera):
    """
    Direct, 100% rootless ADB camera streamer using native Android Camera2 API via scrcpy.
    Connects directly to the Motorola phone via ADB without requiring root, sudo,
    v4l2loopback, or third-party apps like DroidCam.
    Pipes hardware-encoded video frames directly into Python/OpenCV in real-time.
    """
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 15):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._fifo_path = "/tmp/fixi_adb_cam.mkv"
        self._p_scrcpy: Optional[subprocess.Popen] = None
        self._p_ffmpeg: Optional[subprocess.Popen] = None

    @staticmethod
    def is_device_connected() -> bool:
        try:
            res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=2.0)
            return any(len(line.split()) >= 2 and line.split()[1] == "device" for line in res.stdout.splitlines())
        except Exception:
            return False

    def start(self) -> bool:
        if self._running:
            return True
        if not self.is_device_connected():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

        # Wait up to 12 seconds for the first live frame
        for _ in range(60):
            if self._latest_frame is not None:
                print("AdbMotorolaCamera: Live Motorola camera stream acquired successfully over ADB!")
                return True
            time.sleep(0.2)
        # Even if first frame takes slightly longer, keep running so worker continues
        return True

    def _cleanup_procs(self):
        for proc in (self._p_ffmpeg, self._p_scrcpy):
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._p_ffmpeg = None
        self._p_scrcpy = None
        if os.path.exists(self._fifo_path):
            try:
                os.remove(self._fifo_path)
            except Exception:
                pass

    def _capture_worker(self):
        while self._running:
            self._cleanup_procs()
            if not self.is_device_connected():
                time.sleep(2.0)
                continue

            try:
                subprocess.run(
                    ["adb", "shell", "input keyevent KEYCODE_WAKEUP; svc power stayon usb"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3.0, check=False
                )

                if os.path.exists(self._fifo_path):
                    os.remove(self._fifo_path)
                os.mkfifo(self._fifo_path)

                scrcpy_cmd = [
                    "scrcpy", "--video-source=camera", f"--camera-id={self.camera_id}",
                    "--no-audio", "--no-playback", f"--max-fps={self.target_fps}",
                    "-m", str(max(self.width, self.height)),
                    f"--record={self._fifo_path}", "--record-format=mkv"
                ]
                self._p_scrcpy = subprocess.Popen(
                    scrcpy_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

                ffmpeg_cmd = [
                    "ffmpeg", "-v", "error", "-i", self._fifo_path,
                    "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"
                ]
                self._p_ffmpeg = subprocess.Popen(
                    ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )

                buf = b""
                while self._running:
                    r, _, _ = select.select([self._p_ffmpeg.stdout], [], [], 1.5)
                    if not r:
                        if self._p_scrcpy and self._p_scrcpy.poll() is not None:
                            print("AdbMotorolaCamera: scrcpy process exited, reconnecting...")
                            break
                        if self._p_ffmpeg and self._p_ffmpeg.poll() is not None:
                            print("AdbMotorolaCamera: ffmpeg process exited, reconnecting...")
                            break
                        continue
                    chunk = self._p_ffmpeg.stdout.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        a = buf.find(b"\xff\xd8")
                        b = buf.find(b"\xff\xd9")
                        if a != -1 and b != -1 and b > a:
                            jpg = buf[a : b + 2]
                            buf = buf[b + 2 :]
                            arr = np.frombuffer(jpg, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is not None and frame.size > 0:
                                with self._lock:
                                    self._latest_frame = frame
                        else:
                            break
            except Exception as e:
                print(f"AdbMotorolaCamera worker error: {e}")
                time.sleep(1.0)
            finally:
                self._cleanup_procs()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()

        # While initializing the ADB stream, return a clean connecting HUD frame
        connecting_frame = np.full((self.height, self.width, 3), 25, dtype=np.uint8)
        cv2.putText(
            connecting_frame,
            "Connecting to Motorola Camera over ADB...",
            (30, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2
        )
        return True, connecting_frame

    def is_active(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        self._cleanup_procs()


class WebcamCamera(BaseCamera):
    """
    Real-time threaded OpenCV webcam reader.
    Runs a background grabber thread so frame acquisition never blocks main AI inference.
    """
    def __init__(
        self,
        device_index: int = settings.CAMERA_INDEX,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
        fps: int = settings.CAMERA_FPS
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._has_new_frame = False
        self._phone_stream_process: Optional[subprocess.Popen] = None

    def _ensure_phone_camera_stream(self) -> None:
        """Start the local Motorola/scrcpy bridge when an ADB device is available."""
        script = settings.BASE_PATH / "scripts" / "start_camera_stream.sh"
        if not script.exists() or not os.access(script, os.X_OK):
            return
        try:
            device_check = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=2.0, check=False
            )
            has_device = any(
                len(line.split()) >= 2 and line.split()[1] == "device"
                for line in device_check.stdout.splitlines()
            )
            # 1. Forward DroidCam port over USB
            subprocess.run(["adb", "forward", "tcp:4747", "tcp:4747"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            subprocess.run(
                ["adb", "shell", "input keyevent KEYCODE_WAKEUP; am start -n com.dev47apps.obsdroidcam/.MainActivity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
        except (OSError, subprocess.SubprocessError):
            self._phone_stream_process = None

    def start(self) -> bool:
        if self._running:
            return True
        self._ensure_phone_camera_stream()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait up to 1 second for first frame
        for _ in range(10):
            if self._latest_frame is not None:
                return True
            time.sleep(0.1)
        return self.is_active()

    def _open_device(self) -> Optional[cv2.VideoCapture]:
        def configure(cap: cv2.VideoCapture) -> Optional[cv2.VideoCapture]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                return cap
            cap.release()
            return None

        # 1. Try local ADB Motorola phone stream (100% rootless over USB)
        try:
            subprocess.run(["adb", "forward", "tcp:4747", "tcp:4747"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            droid_cap = cv2.VideoCapture("http://127.0.0.1:4747/video")
            if droid_cap.isOpened():
                ret, test_frame = droid_cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print("Connected to live Motorola camera over ADB USB bridge.")
                    return droid_cap
                droid_cap.release()
        except Exception:
            pass

        # 2. Try device index first
        cap = cv2.VideoCapture(self.device_index)
        if cap.isOpened():
            configured = configure(cap)
            if configured is not None:
                return configured

        # 3. Try V4L2 backend with device index
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if cap.isOpened():
            configured = configure(cap)
            if configured is not None:
                return configured

        # 4. Try explicitly /dev/video0
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if cap.isOpened():
            configured = configure(cap)
            if configured is not None:
                return configured

        return None

    def _capture_loop(self) -> None:
        delay = 1.0 / max(1, self.target_fps)
        consecutive_errors = 0
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_device()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(1.0)
                    continue
                consecutive_errors = 0

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    consecutive_errors = 0
                    with self._lock:
                        self._latest_frame = frame
                        self._has_new_frame = True
                    time.sleep(delay)
                else:
                    consecutive_errors += 1
                    time.sleep(0.05)
                    # Only release device after 15 consecutive failed frame reads (~1 second)
                    if consecutive_errors >= 15:
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        consecutive_errors = 0
                        time.sleep(0.5)
            except Exception:
                consecutive_errors += 1
                time.sleep(0.1)
                if consecutive_errors >= 10:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    consecutive_errors = 0

    def push_external_frame(self, frame: np.ndarray) -> None:
        """Allows injecting a frame from browser webcam or snapshot."""
        if frame is not None and frame.size > 0:
            with self._lock:
                self._latest_frame = frame
                self._has_new_frame = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._phone_stream_process and self._phone_stream_process.poll() is None:
            try:
                os.killpg(self._phone_stream_process.pid, 15)
            except OSError:
                pass
        self._phone_stream_process = None

    def is_active(self) -> bool:
        with self._lock:
            has_frame = self._latest_frame is not None
        return self._running and ((self.cap is not None and self.cap.isOpened()) or has_frame)


class SyntheticCamera(BaseCamera):
    """
    Virtual camera for automated testing, headless mode, and hardware-independent demos.
    Generates dynamic frames with a synthetic person avatar and customizable text.
    """
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._running = False
        self.simulated_person_name = "Arun"
        self.draw_person = True

    def start(self) -> bool:
        self._running = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None

        # Create dark gradient background
        frame = np.full((self.height, self.width, 3), 30, dtype=np.uint8)

        # Draw grid
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (45, 45, 45), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (45, 45, 45), 1)

        if self.draw_person:
            # Draw synthetic person silhouette / avatar
            cx, cy = self.width // 2, self.height // 2
            # Head
            cv2.circle(frame, (cx, cy - 60), 45, (180, 160, 140), -1)
            # Eyes
            cv2.circle(frame, (cx - 15, cy - 65), 5, (40, 40, 40), -1)
            cv2.circle(frame, (cx + 15, cy - 65), 5, (40, 40, 40), -1)
            # Smile
            cv2.ellipse(frame, (cx, cy - 50), (18, 10), 0, 0, 180, (40, 40, 40), 2)
            # Torso
            cv2.ellipse(frame, (cx, cy + 80), (80, 100), 0, 0, 360, (70, 130, 180), -1)

            # Label
            cv2.putText(
                frame,
                f"Synthetic Feed ({self.simulated_person_name})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 200),
                2
            )

        return True, frame

    def stop(self) -> None:
        self._running = False

    def is_active(self) -> bool:
        return self._running
