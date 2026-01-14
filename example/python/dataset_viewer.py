import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:  # pragma: no cover
    try:
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QSpinBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PyQt5"
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Qt binding not found. Install one of: PySide6 or PyQt5."
        ) from e

import cv2
import ArducamDepthCamera as ac


MAX_DISTANCE = 4000


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_frame_indices(record_dir: Path) -> list[int]:
    indices: list[int] = []
    if not record_dir.exists():
        return indices
    for p in record_dir.iterdir():
        name = p.name
        if not (name.startswith("depth_") and name.endswith(".npy")):
            continue
        suffix = name[len("depth_") : -len(".npy")]
        if suffix.isdigit():
            indices.append(int(suffix))
    indices.sort()
    return indices


def _dataset_paths(record_dir: Path, idx: int) -> tuple[Path, Path]:
    return (
        record_dir / f"depth_{idx:06d}.npy",
        record_dir / f"confidence_{idx:06d}.npy",
    )


def _to_pixmap_bgr(image_bgr: np.ndarray) -> QPixmap:
    if image_bgr.dtype != np.uint8:
        image_bgr = image_bgr.astype(np.uint8)
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(image_rgb.data, w, h, int(image_rgb.strides[0]), QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _to_pixmap_gray01(image01: np.ndarray) -> QPixmap:
    img = np.nan_to_num(image01)
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).astype(np.uint8)
    h, w = img_u8.shape[:2]
    qimg = QImage(img_u8.data, w, h, int(img_u8.strides[0]), QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


def _compute_preview(depth: np.ndarray, confidence: np.ndarray, depth_range: float, confidence_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    depth_u8 = (depth * (255.0 / float(depth_range))).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_RAINBOW)
    conf = np.nan_to_num(confidence)
    colored[conf < float(confidence_threshold)] = (0, 0, 0)

    conf_vis = conf.copy().astype(np.float32)
    cv2.normalize(conf_vis, conf_vis, 1.0, 0.0, cv2.NORM_MINMAX)
    return colored, conf_vis


@dataclass
class Dataset:
    path: Path
    meta: dict
    indices: list[int]

    @property
    def range(self) -> float:
        return float(self.meta.get("range", MAX_DISTANCE))


class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("Arducam ToF Dataset Viewer")

        self.args = args

        self.confidence_threshold = 30
        self._camera: Optional[ac.ArducamCamera] = None
        self._camera_range: float = float(MAX_DISTANCE)
        self._recording: bool = False
        self._record_dir: Optional[Path] = None
        self._record_count: int = 0
        self._record_ts: list[float] = []

        self._dataset: Optional[Dataset] = None
        self._playback_idx_pos: int = 0
        self._playback_playing: bool = False

        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._on_live_tick)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        central = QWidget(self)
        root = QHBoxLayout(central)

        root.addWidget(self._build_left_panel(), 1)
        root.addWidget(self._build_right_panel(), 3)

        self.setCentralWidget(central)

        self.status_label = QLabel(f"Qt: {QT_BINDING}")
        self.statusBar().addWidget(self.status_label)

        self._last_status_update_ts: float = 0.0

        if self.args.datasets_root is not None:
            self.datasets_root_edit.setText(self.args.datasets_root)
            self._refresh_dataset_list()

    def _build_left_panel(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        root_box = QGroupBox("Datasets")
        root_layout = QVBoxLayout(root_box)

        row = QHBoxLayout()
        self.datasets_root_edit = QLineEdit()
        self.datasets_root_btn = QPushButton("Browse")
        self.datasets_root_btn.clicked.connect(self._choose_datasets_root)
        row.addWidget(self.datasets_root_edit, 1)
        row.addWidget(self.datasets_root_btn)
        root_layout.addLayout(row)

        self.datasets_list = QListWidget()
        self.datasets_list.itemSelectionChanged.connect(self._on_dataset_selected)
        root_layout.addWidget(self.datasets_list, 1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_dataset_list)
        root_layout.addWidget(self.refresh_btn)

        layout.addWidget(root_box, 1)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "Live")
        self.tabs.addTab(self._build_playback_tab(), "Playback")
        layout.addWidget(self.tabs, 0)

        views = QHBoxLayout()
        self.depth_label = QLabel("Depth")
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setMinimumSize(320, 240)

        self.conf_label = QLabel("Confidence")
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_label.setMinimumSize(320, 240)

        views.addWidget(self.depth_label, 1)
        views.addWidget(self.conf_label, 1)
        layout.addLayout(views, 1)

        return w

    def _build_live_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        cfg_row = QHBoxLayout()
        self.cfg_edit = QLineEdit(self.args.cfg or "")
        self.cfg_btn = QPushButton("Choose cfg")
        self.cfg_btn.clicked.connect(self._choose_cfg)
        cfg_row.addWidget(QLabel("cfg:"))
        cfg_row.addWidget(self.cfg_edit, 1)
        cfg_row.addWidget(self.cfg_btn)
        layout.addLayout(cfg_row)

        controls_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_camera)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._disconnect_camera)
        controls_row.addWidget(self.connect_btn)
        controls_row.addWidget(self.disconnect_btn)
        layout.addLayout(controls_row)

        record_row = QHBoxLayout()
        self.record_dir_edit = QLineEdit(self.args.record_dir or "")
        self.record_dir_btn = QPushButton("Record dir")
        self.record_dir_btn.clicked.connect(self._choose_record_dir)
        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self._toggle_record)
        record_row.addWidget(self.record_dir_edit, 1)
        record_row.addWidget(self.record_dir_btn)
        record_row.addWidget(self.record_btn)
        layout.addLayout(record_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Confidence thr:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(255)
        self.conf_slider.setValue(self.confidence_threshold)
        self.conf_slider.valueChanged.connect(self._on_conf_thr_changed)
        thr_row.addWidget(self.conf_slider, 1)
        layout.addLayout(thr_row)

        return w

    def _build_playback_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        open_row = QHBoxLayout()
        self.open_dataset_btn = QPushButton("Open dataset")
        self.open_dataset_btn.clicked.connect(self._open_dataset_dialog)
        self.dataset_path_label = QLabel("-")
        self.dataset_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_row.addWidget(self.open_dataset_btn)
        open_row.addWidget(self.dataset_path_label, 1)
        layout.addLayout(open_row)

        play_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(1)
        self.fps_spin.setMaximum(120)
        self.fps_spin.setValue(int(self.args.playback_fps))
        self.fps_spin.valueChanged.connect(self._update_play_timer)
        play_row.addWidget(self.play_btn)
        play_row.addWidget(QLabel("FPS:"))
        play_row.addWidget(self.fps_spin)
        layout.addLayout(play_row)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self.frame_slider)

        return w

    def _show_error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def _choose_datasets_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose datasets root")
        if not path:
            return
        self.datasets_root_edit.setText(path)
        self._refresh_dataset_list()

    def _choose_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose camera cfg", filter="Config (*.cfg);;All files (*)")
        if not path:
            return
        self.cfg_edit.setText(path)

    def _choose_record_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose record dir")
        if not path:
            return
        self.record_dir_edit.setText(path)

    def _refresh_dataset_list(self) -> None:
        self.datasets_list.clear()
        root = self.datasets_root_edit.text().strip()
        if not root:
            return
        root_path = Path(root)
        if not root_path.exists():
            return

        items: list[Path] = []
        for p in root_path.iterdir():
            if not p.is_dir():
                continue
            if (p / "meta.json").exists() or any(p.glob("depth_*.npy")):
                items.append(p)

        items.sort(key=lambda x: x.name)
        for p in items:
            self.datasets_list.addItem(p.name)

    def _on_dataset_selected(self) -> None:
        root = self.datasets_root_edit.text().strip()
        if not root:
            return
        selected = self.datasets_list.selectedItems()
        if not selected:
            return
        name = selected[0].text()
        self._load_dataset(Path(root) / name)

    def _open_dataset_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open dataset")
        if not path:
            return
        self._load_dataset(Path(path))

    def _load_dataset(self, path: Path) -> None:
        indices = _list_frame_indices(path)
        if len(indices) == 0:
            self._show_error("Dataset", "No frames found in selected dataset folder")
            return

        meta_path = path / "meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        self._dataset = Dataset(path=path, meta=meta, indices=indices)
        self.dataset_path_label.setText(str(path))

        self._playback_idx_pos = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(indices) - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self._render_dataset_frame(0)
        self.status_label.setText(f"Loaded dataset: {path.name} ({len(indices)} frames)")

    def _update_play_timer(self) -> None:
        fps = max(1, int(self.fps_spin.value()))
        self._play_timer.setInterval(max(1, int(1000.0 / float(fps))))

    def _toggle_play(self) -> None:
        if self._dataset is None:
            self._show_error("Playback", "No dataset loaded")
            return
        self._playback_playing = not self._playback_playing
        if self._playback_playing:
            self.play_btn.setText("Pause")
            self._update_play_timer()
            self._play_timer.start()
        else:
            self.play_btn.setText("Play")
            self._play_timer.stop()

    def _on_frame_slider_changed(self, value: int) -> None:
        if self._dataset is None:
            return
        self._playback_idx_pos = int(value)
        self._render_dataset_frame(self._playback_idx_pos)

    def _on_play_tick(self) -> None:
        if self._dataset is None:
            self._play_timer.stop()
            return
        next_pos = self._playback_idx_pos + 1
        if next_pos >= len(self._dataset.indices):
            next_pos = 0
        self._playback_idx_pos = next_pos
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_pos)
        self.frame_slider.blockSignals(False)
        self._render_dataset_frame(next_pos)

    def _render_dataset_frame(self, pos: int) -> None:
        if self._dataset is None:
            return
        idx = self._dataset.indices[int(pos)]
        depth_path, conf_path = _dataset_paths(self._dataset.path, idx)
        depth = np.load(depth_path)
        conf = np.load(conf_path)
        preview, conf_vis = _compute_preview(depth, conf, self._dataset.range, self.confidence_threshold)
        self._show_frames(preview, conf_vis)

    def _on_conf_thr_changed(self, value: int) -> None:
        self.confidence_threshold = int(value)
        if self._dataset is not None:
            self._render_dataset_frame(self._playback_idx_pos)

    def _connect_camera(self) -> None:
        if self._camera is not None:
            return

        cam = ac.ArducamCamera()
        cfg = self.cfg_edit.text().strip() or None

        if cfg is not None:
            ret = cam.openWithFile(cfg, 0)
        else:
            ret = cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            self._show_error("Camera", f"Failed to open camera. Error code: {ret}")
            return

        ret = cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            cam.close()
            self._show_error("Camera", f"Failed to start camera. Error code: {ret}")
            return

        cam.setControl(ac.Control.RANGE, int(self.args.max_distance))
        self._camera_range = float(cam.getControl(ac.Control.RANGE))

        self._camera = cam
        # Keep UI responsive: pull frames ~30 FPS in the GUI thread.
        self._live_timer.start(33)
        self.status_label.setText("Camera connected")

    def _disconnect_camera(self) -> None:
        self._stop_recording(save_timestamps=True)

        if self._camera is None:
            return

        self._live_timer.stop()
        try:
            self._camera.stop()
        finally:
            self._camera.close()
        self._camera = None
        self.status_label.setText("Camera disconnected")

    def _toggle_record(self) -> None:
        if self._camera is None:
            self._show_error("Record", "Connect camera first")
            return

        if self._recording:
            self._stop_recording(save_timestamps=True)
            return

        record_dir_raw = self.record_dir_edit.text().strip()
        if not record_dir_raw:
            self._show_error("Record", "Choose record dir")
            return

        record_dir = Path(record_dir_raw)
        _ensure_dir(record_dir)
        append_mode = False
        existing = list(record_dir.glob("depth_*.npy"))
        if existing:
            msg = QMessageBox(self)
            msg.setWindowTitle("Record")
            msg.setText("Selected folder already contains a dataset. Choose what to do:")
            btn_new = msg.addButton("Create new subfolder", QMessageBox.ButtonRole.AcceptRole)
            btn_append = msg.addButton("Append", QMessageBox.ButtonRole.ActionRole)
            btn_cancel = msg.addButton(QMessageBox.StandardButton.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_cancel:
                return
            if clicked == btn_new:
                base = record_dir
                stamp = time.strftime("run_%Y%m%d_%H%M%S")
                record_dir = base / stamp
                suffix = 1
                while record_dir.exists() and any(record_dir.iterdir()):
                    record_dir = base / f"{stamp}_{suffix}"
                    suffix += 1
                _ensure_dir(record_dir)
            elif clicked == btn_append:
                append_mode = True

        meta_path = record_dir / "meta.json"
        if not meta_path.exists():
            meta = {
                "range": float(self._camera_range),
                "max_distance": int(self.args.max_distance),
                "device_type": str(self._camera.getCameraInfo().device_type),
                "time": time.time(),
            }
            _write_json(meta_path, meta)

        self._recording = True
        self._record_dir = record_dir
        if append_mode:
            indices = _list_frame_indices(record_dir)
            self._record_count = (indices[-1] + 1) if len(indices) > 0 else 0
            ts_path = record_dir / "timestamps.npy"
            if ts_path.exists():
                try:
                    self._record_ts = [float(x) for x in np.load(ts_path).reshape(-1).tolist()]
                except Exception:
                    self._record_ts = []
            else:
                self._record_ts = []
        else:
            self._record_count = 0
            self._record_ts = []
        self.record_btn.setText("Stop")
        if append_mode:
            self.status_label.setText(f"Recording (append) to: {record_dir}")
        else:
            self.status_label.setText(f"Recording to: {record_dir}")

    def _stop_recording(self, save_timestamps: bool) -> None:
        if not self._recording:
            return
        if save_timestamps and self._record_dir is not None and len(self._record_ts) > 0:
            np.save(self._record_dir / "timestamps.npy", np.asarray(self._record_ts, dtype=np.float64))

        self._recording = False
        self._record_dir = None
        self._record_count = 0
        self._record_ts = []
        self.record_btn.setText("Record")
        self.status_label.setText("Recording stopped")

    def _on_live_tick(self) -> None:
        if self._camera is None:
            return

        frame = self._camera.requestFrame(200)
        if frame is None or not isinstance(frame, ac.DepthData):
            return

        depth = frame.depth_data
        conf = frame.confidence_data

        preview, conf_vis = _compute_preview(depth, conf, self._camera_range, self.confidence_threshold)
        self._show_frames(preview, conf_vis)

        if self._recording and self._record_dir is not None:
            try:
                depth_path, conf_path = _dataset_paths(self._record_dir, self._record_count)
                np.save(depth_path, np.asarray(depth))
                np.save(conf_path, np.asarray(conf))
                self._record_ts.append(time.time())
                self._record_count += 1

                now = time.time()
                # Throttle status updates to avoid UI overhead.
                if now - self._last_status_update_ts >= 0.25:
                    self.status_label.setText(
                        f"REC {self._record_count} frames -> {self._record_dir}"
                    )
                    self._last_status_update_ts = now
            except Exception as e:
                record_dir = self._record_dir
                self._stop_recording(save_timestamps=False)
                self._show_error(
                    "Record",
                    "Failed to save dataset frame.\n"
                    f"Directory: {record_dir}\n"
                    f"Error: {type(e).__name__}: {e}",
                )
                self._camera.releaseFrame(frame)
                return

        self._camera.releaseFrame(frame)

    def _show_frames(self, preview_bgr: np.ndarray, conf_vis01: np.ndarray) -> None:
        pm_depth = _to_pixmap_bgr(preview_bgr)
        pm_conf = _to_pixmap_gray01(conf_vis01)

        self.depth_label.setPixmap(pm_depth.scaled(self.depth_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.conf_label.setPixmap(pm_conf.scaled(self.conf_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):  # noqa: N802
        try:
            self._play_timer.stop()
            self._live_timer.stop()
            self._disconnect_camera()
        finally:
            event.accept()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=None)
    p.add_argument("--max-distance", type=int, default=MAX_DISTANCE)
    p.add_argument("--datasets-root", type=str, default=None)
    p.add_argument("--record-dir", type=str, default=None)
    p.add_argument("--playback-fps", type=float, default=30.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication([])
    win = MainWindow(args)
    win.resize(1200, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
