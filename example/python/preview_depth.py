import argparse
import json
import os
import time

import cv2
import numpy as np
import ArducamDepthCamera as ac

# MAX_DISTANCE value modifiable  is 2000 or 4000
MAX_DISTANCE=4000


class UserRect:
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    @property
    def rect(self):
        return (
            self.start_x,
            self.start_y,
            self.end_x - self.start_x,
            self.end_y - self.start_y,
        )

    @property
    def slice(self):
        return (slice(self.start_y, self.end_y), slice(self.start_x, self.end_x))

    @property
    def empty(self):
        return self.start_x == self.end_x and self.start_y == self.end_y


confidence_value = 30
selectRect, followRect = UserRect(), UserRect()


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < confidence_value] = (0, 0, 0)
    return preview


def on_mouse(event, x, y, flags, param):
    global selectRect, followRect

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectRect.start_x = x - 4
        selectRect.start_y = y - 4
        selectRect.end_x = x + 4
        selectRect.end_y = y + 4
    else:
        followRect.start_x = x - 4
        followRect.start_y = y - 4
        followRect.end_x = x + 4
        followRect.end_y = y + 4


def on_confidence_changed(value):
    global confidence_value
    confidence_value = value


def usage(argv0):
    print("Usage: python " + argv0 + " [options]")
    print("Available options are:")
    print(" -d        Choose the video to use")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_frame_indices(record_dir: str) -> list[int]:
    indices = []
    for name in os.listdir(record_dir):
        if not (name.startswith("depth_") and name.endswith(".npy")):
            continue
        suffix = name[len("depth_") : -len(".npy")]
        if suffix.isdigit():
            indices.append(int(suffix))
    indices.sort()
    return indices


def _dataset_paths(record_dir: str, idx: int) -> tuple[str, str]:
    return (
        os.path.join(record_dir, f"depth_{idx:06d}.npy"),
        os.path.join(record_dir, f"confidence_{idx:06d}.npy"),
    )


def main():
    print("Arducam Depth Camera Demo.")
    print("  SDK version:", ac.__version__)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--max-distance", type=int, default=MAX_DISTANCE)
    parser.add_argument("--record-dir", type=str, default=None)
    parser.add_argument("--playback-dir", type=str, default=None)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    args = parser.parse_args()

    if args.record_dir is not None and args.playback_dir is not None:
        raise SystemExit("Use only one of --record-dir or --playback-dir")

    cam = None
    cfg_path = args.cfg

    black_color = (0, 0, 0)
    white_color = (255, 255, 255)

    record_dir = args.record_dir
    playback_dir = args.playback_dir
    is_playback = playback_dir is not None

    if is_playback:
        meta_path = os.path.join(playback_dir, "meta.json")
        if os.path.exists(meta_path):
            meta = _read_json(meta_path)
            r = float(meta.get("range", MAX_DISTANCE))
            device_type = meta.get("device_type", None)
        else:
            meta = {}
            r = float(MAX_DISTANCE)
            device_type = None

        indices = _list_frame_indices(playback_dir)
        if len(indices) == 0:
            raise SystemExit("No frames found in playback dir")

        sample_depth_path, sample_conf_path = _dataset_paths(playback_dir, indices[0])
        depth_buf0 = np.load(sample_depth_path)
        confidence_buf0 = np.load(sample_conf_path)
        info_width, info_height = int(depth_buf0.shape[1]), int(depth_buf0.shape[0])
        print(f"Playback resolution: {info_width}x{info_height}")
    else:
        cam = ac.ArducamCamera()
        ret = 0
        if cfg_path is not None:
            ret = cam.openWithFile(cfg_path, 0)
        else:
            ret = cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            print("Failed to open camera. Error code:", ret)
            return

        ret = cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            print("Failed to start camera. Error code:", ret)
            cam.close()
            return

        cam.setControl(ac.Control.RANGE, int(args.max_distance))

        r = cam.getControl(ac.Control.RANGE)

        info = cam.getCameraInfo()
        device_type = info.device_type
        print(f"Camera resolution: {info.width}x{info.height}")

    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("preview", on_mouse)

    if device_type == ac.DeviceType.VGA:
        # Only VGA support confidence
        cv2.createTrackbar(
            "confidence", "preview", confidence_value, 255, on_confidence_changed
        )

    recording = record_dir is not None
    record_count = 0
    record_ts = []
    if recording:
        _ensure_dir(record_dir)
        _write_json(
            os.path.join(record_dir, "meta.json"),
            {
                "range": float(r),
                "max_distance": int(args.max_distance),
                "device_type": str(device_type),
                "time": time.time(),
            },
        )

    while True:
        if is_playback:
            idx = indices[record_count % len(indices)]
            depth_path, conf_path = _dataset_paths(playback_dir, idx)
            depth_buf = np.load(depth_path)
            confidence_buf = np.load(conf_path)
        else:
            frame = cam.requestFrame(2000)
            if frame is None or not isinstance(frame, ac.DepthData):
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                continue

            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data

        result_image = (depth_buf * (255.0 / float(r))).astype(np.uint8)
        result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
        result_image = getPreviewRGB(result_image, confidence_buf)

        confidence_vis = confidence_buf.copy()
        cv2.normalize(confidence_vis, confidence_vis, 1, 0, cv2.NORM_MINMAX)

        cv2.imshow("preview_confidence", confidence_vis)

        cv2.rectangle(result_image, followRect.rect, white_color, 1)
        if not selectRect.empty:
            cv2.rectangle(result_image, selectRect.rect, black_color, 2)
            print("select Rect distance:", np.mean(depth_buf[selectRect.slice]))

        if recording:
            cv2.putText(
                result_image,
                f"REC {record_count}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        if is_playback:
            cv2.putText(
                result_image,
                f"PLAY {record_count+1}/{len(indices)}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("preview", result_image)

        if not is_playback and frame is not None:
            if recording:
                depth_path, conf_path = _dataset_paths(record_dir, record_count)
                np.save(depth_path, np.asarray(depth_buf))
                np.save(conf_path, np.asarray(confidence_buf))
                record_ts.append(time.time())
                record_count += 1

            cam.releaseFrame(frame)

        if is_playback:
            key = cv2.waitKey(max(1, int(1000.0 / max(1e-3, float(args.playback_fps)))))
            record_count += 1
        else:
            key = cv2.waitKey(1)

        if key == ord("q"):
            break
        if key == ord("r") and record_dir is not None:
            recording = not recording

    if record_dir is not None and len(record_ts) > 0:
        np.save(os.path.join(record_dir, "timestamps.npy"), np.asarray(record_ts, dtype=np.float64))

    if cam is not None:
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()
