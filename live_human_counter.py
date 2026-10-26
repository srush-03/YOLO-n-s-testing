from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Detect and count unique tracked people from a live camera."
    )
    parser.add_argument(
        "--model",
        default=str(project_dir / "yolo11n.pt"),
        help="Ultralytics model with a person/human class (default: yolo11n.pt).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--min-person-height",
        type=float,
        default=0.15,
        help="Minimum person-box height as a fraction of the camera frame.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=15,
        help="Frames a track must be observed before it is counted.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    person_class_ids = [
        class_id
        for class_id, name in model.names.items()
        if str(name).casefold() in {"person", "human"}
    ]
    if not person_class_ids:
        raise ValueError(
            f"{args.model!r} has no 'person' or 'human' class. "
            "Use yolo11n.pt, or train a model with human/person labels."
        )

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check camera permissions "
            "or try another index with --camera."
        )

    seen_track_ids: set[int] = set()
    track_hits: dict[int, tuple[int, int]] = {}
    frame_number = 0
    print("Camera started. Press q in the video window to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped supplying frames.")
            frame_number += 1

            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=person_class_ids,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                min_height = frame.shape[0] * args.min_person_height
                for track_id, (x1, y1, x2, y2) in zip(ids, boxes):
                    box_height = y2 - y1
                    box_width = x2 - x1
                    if box_height < min_height or box_height < box_width * 1.15:
                        continue

                    previous = track_hits.get(track_id)
                    if previous is not None and frame_number - previous[1] <= 5:
                        hits = previous[0] + 1
                    else:
                        hits = 1
                    track_hits[track_id] = (hits, frame_number)
                    if hits >= args.min_track_frames:
                        seen_track_ids.add(track_id)

            display = result.plot()
            cv2.putText(
                display,
                f"Unique people this session: {len(seen_track_ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Live human counter (press q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Unique tracked people this session: {len(seen_track_ids)}")


if __name__ == "__main__":
    main()
