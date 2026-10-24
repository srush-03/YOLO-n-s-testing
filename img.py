import math
import cv2
from ultralytics import YOLO

# 1. Load YOLO11 Nano model
model = YOLO("yolov11n.pt")

# 2. Configure external camera source
CAMERA_SOURCE = 1
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open camera feed at source {CAMERA_SOURCE}.")
    exit()

# Global Permanent Registry: stores all unique humans ever seen in this session
registered_people = {}  # Format: { id: {"center": (cx, cy)} }
next_person_id = 1

# Generous distance threshold to recognize you even if you walk across the room
MAX_MATCH_DISTANCE = 400

print("Starting Global Re-ID Unique Counter...")
print("You and your father will now stay locked to your permanent IDs.")
print("Press 'q' to exit and view final total.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Run YOLO detection
    results = model(frame, classes=[0], conf=0.6, verbose=False)

    current_detections = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # FILTER: Ignore tiny edge boxes (area less than 15,000 pixels)
            if box_area < 15000:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            current_detections.append({"center": (cx, cy), "box": box})

    matched_registered_ids = set()
    detection_assignments = {}

    # Global matching against ALL registered people in the session
    if registered_people and current_detections:
        distances = []
        for d_idx, det in enumerate(current_detections):
            dx, dy = det["center"]
            for pid, pdata in registered_people.items():
                px, py = pdata["center"]
                dist = math.hypot(dx - px, dy - py)
                distances.append((dist, d_idx, pid))

        # Sort by shortest distance first to pair up correctly
        distances.sort(key=lambda x: x[0])

        assigned_detections = set()
        for dist, d_idx, pid in distances:
            if (
                d_idx not in assigned_detections
                and pid not in matched_registered_ids
            ):
                if dist < MAX_MATCH_DISTANCE:
                    assigned_detections.add(d_idx)
                    matched_registered_ids.add(pid)
                    detection_assignments[d_idx] = pid
                    # Update their last known position
                    registered_people[pid]["center"] = current_detections[d_idx][
                        "center"
                    ]

    # Any detection that couldn't be matched to an existing person gets a new ID
    for d_idx, det in enumerate(current_detections):
        if d_idx not in detection_assignments:
            new_id = next_person_id
            registered_people[new_id] = {"center": det["center"]}
            detection_assignments[d_idx] = new_id
            print(f"[NEW UNIQUE HUMAN] Assigned Permanent ID: #{new_id}")
            next_person_id += 1

    active_visible_count = len(current_detections)
    total_unique_count = len(registered_people)

    # Draw boxes and assigned IDs on screen
    for d_idx, det in enumerate(current_detections):
        assigned_id = detection_assignments[d_idx]
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Person ID: {assigned_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Render live stats
    cv2.putText(
        frame,
        f"Currently Visible: {active_visible_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Total Unique Count: {total_unique_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Global Re-ID Unique Human Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("SESSION SUMMARY")
print(
    f"Total distinct humans counted across entire session:"
    f" {len(registered_people)}"
)
print("=" * 50)