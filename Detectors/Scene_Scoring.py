import cv2
import numpy as np

def get_scene_scores(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    scores = []
    times = []
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale — faster, scene change doesn't need color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Absolute difference between frames
            diff = cv2.absdiff(gray, prev_frame)
            score = float(np.mean(diff))
        else:
            score = 0.0

        timestamp = frame_idx / fps
        times.append(round(timestamp, 3))
        scores.append(score)

        prev_frame = gray
        frame_idx += 1

    cap.release()
    return times, scores


def create_scene_window_scores(video_path, window_size=5, step_size=2):
    times, scores = get_scene_scores(video_path)
    times = np.array(times)
    scores = np.array(scores)

    # Normalize to 0-1
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    window_scores = []
    duration = times[-1]
    start = 0.0

    while start + window_size <= duration:
        end = start + window_size
        mask = (times >= start) & (times < end)
        window_energy = scores[mask]

        if len(window_energy) > 0:
            window_scores.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": float(np.mean(window_energy))
            })

        start += step_size

    return times,window_scores