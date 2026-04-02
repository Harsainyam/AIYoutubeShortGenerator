import json

import numpy as np

def is_overlapping(c1, c2):
    return not (c1["end"] <= c2["start"] or c1["start"] >= c2["end"])

def combine_and_select(
    audio_clips,      # from get_top_audio_clips()
    llm_clips,        # from get_nlp_scores_groq()
    scene_windows,    # from create_scene_window_scores()
    video_duration,   # total seconds of video
    num_clips=3,
    clip_len=60,
    audio_weight=0.3,
    llm_weight=0.5,
    scene_weight=0.2,
):
    # Build per-second score arrays for entire video
    duration = int(video_duration) + 1
    audio_arr = np.zeros(duration)
    llm_arr   = np.zeros(duration)
    scene_arr = np.zeros(duration)

    # Fill audio scores — each clip covers a range of seconds
    for clip in audio_clips:
        s, e = int(clip["start"]), int(clip["end"])
        audio_arr[s:e] += clip["score"]

    # Fill LLM scores
    for clip in llm_clips:
        s, e = int(clip["start"]), int(clip["end"])
        llm_arr[s:e] += clip["score"]

    # Fill scene scores — already windowed
    for w in scene_windows:
        s, e = int(w["start"]), int(w["end"])
        scene_arr[s:e] += w["score"]

    # Normalize each to 0-1
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    audio_arr = norm(audio_arr)
    llm_arr   = norm(llm_arr)
    scene_arr = norm(scene_arr)

    # Combine into single score array
    combined = (
        audio_weight * audio_arr +
        llm_weight   * llm_arr   +
        scene_weight * scene_arr
    )

    # Slide a window of clip_len seconds, sum scores inside
    candidates = []
    for start in range(0, duration - clip_len, 2):
        end = start + clip_len
        window_score = float(combined[start:end].mean())
        candidates.append({
            "start": float(start),
            "end":   float(end),
            "length": clip_len,
            "score": round(window_score, 6)
        })

    # Sort and pick top N non-overlapping
    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for c in candidates:
        if len(selected) >= num_clips:
            break
        if not any(is_overlapping(c, s) for s in selected):
            selected.append(c)

    return selected