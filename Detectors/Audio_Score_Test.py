import librosa
import numpy as np

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def smooth_signal(signal, window_size=5):
    return np.convolve(
        signal,
        np.ones(window_size) / window_size,
        mode='same'
    )


def is_overlapping(c1, c2):
    return not (c1["end"] <= c2["start"] or c1["start"] >= c2["end"])

def extract_audio_scores(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    rms_norm = normalize(rms)
    centroid_norm = normalize(centroid)
    zcr_norm = normalize(zcr)

    combined_score = (
        0.5 * rms_norm +
        0.3 * centroid_norm +
        0.2 * zcr_norm
    )

    smoothed_score = smooth_signal(combined_score, window_size=5)

    hop_length = 512
    times = librosa.frames_to_time(
        np.arange(len(smoothed_score)),
        sr=sr,
        hop_length=hop_length
    )

    return times, smoothed_score


def create_window_scores(times, scores, window_size=5, step_size=2):
    window_scores = []
    duration = times[-1]
    start = 0

    while start + window_size <= duration:
        end = start + window_size

        mask = (times >= start) & (times < end)
        window_energy = scores[mask]

        if len(window_energy) > 0:
            score = float(np.mean(window_energy))

            window_scores.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": score
            })

        start += step_size

    return window_scores



def generate_clips(window_scores,
                   min_len=50,
                   max_len=60,
                   clip_step=5):

    clips = []

    if not window_scores:
        return clips

    duration = window_scores[-1]["end"]
    clip_start = 0

    while clip_start + min_len <= duration:

        for length in range(min_len, max_len + 1, 2):
            clip_end = clip_start + length

            if clip_end > duration:
                continue

            score_sum = sum(
                w["score"]
                for w in window_scores
                if w["start"] >= clip_start and w["end"] <= clip_end
            )

            # Normalize by length so longer clips aren't unfairly favored
            normalized_score = score_sum / length

            clips.append({
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "length": length,
                "score": float(normalized_score)
            })

        clip_start += clip_step

    return clips



def select_top_k_clips(clips, k=3):
    sorted_clips = sorted(
        clips,
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    for clip in sorted_clips:
        if len(selected) >= k:
            break

        overlap = any(is_overlapping(clip, s) for s in selected)

        if not overlap:
            selected.append(clip)

    return selected


def get_top_audio_clips(audio_path,num_shorts=3,min_len=50,max_len=60):

    times, scores = extract_audio_scores(audio_path)

    window_scores = create_window_scores(
        times,
        scores,
        window_size=5,
        step_size=2
    )

    clips = generate_clips(
        window_scores,
        min_len=min_len,
        max_len=max_len
    )

    top_clips = select_top_k_clips(
        clips,
        k=num_shorts
    )

    return top_clips
