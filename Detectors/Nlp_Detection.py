import pysrt
from transformers import pipeline

# -----------------------------
# Emotion model
# -----------------------------

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# -----------------------------
# Keyword lists
# -----------------------------

viral_keywords = [
    "crazy",
    "secret",
    "insane",
    "unbelievable",
    "wait",
    "listen",
    "important",
    "shocking",
    "truth",
    "never knew",
    "trick",
    "this changed",
    "most people",
]

hooks = [
    "did you know",
    "what if",
    "here's why",
    "the reason",
    "but wait",
    "listen carefully",
    "let me show you",
    "Oh my god",
    "Oh my gosh!",
]


def time_to_seconds(t):
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000


def keyword_score(text):

    text = text.lower()
    score = 0

    for word in viral_keywords:
        if word in text:
            score += 1

    return score

def hook_score(text):

    text = text.lower()

    for h in hooks:
        if h in text:
            return 1

    return 0

def emotion_score(text):

    results = emotion_model(text[:500])

    score = 0

    for r in results[0]:
        if r["label"] in ["joy", "surprise", "anger"]:
            score += r["score"]

    return score

def nlp_score(text):

    k = keyword_score(text)
    h = hook_score(text)
    e = emotion_score(text)

    final_score = (
        0.4 * k +
        0.2 * h +
        0.4 * e
    )

    return final_score



def build_segments_from_srt(srt_path, min_len=50, max_len=60):

    subs = pysrt.open(srt_path)

    if not subs:
        return []

    segments = []
    current_text = ""
    start_time = None
    end_time = None

    for sub in subs:
        sub_start = time_to_seconds(sub.start)
        sub_end = time_to_seconds(sub.end)

        if start_time is None:
            start_time = sub_start

        current_text += " " + sub.text
        end_time = sub_end
        length = end_time - start_time

        # Hard cap — flush if exceeding max_len
        if length >= max_len:
            segments.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": current_text.strip()
            })
            current_text = ""
            start_time = None
            end_time = None

        # Normal flush at min_len
        elif length >= min_len:
            segments.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": current_text.strip()
            })
            current_text = ""
            start_time = None
            end_time = None

    # Fix — save leftover segment instead of dropping it
    if current_text.strip() and start_time is not None:
        segments.append({
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "text": current_text.strip()
        })

    return segments


def get_nlp_scores(srt_path, num_clips=3, min_len=50, max_len=60):

    segments = build_segments_from_srt(srt_path, min_len, max_len)

    results = []

    for seg in segments:

        score = nlp_score(seg["text"])

        results.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "length": round(seg["end"] - seg["start"], 2),
            "score": score
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:num_clips]
