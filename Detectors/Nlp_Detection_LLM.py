from groq import Groq
import json
import pysrt


def time_to_seconds(t):
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000

def get_nlp_scores_groq(srt_path, num_clips=3, min_len=50, max_len=60, api_key=None):

    client = Groq(api_key=api_key)

    subs = pysrt.open(srt_path)
    formatted = "\n".join([
        f"[{time_to_seconds(sub.start):.2f}s - {time_to_seconds(sub.end):.2f}s]: {sub.text.strip()}"
        for sub in subs
    ])

    prompt = f"""
    You are a viral video editor. Analyze this transcript and find the {num_clips} most engaging clips.
    
    Rules:
    - Each clip MUST be between {min_len} and {max_len} seconds long
    - Pick moments that are funny, shocking, emotional, insightful or have strong hooks
    - Score each clip from 0.0 to 1.0 based on viral potential
    - Return ONLY a valid JSON array, no explanation, no markdown

    Format:
    [{{"start": 12.5, "end": 65.0, "length": 52.5, "score": 0.87}}]

    Transcript:
    {formatted}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)