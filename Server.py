import os
import re
import glob
import json
import uuid
import torch
import ffmpeg
import yt_dlp
import shutil
import asyncio
import subprocess
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from collections import defaultdict

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

from Detectors.Make_Shorts import make_shorts
from Detectors.Audio_Score_Test import get_top_audio_clips
from Detectors.Nlp_Detection import get_nlp_scores
from Detectors.Nlp_Detection_LLM import get_nlp_scores_groq
from Detectors.Combine_all_scores import combine_and_select
from Detectors.Scene_Scoring import create_scene_window_scores
# NOTE: Add_Captions_on_shorts is NOT imported here — it loads Whisper at module
# level, which would waste ~3 GB loading it a second time. Caption logic is
# inlined below using the single global pipe.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Whisper ONCE ────────────────────────────────────────────────────────
pipe = None

def get_pipe():
    global pipe

    if pipe is None:
        print("Loading Whisper model...")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-medium"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        print("Whisper loaded.")

    return pipe

# ── In-memory job store ──────────────────────────────────────────────────────
jobs: dict = defaultdict(lambda: {"status": "pending", "logs": [], "output_files": []})


class JobRequest(BaseModel):
    url: str
    num_shorts: int = 3
    min_len: int = 50
    max_len: int = 60
    audio_weight: float = 0.3
    llm_weight: float = 0.5
    scene_weight: float = 0.2
    groq_key: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(job_id: str, msg: str):
    print(f"[{job_id[:8]}] {msg}")
    jobs[job_id]["logs"].append(msg)


def sanitize_filename(video_path: str) -> str:
    directory = os.path.dirname(video_path)
    name, ext = os.path.splitext(os.path.basename(video_path))
    clean_name = re.sub(r'[\\/*?:"<>|\'"]', "", name)
    clean_name = re.sub(r'\s+', " ", clean_name).strip()
    new_path = os.path.join(directory, clean_name + ext)
    if video_path != new_path:
        os.rename(video_path, new_path)
    return new_path


def format_time(seconds: float) -> str:
    hrs  = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms   = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


# ── Caption helpers (inlined to use the single global pipe) ──────────────────

def _generate_srt_for_short(audio_path: str, srt_dir: str, words_per_line: int = 3) -> str:
    """Transcribe a short clip and write a word-grouped SRT into srt_dir."""
    os.makedirs(srt_dir, exist_ok=True)
    out_srt = os.path.join(srt_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".srt")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = get_pipe()(
        audio_path,
        return_timestamps=True,
        chunk_length_s=10,
        stride_length_s=2,
        generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0},
    )

    # Clean and clamp chunks so there are no overlaps / gaps
    cleaned = []
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        if not text or start is None:
            continue
        if end is None or end <= start:
            end = start + max(0.5, len(text.split()) * 0.3)
        cleaned.append({"text": text, "start": start, "end": end})

    for i in range(len(cleaned) - 1):
        ns = cleaned[i + 1]["start"]
        if cleaned[i]["end"] > ns:
            cleaned[i]["end"] = ns
        elif cleaned[i]["end"] < ns - 0.1:
            cleaned[i]["end"] = ns

    srt_lines, index = [], 1
    for chunk in cleaned:
        start, end = chunk["start"], chunk["end"]
        words = chunk["text"].split()
        if not words:
            continue
        tpw = (end - start) / len(words)
        for i in range(0, len(words), words_per_line):
            group = words[i : i + words_per_line]
            gs = start + i * tpw
            ge = min(gs + len(group) * tpw, end)
            if ge - gs < 0.4:
                ge = gs + 0.4
            srt_lines += [str(index), f"{format_time(gs)} --> {format_time(ge)}", " ".join(group).upper(), ""]
            index += 1

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    return out_srt


def _burn_subtitles(video_path: str, srt_path: str, final_dir: str) -> str:
    """Convert SRT → ASS with custom style, then burn into video."""
    os.makedirs(final_dir, exist_ok=True)
    base     = os.path.splitext(os.path.basename(video_path))[0]
    ass_path = os.path.join(final_dir, base + ".ass")
    out_mp4  = os.path.join(final_dir, base + ".mp4")

    subprocess.run(
        ["ffmpeg", "-y", "-i", srt_path, "-c:s", "ass", ass_path],
        check=True, capture_output=True,
    )

    with open(ass_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("Style: Default"):
            lines[i] = (
                "Style: Default,Montserrat,10,"
                "&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
                "-1,0,0,0,100,100,0,0,1,1,0,2,0,0,70,1\n"
            )
            break
    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    safe_ass = ass_path.replace("\\", "/")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"ass='{safe_ass}'",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19",
            "-c:a", "copy",
            out_mp4,
        ],
        check=True, capture_output=True,
    )
    os.remove(ass_path)
    return out_mp4


# ── Core pipeline (sync — runs in a thread so the event loop stays free) ────

def _run_pipeline_sync(job_id: str, req: JobRequest):
    work_dir       = f"jobs/{job_id}"
    videos_dir     = f"{work_dir}/Videos"
    audio_dir      = f"{work_dir}/Audio"
    subtitles_dir  = f"{work_dir}/Subtitles"
    shorts_dir     = f"{work_dir}/Shorts"
    final_dir      = f"{work_dir}/Final_Videos"
    json_dir       = f"{work_dir}/Json_Files"
    temp_audio_dir = f"{work_dir}/temp_audio"
    temp_srt_dir   = f"{work_dir}/temp_srt"

    for d in [videos_dir, audio_dir, subtitles_dir, shorts_dir, final_dir, json_dir]:
        os.makedirs(d, exist_ok=True)

    try:
        # 1. Download ──────────────────────────────────────────────────────────
        log(job_id, "Downloading video...")
        ydl_opts = {
            "format": "best",
            "outtmpl": f"{videos_dir}/%(title)s.%(ext)s",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(req.url, download=True)
            video_path = ydl.prepare_filename(info)
        video_path = sanitize_filename(video_path)
        log(job_id, f"Downloaded: {os.path.basename(video_path)}")

        # 2. Extract audio ─────────────────────────────────────────────────────
        log(job_id, "Extracting audio...")
        audio_path = os.path.join(audio_dir, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True, quiet=True)
        log(job_id, "Audio extracted")

        # 3. Transcribe (full video — used for NLP scoring) ────────────────────
        log(job_id, "Transcribing with Whisper...")
        srt_path = os.path.join(subtitles_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".srt")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = get_pipe()(
            audio_path,
            return_timestamps=True,
            chunk_length_s=10,
            stride_length_s=2,
            generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0},
        )
        srt_lines, index = [], 1
        for chunk in result["chunks"]:
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()
            if not text or start is None:
                continue
            if end is None:
                end = start + 1.5
            srt_lines += [str(index), f"{format_time(start)} --> {format_time(end)}", text, ""]
            index += 1
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
        log(job_id, "Transcript done")

        # 4. Audio scoring ─────────────────────────────────────────────────────
        log(job_id, "Scoring audio energy...")
        audio_clips = get_top_audio_clips(
            audio_path, num_shorts=req.num_shorts + 5, min_len=req.min_len, max_len=req.max_len
        )

        # 5. NLP scoring ───────────────────────────────────────────────────────
        log(job_id, "Running NLP scoring...")
        try:
            if not req.groq_key:
                raise ValueError("No Groq key provided")
            nlp_clips = get_nlp_scores_groq(
                srt_path, num_clips=req.num_shorts + 5,
                api_key=req.groq_key, min_len=req.min_len, max_len=req.max_len,
            )
            log(job_id, "NLP: used Groq LLM")
        except Exception as e:
            log(job_id, f"NLP: Groq failed ({e}), falling back to local model")
            nlp_clips = get_nlp_scores(srt_path, num_clips=req.num_shorts + 5, min_len=req.min_len, max_len=req.max_len)

        # 6. Scene scoring ─────────────────────────────────────────────────────
        log(job_id, "Scoring scene changes...")
        scene_times, scene_windows = create_scene_window_scores(video_path)

        # 7. Combine scores ────────────────────────────────────────────────────
        log(job_id, "Combining scores...")
        final_clips = combine_and_select(
            audio_clips=audio_clips,
            llm_clips=nlp_clips,
            scene_windows=scene_windows,
            video_duration=scene_times[-1],
            num_clips=req.num_shorts,
            clip_len=req.max_len,
            audio_weight=req.audio_weight,
            llm_weight=req.llm_weight,
            scene_weight=req.scene_weight,
        )
        clips_path = os.path.join(json_dir, "final_clips.json")
        with open(clips_path, "w") as f:
            json.dump(final_clips, f, indent=2)

        # 8. Cut shorts ────────────────────────────────────────────────────────
        log(job_id, "Cutting shorts...")
        make_shorts(video_path=video_path, final_clips_path=clips_path, output_path=shorts_dir)

        # 9. Caption each short (uses job-specific dirs — no path collisions) ──
        log(job_id, "Adding captions...")
        shorts = glob.glob(os.path.join(shorts_dir, "*.mp4"))
        for short_path in shorts:
            short_name = os.path.basename(short_path)
            try:
                short_audio = os.path.join(
                    temp_audio_dir,
                    os.path.splitext(short_name)[0] + ".wav",
                )
                os.makedirs(temp_audio_dir, exist_ok=True)
                ffmpeg.input(short_path).output(short_audio).run(overwrite_output=True, quiet=True)

                short_srt = _generate_srt_for_short(short_audio, temp_srt_dir)
                _burn_subtitles(short_path, short_srt, final_dir)

                os.remove(short_audio)
                os.remove(short_srt)
                log(job_id, f"Captioned: {short_name}")
            except Exception as e:
                log(job_id, f"Caption failed for {short_name}: {e}")

        # 10. Cleanup temp dirs ────────────────────────────────────────────────
        shutil.rmtree(temp_audio_dir, ignore_errors=True)
        shutil.rmtree(temp_srt_dir,   ignore_errors=True)
        shutil.rmtree(shorts_dir,     ignore_errors=True)

        output_files = list(Path(final_dir).glob("*.mp4"))
        jobs[job_id]["output_files"] = [str(f) for f in output_files]
        jobs[job_id]["status"] = "done"
        log(job_id, f"Done! {len(output_files)} shorts ready.")

    except Exception:
        jobs[job_id]["status"] = "error"
        log(job_id, f"ERROR:\n{traceback.format_exc()}")


# ── Async wrapper — offloads the blocking pipeline to a thread ───────────────

async def run_pipeline(job_id: str, req: JobRequest):
    jobs[job_id]["status"] = "running"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_pipeline_sync, job_id, req)


# ── API routes ────────────────────────────────────────────────────────────────

@app.post("/start")
async def start_job(req: JobRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "logs": [], "output_files": []}
    asyncio.create_task(run_pipeline(job_id, req))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return {
        "status": job["status"],
        "logs": job["logs"],
        "output_files": [os.path.basename(f) for f in job["output_files"]],
    }


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    for f in job["output_files"]:
        if os.path.basename(f) == filename:
            return FileResponse(f, media_type="video/mp4", filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/test")
async def test():
    return {"ok": True}

app.mount("/", StaticFiles(directory="static", html=True), name="static")