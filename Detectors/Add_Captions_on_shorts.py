import os
import glob
import torch
import ffmpeg
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def Extract_Audio(video_path,output_path="Audio"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        final_AudioPath = f'{output_path}/{os.path.splitext(os.path.basename(video_path))[0]}.wav'
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, final_AudioPath)
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Audio extracted successfully: {final_AudioPath}")
        return final_AudioPath
    except ffmpeg.Error as e:
        print(f"An error occurred while extracting audio: {e}")
        return None

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def Generate_SRT(audio_path, output_path="Subtitles", words_per_line=3):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = f'{output_path}/{os.path.splitext(os.path.basename(audio_path))[0]}.srt'

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = pipe(
            audio_path,
            return_timestamps=True,
            chunk_length_s=10,
            stride_length_s=2,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "temperature": 0,
            }
        )

        chunks = result['chunks']

        # Clean chunks first
        cleaned = []
        for chunk in chunks:
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()
            if not text or start is None:
                continue
            if end is None or end <= start:
                end = start + max(0.5, len(text.split()) * 0.3)
            cleaned.append({"text": text, "start": start, "end": end})

        # Clamp each chunk end to next chunk start — no gaps no overlaps
        for i in range(len(cleaned) - 1):
            next_start = cleaned[i + 1]["start"]
            if cleaned[i]["end"] > next_start:
                cleaned[i]["end"] = next_start
            elif cleaned[i]["end"] < next_start - 0.1:
                cleaned[i]["end"] = next_start

        srt_lines = []
        index = 1

        for chunk in cleaned:
            start = chunk["start"]
            end = chunk["end"]
            words = chunk["text"].split()

            if not words:
                continue

            duration = end - start
            time_per_word = duration / len(words)

            for i in range(0, len(words), words_per_line):
                group = words[i:i + words_per_line]

                group_start = start + i * time_per_word
                group_end = min(group_start + len(group) * time_per_word, end)

                if group_end - group_start < 0.4:
                    group_end = group_start + 0.4

                srt_lines.append(str(index))
                srt_lines.append(f"{format_time(group_start)} --> {format_time(group_end)}")
                srt_lines.append(" ".join(group).upper())
                srt_lines.append("")
                index += 1

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        return output_file

    except Exception as e:
        print(f"Error generating SRT: {e}")
        return None

def addSubtitiles(video_path,srt_path,output_path="Final_Videos"):
    if os.path.exists(video_path) and os.path.exists(srt_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            ass_path = os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0] + ".ass")
            output_file = os.path.join(output_path, base_name + ".mp4")
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", srt_path,
                "-c:s", "ass",
                ass_path
            ], check=True)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.readlines()

            for i, line in enumerate(content):
                if line.startswith("Style: Default"):
                    content[i] = (
                        "Style: Default,Montserrat,10,"
                        "&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
                        "-1,0,0,0,100,100,0,0,1,1,0,2,0,0,70,1\n"
                    )
                    break

            with open(ass_path, "w", encoding="utf-8") as f:
                f.writelines(content)
            safe_ass_path = ass_path.replace("\\", "/")
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vf", f"ass='{safe_ass_path}'",
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "19",
                "-c:a", "copy",
                output_file
            ], check=True)
            os.remove(safe_ass_path)
        except Exception as e:
            print(f"An error occurred while adding subtitles: {e}")
            return None

def process_shorts_folder(shorts_folder="Shorts"):
    
    shorts = glob.glob(os.path.join(shorts_folder, "*.mp4"))
    
    if not shorts:
        print("No shorts found in folder")
        return

    print(f"Found {len(shorts)} shorts to process")

    for video_path in shorts:
        base = os.path.splitext(video_path)[0]
        srt_path = base + ".srt"

        print(f"\nProcessing: {os.path.basename(video_path)}")

        # Extract audio
        print("  Extracting audio...")
        audio_path = Extract_Audio(video_path, output_path="Shorts/temp_audio")
        if not audio_path:
            print("  Audio extraction failed, skipping")
            continue

        # Generate SRT
        print("  Generating SRT...")
        srt_path = Generate_SRT(audio_path, output_path="Shorts/temp_srt")
        if not srt_path:
            print("  SRT generation failed, skipping")
            os.remove(audio_path)
            continue

        # Add subtitles — overwrites original short with captioned version
        print("  Adding captions...")
        addSubtitiles(video_path, srt_path, output_path="Final_Videos")

        # Cleanup temp files
        os.remove(audio_path)
        os.remove(srt_path)
        print(f"  Done: {os.path.basename(video_path)}")
        os.remove(video_path)

    print(f"\nAll shorts captioned → Final_Videos/")