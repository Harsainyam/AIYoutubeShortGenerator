import os
import json
import subprocess

import os
import json
import subprocess

def make_shorts(
    video_path,
    final_clips_path="Json_Files/final_clips.json",
    output_path="Shorts",
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(final_clips_path, "r") as f:
        clips = json.load(f)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i, clip in enumerate(clips):
        start  = clip["start"]
        end    = clip["end"]
        length = end - start

        out_file = os.path.join(output_path, f"{base_name}_short{i+1}.mp4")

        print(f"Processing short {i+1}: {start}s → {end}s")

        filter_complex = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "gblur=sigma=30[blurred];"
            "[fg]scale=1080:-2[scaled];"
            "[blurred][scaled]overlay=(W-w)/2:(H-h)/2[out]"
        )

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "19",
            "-c:a", "aac",
            "-t", str(length),
            out_file
        ], check=True)

        print(f"Saved: {out_file}")

    print(f"\nAll {len(clips)} shorts done!")