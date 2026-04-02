import os
import re
import json
import torch
import ffmpeg
import yt_dlp
import shutil
from Detectors.Make_Shorts import make_shorts
from Detectors.Nlp_Detection import get_nlp_scores
from Detectors.Audio_Score_Test import get_top_audio_clips
from Detectors.Nlp_Detection_LLM import get_nlp_scores_groq
from Detectors.Combine_all_scores import combine_and_select
from Detectors.Scene_Scoring import create_scene_window_scores
from Detectors.Add_Captions_on_shorts import process_shorts_folder
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

GROQ_API_KEY = "//Enter your Groq Api key"

#/Openai Whisper-Large-v3 importing model once Globally to save time and resources for multiple transcriptions
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

def sanitize_video_filename(video_path):
    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    name, ext = os.path.splitext(filename)
    clean_name = re.sub(r'[\\/*?:"<>|\'"]', "", name)
    clean_name = re.sub(r'\s+', " ", clean_name).strip()
    new_path = os.path.join(directory, clean_name + ext)
    if video_path != new_path:
        os.rename(video_path, new_path)
    return new_path

def download_yt_video(url, output_path="Videos"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    final_FilePath = None
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writedescription': False,
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            final_FilePath = ydl.prepare_filename(info_dict)
        print(f"Video downloaded successfully: {final_FilePath}")
        return sanitize_video_filename(final_FilePath)
    except yt_dlp.DownloadError as e:
        print(f"An error occurred while downloading the video: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



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

def Generate_SRT(audio_path, output_path="Subtitles"):
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

        srt_lines = []
        index = 1

        for chunk in chunks:
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()

            if not text or start is None:
                continue
            if end is None:
                end = start + 1.5

            srt_lines.append(f"{index}")
            srt_lines.append(f"{format_time(start)} --> {format_time(end)}")
            srt_lines.append(text)
            srt_lines.append("")
            index += 1

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        return output_file

    except Exception as e:
        print(f"Error generating SRT: {e}")
        return None
    

if __name__ == "__main__":
    Path = input("Enter the YouTube video URL: ")
    number_of_shorts = int(input("Enter the number of shorts to generate: "))
    print("Downloading video...") 
    download_video_path = download_yt_video(Path)
    if download_video_path:
        print("Extracting audio...")
        download_audio_path = Extract_Audio(download_video_path)
        if download_audio_path:
            print("Generating subtitles...")
            srt_path = Generate_SRT(download_audio_path)
            if srt_path:


                print("Running audio scoring...")
                results = get_top_audio_clips(
                    download_audio_path,
                    num_shorts=number_of_shorts+5,
                    min_len=50,
                    max_len=60
                )
                with open("Json_Files/audio_scores.json", "w") as f:
                    json.dump(results, f, indent=4)



                print("Running nlp scoring...")
                try:
                    clips = get_nlp_scores_groq(srt_path, num_clips=number_of_shorts+5, api_key=GROQ_API_KEY)
                    print("Used Groq")
                except Exception as e:
                    print(f"Groq failed: {e} — falling back to local NLP")
                    clips = get_nlp_scores(srt_path, num_clips=number_of_shorts+5)
                with open("Json_Files/nlp_scores.json", "w") as f:
                    json.dump(clips, f, indent=4)


                print("Running Scene scoring...")
                scene_times,Scene_scores = create_scene_window_scores(download_video_path, window_size=5, step_size=2)
                with open("Json_Files/Scene_scores.json", "w") as f:
                    json.dump(Scene_scores, f, indent=4)

                print("Combining scores...")
                final_clips = combine_and_select(
                    audio_clips=results,
                    llm_clips=clips,
                    scene_windows=Scene_scores,
                    video_duration=scene_times[-1],
                    num_clips=number_of_shorts,
                    clip_len=60,
                )
                with open("Json_Files/final_clips.json", "w") as f:
                    json.dump(final_clips, f, indent=4)
                
                print("Generating shorts...")
                make_shorts(
                    video_path=download_video_path,
                    final_clips_path="Json_Files/final_clips.json",
                    output_path="Shorts"
                )

                print("Adding captions to shorts...")
                process_shorts_folder(shorts_folder="Shorts")
                os.remove(download_video_path)
                os.remove(download_audio_path)
                os.remove(srt_path)
                os.remove("Json_Files/audio_scores.json")
                os.remove("Json_Files/nlp_scores.json")
                os.remove("Json_Files/Scene_scores.json")
                os.remove("Json_Files/final_clips.json")
                shutil.rmtree("Shorts/temp_audio", ignore_errors=True)
                shutil.rmtree("Shorts/temp_srt", ignore_errors=True)
                print("Done.")