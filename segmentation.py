from typing import Optional, Sequence, Dict, List, Union
from argparse import ArgumentParser
from transformers import Pipeline, pipeline
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import torchaudio
import json
import numpy as np

SAMPLE_RATE = 16000

"""
Pyannote and HuggingFace entry points
"""

def perform_sli(
        in_fp: str,
        pipe: Optional[Pipeline]= None,
    ) -> List[Dict[str, Union[str, float]]]:
        wav = load_and_resample(in_fp)
        wav = wav[0].numpy() # hf pipeline expects 1D numpy array


        if not pipe:
            pipe = pipeline("audio-classification", model="markjosims/mms-lid-tira")

        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))

        result = pipe(wav)
        return result

def perform_vad(
        in_fp: str,
        pipe: Optional[PyannotePipeline] = None,
    ):
    wav = load_and_resample(in_fp)

    if not pipe:
        pipe = PyannotePipeline.from_pretrained("pyannote/voice-activity-detection")

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": wav, "sample_rate": SAMPLE_RATE},
            hook=hook,
        )
    return result

def pyannote_result_to_json(result)  -> List[Dict[str, float]]:
    segments = []
    for track, _ in result.itertracks():
        segment = {'start': track.start, 'end': track.end}
        segments.append(segment)
    return segments

def diarize(
        in_fp: str,
        pipe: Optional[PyannotePipeline] = None,
        num_speakers: int = 1,
    ):
    wav = load_and_resample(in_fp)

    if not pipe:
        pipe = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": wav, "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            hook=hook,
        )
    return result

"""
Audio handling methods
"""

def load_and_resample(fp: str) -> torch.tensor:
    wav_orig, sr_orig = torchaudio.load(fp)
    wav = torchaudio.functional.resample(wav_orig, sr_orig, SAMPLE_RATE)
    return wav

def sec_to_samples(time_sec: float):
    return int(time_sec*SAMPLE_RATE)

def remove_segments(
        audio: Union[torch.Tensor, np.ndarray],
        out_segments: List[Dict[str, float]]
    ) -> torch.tensor:
    if len(audio.shape) == 2:
        audio = audio[0]
    if type(audio) is torch.Tensor:
        audio = audio.numpy()

    for segment in out_segments:
        # set segment to NaN
        start_sample = sec_to_samples(segment['start'])
        end_sample = sec_to_samples(segment['end'])
        audio[start_sample:end_sample] = np.nan
    
    # drop all NaN
    audio = audio[~np.isnan(audio)]

    # reshape to 2D torch tensor
    audio = torch.unsqueeze(torch.from_numpy(audio), 0)

    return audio

"""

Main script
"""

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("VAD, LID and diarization runner")
    parser.add_argument(
        "TASK",
        help="Task to run",
        choices=["VAD", "LID", "DRZ"]
    )
    parser.add_argument("-i", "--input", help="Filepath to run VAD on")
    parser.add_argument("-o", "--output", help="Filepath to save predictions to")
    parser.add_argument(
        "-n",
        "--num_speakers",
        help="Number of speakers in file (only use for diarization)",
        default=1
    )
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    if args.TASK == 'SLI':
        labels = perform_sli(args.input)
    elif args.TASK == 'VAD':
        result = perform_vad(args.input)
        labels = pyannote_result_to_json(result)
    elif args.TASK == 'DRZ':
        result = diarize(args.input, num_speakers=args.num_speakers)
        labels = pyannote_result_to_json(result)

    out_fp = args.output or args.input.replace('.wav', '.json')
    with open(out_fp, 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    return 0



if __name__ == '__main__':
    main()