from typing import Optional, Sequence
from argparse import ArgumentParser
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import torchaudio

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("PyAnnote VAD runner")
    parser.add_argument("-i", "--input", help="Filepath to run VAD on")
    parser.add_argument("-o", "--output", help="Filepath to save predictions to")
    #parser.add_argument("-n", "--num_speakers", help="Number of speakers in file", default=1)
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    
    wav, sr = torchaudio.load(args.input)
    with ProgressHook() as hook:
        output = pipeline(
            {"waveform": wav, "sample_rate": sr},
            #num_speakers=args.num_speakers,
            hook=hook,
        )
    breakpoint()

    return 0



if __name__ == '__main__':
    main()