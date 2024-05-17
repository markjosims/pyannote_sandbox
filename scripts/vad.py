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
            hook=hook,
        )
    
    with open(
        args.output or args.input.replace('.wav', '.lab'),
        'w',
    ) as f:
        output.write_lab(f)

    return 0



if __name__ == '__main__':
    main()