from typing import Optional, Sequence
from argparse import ArgumentParser
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import torchaudio
from numpy import ndarray

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("PyAnnote VAD runner")
    parser.add_argument("-i", "--input", help="Filepath to run VAD on")
    parser.add_argument("-o", "--output", help="Filepath to save predictions to")
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    pipe = pipeline("audio-classification", model="markjosims/mms-lid-tira")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    
    wav, sr = torchaudio.load(args.input)
    # TODO: resample wav to 16kHz!!!
    output = pipe(wav.numpy())
    
    print(output)

    return 0



if __name__ == '__main__':
    #main()
    processor = AutoProcessor.from_pretrained("markjosims/mms-lid-tira", force_download=True)
    model = AutoModelForAudioClassification.from_pretrained("markjosims/mms-lid-tira")