from transformers import pipeline, Pipeline
from segmentation import *
from tqdm import tqdm
import tempfile
import os

# %load_ext autoreload
# %autoreload 2

# +
SAMPLE_RATE = 16000

# Load in full wav clip
elan_path = 'data/HH20220227-1-AC.eaf'
wav_path = 'data/HH20220227-1.wav'
clip_path = 'data/tira_clip.wav'

# Get IPA labels
ipa_segments = get_ipa_labels(elan_path) # List of start-end-value dicts
# -

vad = perform_vad(wav_path) # pyannote.core.annotation.Annotation
vad_json = pyannote_result_to_json(vad) # list of dictionaries with start and end
vad_json

wav = load_and_resample(wav_path)[0] # Torch.tensor object

# +
# Runs sli individually on first ten clips of vad_json

sli_list = list()
for clip in vad_json[:10]:
    start = sec_to_samples(clip['start'])
    end = sec_to_samples(clip['end'])
    sli_result = perform_sli(wav = wav[start:end])
    sli_list.append(sli_result)
    
sli_list
# -

print(tempfile.mkdtemp())

# create a temporary directory using the context manager
with tempfile.TemporaryDirectory(prefix='temp') as tmpdirname:
    print('created temporary directory', tmpdirname)
    
    file_names = list()
    
    for i in range(len(vad_json)):
        start = sec_to_samples(vad_json[i]['start'])
        end = sec_to_samples(vad_json[i]['end'])
        clip = np.array(wav[start:end])
        
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(clip)
            name = f'segment{i}'
            fp.name = name
            wav_path = os.path.join(tmpdirname, name)
            file_names.append(wav_path)
    
    output = perform_sli(file_names)



