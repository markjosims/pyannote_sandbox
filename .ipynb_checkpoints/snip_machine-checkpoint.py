from transformers import pipeline, Pipeline
from segmentation import *
from tqdm import tqdm
import tempfile

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
vad_json = pyannote_result_to_json(vad)
vad_json

wav = load_and_resample(wav_path)[0]
wav

start = sec_to_samples(vad_json[0]['start'])
end = sec_to_samples(vad_json[0]['end'])
perform_sli(wav = wav[start:end])

sli_list = list()
for clip in vad_json[:10]:
    start = sec_to_samples(clip['start'])
    end = sec_to_samples(clip['end'])
    sli_result = perform_sli(wav = wav[start:end])
    sli_list.append(sli_result)

sli_in = list()
for clip in vad_json[:10]:
    start = sec_to_samples(clip['start'])
    end = sec_to_samples(clip['end'])
    sli_in.append(wav[start:end])
sli_out = perform_sli(wav=sli_in[:3])

sli_list



# +
# result = perform_sli(in_fp = wav_path)
# print(result)

# +
# Perform vad on CLIP
clip_vad = perform_vad(clip_path) # pyannote.core.annotation.Annotation
clip_vad_json = pyannote_result_to_json(clip_vad) # list of start-end dicts
clip_bool_arr = segments_to_array(clip_vad_json)

# remove_segments_from_audio(wav_tensor, ipa_segments, sli_segments)
# -
clip = load_and_resample(clip_path)[0]
perform_sli(wav = clip)




# +
# pyannote_result_to_json(vad)

# +
# perform_sli(pyannote_result_to_json(vad))

# +
# main()
# -
start = sec_to_samples(clip['start'])
end = sec_to_samples(clip['end'])
test = wav[start:end]



vad_json



# +


# create a temporary directory using the context manager
with tempfile.TemporaryDirectory(prefix='temp') as tmpdirname:
    print('created temporary directory', tmpdirname)
    
    file_names = list()
    
    for i in tqdm(range(len(vad_json))):
        start = sec_to_samples(vad_json[i]['start'])
        end = sec_to_samples(vad_json[i]['end'])
        clip = np.array(wav[start:end])
        
        with tempfile.TemporaryFile() as fp:
            fp.write(clip)
            name = f'segment{i}'
            print(fp.name)
            file_names.append(tmpdirname + name)
    
    output = perform_sli(file_names)



# -

#tempfile, naming wav files
for segment, track in result.itertracks():
    counts = str(count)
    segment_count = 'segments' + counts
    print(segment_count)
    segment_count = segment
    count +=1

