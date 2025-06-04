
from pyannote.audio import Pipeline



def pyannote(audio_path, pipeline):
    vad = pipeline(audio_path)

    debtor_ranges = []
    for segment, _, label in vad.itertracks(yield_label=True):
        start = int(segment.start * 1000)
        end = int(segment.end * 1000)
        debtor_ranges.append([start, end])

    return debtor_ranges


AUDIO_FILE = f"/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/data/splited_mono_data/split_mono_left/103-1677024213.48_stereo_mono_left.wav"
# look ma: no hands!
offline_vad = Pipeline.from_pretrained("/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/src/engine/pyannotate/config.yaml")
# output = offline_vad(audio_in_memory)

print(pyannote(AUDIO_FILE, offline_vad))

