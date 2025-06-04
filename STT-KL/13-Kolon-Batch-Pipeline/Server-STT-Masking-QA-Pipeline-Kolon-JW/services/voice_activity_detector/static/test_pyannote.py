
import torch
import torchaudio
from pyannote.audio import Pipeline
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# offline_vad = Pipeline.from_pretrained("/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/src/engine/pyannotate/config.yaml")
offline_vad = Pipeline.from_pretrained("/data/06-Kolon-Labeling-Processing/Server-STT-TAs-Docker-Compose-Unicef/services/voice_activity_detector/static/config.yaml")
offline_vad = offline_vad.to(0)

# 4. apply pretrained pipeline
diarization = offline_vad("/home/metanet/Workspace/08-Kolon-Prj/temp_data/103-1677111254.82_stereo_mono_left.wav")

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")