import os
import io
import soundfile as sf
from pydub import AudioSegment


def convert_to_mono(raw_input_data_path, split_mono_output_path):
    wav_files = []
    split_mono_output_path_left = os.path.join(split_mono_output_path, "split_mono_left")
    split_mono_output_path_right = os.path.join(split_mono_output_path, "split_mono_right")

    if not os.path.exists(split_mono_output_path_left):
        os.makedirs(split_mono_output_path_left)

    if not os.path.exists(split_mono_output_path_right):
        os.makedirs(split_mono_output_path_right)

    for root, dirs, files in os.walk(raw_input_data_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
                data, samplerate = sf.read(os.path.join(root, file))

                tmp_memory_file = io.BytesIO( )
                sf.write(tmp_memory_file, data, samplerate, format="wav")

                stereo_audio = AudioSegment.from_file(tmp_memory_file, format="wav")
                mono_audios = stereo_audio.split_to_mono()
                mono_left = mono_audios[0].export(os.path.join(split_mono_output_path_left, "{0}_mono_left.wav".format(file[:-4])),format="wav")
                mono_right = mono_audios[1].export(os.path.join(split_mono_output_path_right, "{0}_mono_right.wav".format(file[:-4])),format="wav")

    return wav_files

def get_left_right_audio_from_pcm_s4le(input_ms_adpcm_audio):  # Input file MS_ADPCM path

    assert input_ms_adpcm_audio.endswith(".wav"), f"Input audio file must end with .wav, got: {input_ms_adpcm_audio}"
    

    data, samplerate = sf.read(input_ms_adpcm_audio)

    tmp_memory_file = io.BytesIO( )
    sf.write(tmp_memory_file, data, samplerate, format="wav")

    stereo_audio = AudioSegment.from_file(tmp_memory_file, format="wav")
    mono_audios = stereo_audio.split_to_mono()
    mono_left = mono_audios[0]
    mono_right = mono_audios[1]

    # FOr KOLON
    # RX = LEFT
    # TX = RIGHT

    return mono_left, mono_right, samplerate