from xmlrpc.client import MAXINT
import librosa
import matplotlib.pyplot as plt
import librosa.display
import matplotlib.patches as patches
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, detect_silence
import itertools
import os
import fnmatch

import multiprocessing
import functools
import json
import traceback
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from api.utils.audio_utils import get_left_right_audio_from_pcm_s4le
import numpy as np
from pyannote.audio import Pipeline
import torch
import torchaudio
import io
import logging
MAX_SECOND = 16000
import os
import config

MODEL_CACHE = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VADPyannoModel:

    @staticmethod
    def is_rx_counselor(rx_audio_file_path):
        filename = os.path.basename(rx_audio_file_path)
        if filename.find("_") > 5:
            return False
        else:
            return True

    @staticmethod
    def pyannote(audio_path, pipeline):
        vad = pipeline(audio_path)

        debtor_ranges = []
        for segment, _, label in vad.itertracks(yield_label=True):
            start = int(segment.start * 1000)
            end = int(segment.end * 1000)
            debtor_ranges.append([start, end])

        return debtor_ranges

    @staticmethod
    def pyannote_file_like_obj(file_like_obj, pipeline):  # https://github.com/pyannote/pyannote-audio/blob/master/tutorials/applying_a_pipeline.ipynb 

        waveform, sample_rate = torchaudio.load(file_like_obj)   # https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-from-file-like-object:~:text=%2C%20sample_rate%20%3D-,torchaudio.load(,-response.raw

        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        vad = pipeline(audio_in_memory)

        debtor_ranges = []
        for segment, _, label in vad.itertracks(yield_label=True):
            start = int(segment.start * 1000)
            end = int(segment.end * 1000)
            debtor_ranges.append([start, end])

        return debtor_ranges

    @staticmethod
    def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1, non_silence_range=None, return_timestamp=False):
        """
        Returns list of audio segments from splitting audio_segment on silent sections

        audio_segment - original pydub.AudioSegment() object

        min_silence_len - (in ms) minimum length of a silence to be used for
            a split. default: 1000ms

        silence_thresh - (in dBFS) anything quieter than this will be
            considered silence. default: -16dBFS

        keep_silence - (in ms or True/False) leave some silence at the beginning
            and end of the chunks. Keeps the sound from sounding like it
            is abruptly cut off.
            When the length of the silence is less than the keep_silence duration
            it is split evenly between the preceding and following non-silent
            segments.
            If True is specified, all the silence is kept, if False none is kept.
            default: 100ms

        seek_step - step size for interating over the segment in ms
        """

        # from the itertools documentation
        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        if isinstance(keep_silence, bool):
            keep_silence = len(audio_segment) if keep_silence else 0

        if non_silence_range:
            output_ranges = [
                    [ start - keep_silence, end + keep_silence ]
                    for (start,end) in non_silence_range
                ]
        else:
            output_ranges = [
                [ start - keep_silence, end + keep_silence ]
                for (start,end)
                    in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
            ]

        for range_i, range_ii in pairwise(output_ranges):
            last_end = range_i[1]
            next_start = range_ii[0]
            if next_start < last_end:
                range_i[1] = (last_end+next_start)//2
                range_ii[0] = range_i[1]

        if return_timestamp:
            return [
                audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
                for start,end in output_ranges
            ], output_ranges

        return [
            audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
            for start,end in output_ranges
        ]

    @staticmethod
    def save_vizualization_figure(
                    samplerate,
                    customer_audio_segment, 
                    counselor_audio_segment,
                    customer_silence_thresh,
                    counselor_silence_thresh,
                    customer_ranges,
                    counselor_ranges,
                    refined_customer_ranges,
                    refined_counselor_ranges,
                    fig_output_file
                    ):

        # Ref: https://github.com/phrasenmaeher/audio-transformation-visualization/blob/main/visualize_transformation.py
        samples = [customer_audio_segment.get_array_of_samples()]
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        customer_y = fp_arr[:, 0]
        customer_sr = samplerate

        # Ref: https://github.com/phrasenmaeher/audio-transformation-visualization/blob/main/visualize_transformation.py
        samples = [counselor_audio_segment.get_array_of_samples()]
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        counselor_y = fp_arr[:, 0]
        counselor_sr = samplerate

        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(30, 10))
        # Draw rx wave
        librosa.display.waveshow(customer_y, sr=customer_sr, ax=ax[1],  color="blue")
        ax[1].set(title=f"customer avg: {customer_audio_segment.dBFS:.2f} thres: {customer_silence_thresh:.2f}")
        ax[1].label_outer()
        # Draw tx wave
        librosa.display.waveshow(counselor_y, sr=counselor_sr, ax=ax[0],  color="blue")
        ax[0].set(title=f"counselor avg: {counselor_audio_segment.dBFS:.2f} thresL {counselor_silence_thresh:.2f}")
        ax[0].label_outer()
        # Draw combine wave
        librosa.display.waveshow(counselor_y, sr=counselor_sr, alpha=0.5, ax=ax[2],  color="blue")
        librosa.display.waveshow(customer_y, sr=customer_sr, color='red', alpha=0.5, ax=ax[2])

        for idx, out_range in enumerate(customer_ranges):
            start = out_range[0] / 1000
            stop = out_range[1] / 1000

            rect = patches.Rectangle((start, -1), stop-start, 1.5, lw=0.5, edgecolor='red', facecolor='red', alpha=0.2)
            ax[1].add_patch(rect)
            ax[1].annotate(f"{idx}", (start, 0.3), color='black', fontsize=16)

        for idx, out_range in enumerate(counselor_ranges):
            start = out_range[0] / 1000
            stop = out_range[1] / 1000

            rect = patches.Rectangle((start, -1), stop-start, 1.5, lw=0.5, edgecolor='red', facecolor='red', alpha=0.2)
            ax[0].add_patch(rect)
            ax[0].annotate(f"{idx}", (start, 0.3), color='black', fontsize=16)


        if refined_customer_ranges is not None and refined_counselor_ranges is not None:
            for idx, out_range in enumerate(refined_customer_ranges):
                start = out_range[0] / 1000
                stop = out_range[1] / 1000

                rect = patches.Rectangle((start, 0), stop-start, 2, lw=0.5, edgecolor='green', facecolor='green', alpha=0.2)
                ax[1].add_patch(rect)

            for idx, out_range in enumerate(refined_counselor_ranges):
                start = out_range[0] / 1000
                stop = out_range[1] / 1000

                rect = patches.Rectangle((start, 0), stop-start, 2, lw=0.5, edgecolor='green', facecolor='green', alpha=0.2)
                ax[0].add_patch(rect)

        fig_output_folder = os.path.dirname(fig_output_file)
        if not os.path.isdir(fig_output_folder):
            os.makedirs(fig_output_folder)
        plt.savefig(fig_output_file)
        plt.close(fig)


    
    @staticmethod
    def generate_audio_wave(ms_adpcm_audio_file_path, offline_vad, output_folder, save_fig=True, save_combined_audio=True, fix_silence_thresh=False, min_audio_length=20):
        call_id = os.path.basename(ms_adpcm_audio_file_path).replace(".wav", "")
        curr_output_folder = os.path.join(output_folder, call_id)

        fig_output_file = os.path.join(curr_output_folder, "a_wave_image.png")
        combine_audio_file_path = os.path.join(curr_output_folder, "a_combine_audio.mp3")
        information_json = os.path.join(curr_output_folder, "info.json")

        # Return previous result if call already processed
        if os.path.isfile(information_json):
            with open(information_json, "r", encoding="utf-8") as f:
                information_dict = json.load(f)
                return information_dict
        
        # Create output folder if it isn't exist
        if not os.path.isdir(curr_output_folder):
            os.makedirs(curr_output_folder)

        # For MetaM case, The RX is Counselor
        is_rx_counselor = True

        informations = {
                    "call_id": call_id,
                    "ms_adpcm_audio_file_path" : ms_adpcm_audio_file_path,
                    "split_output_folder" : curr_output_folder,
                    "is_rx_counselor" : is_rx_counselor,
                }


        try:

            audio_rx, audio_tx, samplerate = get_left_right_audio_from_pcm_s4le(ms_adpcm_audio_file_path)
            if offline_vad is None:

                if "offline_vad" in MODEL_CACHE:
                    offline_vad = MODEL_CACHE["offline_vad"]
                else:
                    
                    # offline_vad = Pipeline.from_pretrained("/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/src/engine/pyannotate/config.yaml")
                    offline_vad = Pipeline.from_pretrained(config.as_dict()['PYANNOTATE_CONFIG_PATH'])
                    # offline_vad = offline_vad.to(device)
                    MODEL_CACHE["offline_vad"] = offline_vad

            informations["length"] = audio_rx.duration_seconds
            # Check threshold and ignore short audio call
            if audio_rx.duration_seconds < min_audio_length:
                informations["successYN"] = False
                informations["error"] = "TOO SHORT"

                with open(information_json, 'w') as fp:
                    json.dump(informations, fp, indent=4)
                output_error_file = os.path.join(output_folder, f"short_{call_id}.json")
                with open(output_error_file, 'w') as fp:
                    json.dump(informations, fp, indent=4)
                return informations


            STD_SILENT_THRES = -50

            customer_audio = audio_tx
            counselor_audio = audio_rx


            counselor_silence_thresh = 0
            customer_silence_thresh = 0

            rx_memory_file = io.BytesIO()
            audio_rx.export(rx_memory_file, format='wav')

            tx_memory_file = io.BytesIO()
            audio_tx.export(tx_memory_file, format='wav')

            customer_ranges = VADPyannoModel.pyannote_file_like_obj(tx_memory_file, offline_vad)     # https://github.com/jiaaro/pydub/issues/220
            counselor_ranges = VADPyannoModel.pyannote_file_like_obj(rx_memory_file, offline_vad)

            if save_combined_audio:
                # Save combine sound:
                combined_sound = audio_rx.overlay(audio_tx)
                combined_sound.export(combine_audio_file_path, format='mp3')

            # If detect nonsilent give us ranges of 0 or 1 item => Just save whole audio
            if len(counselor_ranges) in [0, 1] or len (customer_ranges) in [0, 1]:
                counselor_audio.export(curr_output_folder + "/chunk{0}_counselor_{1}_{2}.wav".format(str(0).rjust(5, '0'),0, int(audio_rx.duration_seconds*1000)), format = "wav")
                customer_audio.export(curr_output_folder + "/chunk{0}_customer_{1}_{2}.wav".format(str(1).rjust(5, '0'),1, int(audio_rx.duration_seconds*1000)), format = "wav")

                if save_fig:
                    VADPyannoModel.save_vizualization_figure(samplerate=samplerate,
                                            customer_audio_segment=customer_audio,
                                            counselor_audio_segment=counselor_audio,
                                            customer_silence_thresh=customer_silence_thresh,
                                            counselor_silence_thresh=counselor_silence_thresh,
                                            customer_ranges=customer_ranges,
                                            counselor_ranges=counselor_ranges,
                                            refined_counselor_ranges=None,
                                            refined_customer_ranges=None,
                                            fig_output_file=fig_output_file
                                            )

                informations["successYN"] = True
                informations["counselor_info"] = {
                    # "voice_avg" : counselor_audio.dBFS,
                    "silence_thresh" : counselor_silence_thresh,
                    "ranges" : counselor_ranges
                    }
                informations["customer_info"] = {
                    # "voice_avg" : customer_audio.dBFS,
                    "silence_thresh" : customer_silence_thresh,
                    "ranges" : customer_ranges,
                }

                with open(information_json, 'w') as fp:
                    json.dump(informations, fp, indent=4)

                return informations
            else:
                
                # Sort all audio ranges from customer channel and counselor channel into one
                sorted_range_mask = []
                sorted_type_mask = []
                sorted_index_mask = []

                counselor_index = 0
                customer_index = 0
                while counselor_index < len(counselor_ranges) or customer_index < len(customer_ranges):

                    if counselor_index == len(counselor_ranges):
                        customer_ts = customer_ranges[customer_index][0]
                        counselor_ts =MAXINT
                    elif customer_index == len (customer_ranges):
                        customer_ts = MAXINT
                        counselor_ts = counselor_ranges[counselor_index][0]
                    else:
                        customer_ts = customer_ranges[customer_index][0]
                        counselor_ts = counselor_ranges[counselor_index][0]
                    
                    if counselor_ts < customer_ts and counselor_index < len(counselor_ranges):
                        sorted_range_mask.append(counselor_ranges[counselor_index])
                        sorted_type_mask.append("counselor")
                        sorted_index_mask.append(counselor_index)
                        counselor_index += 1

                    elif customer_index < len(customer_ranges):
                        sorted_range_mask.append(customer_ranges[customer_index])
                        sorted_type_mask.append("customer")
                        sorted_index_mask.append(customer_index)
                        customer_index += 1
            


                # Refine the ranges, combine two consecutive ranges if they are same type AND not larger than MAX_SECOND time
                start_index = 0
                stop_index = 0
                curr_length_count = 0
                new_combined_customer_ranges = []
                new_combined_counselor_ranges = []
                start_type = sorted_type_mask[start_index]
        
                while stop_index < len(sorted_range_mask):
                    curr_length_count += sorted_range_mask[stop_index][1] - sorted_range_mask[start_index][0]
                    if stop_index == len(sorted_range_mask)-1 or sorted_type_mask[stop_index+1] != start_type or curr_length_count >= MAX_SECOND:
                        
                        # Assign sorted range mask into counselor and customer
                        combined_start_ts = sorted_range_mask[start_index][0]
                        combined_stop_ts = sorted_range_mask[stop_index][1]
                        if start_type == "counselor":
                            new_combined_counselor_ranges.append((combined_start_ts, combined_stop_ts))
                        else:
                            new_combined_customer_ranges.append((combined_start_ts, combined_stop_ts))

                        if stop_index != len(sorted_range_mask)-1:
                            start_index = stop_index + 1
                            start_type = sorted_type_mask[start_index]
                            curr_length_count = 0

                    stop_index += 1

                if save_fig:
                    VADPyannoModel.save_vizualization_figure(samplerate=samplerate,
                                            customer_audio_segment=customer_audio,
                                            counselor_audio_segment=counselor_audio,
                                            customer_silence_thresh=customer_silence_thresh,
                                            counselor_silence_thresh=counselor_silence_thresh,
                                            customer_ranges=customer_ranges,
                                            counselor_ranges=counselor_ranges,
                                            refined_customer_ranges=new_combined_customer_ranges,
                                            refined_counselor_ranges=new_combined_counselor_ranges,
                                            fig_output_file=fig_output_file
                                            )

                counselor_ranges = new_combined_counselor_ranges
                customer_ranges = new_combined_customer_ranges

                # Get audio chunks from our new combined ranges
                counselor_chunks, counselor_ranges_ext = VADPyannoModel.split_on_silence(counselor_audio, non_silence_range=counselor_ranges, return_timestamp=True, keep_silence=500)
                customer_chunks, customer_ranges_ext = VADPyannoModel.split_on_silence(customer_audio, non_silence_range=customer_ranges, return_timestamp=True, keep_silence=500)
                
                if len(counselor_chunks) == 1 or len (customer_chunks) == 1:
                    counselor_audio.export(curr_output_folder + "/chunk{0}_counselor_{1}_{2}.wav".format(str(0).rjust(5, '0'),0, int(audio_rx.duration_seconds*1000)), format = "wav")
                    customer_audio.export(curr_output_folder + "/chunk{0}_customer_{1}_{2}.wav".format(str(1).rjust(5, '0'),1, int(audio_rx.duration_seconds*1000)), format = "wav")
                else:
                    counselor_index = 0
                    customer_index = 0
                    while counselor_index < len(counselor_chunks) or customer_index < len(customer_chunks):

                        if counselor_index == len(counselor_chunks):
                            customer_ts = customer_ranges[customer_index][0]
                            counselor_ts =MAXINT
                        elif customer_index == len (customer_chunks):
                            customer_ts = MAXINT
                            counselor_ts = counselor_ranges[counselor_index][0]
                        else:
                            customer_ts = customer_ranges[customer_index][0]
                            counselor_ts = counselor_ranges[counselor_index][0]
                        
                        if counselor_ts < customer_ts and counselor_index < len(counselor_chunks):
                            chunk = counselor_chunks[counselor_index]
                            chunk = chunk.set_frame_rate(16000)

                            chunk_start_ts = max(0, counselor_ranges_ext[counselor_index][0])
                            chunk_end_ts = min(int(audio_rx.duration_seconds*1000), counselor_ranges_ext[counselor_index][1])

                            chunk.export(
                                curr_output_folder + "/chunk{0}_counselor_{1}_{2}.wav".format(str(counselor_index+customer_index).rjust(5, '0'), chunk_start_ts, chunk_end_ts),
                                format = "wav"
                            )
                            counselor_index += 1

                        elif customer_index < len(customer_chunks):
                            chunk = customer_chunks[customer_index]
                            chunk = chunk.set_frame_rate(16000)

                            chunk_start_ts = max(0, customer_ranges_ext[customer_index][0])
                            chunk_end_ts = min(int(audio_rx.duration_seconds*1000), customer_ranges_ext[customer_index][1])

                            chunk.export(
                                curr_output_folder + "/chunk{0}_customer_{1}_{2}.wav".format(str(counselor_index+customer_index).rjust(5, '0'),chunk_start_ts, chunk_end_ts),
                                format = "wav"
                            )
                            customer_index += 1


                informations["successYN"] = True
                informations["counselor_info"] = {
                        # "voice_avg" : counselor_audio.dBFS,
                        "silence_thresh" : counselor_silence_thresh,
                        "ranges" : counselor_ranges,
                        "ranges_ext" : counselor_ranges_ext
                    }  
                informations["customer_info"] = {
                        # "voice_avg" : customer_audio.dBFS,
                        "silence_thresh" : customer_silence_thresh,
                        "ranges" : customer_ranges,
                        "ranges_ext" :customer_ranges_ext
                    }

                with open(information_json, 'w') as fp:
                    json.dump(informations, fp, indent=4)
                return informations

        except Exception as e:
            error_msg = {
                "call_id": call_id,
                "ms_adpcm_audio_file_path" : ms_adpcm_audio_file_path,
                "split_output_folder" : curr_output_folder,
                "successYN" : False,
                "error" : str(e),
                "traceback": traceback.format_exc()
            }

            with open(information_json, 'w') as fp:
                    json.dump(error_msg, fp, indent=4)

            output_error_file = os.path.join(output_folder, f"error_{call_id}.json")
            with open(output_error_file, 'w') as fp:
                json.dump(error_msg, fp, indent=4)

            return error_msg