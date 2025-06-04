# This class should include the necessary methods for loading the model from disk and performing predictions.
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

MAX_SECOND = 16000

MAX_CHUNKS_SIZE = 90000

class VADRuleBasedModel:

    @staticmethod
    def check_length_and_save(input_audio_segment, save_audio_path, format, chunk_duration=60000):
        """
        Check the audio chunk and split the input segment audio file into chunks, each chunk not longer than 1 minute (60,000 milliseconds).
        
        Parameters:
            input_file_path (str): The path of the input WAV audio file.
            chunk_duration (int): The duration of each chunk in milliseconds. Default is 60,000 milliseconds (1 minute).
        """
        # Calculate the number of chunks needed
        total_duration = len(input_audio_segment)

        if total_duration < MAX_CHUNKS_SIZE:
            input_audio_segment.export(save_audio_path, format=format)
        else:
            # Calculate the number of chunks needed
            num_chunks = (total_duration + chunk_duration - 1) // chunk_duration
            
            # Split the audio into chunks and save each chunk
            chunk_paths = []
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = (i + 1) * chunk_duration
                if end_time > total_duration:
                    end_time = total_duration
                chunk = input_audio_segment[start_time:end_time]
                chunk_path = save_audio_path.replace(".wav", f"_{i}.wav")
                chunk.export(chunk_path, format=format)

    @staticmethod
    def is_rx_counselor(rx_audio_file_path):
        filename = os.path.basename(rx_audio_file_path)
        if filename.find("_") > 5:
            return False
        else:
            return True

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
        librosa.display.waveshow(customer_y, sr=customer_sr, ax=ax[1])
        ax[1].set(title=f"customer avg: {customer_audio_segment.dBFS:.2f} thres: {customer_silence_thresh:.2f}")
        ax[1].label_outer()
        # Draw tx wave
        librosa.display.waveshow(counselor_y, sr=counselor_sr, ax=ax[0])
        ax[0].set(title=f"counselor avg: {counselor_audio_segment.dBFS:.2f} thresL {counselor_silence_thresh:.2f}")
        ax[0].label_outer()
        # Draw combine wave
        librosa.display.waveshow(counselor_y, sr=counselor_sr, alpha=0.5, ax=ax[2])
        librosa.display.waveshow(customer_y, sr=customer_sr, color='r', alpha=0.5, ax=ax[2])

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

    def generate_audio_wave(self, ms_adpcm_audio_file_path, output_folder, save_fig=False, save_combined_audio=True, fix_silence_thresh=True, min_audio_length=20):
        call_id = os.path.basename(ms_adpcm_audio_file_path).replace(".wav", "")
        curr_output_folder = os.path.join(output_folder, call_id)

        fig_output_file = os.path.join(curr_output_folder, "a_wave_image.png")
        combine_audio_file_path = os.path.join(curr_output_folder, "a_combine_audio.mp3")
        information_json = os.path.join(curr_output_folder, "info.json")

        # Return previous result if call already processed
        if os.path.isfile(information_json):
            with open(information_json, "r", encoding="utf-8") as f:
                information_dict = json.load(f)

                if information_dict["successYN"]:
                    del information_dict["customer_info"]
                    del information_dict["counselor_info"]
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
            if is_rx_counselor:
                customer_audio = audio_rx
                customer_silence_thresh = audio_rx.dBFS * 1.8
                counselor_audio = audio_tx
                counselor_silence_thresh = audio_tx.dBFS * 1.8
                
            else:
                customer_audio = audio_tx
                counselor_audio = audio_rx

            counselor_silence_thresh = 0
            customer_silence_thresh = 0

            if fix_silence_thresh:
                counselor_silence_thresh = STD_SILENT_THRES
                customer_silence_thresh = STD_SILENT_THRES
                counselor_ranges = detect_nonsilent(counselor_audio, 200, STD_SILENT_THRES, 1)
                customer_ranges = detect_nonsilent(customer_audio, 200, STD_SILENT_THRES, 1)
            else:
                # Find best silence_thresh

                # Check all threadhold in range and select one which give highest number of ranges
                len_counselor_ranges = 0
                for silence_thresh in range(int(counselor_audio.dBFS)-10, STD_SILENT_THRES, -2):
                    cur_counselor_ranges = detect_nonsilent(counselor_audio, 200, silence_thresh, 1)

                    if len(cur_counselor_ranges) > len_counselor_ranges:
                        len_counselor_ranges = len(    cur_counselor_ranges)
                        counselor_ranges = cur_counselor_ranges
                        counselor_silence_thresh = silence_thresh

                if counselor_silence_thresh == 0:
                    counselor_ranges = detect_nonsilent(counselor_audio, 200, STD_SILENT_THRES, 1)

                # Check all threadhold in range and select one which give highest number of ranges
                len_customer_ranges = 0
                for silence_thresh in range(int(customer_audio.dBFS)-10, STD_SILENT_THRES, -2):
                    cur_customer_ranges = detect_nonsilent(customer_audio, 200, silence_thresh, 1)

                    if len(cur_customer_ranges) > len_customer_ranges:
                        len_customer_ranges = len(cur_customer_ranges)
                        customer_ranges = cur_customer_ranges
                        customer_silence_thresh = silence_thresh
                    
                if customer_silence_thresh == 0:
                    customer_ranges = detect_nonsilent(customer_audio, 200, STD_SILENT_THRES, 1)

            if save_combined_audio:
                # Save combine sound:
                combined_sound = audio_rx.overlay(audio_tx)
                combined_sound.export(combine_audio_file_path, format='mp3')


            # If detect nonsilent give us ranges of 0 or 1 item => Just save whole audio
            if len(counselor_ranges) in [0, 1] or len (customer_ranges) in [0, 1]:
                if is_rx_counselor:
                    # counselor_audio.export(curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")
                    # customer_audio.export(curr_output_folder + "/chunk{0}_customer_{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                    VADRuleBasedModel.check_length_and_save(counselor_audio, curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(1).rjust(5, '0'),1),format = "wav")
                    VADRuleBasedModel.check_length_and_save(customer_audio, curr_output_folder + "/chunk{0}_customer_{1}.wav".format(str(0).rjust(5, '0'),0),format = "wav")
                else:
                    # counselor_audio.export(curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                    # customer_audio.export(curr_output_folder + "/chunk{0}_customer_{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")
                    VADRuleBasedModel.check_length_and_save(counselor_audio, curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                    VADRuleBasedModel.check_length_and_save(customer_audio, curr_output_folder + "/chunk{0}_customer_{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")

                if save_fig:
                    VADRuleBasedModel.save_vizualization_figure(
                                            samplerate=samplerate,
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

                del informations["customer_info"]
                del informations["counselor_info"]
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
                    VADRuleBasedModel.save_vizualization_figure(samplerate=samplerate,
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
                counselor_chunks, counselor_ranges_ext = VADRuleBasedModel.split_on_silence(counselor_audio, non_silence_range=counselor_ranges, return_timestamp=True, keep_silence=500)
                customer_chunks, customer_ranges_ext = VADRuleBasedModel.split_on_silence(customer_audio, non_silence_range=customer_ranges, return_timestamp=True, keep_silence=500)
                
                if len(counselor_chunks) == 1 or len (customer_chunks) == 1:
                    if is_rx_counselor:
                        # counselor_audio.export(curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")
                        # customer_audio.export(curr_output_folder + "/chunk{0}_customer__{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                        VADRuleBasedModel.check_length_and_save(counselor_audio, curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")
                        VADRuleBasedModel.check_length_and_save(customer_audio, curr_output_folder + "/chunk{0}_customer__{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                    else:
                        # counselor_audio.export(curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                        # customer_audio.export(curr_output_folder + "/chunk{0}_customer__{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")
                        VADRuleBasedModel.check_length_and_save(counselor_audio, curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(0).rjust(5, '0'),0), format = "wav")
                        VADRuleBasedModel.check_length_and_save(customer_audio, curr_output_folder + "/chunk{0}_customer__{1}.wav".format(str(1).rjust(5, '0'),1), format = "wav")
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
                            VADRuleBasedModel.check_length_and_save(
                                chunk,
                                curr_output_folder + "/chunk{0}_counselor_{1}.wav".format(str(counselor_index+customer_index).rjust(5, '0'),counselor_ts),
                                format = "wav"
                            )
                            counselor_index += 1

                        elif customer_index < len(customer_chunks):
                            chunk = customer_chunks[customer_index]
                            chunk = chunk.set_frame_rate(16000)
                            VADRuleBasedModel.check_length_and_save(
                                chunk,
                                curr_output_folder + "/chunk{0}_customer_{1}.wav".format(str(counselor_index+customer_index).rjust(5, '0'),customer_ts),
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

                del informations["customer_info"]
                del informations["counselor_info"]
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