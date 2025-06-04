# This file will contain the service responsible for handling the prediction logic. 

import os
import fnmatch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import functools
import torch

class PyAnnoPredictionService:
    def __init__(self, model):
        self.model = model

    def split_single(self, data_input_folder, data_output_folder):
        ms_adpcm_audio_file_paths = []

        # Search for all .wav audio files in data_input_folder
        for root_dir, dirnames, filenames in os.walk(data_input_folder):
            for filename in fnmatch.filter(filenames, "*.wav"):
                ms_adpcm_audio_file_path = os.path.join(root_dir, filename)
                ms_adpcm_audio_file_paths.append(ms_adpcm_audio_file_path)

        if len(ms_adpcm_audio_file_paths) == 0:
            return []
            
        records = []
        for input_file in ms_adpcm_audio_file_paths:
            result = VADPyannoModel.generate_audio_wave(input_file, data_output_folder, fix_silence_thresh=True)
            if result is not None:
                records.append(result)


        return records

    def split_paralell(self, data_input_folder, data_output_folder):

        ms_adpcm_audio_file_paths = []

        # Search for all .wav audio files in data_input_folder
        for root_dir, dirnames, filenames in os.walk(data_input_folder):
            for filename in fnmatch.filter(filenames, "*.wav"):
                ms_adpcm_audio_file_path = os.path.join(root_dir, filename)
                ms_adpcm_audio_file_paths.append(ms_adpcm_audio_file_path)


        if len(ms_adpcm_audio_file_paths) == 0:
            return []

        num_workers = 8
        if len(ms_adpcm_audio_file_paths) < num_workers:
            num_workers = len(ms_adpcm_audio_file_paths)

        records = []
        # with multiprocessing.Pool(num_workers) as p:
        with multiprocessing.get_context("spawn").Pool(num_workers) as p:
            processing_func = functools.partial(self.model.generate_audio_wave,offline_vad=None, output_folder=data_output_folder, save_fig=True, fix_silence_thresh=True)
            results = p.imap_unordered(processing_func, ms_adpcm_audio_file_paths)
            for result in results:
                if result is not None:
                    records.append(result)
        return records


