# This file will contain the service responsible for handling the prediction logic. 

import os
import fnmatch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import functools

class PredictionService:
    def __init__(self, model):
        self.model = model

    def split_single(self, data_input_folder, data_output_folder, ms_adpcm_audio_file_paths=None):

        if ms_adpcm_audio_file_paths is None:
            ms_adpcm_audio_file_paths = []

            # Search for all .wav audio files in data_input_folder
            for root_dir, dirnames, filenames in os.walk(data_input_folder):
                for filename in fnmatch.filter(filenames, "*.wav"):
                    ms_adpcm_audio_file_path = os.path.join(root_dir, filename)
                    ms_adpcm_audio_file_paths.append(ms_adpcm_audio_file_path)

        records = []
        for input_file in ms_adpcm_audio_file_paths:
            result = self.model.generate_audio_wave(input_file, data_output_folder, fix_silence_thresh=True)
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

        # results = []
        # with ThreadPoolExecutor(max_workers=30) as executor:
        #     # Submit the process_audio function for each audio file
        #     futures = [executor.submit(self.model.generate_audio_wave, audio_path, data_output_folder) for audio_path in ms_adpcm_audio_file_paths]
        #     # Gather the results as they become available
        #     for future in futures:
        #         result = future.result()
        #         results.append(result)
        # return results

        if len(ms_adpcm_audio_file_paths) < 2:
            return self.split_single(data_input_folder, data_output_folder, ms_adpcm_audio_file_paths)

        num_workers = os.cpu_count() - 5

        if len(ms_adpcm_audio_file_paths) < num_workers:
            num_workers = len(ms_adpcm_audio_file_paths)

        records = []
        with multiprocessing.Pool(num_workers) as p:
            processing_func = functools.partial(self.model.generate_audio_wave, output_folder=data_output_folder, save_fig=False, fix_silence_thresh=True)
            results = p.imap_unordered(processing_func, ms_adpcm_audio_file_paths)
            for result in results:
                if result is not None:
                    records.append(result)
        return records


