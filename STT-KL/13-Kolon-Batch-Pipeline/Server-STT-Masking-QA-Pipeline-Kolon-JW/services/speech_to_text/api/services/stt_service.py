# This file will contain the service responsible for handling the prediction logic. 

import os
import fnmatch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import functools

class PredictionService:
    def __init__(self, model):
        self.model = model


    def transcribe(self, diarization_records, data_output_folder, logger):
        
        diarization_stt_records = self.model.stt_transcribe(diarization_records, data_output_folder, logger, cuda=-1)
        return diarization_stt_records

    def transcribe_gpu(self, diarization_records, data_output_folder, logger):

        diarization_stt_records = self.model.stt_transcribe(diarization_records, data_output_folder, logger)

        self.model.release_stt_model()
        return diarization_stt_records
    

    def transcribe_gpu_lm(self, diarization_records, data_output_folder, logger):

        diarization_stt_records = self.model.stt_transcribe(diarization_records, data_output_folder, logger, use_kenlm=True)

        self.model.release_stt_model()
        return diarization_stt_records


