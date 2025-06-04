# This class should include the necessary methods for loading the model from disk and performing predictions.
import sys
import fnmatch
import contextlib
import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

import ctc_decoders   # To fix KENLM Hanging issue
import nemo.collections.asr as nemo_asr

from nemo.collections.asr.parts.submodules import ctc_beam_decoding

from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.transcribe_utils import transcribe_partial_audio
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils

from nemo.collections.asr.parts.submodules import rnnt_beam_decoding as beam_decode

from PIL import Image
import time
import traceback
import datetime
import config

config_dict = config.as_dict()

MODEL_CACHE = {}

import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

import re



@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    # rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1, strategy="beam", beam=beam_decode.BeamRNNTInferConfig(beam_size=2))
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()
    

class STTModel:

    @staticmethod
    @hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
    def main_inference(cfg: TranscriptionConfig):
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if cfg.model_path is None and cfg.pretrained_name is None:
            raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
        if cfg.audio_dir is None and cfg.dataset_manifest is None:
            raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

        # setup GPU
        if cfg.cuda is None:
            if torch.cuda.is_available():
                device = [0]  # use 0th CUDA device
                accelerator = 'gpu'
            else:
                device = 1
                accelerator = 'cpu'
        else:

            if cfg.cuda < 0:
                device = 1
                accelerator = 'cpu'

            else:
                device = [cfg.cuda]
                accelerator = 'gpu'


        print("device", device, "accelerator", accelerator, "cfg.cuda", cfg.cuda, "torch.cuda.is_available()", torch.cuda.is_available())
        map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
        if cfg.model_path not in MODEL_CACHE:

            print("AAAAAAAAAAAAAAAA MODEL is not in MODEL_CACHE, Load model AAAAAAAAAAAAAAAAAAAAAAAAAA")

            # setup model
            if cfg.model_path is not None:
                # restore model from .nemo file path
                model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
                classpath = model_cfg.target  # original class path
                imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
                logging.info(f"Restoring model : {imported_class.__name__}")
                asr_model = imported_class.restore_from(
                    restore_path=cfg.model_path, map_location=map_location
                )  # type: ASRModel
                model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
            else:
                # restore model by name
                asr_model = ASRModel.from_pretrained(
                    model_name=cfg.pretrained_name, map_location=map_location
                )  # type: ASRModel
                model_name = cfg.pretrained_name

            trainer = pl.Trainer(devices=device, accelerator=accelerator)
            asr_model.set_trainer(trainer)
            asr_model = asr_model.eval()
            
            # cache model
            MODEL_CACHE[cfg.model_path] = asr_model

        asr_model = MODEL_CACHE[cfg.model_path]
        partial_audio = False

        # Setup decoding strategy
        if hasattr(asr_model, 'change_decoding_strategy'):
            # asr_model.change_decoding_strategy(cfg.rnnt_decoding)
            # cfg.ctc_decoding.compute_timestamps = cfg.compute_timestamps
            asr_model.change_decoding_strategy(cfg.ctc_decoding)

        # get audio filenames
        if cfg.audio_dir is not None:
            filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"*.{cfg.audio_type}")))
            filepaths = sorted(filepaths)
        else:
            # get filenames from manifest
            filepaths = []
            if os.stat(cfg.dataset_manifest).st_size == 0:
                logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
                return None

            with open(cfg.dataset_manifest, 'r') as f:
                has_two_fields = []
                for line in f:
                    item = json.loads(line)
                    if "offset" in item and "duration" in item:
                        has_two_fields.append(True)
                    else:
                        has_two_fields.append(False)
                    filepaths.append(item['audio_filepath'])
            partial_audio = all(has_two_fields)

        logging.info(f"\nTranscribing {len(filepaths)} files...\n")

        # setup AMP (optional)
        if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logging.info("AMP enabled!\n")
            autocast = torch.cuda.amp.autocast
        else:

            @contextlib.contextmanager
            def autocast():
                yield

        # Compute output filename
        if cfg.output_filename is None:
            # create default output filename
            if cfg.audio_dir is not None:
                cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
            else:
                cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

        # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
        if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
            logging.info(
                f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
                f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
            )

            return cfg

        # transcribe audio
        with autocast():
            with torch.no_grad():
                if partial_audio:
                    if isinstance(asr_model, EncDecCTCModel):
                        transcriptions = transcribe_partial_audio(
                            asr_model=asr_model,
                            path2manifest=cfg.dataset_manifest,
                            batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                        )
                    else:
                        logging.warning(
                            "RNNT models do not support transcribe partial audio for now. Transcribing full audio."
                        )
                        transcriptions = asr_model.transcribe(
                            paths2audio_files=filepaths, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        )
                else:
                    transcriptions = asr_model.transcribe(
                        paths2audio_files=filepaths, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                    )

        logging.info(f"Finished transcribing {len(filepaths)} files !")

        logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

        # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis

        if type(transcriptions) == tuple and len(transcriptions) == 2:
            transcriptions = transcriptions[0]
        # write audio transcriptions
        with open(cfg.output_filename, 'w', encoding='utf-8') as f:
            if cfg.audio_dir is not None:
                for idx, text in enumerate(transcriptions):
                    item = {'audio_filepath': filepaths[idx], 'pred_text': text}
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                with open(cfg.dataset_manifest, 'r', encoding='utf-8') as fr:
                    for idx, line in enumerate(fr):
                        item = json.loads(line)
                        item['pred_text'] = transcriptions[idx]
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logging.info("Finished writing predictions !")

        results = []
        for idx, text in enumerate(transcriptions):
            item = {'audio_filepath': filepaths[idx], 'pred_text': text}
            results.append(item)

        return results

    def release_stt_model(self):
        torch.cuda.empty_cache()

    def stt_transcribe(self, diarization_records, data_ouput_stt_dir, logger, cuda=0, use_kenlm=False):
        diarization_stt_records = []

        print("Start time:", datetime.datetime.now(), use_kenlm)

        
        for diarization_record in diarization_records:
            diarization_stt_record = {}
            diarization_stt_record['call_id'] = diarization_record['call_id']
            diarization_stt_record['diarization_record'] = diarization_record

            call_Id = diarization_record['call_id']
            split_output_folder = diarization_record['split_output_folder']
            combine_audio_path = os.path.join(split_output_folder, "a_combine_audio.mp3")
            wave_img_path = os.path.join(split_output_folder, "a_wave_image.png")
            info_path = os.path.join(split_output_folder,f"info.json")

            call_date = call_Id
            stt_output_folder_dir = os.path.join(data_ouput_stt_dir, call_date)
            stt_json_output = os.path.join(stt_output_folder_dir, f"{call_Id}_stt.json")
            all_json_output = os.path.join(stt_output_folder_dir, f"{call_Id}_all.json")

            if not os.path.exists(stt_output_folder_dir):
                os.makedirs(stt_output_folder_dir)

            if os.path.isfile(all_json_output) :
                logger.info(f"Exist! {call_Id} at {all_json_output}")
                with open(all_json_output, 'r', encoding="utf-8") as fp:
                    data = json.load(fp)
                diarization_stt_record['stt_record'] = data
                diarization_stt_records.append(diarization_stt_record)
                continue

            stt_record = {
                        "call_id": call_Id,
                        "call_folder_path": split_output_folder,
                        "combine_audio_path": combine_audio_path,
                        "wave_img_path": wave_img_path,
                        "info_path": info_path,
                        "stt_json_output": stt_json_output,
                        "call_date": call_date
                    }

            if diarization_record['successYN']:
                start_time = time.time()


                if use_kenlm:
                        ctc_decoding: CTCDecodingConfig = CTCDecodingConfig(
                                                                                strategy="beam",
                                                                                beam= ctc_beam_decoding.BeamCTCInferConfig(
                                                                                    beam_size=4,
                                                                                    beam_alpha=1.0,
                                                                                    beam_beta=1.0,
                                                                                    kenlm_path=config_dict["KENLM_MODEL_CHECKPOINT"]
                                                                                    )
                                                                            )
                else:
                    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()


                try:

                    try:
                        cfg = TranscriptionConfig(
                        model_path=config_dict["STT_MODEL_CHECKPOINT"],
                        audio_dir=split_output_folder,
                        output_filename=stt_json_output,
                        batch_size=64,
                        cuda=cuda,
                        amp=False,
                        ctc_decoding=ctc_decoding
                        )

                        stt_engine_output = STTModel.main_inference(cfg)

                    except RuntimeError:
                        try:
                            logger.warning(f"OOM with 64, try 32 {split_output_folder}")

                            torch.cuda.empty_cache()
                            cfg = TranscriptionConfig(
                            model_path=config_dict["STT_MODEL_CHECKPOINT"],
                            audio_dir=split_output_folder,
                            output_filename=stt_json_output,
                            batch_size=32,
                            cuda=cuda,
                            amp=False,
                            ctc_decoding=ctc_decoding
                            )

                            stt_engine_output = STTModel.main_inference(cfg)

                        except RuntimeError:

                            try:
                                logger.warning(f"OOM with 32, try 4 {split_output_folder}")

                                torch.cuda.empty_cache()
                                cfg = TranscriptionConfig(
                                model_path=config_dict["STT_MODEL_CHECKPOINT"],
                                audio_dir=split_output_folder,
                                output_filename=stt_json_output,
                                batch_size=4,
                                cuda=cuda,
                                amp=False,
                                ctc_decoding=ctc_decoding
                                )

                                stt_engine_output = STTModel.main_inference(cfg)
                            except RuntimeError:
                                try:
                                    logger.warning(f"OOM with 4, try 1 {split_output_folder}")

                                    torch.cuda.empty_cache()
                                    cfg = TranscriptionConfig(
                                    model_path=config_dict["STT_MODEL_CHECKPOINT"],
                                    audio_dir=split_output_folder,
                                    output_filename=stt_json_output,
                                    batch_size=1,
                                    cuda=cuda,
                                    amp=False,
                                    ctc_decoding=ctc_decoding
                                    )

                                    stt_engine_output = STTModel.main_inference(cfg)
                                except Exception as e:

                                    stt_record['successYN'] = False
                                    stt_record['error'] = str(e)
                                    stt_record['traceback'] = traceback.format_exc()

                                    output_error_file = os.path.join(data_ouput_stt_dir, f"error_{call_Id}.json")
                                    with open(output_error_file, 'w', encoding="utf-8") as fp:
                                        json.dump(stt_record, fp, indent=4)

                                    with open(all_json_output, 'w', encoding="utf-8") as fp:
                                        json.dump(stt_record, fp, indent=4)
                                    diarization_stt_record['stt_record'] = stt_record
                                    diarization_stt_records.append(diarization_stt_record)
                                    continue


                    inference_time = time.time() - start_time


                    # Update stt_engine_output:

                    # Extract and add info to each element
                    for element in stt_engine_output:
                        audio_filepath = element["audio_filepath"]
                        
                        # Extract chunk number, channel, start time, and end time using regex
                        match = re.search(r'/chunk(\d{5})_(customer|counselor)_(\d+)_(\d+).wav', audio_filepath)
                        if match:
                            chunk_number = match.group(1)
                            channel = match.group(2)
                            start_time = match.group(3)
                            end_time = match.group(4)
                            
                            # Add extracted info to the current element
                            element["channel"] = channel
                            element["start_time"] = int(start_time)
                            element["end_time"] = int(end_time)

                    stt_record['successYN'] = True
                    stt_record['inference_time'] = inference_time
                    stt_record['stt_engine_output'] = stt_engine_output
                    stt_record["use_kenlm"] = use_kenlm
                    with open(all_json_output, 'w', encoding="utf-8") as fp:
                        json.dump(stt_record, fp, indent=4, ensure_ascii=False)
                    diarization_stt_record['stt_record'] = stt_record
                    diarization_stt_records.append(diarization_stt_record)

                except Exception as e:

                    stt_record['successYN'] = False
                    stt_record['error'] = str(e)
                    stt_record['traceback'] = traceback.format_exc()

                    logger.warning(f"Exception occur in STT Service: {stt_record}")

                    output_error_file = os.path.join(data_ouput_stt_dir, f"error_{call_Id}.json")
                    with open(output_error_file, 'w', encoding="utf-8") as fp:
                        json.dump(stt_record, fp, indent=4)
                        
                    with open(all_json_output, 'w', encoding="utf-8") as fp:
                        json.dump(stt_record, fp, indent=4)
                    diarization_stt_record['stt_record'] = stt_record
                    diarization_stt_records.append(diarization_stt_record)
            else:
                stt_record['successYN'] = False
                stt_record['error'] = "SPLIT ERROR - " + diarization_record['error']
                output_error_file = os.path.join(data_ouput_stt_dir, f"error_{call_Id}.json")
                with open(output_error_file, 'w', encoding="utf-8") as fp:
                    json.dump(stt_record, fp, indent=4)
                    
                with open(all_json_output, 'w', encoding="utf-8") as fp:
                    json.dump(stt_record, fp, indent=4)
                diarization_stt_record['stt_record'] = stt_record
                diarization_stt_records.append(diarization_stt_record)


        return diarization_stt_records

    