import fnmatch
import os
import json

# vad, stt 결과 파일 복사 후 해당 경로로 변경
VOICE_SPLIT_OUTPUT_DIR = "/data12t/03_STT_FOR_SK/02_SON/labeling-test/input-files/output_vad_folder/2025-05-08_14-44-33.468264_50bf0ef7-ce20-4fec-8462-94f7f733dfdb"
STT_OUTPUT_DIR = "/data12t/03_STT_FOR_SK/02_SON/labeling-test/input-files/output_stt_ta_folder/2025-05-08_14-44-33.468264_50bf0ef7-ce20-4fec-8462-94f7f733dfdb"

DIR_SOURCE_PATTERN = "/workspace/shared_folder/"   # Fixed value for STI system
DIR_REPLACE_PATTERN = "/data12t/03_STT_FOR_SK/02_SON/labeling-test/input-files/"  # New DIR where the STI output store


def replace_text_in_file(file_path, source_pattern, replace_pattern):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        # Replace all occurrences of source_pattern with replace_pattern
        modified_content = file_content.replace(source_pattern, replace_pattern)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        return True
    except Exception as e:
        print("Error:", e)
        return False


### Replace _all.json output files
for root, dirnames, filenames in os.walk(STT_OUTPUT_DIR):
    for filename in fnmatch.filter(filenames, "*_all.json"):
        all_stt_output_file_path = os.path.join(root, filename)
        replace_text_in_file(all_stt_output_file_path, DIR_SOURCE_PATTERN, DIR_REPLACE_PATTERN)

### Replace _stt.json output files
for root, dirnames, filenames in os.walk(STT_OUTPUT_DIR):
    for filename in fnmatch.filter(filenames, "*_stt.json"):
        stt_output_file_path = os.path.join(root, filename)
        replace_text_in_file(stt_output_file_path, DIR_SOURCE_PATTERN, DIR_REPLACE_PATTERN)


### Replace in Split JSON output
for root, dirnames, filenames in os.walk(VOICE_SPLIT_OUTPUT_DIR):
    for filename in fnmatch.filter(filenames, "*nfo.json"):
        split_output_file_path = os.path.join(root, filename)
        replace_text_in_file(split_output_file_path, DIR_SOURCE_PATTERN, DIR_REPLACE_PATTERN)