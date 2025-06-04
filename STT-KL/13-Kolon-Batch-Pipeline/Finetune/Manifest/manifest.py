"""
레이블러들이 작업한거 manifest 데이터로 만들기
"""

import os
import fnmatch
import json
import librosa
import ast
from tqdm import tqdm

def read_label_file(input_label):
    with open(input_label, "r", encoding="utf-8") as f:
        data  = json.load(f)
        return data

def convert_info(metainfo:list) -> dict:
    dict1={}
    metainfo["text"] = metainfo.pop("label_text")
    
    error_dict_list=[]
    for check_char in ['\n', ' ', '2', '4', 'ㄹ']:
        if check_char in metainfo["text"]: 
            stt_label_error = metainfo.copy()
            stt_label_error["reason_contains"] = check_char
            error_dict_list.append(stt_label_error)
            
    dict1['duration']=librosa.core.get_duration(path=metainfo["audio_filepath"])
    
    return {**metainfo, **dict1} #, error_dict_list


# key_list=['audio_filepath', 'text', 'duration', 'pred_text']

result_list=[]
count = 0 
records = []
no_edit_dict_list = []
edit_count_list = []
error_dict_list = []

STT_OUTPUT_DIR_LIST=["/data12t/03_STT_FOR_SK/03_KIM/labeling-test/10-OP-labeling-tool-wisely-handover/labeling_tools/label_output_dir"]

# STT_OUTPUT_DIR_LIST= ["/data/20-Mario/03-Labeling-Tools-2024-04-26/labeling_tools/label_output_dir_backup/label_output_dir_202412091706"]

call_count=0
for STT_OUTPUT_DIR in STT_OUTPUT_DIR_LIST:
    print("---------데이터 변환중------------")
    for root, dirnames, filenames in os.walk(STT_OUTPUT_DIR):
        print(filenames)
        call_count+=len(filenames)
        # print(root, dirnames, filenames)
        for idx, filename in enumerate(fnmatch.filter(filenames, "*_label.json")):
            file_path = os.path.join(root, filename)
            file_name = os.path.basename(file_path)
            # print(file_path, file_name)
            data = read_label_file(file_path)
            result=list(map(lambda x:convert_info(x),data["stt_label"]))
            # print(result)
            result_list.extend(result)


manifest_path="/"+os.path.join(*STT_OUTPUT_DIR.split("/")[:-1])+"/data_manifest"
os.makedirs(manifest_path,exist_ok=True)
manifest_ouput_name ="/data12t/03_STT_FOR_SK/03_KIM/manifest/data_manifest_all_250516.json"

total_duration=150*60*60
sum_duration=0
print("---------데이터 저장중------------")
with open(os.path.join(manifest_path,manifest_ouput_name), "w",  encoding='utf-8') as file:
    for idx,m in tqdm(enumerate(result_list)):
        sum_duration+=m['duration']
        
        if m['text'].strip():
            mm=json.dumps(m, ensure_ascii=False)
            file.write(mm+"\n")
        
        if sum_duration>=total_duration:
            print(f"끝난 파일명 및 인덱스 번호 :{idx} {m['audio_filepath']}")
            break
        
print(f"저장한 데이터셋 총 시간 : {sum_duration/(60*60)}")
print(f"---------{manifest_ouput_name} 데이터 저장 완료------------")