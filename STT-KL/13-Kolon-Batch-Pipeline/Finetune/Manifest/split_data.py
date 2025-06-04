import json
import random
from tqdm import tqdm

# # 데이터 파일 경로
# input_file = "/data/20-Mario/03-Labeling-Tools-2024-04-26/labeling_tools/data_manifest/clean_sk_manifest.json"  # 원본 JSON 파일 경로
# output_train = "/data/20-Mario/SK-STT/dataset/clean_sk_manifest_train.json"
# output_valid = "/data/20-Mario/SK-STT/dataset/clean_sk_manifest_validation.json"
# output_test = "/data/20-Mario/SK-STT/dataset/clean_sk_manifest_test.json"


input_file = "/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/Manifest/data_manifest_all_250516.json"  # 원본 JSON 파일 경로 (mainifest에서 나온 모든 데이터) 
output_train = "/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/Manifest/filtering_SKShieldus_manifest_train.json"  # train으로 나뉘어져 나오는 파일 경로 설정 
output_valid = "/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/Manifest/filtering_SKShieldus_manifest_validation.json"  # validation으로 나뉘어져 나오는 파일 경로 설정
output_test = "/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/Manifest/filtering_SKShieldus_manifest_test.json"    # test으로 나뉘어져 나오는 파일 경로 설정

# JSON 파일 불러오기
with open(input_file, "r",encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 데이터 셔플링
random.shuffle(data)

# 데이터 나누기 (8:1:1 비율)
total_count = len(data)
train_count = int(total_count * 0.8)
valid_count = int(total_count * 0.1)

train_data = data[:train_count]
validation_data = data[train_count:train_count + valid_count]
test_data = data[train_count + valid_count:]
print(train_count,valid_count,len(test_data))


# 나눈 데이터를 각각 JSON 파일로 저장
with open(output_train, "w",encoding='utf-8') as f:
    for item in tqdm(train_data):
        f.write(json.dumps(item,ensure_ascii=False) + "\n")

with open(output_valid, "w",encoding='utf-8') as f:
    for item in tqdm(validation_data):
        f.write(json.dumps(item,ensure_ascii=False) + "\n")

with open(output_test, "w",encoding='utf-8') as f:
    for item in tqdm(test_data):
        f.write(json.dumps(item,ensure_ascii=False) + "\n")

print("데이터가 성공적으로 나누어졌습니다!")
