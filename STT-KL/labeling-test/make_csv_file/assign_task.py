import os
import json
import fnmatch
import csv

class DataDistributor:
    def __init__(self, voice_dir, stt_dir, output_dir, labeler_count):
        self.voice_dir = voice_dir
        self.stt_dir = stt_dir
        self.output_dir = output_dir
        self.labeler_count = labeler_count
        self.length_dict = {}
        self.excluded_files = []

    def get_duration_from_id(self, current_call_id):
        try:
            infojson_path = os.path.join(self.voice_dir, current_call_id, "info.json")
            with open(infojson_path, "r", encoding="utf-8") as f:
                split_info_data = json.load(f)
                return int(split_info_data['length'])
        except:
            return 0

    def load_data(self):
        for root, _, filenames in os.walk(self.stt_dir):
            for filename in fnmatch.filter(filenames, "*_all.json"):
                stt_output_file_path = os.path.join(root, filename)
                with open(stt_output_file_path, "r", encoding="utf-8") as f:
                    stt_data = json.load(f)
                    info_file_name = os.path.basename(stt_output_file_path).replace("_all.json", "")
                    duration = self.get_duration_from_id(info_file_name)

                    # 'stt_engine_output' 키가 존재하는지 확인
                    stt_output_length = len(stt_data.get('stt_engine_output', ""))

                    if stt_data.get("successYN") and 30 < duration < 420 and stt_output_length >= 20:
                        self.length_dict[info_file_name] = duration
                    else:
                        self.excluded_files.append((info_file_name, duration, stt_data.get("successYN", False), stt_output_length))


    def distribute(self):
        labelers = {f"Labeler_{i+1}": {} for i in range(self.labeler_count)}
        labeler_durations = [0] * self.labeler_count
        
        # duration 내림차순 정렬 (긴 파일부터 분배)
        sorted_files = sorted(self.length_dict.items(), key=lambda x: x[1], reverse=True)

        for filename, duration in sorted_files:
            min_index = labeler_durations.index(min(labeler_durations))
            labelers[f"Labeler_{min_index + 1}"][filename] = duration
            labeler_durations[min_index] += duration

        return labelers

    def print_distribution_summary(self, labelers):
        print("\n=== 분배 요약 ===")
        for labeler, files in labelers.items():
            total_files = len(files)
            total_duration = sum(files.values()) / 60  # 분 단위로 변환
            print(f"{labeler}: {total_files}개 파일, 총 {total_duration:.2f}분")

    def save_distribution(self, labelers):
        os.makedirs(self.output_dir, exist_ok=True)
        # 분배된 데이터 저장
        csv_path = os.path.join(self.output_dir, 'labeling_task_separate_test.csv')
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(labelers.keys())
            max_len = max(len(v) for v in labelers.values())
            for i in range(max_len):
                row = []
                for v in labelers.values():
                    row.append(list(v.keys())[i] if i < len(v) else '')
                writer.writerow(row)
        print(f"\n분배 결과가 '{csv_path}'에 저장되었습니다.")
        
        # 사용되지 않은 데이터 저장
        excluded_csv_path = os.path.join(self.output_dir, 'excluded_files_test.csv')
        with open(excluded_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Filename", "Duration (s)", "SuccessYN", "STT Output Length"])
            for file_info in self.excluded_files:
                writer.writerow(file_info)
        print(f"사용되지 않은 파일 목록이 '{excluded_csv_path}'에 저장되었습니다.")


if __name__ == '__main__':
    # 사용자 지정
    VOICE_SPLIT_OUTPUT_DIR = "/data12t/03_STT_FOR_SK/03_KIM/labeling-test/input-files/output_vad_folder/2025-05-09_10-48-00.343275_f03c9c1a-eb99-4dcc-a2fa-5ecc04775cd1"
    STT_OUTPUT_DIR = "/data12t/03_STT_FOR_SK/03_KIM/labeling-test/input-files/output_stt_ta_folder/2025-05-09_10-48-00.343275_f03c9c1a-eb99-4dcc-a2fa-5ecc04775cd1"
    OUTPUT_DIR = "/data12t/03_STT_FOR_SK/03_KIM/labeling-test/10-OP-labeling-tool-wisely-handover/labeling_tools"
    LABELER_COUNT = 4

    distributor = DataDistributor(VOICE_SPLIT_OUTPUT_DIR, STT_OUTPUT_DIR, OUTPUT_DIR, LABELER_COUNT)
    distributor.load_data()
    labelers = distributor.distribute()
    distributor.print_distribution_summary(labelers)
    distributor.save_distribution(labelers)
