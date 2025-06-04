import requests
import threading
import time
import csv
import uuid
import numpy as np
from datetime import datetime
import os
from io import BytesIO
from pydub import AudioSegment

# Define the API endpoint and payload
url = "http://127.0.0.1:9889/tts"
payload = {
    # "text": "제품을 확인해 보니 언제 주문하셨나요? 지난주 화요일에 주문했어요",
    "text": " 안녕하세요, 고객님. 먼저 불편을 드려 죄송합니다. 어떤 제품에 문제가 생기셨나요? 불편을 겪으셔서 정말 죄송합니다. 제품을 확인해 보니 언제 주문하셨나요? ",
    "speaker": "KR",
    "speed": 1.0,
    "media_type": "wav"
}
headers = {"Content-Type": "application/json"}

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Output directories
output_dir = f"/data2/02-TTS/01-Inference-Server/00-MeloTTS/script/TEST-OUTPUT-OFFLINE_{current_datetime}"
statistic_output_dir = f"/data2/02-TTS/01-Inference-Server/00-MeloTTS/script/statistic_output/offline_{current_datetime}"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(statistic_output_dir, exist_ok=True)

# Global variables
lock = threading.Lock()
results = []

def send_request(connection_id, max_connection, is_save_audio=True):
    global results
    start_time = time.time()
    request_uid = str(uuid.uuid4())

    print(f"Request ID: {request_uid} - Starting request...")
    
    payload_with_id = payload.copy()
    payload_with_id["request_uid"] = request_uid

    response = requests.post(url, headers=headers, json=payload_with_id)
    received_time = time.time()
    elapsed_time = received_time - start_time

    if response.status_code == 200:
        try:
            audio_data = BytesIO(response.content)
            audio_segment = AudioSegment.from_file(audio_data, format="wav")
            audio_duration = len(audio_segment) / 1000.0  # Duration in seconds
            rtf = elapsed_time / audio_duration if audio_duration > 0 else None
            
            if is_save_audio:
                output_path = os.path.join(output_dir, f"{max_connection}_audio_{connection_id}_{request_uid}.wav")
                audio_segment.export(output_path, format="wav")

            response_time_dt = datetime.fromtimestamp(received_time)
            detailed_time = response_time_dt.strftime('%H:%M:%S') + f".{response_time_dt.microsecond // 1000:03d}"
            
            print(f"UID: {request_uid} | Connection {connection_id}: Success (RTF={rtf:.2f}, Duration={audio_duration:.2f}s, Elapsed={elapsed_time:.3f}s, Received={detailed_time})")
            
            with lock:
                results.append([request_uid, connection_id, "Success", elapsed_time, rtf, audio_duration, detailed_time])
        except Exception as e:
            print(f"Connection {connection_id}: Failed to process audio - {e}")
            with lock:
                results.append([request_uid, connection_id, "Error", None, None, None, None])
    else:
        print(f"Connection {connection_id}: Request failed with status {response.status_code}")
        with lock:
            results.append([request_uid, connection_id, "HTTP Error", None, None, None, None])

def stress_test(max_connections):
    threads = []
    for i in range(max_connections):
        thread = threading.Thread(target=send_request, args=(i, max_connections,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

def calculate_percentiles(data, percentiles=[99, 95, 90, 85, 80, 75]):
    return {f"P{p}": np.percentile(data, p) for p in percentiles}

def save_results_to_csv(count):
    output_csv = os.path.join(statistic_output_dir, f"tts_test_results_{count}.csv")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Request ID", "Connection ID", "Status", "Elapsed Time (s)", "RTF", "Audio Duration (s)", "Received Time"])
        writer.writerows(results)
    print(f"Results saved to {output_csv}")

def save_percentiles_to_csv(percentile_results, connection_counts, metric_name):
    output_csv = os.path.join(statistic_output_dir, f"tts_percentile_results_{metric_name}.csv")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        headers = ["Percentile"] + [str(count) for count in connection_counts]
        writer.writerow(headers)
        
        for percentile in [99, 95, 90, 85, 80, 75]:
            row = [f"P{percentile}"] + [percentile_results[str(count)].get(f"P{percentile}", None) for count in connection_counts]
            writer.writerow(row)
    print(f"{metric_name} Percentile results saved to {output_csv}")

if __name__ == "__main__":
    connection_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    percentile_results = {
        "Elapsed Time (s)": {},
        "RTF": {},
    }
    
    for count in connection_counts:
        results = []
        print(f"\nStarting test with {count} concurrent connections...")
        stress_test(count)
        save_results_to_csv(count)
        
        elapsed_times = [r[3] for r in results if r[3] is not None]
        rtfs = [r[4] for r in results if r[4] is not None]
        
        percentile_results["Elapsed Time (s)"][str(count)] = calculate_percentiles(elapsed_times)
        percentile_results["RTF"][str(count)] = calculate_percentiles(rtfs)
    
    save_percentiles_to_csv(percentile_results["Elapsed Time (s)"], connection_counts, "Elapsed_Time")
    save_percentiles_to_csv(percentile_results["RTF"], connection_counts, "RTF")
