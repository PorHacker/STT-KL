import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import datetime
import fnmatch
from utils import local_css, wer, colored

st.set_page_config(
    page_title="AItheNutrigene - Smart STT Labeling Tool",
    page_icon="chart_with_upwards_trend",
    layout='wide'
    )

local_css("style.css")

# ROOT_DIR = "/data2/mario/sk_shielders/02.labeling/labeling_tools"
# VOICE_SPLIT_OUTPUT_DIR = "/data2/mario/sk_shielders/02.labeling/shared_folder/output_vad_folder/2025-01-15_15-02-04.518087_de918582-cd16-4808-b94a-ed6fda1fa344"
# STT_OUTPUT_DIR = "/data2/mario/sk_shielders/02.labeling/shared_folder/output_stt_ta_folder/2025-01-15_15-02-04.518087_de918582-cd16-4808-b94a-ed6fda1fa344"

# LABEL_OUTPUT_DIR = f"{ROOT_DIR}"
# LABEL_TASK_SEPARATE_FILE = "labeling_task_separate.csv"


ROOT_DIR = "/data2/mario/sk_shielders/02.labeling/labeling_tools"
LABEL_OUTPUT_DIR = f'{ROOT_DIR}/label_output_dir_0116'
LABEL_TASK_SEPARATE_FILE = "labeling_task_separate_0116.csv"
VOICE_SPLIT_OUTPUT_DIR = "/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/data_pyannotate/splited_speaker_data_metam_pyannotate_change_rx_tx/split_mono_left/mono"
STT_OUTPUT_DIR = "/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/data_pyannotate/stt_output_data_ctc_decoding_metam_pyannotate_change_rx_tx/split_mono_left"


task_separate = pd.read_csv(os.path.join(ROOT_DIR, LABEL_TASK_SEPARATE_FILE), keep_default_na=False)
user_ids = list(task_separate.columns)


def check_password(password_correct_key, screts_key):
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets[screts_key]:
            st.session_state[password_correct_key] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state[password_correct_key] = False

    if password_correct_key not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state[password_correct_key]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Convert datetime format
def get_duration_from_id(current_call_id):
    try:
        infojson_path = os.path.join(VOICE_SPLIT_OUTPUT_DIR, current_call_id, "info.json")
        with open(infojson_path, "r", encoding="utf-8") as f:
            split_info_data = json.load(f)
            return int(split_info_data['length'])
    except:
        return 0
    

def convert_duration(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


# Convert datetime format
def get_label_status(current_call_id):
    try:
        infojson_path = os.path.join(VOICE_SPLIT_OUTPUT_DIR, current_call_id, "info.json")
        with open(infojson_path, "r", encoding="utf-8") as f:
            split_info_data = json.load(f)
            return int(split_info_data['length'])
    except:
        return 0

def get_label_value_if_have(all_labelled_data_dict, call_id, key):
    if call_id in all_labelled_data_dict:
        data =  all_labelled_data_dict[call_id][key]
        return str(data)
    else:
        return "-"
    

def show():
    if check_password("monitor_page_pw_correct", "password_monitoring"):
        for user_id in user_ids:
            current_user_pd = task_separate[[user_id]].copy()
            indexEmptyRow = current_user_pd[ (current_user_pd[user_id] == "")].index
            current_user_pd.drop(indexEmptyRow, inplace=True)

            current_user_pd["Second"] = current_user_pd[user_id].map(lambda x: get_duration_from_id(x))
            current_user_pd["Duration"] = current_user_pd["Second"].map(lambda x: convert_duration(x))


            all_labelled_data_dict = {}
            user_id_folder = os.path.join(LABEL_OUTPUT_DIR, user_id)
            for root, dirnames, filenames in os.walk(user_id_folder):
                for filename in fnmatch.filter(filenames, "*.json"):
                    file_path = os.path.join(root, filename)

                    with open(file_path, "r", encoding="utf-8") as f:
                            curr_label_info_data = json.load(f)

                    call_id = curr_label_info_data["call_id"]

                    interesting_keys = ["is_ignored", "label_time", "sentiment"]
                    current_call_id_interesting_dict = {x: curr_label_info_data[x] for x in interesting_keys if x in curr_label_info_data}
                    current_call_id_interesting_dict["file_path"] = file_path

                    all_labelled_data_dict[call_id] = current_call_id_interesting_dict

            current_user_pd["Status"] = current_user_pd[user_id].map(lambda x: x in all_labelled_data_dict)
            current_user_pd["Ignored"] = current_user_pd[user_id].map(lambda x: get_label_value_if_have(all_labelled_data_dict, x, "is_ignored"))
            current_user_pd["Submit Time"] = current_user_pd[user_id].map(lambda x: get_label_value_if_have(all_labelled_data_dict, x, "label_time"))
            current_user_pd["Sentiment"] = current_user_pd[user_id].map(lambda x: get_label_value_if_have(all_labelled_data_dict, x, "sentiment"))
            current_user_pd["File Location"] = current_user_pd[user_id].map(lambda x: get_label_value_if_have(all_labelled_data_dict, x, "file_path"))

            current_user_all_time = current_user_pd["Second"].sum()
            current_user_total_finished_time = current_user_pd.loc[current_user_pd['Status'],'Second'].sum()

            st.write(f"### Progress on user {user_id}")
            # st.write("Finished Duration: ", datetime.timedelta(seconds=float(current_user_total_finished_time)))
            # st.write("Total Duration: ", datetime.timedelta(seconds=float(current_user_all_time)))

            st.write("Finished Duration: ", convert_duration(current_user_total_finished_time))
            st.write("Total Duration: ", convert_duration(current_user_all_time))



            st.write("")
            st.dataframe(current_user_pd)


if __name__ == "__main__":
    show()
