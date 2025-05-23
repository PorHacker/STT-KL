from genericpath import isdir
import os
import random
from re import S
from turtle import onclick
import fnmatch
import streamlit as st
import json
import pandas as pd
from torch import is_distributed
from utils import local_css, wer, colored
from PIL import Image
import streamlit.components.v1 as components
import csv
import datetime


# TEST
# VOICE_SPLIT_OUTPUT_DIR = "/home/metanet/Workspace/08-Kolon-Prj/02-Labeling-Tools-Design-Upgrade/unicef_origin_subset_test_split"
# STT_OUTPUT_DIR = "/home/metanet/Workspace/08-Kolon-Prj/02-Labeling-Tools-Design-Upgrade/unicef_origin_subset_test_stt"

## ~2024-04-19
# VOICE_SPLIT_OUTPUT_DIR = "/data/07-Kolon-Processed-Data/20240419/output_vad_folder/2024-04-19_18-01-18.044667_3bf76aa8-982d-49d0-a95a-9373e6b1909f"
# STT_OUTPUT_DIR = "/data/07-Kolon-Processed-Data/20240419/output_stt_ta_folder/2024-04-19_18-01-18.044667_3bf76aa8-982d-49d0-a95a-9373e6b1909f"
# LABEL_OUTPUT_DIR = "/home/metanet/Workspace/08-Kolon-Prj/03-Labeling-Tools-2024-04-19/labeling_tools/label_output_dir"

# ~2024-04-26
# VOICE_SPLIT_OUTPUT_DIR = "/data2/mario/sk_shielders/02.labeling/shared_folder/output_vad_folder/2025-01-15_15-02-04.518087_de918582-cd16-4808-b94a-ed6fda1fa344"
# STT_OUTPUT_DIR = "/data2/mario/sk_shielders/02.labeling/shared_folder/output_stt_ta_folder/2025-01-15_15-02-04.518087_de918582-cd16-4808-b94a-ed6fda1fa344"
# LABEL_OUTPUT_DIR = '/data2/mario/sk_shielders/02.labeling/labeling_tools/label_output_dir'
# LABEL_TASK_SEPARATE_FILE = "labeling_task_separate.csv"

# 250116 테스트
# VOICE_SPLIT_OUTPUT_DIR = "/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/data_pyannotate/splited_speaker_data_metam_pyannotate_change_rx_tx/split_mono_left/mono"
# STT_OUTPUT_DIR = "/nas2/voice/data/kynd/AItheDaisy/08-1-MetaM_Labeling_Prepare/data_pyannotate/stt_output_data_ctc_decoding_metam_pyannotate_change_rx_tx/split_mono_left"

# ROOT_DIR = "/data2/mario/sk_shielders/02.labeling/labeling_tools"
# LABEL_OUTPUT_DIR = f'{ROOT_DIR}/label_output_dir_0116'
# LABEL_TASK_SEPARATE_FILE = "labeling_task_separate_0116.csv"

# 250509 kim test
VOICE_SPLIT_OUTPUT_DIR = "/data12t/03_STT_FOR_SK/03_KIM/labeling-test/input-files/output_vad_folder/2025-05-09_10-48-00.343275_f03c9c1a-eb99-4dcc-a2fa-5ecc04775cd1"
STT_OUTPUT_DIR = "/data12t/03_STT_FOR_SK/03_KIM/labeling-test/input-files/output_stt_ta_folder/2025-05-09_10-48-00.343275_f03c9c1a-eb99-4dcc-a2fa-5ecc04775cd1"

ROOT_DIR = "/data12t/03_STT_FOR_SK/03_KIM/labeling-test/10-OP-labeling-tool-wisely-handover/labeling_tools"
LABEL_OUTPUT_DIR = f'{ROOT_DIR}/label_output_dir'
LABEL_TASK_SEPARATE_FILE = "labeling_task_separate_test.csv"



sentiment_labels = ["부정", "중립", "긍정"]
topic_labels = ["구매", "재구매", "환불", "교환", "취소"]

LABELLING_STATUS_FILE = "labeling_status.csv"

## 메타데이터 이용시
# labeling_meta_path='/data/20-Mario/03-Labeling-Tools-2024-04-26/labeling_tools2/sk_shielders_metadata.json'
# with open(labeling_meta_path,'r') as f:
#     LABELING_METAINFO_DICT=json.load(f)


st.set_page_config(
    page_title="AItheNutrigene - Smart STT Labeling Tool",
    page_icon="chart_with_upwards_trend",
    layout='wide'
    )

local_css("style.css")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def show():


    if check_password():

        task_separate = pd.read_csv(LABEL_TASK_SEPARATE_FILE, keep_default_na=False)

        user_ids = list(task_separate.columns)
        
        # Create user output folder
        user_id_to_folder_dict = {}
        for user_id in user_ids:
            user_id_folder = os.path.join(LABEL_OUTPUT_DIR, user_id)
            user_id_to_folder_dict[user_id] = user_id_folder

            # Create user output folder if not exist
            if not os.path.exists(user_id_folder):
                os.makedirs(user_id_folder)


        def init_reset_state():
            st.session_state.stt_engine_output_list = []
            st.session_state['sentiment'] = sentiment_labels[0]

            st.session_state['topic_labels_0'] = False
            st.session_state['topic_labels_1'] = False
            st.session_state['topic_labels_2'] = False
            st.session_state['topic_labels_3'] = False
            st.session_state['topic_labels_4'] = False

            output_label_name = st.session_state.current_call_id + "_label.json"
            previous_label_path = os.path.join(st.session_state.curr_user_folder, output_label_name)
            if os.path.exists(previous_label_path):
                with open(previous_label_path, "r",  encoding='utf-8') as fin:
                    previous_data = json.load(fin)
                    st.session_state.stt_engine_output_list = previous_data["stt_label"]
                    st.session_state.previous_sentiment = previous_data["sentiment"]
                    st.session_state.previous_topic = previous_data["topic"]
            
                    st.session_state['sentiment'] = previous_data["sentiment"]
                    st.session_state['topic'] = previous_data["topic"]

                    for idx, topic in enumerate(topic_labels):
                        if topic in st.session_state['topic']:
                            st.session_state[f"topic_labels_{idx}"] = True
                        else:
                            st.session_state[f"topic_labels_{idx}"] = False
                        
                    st.warning("Loaded from labelled call", icon="⚠️")

            else:
                output_label_name = st.session_state.current_call_id + "_label_ignored.json"
                previous_label_path = os.path.join(st.session_state.curr_user_folder, output_label_name)
                if os.path.exists(previous_label_path):
                    with open(previous_label_path, "r",  encoding='utf-8') as fin:
                        previous_data = json.load(fin)
                        st.session_state.stt_engine_output_list = previous_data["stt_label"]
                        st.session_state.previous_sentiment = previous_data["sentiment"]
                        st.session_state.previous_topic = previous_data["topic"]
                
                        st.session_state['sentiment'] = previous_data["sentiment"]
                        st.session_state['topic'] = previous_data["topic"]

                        for idx, topic in enumerate(topic_labels):
                            if topic in st.session_state['topic']:
                                st.session_state[f"topic_labels_{idx}"] = True
                            else:
                                st.session_state[f"topic_labels_{idx}"] = False

                        if previous_data["is_ignored"]:
                            st.warning("Loaded from ignored label", icon="🚨")

            # Update Selected Call ID Select Box
            st.session_state.selected_call_id_selectbox = st.session_state.current_call_id


        def on_current_call_id_change():
            st.session_state.current_call_id = st.session_state.selected_call_id_selectbox
            init_reset_state()


        def on_user_id_change():
            if "selected_user_id" not in st.session_state:
                st.session_state.curr_user_folder = user_id_to_folder_dict[user_ids[0]]
                st.session_state.selected_user_id = user_ids[0]
            else:
                st.session_state.curr_user_folder = user_id_to_folder_dict[st.session_state.selected_user_id]
            
            st.session_state.saved_section_selected_user_id = st.session_state.selected_user_id

            st.session_state.total_call_tasks = [str(value) for value in list(task_separate[st.session_state.selected_user_id]) if value != ""]

            if not os.path.exists(os.path.join(st.session_state.curr_user_folder, LABELLING_STATUS_FILE)):

                st.session_state.remaining_files = [str(value) for value in list(task_separate[st.session_state.selected_user_id]) if value != ""]

                st.session_state.annotations = []
            else: 
                labeling_status_pd = pd.read_csv(os.path.join(st.session_state.curr_user_folder, LABELLING_STATUS_FILE), keep_default_na=False, header=0)

                st.session_state.remaining_files = [str(value) for value in list(labeling_status_pd["Remaining ID"]) if value != ""]

                st.session_state.annotations = [str(value) for value in list(labeling_status_pd["Finish ID"]) if value != ""]


            if len(st.session_state.remaining_files) > 0:
                st.session_state.current_call_id = st.session_state.remaining_files[-1]
            else:
                st.warning("Finished!!!")
                
            init_reset_state()

        def ignore_and_move():
            save_and_move(True)

        def save_and_move(is_ignored = False):

            selected_topics = []
            for idx, topic in enumerate(topic_labels):
                if st.session_state[f"topic_labels_{idx}"]:
                    selected_topics.append(topic)

            cur_annotation = {
                "is_ignored" : is_ignored,
                "labeler_id" : st.session_state.selected_user_id,
                "call_id" : st.session_state.current_call_id,
                "label_time": str(datetime.datetime.now()),
                "sentiment": st.session_state.sentiment,
                "topic": selected_topics,
                "stt_label": stt_label
            }

            if not is_ignored:
                output_label_name = st.session_state.current_call_id + "_label.json"
            else:
                output_label_name = st.session_state.current_call_id + "_label_ignored.json"

            if not os.path.exists(st.session_state.curr_user_folder):
                os.makedirs(st.session_state.curr_user_folder)

            with open(os.path.join(st.session_state.curr_user_folder, output_label_name), "w",  encoding='utf-8') as fout:
                json.dump(cur_annotation, fout, ensure_ascii=False, indent=4)

            
            ## Reset all radio selection in Emotion

            for idx in range(0, len(stt_label)):
                st.session_state[f"emotion_{idx}"] = sentiment_labels[1]

            if st.session_state.current_call_id not in st.session_state.annotations:
                # For newly submit only, if edit => the call already in annotations list
                # Add current call id to annotations list
                st.session_state.annotations.append(st.session_state.current_call_id)

            # Sometime, call is already annotate and just update (edit) => already removed from remaining_files
            if st.session_state.current_call_id in st.session_state.remaining_files:
                # Remove already labled id:
                st.session_state.remaining_files.remove(st.session_state.current_call_id)

            save_pd = pd.DataFrame({"Finish ID": pd.Series(st.session_state.annotations), "Remaining ID": pd.Series(st.session_state.remaining_files)})
            save_pd.to_csv(os.path.join(st.session_state.curr_user_folder, LABELLING_STATUS_FILE), header=True, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

            if st.session_state.remaining_files:
                st.session_state.current_call_id = st.session_state.remaining_files[-1]
                init_reset_state()

            else: 
                st.session_state.current_call_id = st.session_state.total_call_tasks[0]
                st.warning("Finish!!!")

        def previous_call_bt():
            
            if len(st.session_state.annotations) > 0:
            # Get last call id from annotations list
                st.session_state.current_call_id = st.session_state.annotations[-1]

            # # Remove already labled id:
            # st.session_state.remaining_files.append(st.session_state.current_call_id)
            # save_pd = pd.DataFrame({"Finish ID": pd.Series(st.session_state.annotations), "Remaining ID": pd.Series(st.session_state.remaining_files)})
            # save_pd.to_csv(os.path.join(st.session_state.curr_user_folder, LABELLING_STATUS_FILE), header=True, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

            init_reset_state()

        if "annotations" in st.session_state and len(st.session_state.annotations) > 0:

            if st.button('Nothing Here'):
                st.write('Heyyy! Refresh!!')

            st.button('< Previous Call', on_click=previous_call_bt)

        st.write(
            """
            ## AItheNutrigene - Smart STT Labeling Tool
            """)


        if "saved_section_selected_user_id" in st.session_state:
            st.session_state.selected_user_id = st.session_state.saved_section_selected_user_id

        st.selectbox('Hello! Please select your ID:', user_ids, on_change=on_user_id_change, key='selected_user_id')

    
        if "annotations" not in st.session_state:
            on_user_id_change()

        # Need to place after select user ID
        st.session_state.selected_call_id_selectbox = st.session_state.current_call_id
        st.selectbox('Current CallID:', st.session_state.total_call_tasks, on_change=on_current_call_id_change, key='selected_call_id_selectbox')

            
        # Progress bar
        st.write("Annotated:",
                    len(st.session_state.annotations),
                    "– Total:",
                    len(st.session_state.remaining_files) + len(st.session_state.annotations),
                )


        # st.session_state.vizimage = os.path.join(VOICE_SPLIT_OUTPUT_DIR, st.session_state.current_call_id, "a_wave_image.png")
        # image = Image.open(st.session_state.vizimage)
        # st.image(image)

        # st.session_state.audio_file = open(os.path.join(VOICE_SPLIT_OUTPUT_DIR, st.session_state.current_call_id, "a_combine_audio.mp3"), 'rb')
        # st.session_state.audio_bytes = st.session_state.audio_file.read()
        # st.audio(st.session_state.audio_bytes, format='audio/mp3')


        st.session_state.infojson = os.path.join(VOICE_SPLIT_OUTPUT_DIR, st.session_state.current_call_id, "info.json")
        with open(st.session_state.infojson, "r", encoding="utf-8") as f:
                        st.session_state.split_info_data = json.load(f)

        st.write("Current Call_ID: ",
                    st.session_state.current_call_id,
        )
        st.write("(Duration: ",
                    datetime.timedelta(seconds=st.session_state.split_info_data['length']),
                    "   |  ",
                    st.session_state.split_info_data['length'],
                    "seconds )")

        # call_type_list=LABELING_METAINFO_DICT.get(st.session_state.current_call_id,["-","-","-"])
        # st.write(f"- 인아웃 : {call_type_list[0]}")
        # st.write(f"- 상담유형_대 : {call_type_list[1]}")
        # st.write(f"- 상담유형_중 : {call_type_list[2]}")


        st.write("### STT Labeling")


        if len(st.session_state.stt_engine_output_list) == 0:
            st.session_state.sttjson = os.path.join(STT_OUTPUT_DIR,st.session_state.current_call_id, st.session_state.current_call_id + "_all.json")
            with open(st.session_state.sttjson, "r", encoding="utf-8") as f:
                            st.session_state.stt_data = json.load(f)
                            st.session_state.stt_engine_output_list = st.session_state.stt_data['stt_engine_output']

        for idx, item in enumerate(st.session_state.stt_engine_output_list):
            col0, col1, col2 = st.columns((1, 5, 5))

            if "counselor" in item['audio_filepath']:
                col0.write(f"{idx}")
                col0.write("Employee")  
            else:
                col0.write(f"{idx}")
                col0.write("Customer")


            st.session_state.audio_file = open(item['audio_filepath'], 'rb')
            st.session_state.chunk_audio_bytes = st.session_state.audio_file.read()
            col1.audio(st.session_state.chunk_audio_bytes, format='audio/mp3')
            col1.write(item['pred_text'])

            col2.text(' ')

            if "label_text" in item:
                col2.text_area(label="Correct", key=f"correct_{idx}", value=item['label_text'])
                # col2.radio(
                #             "감정:",
                #             sentiment_labels,
                #             key=f"emotion_{idx}",
                #             # index=sentiment_labels.index(item['emotion']),
                #             index=1, # '중립'을 기본으로
                #             horizontal=True,
                #         )
            else:
                col2.text_area(label="Correct", key=f"correct_{idx}", value=item['pred_text'])

                # Fix Warning: The widget with key "emotion_0" was created with a default value but also had its value set via the Session State API.
                # col2.radio(
                #             "감정:",
                #             sentiment_labels,
                #             key=f"emotion_{idx}",
                #             index=1,
                #             horizontal=True,
                #         )
                    

        stt_label = []
        for idx, item in enumerate(st.session_state.stt_engine_output_list):
            data = {
                "audio_filepath": item['audio_filepath'],
                "pred_text": item['pred_text'],
                "label_text": st.session_state[f"correct_{idx}"],
                "emotion": st.session_state.get(f"emotion_{idx}","") # st.session_state[f"emotion_{idx}"]
            }

            stt_label.append(data)
            


        # st.write("#### 1. 감정 분류")
        # st.radio("선택:", sentiment_labels, key='sentiment')
        # st.write("#### 2. 주제 분류")
        # st.checkbox(topic_labels[0], key="topic_labels_0")
        # st.checkbox(topic_labels[1], key="topic_labels_1")
        # st.checkbox(topic_labels[2], key="topic_labels_2")
        # st.checkbox(topic_labels[3], key="topic_labels_3")
        # st.checkbox(topic_labels[4], key="topic_labels_4")
        # st.button('Submit', on_click=save_and_move)
        # st.button("Ignore", on_click=ignore_and_move)

        col0, space, col1 = st.columns((2, 5, 2))

        col0.button('Submit', on_click=save_and_move)
        col1.button("Ignore", on_click=ignore_and_move)

        components.html(
            "<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>",
            width= 0,
            height= 0)


if __name__ == "__main__":
    show()
