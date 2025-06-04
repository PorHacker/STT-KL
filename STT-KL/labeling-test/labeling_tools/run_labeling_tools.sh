#!/bin/bash

source /home/tako/anaconda3/bin/activate nemo_stt
cd /data12t/03_STT_FOR_SK/03_KIM/labeling-test/10-OP-labeling-tool-wisely-handover/labeling_tools

/home/tako/anaconda3/envs/nemo_stt/bin/streamlit run AitheNutrigene_STI_Solution.py --server.port 8708
