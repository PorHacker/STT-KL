#!/bin/bash

source /home/tako/anaconda3/bin/activate nemo_stt
cd /nas2/voice/data/kynd/AItheDaisy/10-OP-labeling-tool-wisely-handover/labeling_tools

/home/tako/anaconda3/envs/nemo_stt/bin/streamlit run Labeling_Page.py --server.port 8507

