import streamlit as st
import requests
import json
import os
import uuid
import time
from PIL import Image


st.set_page_config(
    page_title="AItheNutrigene - Smart STI Demo",
    page_icon="chart_with_upwards_trend",
    layout='wide'
    )



st.write("# Welcome to AItheNutrigene - Smart STI Demo! 👋")

st.sidebar.success("Select a tab above.")

st.markdown(
    """
    #### Smart STT란 ? 

**음성파일(상담 녹취파일) 내용을 텍스트(text)로 변환해 주는 솔루션입니다.**

  - 가장 최신의 Conformer 딥러닝(DeepLearning) 알고리즘을 적용해 95% 이상의 인식률 성능

  - 실시간변환, 한국어-영어 혼용, Fast(1초당/파일), GPU 없어도 변환이 가능함

  - 묵음 필터링, 노이즈(noise) 제거 등의 음성파일 전처리(Pre-processing) 모듈 포함

   
![](http://aithe.io/common/images/solutions/pc_STT01.png "STT PIPELINE")
   

**Smart STT 프로세싱 방법론**

  - 음성 데이터에서 특징을 추출하여 벡터로 변환하고 음성인식 모델 학습에서 입력 데이터로 사용

  - 트랜스포머(Transformer)와 CNN을 함께 사용하여 지역 정보(local feature)와 광역 정보(global feature)를 효율적으로 반영하는 모델을 사용

![](http://aithe.io/common/images/solutions/pc_STT002.png "STT PIPELINE")


**성능평가(Performance Measures)**

![](http://aithe.io/common/images/solutions/pc_STT003.png "STT PIPELINE")


#### Demo 사이트

**아래 사이트를 방문하시면 실제 Smart STT 음성변환을 테스트 해 보실 수 있습니다.**

[\[데모사이트바로가기\]](https://smartstt.aithe.io/)
)

""")
