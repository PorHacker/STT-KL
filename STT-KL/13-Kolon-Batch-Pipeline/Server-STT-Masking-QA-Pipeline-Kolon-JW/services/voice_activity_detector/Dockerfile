# build: docker build -t metam-vad:1.0.0-debug .
# run: sudo docker run -itd --name ${CONTAINER_NAME} \
	# -p 5003:5003	\
	# -v /nas2/voice/data/kynd/AItheDaisy/15-MetaM-System-Design/02-API/Server-STT-TAs-Dockers/shared_folder:/workspace/shared_folder \
	# -v /nas2/voice/data/kynd/AItheDaisy/15-MetaM-System-Design/02-API/Server-STT-TAs-Dockers/services/voice_activity_detector/logs:/workspace/logs \
	# --gpus all \
	#  ${IMAGE_NAME}

# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

# apt install necessary packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq software-properties-common libgl1-mesa-glx wget build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev curl
RUN apt-get install -y ffmpeg
#install python3.10
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get -yq install python3.10 \
    && ln -s /bin/python3.10 /bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm -rf get-pip.py \
    && apt install -yq python3.10-dev

RUN pip install pip==23.0
# RUN pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
RUN pip install pytorch-lightning==1.6.5
RUN pip install -qq https://github.com/pyannote/pyannote-audio/archive/2.1.1.zip
RUN pip install --force-reinstall torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# run app
COPY . .
RUN pip install -r requirements.txt

RUN pip install librosa

RUN chmod +x /workspace/app_run_gunicorn.sh

ENTRYPOINT ["/workspace/app_run_gunicorn.sh"]
EXPOSE 5003