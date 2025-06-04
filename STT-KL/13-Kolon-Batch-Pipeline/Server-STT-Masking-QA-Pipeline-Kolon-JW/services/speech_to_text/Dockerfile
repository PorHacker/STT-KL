# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

FROM nvcr.io/nvidia/nemo:23.08

WORKDIR /workspace
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq software-properties-common libgl1-mesa-glx wget build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev curl

# #install python3.10
# RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
#     && apt-get -yq install python3.10 \
#     && ln -s /bin/python3.10 /bin/python \
#     && wget -q https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm -rf get-pip.py \
#     && apt install -yq python3.10-dev

# RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
# RUN pip install Cython==0.29.36
# RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 torchmetrics==0.11.4 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install PyYAML==6.0.1
# RUN pip install nemo-toolkit['all']==1.21.0 PyYAML==6.0.1

# Issue: https://github.com/NVIDIA/NeMo/issues/6984
# RUN pip install --upgrade --force-reinstall torchmetrics==0.11.4   

# install ctc_decoders
RUN apt-get update && apt-get install -y swig

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

    RUN pip install cmake

    WORKDIR /workspace/nemo

COPY scripts/. /workspace/nemo/scripts/
RUN /bin/bash scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh

WORKDIR /workspace
COPY . .
RUN pip install -r requirements.txt


# run app
RUN chmod +x /workspace/app_run_gunicorn.sh
ENTRYPOINT ["/workspace/app_run_gunicorn.sh"]
EXPOSE 5004