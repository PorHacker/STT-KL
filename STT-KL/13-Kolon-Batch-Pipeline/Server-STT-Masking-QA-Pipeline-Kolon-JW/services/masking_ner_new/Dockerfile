FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

WORKDIR /workspace

# apt install necessary packages
ENV DEBIAN_FRONTEND noninteractive

# Fix public key not available error: https://github.com/NVIDIA/nvidia-container-toolkit/issues/257#issuecomment-1903943680
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get install -y apt-utils \
    wget \
    git \
    gcc \
    build-essential \
    cmake \
    libpq-dev \
    libsndfile-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libsndfile1-dev \
    libopenblas-dev \
    libfftw3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgl1-mesa-glx \
    libomp-dev


RUN pip install wheel setuptools pip --upgrade
RUN pip install pybind11

# 1. upgrade pip
RUN pip install --upgrade pip

# # 2. install python-mecab-ko
# RUN pip install python-mecab-ko

# # 3. install pororo
# RUN pip install pororo --ignore-installed PyYAML

RUN pip install --no-cache-dir transformers

# 4. install brainspeech
RUN pip install soundfile \
    pydub

# 5. install etc modules
RUN pip install librosa \
    kollocate \
    koparadigm \
    g2pk \
    fugashi \
    ipadic \
    romkan \
    g2pM \
    jieba \
    opencv-python \
    scikit-image \
    editdistance==0.5.3 \
    epitran==1.2 \
    fastdtw==0.3.4 \
    future \
    Pillow==7.2.0 \
    pinyin==0.4.0 \
    scikit-learn \
    scipy \
    SoundFile==0.10.2 \
    numba==0.48 \
    ko_pron

# run app
# RUN pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY . .
RUN pip install -r requirements.txt

# NER
RUN pip install pyahocorasick==2.0.0

RUN chmod +x /workspace/app_run_gunicorn.sh
ENTRYPOINT ["/workspace/app_run_gunicorn.sh"]
EXPOSE 5002