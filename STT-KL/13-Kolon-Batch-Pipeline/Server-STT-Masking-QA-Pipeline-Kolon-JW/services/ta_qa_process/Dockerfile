# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

# apt install necessary packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq software-properties-common libgl1-mesa-glx wget build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev apt-transport-https curl

#install python3.10
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get -yq install python3.10 \
    && ln -s /bin/python3.10 /bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm -rf get-pip.py \
    && apt install -yq python3.10-dev


RUN pip install wheel setuptools pip --upgrade

RUN pip install --upgrade pip


# 1. Install OpenAI
RUN pip install openai==1.10.0

# 2. Install langchain
RUN pip install langchain==0.1.12

# 3. Install openpyxl
RUN pip install openpyxl==3.1.2

# 4 Install langchain-openai
RUN pip install langchain-openai

# 5. Install chromadb
RUN pip install chromadb

# 6 install sqlite3
RUN pip install pysqlite3-binary

RUN pip install pandas==1.5.3 \
    pdfplumber==0.10.4


COPY . .
RUN pip install -r requirements.txt

RUN chmod +x /workspace/app_run_gunicorn.sh
ENTRYPOINT ["/workspace/app_run_gunicorn.sh"]
EXPOSE 5001