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


# Adding the MariaDB repository
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com 0xcbcb082a1bb943db
RUN curl -LsS https://downloads.mariadb.com/MariaDB/mariadb_repo_setup | bash

# Install the newest Connector/c
RUN apt-get update && apt-get install -y libmariadb3 libmariadb-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the Python MariaDB connector
RUN pip3 install mariadb

RUN apt-get update && apt-get install -y vim
# run app
COPY . .
RUN pip install -r requirements.txt

RUN pip install openai

RUN chmod +x /workspace/app_run_gunicorn.sh

ENTRYPOINT ["/workspace/app_run_gunicorn.sh"]
EXPOSE 5005