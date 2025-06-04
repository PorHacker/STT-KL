# BUILD DOCKER
# docker build -t qa_process:latest .

IMAGE_NAME=qa_process:latest
CONTAINER_NAME=qa_process

sudo docker run -itd --name ${CONTAINER_NAME} \
	-p 5002:5002	\
	-v /data/10-Kolon-QA-Pipeline/Server-STT-Masking-QA-Pipeline-Kolon/shared_folder:/workspace/shared_folder \
	-v /data/10-Kolon-QA-Pipeline/Server-STT-Masking-QA-Pipeline-Kolon/services/qa_process:/workspace \
	--gpus all \
	 ${IMAGE_NAME}