# BUILD DOCKER
# docker build -t masking_ner:latest .

IMAGE_NAME=masking_ner:latest
CONTAINER_NAME=masking_ner

sudo docker run -itd --name ${CONTAINER_NAME} \
	-p 5004:5004	\
	-v /data/10-Kolon-QA-Pipeline/Server-STT-Masking-QA-Pipeline-Kolon/shared_folder:/workspace/shared_folder \
	-v /data/10-Kolon-QA-Pipeline/Server-STT-Masking-QA-Pipeline-Kolon/services/masking_ner:/workspace \
	--gpus all \
	 ${IMAGE_NAME}