APP_NAME=cloudsdetect
CONTAINER_NAME=clouds

build:  ## Build the container
	sudo docker build -t ${APP_NAME} -f Dockerfile .

run: ## Run container
	docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME} \
		-v /mnt/ssd/videoanalytics/clouds_data:/data \
		-v /mnt/ssd/videoanalytics/clouds_dumps:/dumps \
		-v $(shell pwd):/clouds-satellite $(APP_NAME) bash

exec: ## Run a bash in a running container
	docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}
