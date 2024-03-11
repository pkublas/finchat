IMAGE_NAME="finchat"
IMAGE_VERSION="0.1"
CONTAINER_NAME="finchat_app"

run_job:
	docker build -t ${IMAGE_NAME}:${IMAGE_VERSION} -f Dockerfile . && \
	docker run --env-file .env --name ${CONTAINER_NAME} --volume ${PWD}/data:/app/data --volume ${PWD}/reports:/app/reports --rm ${IMAGE_NAME}:${IMAGE_VERSION}
