#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NO_COLOR='\033[0m'

DOCKER_CONTAINER_NAME=personal-website
DOCKER_IMAGE_NAME=personal-website
WEBSITE_VERSION=0.0.1
PORT=9090

function message() {
    T=$(date)
    echo -e "${CYAN} ${T} :: $1 ${NO_COLOR}"
}

function error() {
    T=$(date)
    echo -e "${RED} ${T} :: $1 ${NO_COLOR}"
}

function success() {
    T=$(date)
    echo -e "${GREEN} ${T} :: $1 ${NO_COLOR}"
}

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ ${GIT_BRANCH} != "master" ]]; then
    error "DEPLOYING FROM BRANCH OTHER THAN MASTER :: [ BRANCH -> ${GIT_BRANCH} ]"
    exit 1
else
    success "BRANCH CHECK PASSED! Deploying from master ... "
fi

message "DOWNLOAD DEPENDENIES :: npm install"
npm install

message "BUILDING WEBSITE :: npm run prod"
npm run prod
RETURN_CODE=$(echo $?)

if [ $RETURN_CODE -ne 0 ]; then
    error "Failed To Build Website"
    exit ${RETURN_CODE}
else
    success "BUILT WEBSITE"
fi

message "KILLING DOCKER CONTAINER :: docker"
docker kill personal-website
RETURN_CODE=$(echo $?)

if [ $RETURN_CODE -ne 0 ]; then
    message "Failed to Kill Docker Container. May be Docker Container was Never Run"
else
    message "Removing Docker Container"
    docker rm personal-website
fi

message "Building New Docker Container"
docker build . -t ${DOCKER_CONTAINER_NAME}:${WEBSITE_VERSION}
RETURN_CODE=$(echo $?)

if [ $RETURN_CODE -ne 0 ]; then
    error "Failed To Build Docker Container"
    exit ${RETURN_CODE}
else
    success "BUILT DOCKER CONTAINER"
fi

message "Starting Website in a Docker Image"
docker run -d -p ${PORT}:80 --name ${DOCKER_IMAGE_NAME} ${DOCKER_CONTAINER_NAME}:${WEBSITE_VERSION}
RETURN_CODE=$(echo $?)

if [ $RETURN_CODE -ne 0 ]; then
   error "Failed To Start Docker Image"
   exit ${RETURN_CODE}
fi

success "THE WEBSITE IS UP AND RUNNING"