# Airflow 2.x for ML Pipeline Workshop - PyCon Sweden 2021

This is the corresponding git repo for PyCon Sweden 2021 Airflow 2.x for ML Pipeline Workshop.

## Pre-requirements

The whole workshop will on your local docker environment. You should have docker and docker-compose ready on your computer.

The easiest way to have everything ready for the workshop is to install [Docker Desktop](https://docs.docker.com/desktop/.)

You need make sure docker config with at least 3GB RAM to start all the services.

## Getting started

### Build multi-container applications with Docker

Use docker-compose to build all the required docker images:

```
docker-compose build
```

### Use docker-compose to start the application:

```
docker-compose up
```

### Stop and remove containers, networks, images, and volumes

```
docker-compose down
```

## Access services from browser

### Airflow

UI: http://localhost:8080
Username: airflow
Password: airflow

### MLflow

http://localhost:5000

### Celery Flower

http://localhost:5555

## Setup local environment for debug

If there is needs to have local environment to develop and debug, you can use `conda` to create the environment:

```
conda env create -f environment.yml
conda activate airflow_ml
```