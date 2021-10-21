# Airflow 2.x for ML Pipeline Workshop - PyCon Sweden 2021

This is the corresponding git repo for PyCon Sweden 2021 Airflow 2.x for ML Pipeline Workshop.

## Pre-requisites

The whole workshop will work on your local docker environment. 

You should have `docker` and `docker-compose` installed on your machine !

The easiest way to have everything ready for the workshop is to install [Docker Desktop](https://docs.docker.com/desktop/.)

### Docker resource requirement
Minimum resource requirement for docker to start all the services is mentioned below:

| Resource    | Recommendation |
| ----------- | -------------- |
| Memory      | 3 GB           |
| CPU         | 2 CPU          |


## Getting started


### 1. Clone this repo
```
git clone https://github.com/pycon-ml/airflow_workshop.git
```

### 2. Setup local Airflow environment with Docker

Use docker-compose to build all the required docker images:

```
docker-compose pull
docker-compose build
```

### 3. Use docker-compose to start the applications:

```
docker-compose up
```
### 4. Access services from browser

#### **Airflow**

*UI*: http://localhost:8080

*Username*: airflow

*Password*: airflow

#### **MLflow**

http://localhost:5000

#### **Celery Flower**

http://localhost:5555

### 5. Try out the exercises
- Exercise 1: Checkout to [feature/exercise1_training_DAG](https://github.com/pycon-ml/airflow_workshop/tree/feature/exercise1_training_DAG) branch.
- Exercise 2: Checkout to [feature/exercise2_predict_multi_batch](https://github.com/pycon-ml/airflow_workshop/tree/feature/exercise2_predict_multi_batch) branch.

## Tear down

Stop and remove containers, networks, images, and volumes

```
docker-compose down
```

## Setup local environment for debug

If there is needs to have local environment to develop and debug, you can use `conda` to create the environment:

```
conda env create -f environment.yml
conda activate airflow_ml
```