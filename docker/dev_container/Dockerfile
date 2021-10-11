FROM continuumio/miniconda3:latest

RUN mkdir -p /code
WORKDIR /code

COPY environment.yml ./
RUN conda config --set ssl_verify False
RUN conda env create -f environment.yml

RUN echo "conda activate airflow_ml" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
