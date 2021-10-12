FROM ubuntu:18.04
ARG AIRFLOW_VERSION
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN mkdir -p /opt/airflow/
WORKDIR /opt/airflow/
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate\
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh
RUN conda config --set ssl_verify no

ADD requirements.txt /opt/airflow/
RUN pip install apache-airflow==$AIRFLOW_VERSION
RUN pip install -r /opt/airflow/requirements.txt
