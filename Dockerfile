FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04

ENV ACCEPT_EULA=Y
RUN rm /etc/apt/sources.list.d/mssql-release.list
RUN apt update
RUN yes | apt upgrade
RUN apt dist-upgrade
RUN yes | apt install unattended-upgrades
RUN unattended-upgrade -d

RUN yes | apt-get purge openssh-client
RUN apt-get install git -y

WORKDIR /app
SHELL ["/bin/bash", "--login", "-c"]

ADD requirements.txt .
RUN conda create -n agent_py_env python=3.8
RUN echo "conda activate agent_py_env" > ~/.bashrc
RUN conda init bash
ENV PYTHONPATH "/opt/miniconda/envs/agent_py_env/bin/python"
ENV PATH "$PYTHONPATH:$PATH"

RUN conda activate agent_py_env \
    && pip install --no-cache-dir -i https://repo1.com/artifactory/api/pypi/pypi-virtual/simple/ -r requirements.txt


