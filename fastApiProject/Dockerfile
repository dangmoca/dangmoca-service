# from
FROM ubuntu:latest

# apt init
ENV LANG=C.UTF-8
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y sudo

# Anaconda 설치
RUN apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /usr/local/anaconda && \
    rm miniconda.sh

# 환경 변수 설정
ENV PATH="/usr/local/anaconda/bin:$PATH"

# apt cleanse
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# timezone
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . ./

# workspace
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

# pip upgrade
RUN pip install --upgrade pip

#############
# uvicorn 설치 및 PATH 추가
RUN apt-get update && \
    apt-get install -y gcc && \
    pip install fastapi && \
    pip install uvicorn && \
    pip3 install torch torchvision torchaudio && \
    ln -s /usr/local/anaconda/bin/uvicorn /usr/bin/uvicorn

RUN pip install poetry
RUN pip wheel --no-cache-dir --use-pep517 "gluonnlp (==0.10.0)"
RUN poetry install

COPY requirements.txt . 
RUN pip install --upgrade --no-deps --force-reinstall -r requirements.txt
#############

# for run server
ENV APP_ENV=dev
EXPOSE 8083
CMD uvicorn main:app --host=0.0.0.0 --port=8083
