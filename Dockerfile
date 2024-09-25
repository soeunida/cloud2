FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu20.04
FROM python:3.10-slim

WORKDIR /home/hayoung/cloud/workspace

COPY requirements.txt ./

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache -r requirements.txt

RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation

COPY . .

CMD  ["python", "script.py"]

