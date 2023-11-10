FROM continuumio/miniconda3:23.3.1-0
COPY . /app
RUN conda env create -f /app/environment.yml
WORKDIR /app
