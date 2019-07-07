FROM continuumio/miniconda3

WORKDIR /root
COPY lab.sh .
COPY environment.yaml /tmp/environment.yaml
RUN conda env update -n base -f /tmp/environment.yaml