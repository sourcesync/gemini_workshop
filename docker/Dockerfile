FROM jupyter/datascience-notebook:177037d09156

USER root

#RUN sudo apt-get install libomp-dev

RUN mkdir -p /home/gemini
RUN chmod ugo+rw /home/gemini

USER jovyan
WORKDIR /home/gemini

RUN pip install faiss-cpu

COPY notebooks/Test.ipynb /home/gemini
COPY notebooks/gnl.py /home/gemini

USER root
