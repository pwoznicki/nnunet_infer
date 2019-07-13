FROM ubuntu:18.04
# FROM nvidia/cuda:9.2

MAINTAINER Piotr Woźnicki <piotrekwoznicki@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR /opt/conda
#ENV MODEL_DIR /home/piotr/kidney_seg/example
ENV PATH $CONDA_DIR/bin:$PATH
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libhdf5-dev g++ graphviz libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

RUN useradd -m deepcyst && echo "deepcyst:deepcyst" | chpasswd && adduser deepcyst sudo

RUN mkdir -p $CONDA_DIR && \
    chown deepcyst $CONDA_DIR -R

USER deepcyst

# Python
RUN pip install --upgrade pip && \
    pip install progressbar2 && \
    #pip install scipy && \
       conda clean -yt

RUN mkdir -p /home/deepcyst

RUN mkdir -p /home/deepcyst/github && \
    mkdir -p /home/deepcyst/data && \
    mkdir -p /home/deepcyst/nnUNet_raw && \
    mkdir -p /home/deepcyst/nnUNet_raw_splitted && \
    mkdir -p /home/deepcyst/nnUNet_raw_cropped && \
    cd /home/deepcyst/github/ && \
    git clone https://github.com/piotrekwoznicki/nnunet_infer.git

ADD nnunet_results /home/deepcyst/nnunet_results
ADD single_input /home/deepcyst/single_input

#RUN chown deepcyst  -R

# Running test.sh will download the weights and keep them in docker
RUN cd /home/deepcyst/github/nnunet_infer && \
    pip install -r nnunet/requirements.txt && \
    bash test.sh

EXPOSE 8888

WORKDIR "/home/deepcyst"
# CMD ["/bin/bash"]
ENTRYPOINT ["python","/home/deepcyst/github/nnunet_infer/nnunet/inference/predict_simple.py"]