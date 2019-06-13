FROM ubuntu:18.04
# FROM nvidia/cuda:9.2

MAINTAINER Piotr Woźnicki <piotrekwoznicki@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR /opt/conda
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
    #pip install SimpleITK && \
    #pip install torch && \
    #pip install scikit-image && \
    #pip install batchgenerators>=0.19.3 && \
    #pip install medpy && \
    #pip install numpy && \
    #pip install sklearn && \
    conda clean -yt

RUN mkdir -p /home/deepcyst

RUN echo 'hi'

RUN mkdir -p /home/deepcyst/github && \
    mkdir -p /home/deepcyst/data && \
    cd /home/deepcyst/github/ && \
    git clone https://github.com/piotrekwoznicki/nnunet_infer.git

# Running test.sh will download the weights and keep them in docker
RUN cd /home/deepcyst/github/nnunet_infer && \
    pip install -r nnunet/requirements.txt && \
    bash test.sh

EXPOSE 8888

WORKDIR "/home/deepcyst"
# CMD ["/bin/bash"]
ENTRYPOINT ["python","/home/deepcyst/github/nnunet_infer/nnunet/fit.py"]