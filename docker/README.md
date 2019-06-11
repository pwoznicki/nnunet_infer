# README

## CPU

##### Downloading the docker image from the hub
```
docker pull deepcyst/nnunet-infer-cpu
```

##### Building the docker
```
docker build -t nnunet-infer-cpu -f Dockerfile.cpu .
```

##### Running the docker
```
docker run -t -v [Absolute PATH to the Project Folder]/Prostate-Segmenter/deepcyst/data/test/:/home/deepcyst/data deepcyst/nnunet-infer-cpu --InputVolume /home/deepcyst/data/input.nrrd --OutputLabel /home/deepcyst/data/label_predicted_test.nrrd
```