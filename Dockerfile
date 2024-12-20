# pytorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# set working directory

# copy the project code
COPY . .    