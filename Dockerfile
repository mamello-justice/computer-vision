FROM jupyter/scipy-notebook

# Install additional modules
RUN python -m pip install opencv-python

ENV JUPYTER_ENABLE_LAB=yes