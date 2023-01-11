# Base image
# FROM ubuntu:latest

FROM python:3.8

# Set up for your local zone an UTC information
# ENV TZ=Europe/Paris
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
# RUN apt-get update && apt-get install -y wget

# RUN apt-get update && apt-get install -y software-properties-common && \
#    add-apt-repository ppa:deadsnakes/ppa

# Install the python3.9-venv package
# RUN apt-get update && apt-get install -y python3.9-venv

# Create a virtual environment and activate it
# RUN python3.9 -m venv myenv
# ENV PATH="/myenv/bin:$PATH"

# Install the required libraries
RUN pip install numpy pandas panel matplotlib scikit-learn xgboost \
 category_encoders joblib pandas_profiling eli5 shap seaborn gradio==3.12.0 \
 datasets

#fixer le port de sortie
EXPOSE 5000

#ouvrir un terminal bash
CMD ["bash"]