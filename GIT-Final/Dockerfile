# Base image
# FROM ubuntu:latest

FROM python:3.8

# Installer les librairies
RUN pip install numpy pandas panel matplotlib scikit-learn xgboost \
 category_encoders pandas_profiling eli5 shap seaborn gradio==3.12.0 \
 datasets

#fixer le port de sortie
EXPOSE 5000

#ouvrir un terminal bash
CMD ["bash"]