# Defi-IA : 1001 nights
Please consider the final codes and files that are in **GIT-Final** folder.
## Overview of the project 
This project is for the kaggle competition (https://www.kaggle.com/competitions/defi-ia-2023).  <br /> 
The aim of the project is to accurately predict the price of a hotel room for a given night, taking into account various factors such as the location of the hotel, the language spoken by the person requesting the room, or the date of the request.


To explain our work for this project, you will find different scripts: <br /> 
      - *train.py* : This script is used to train a XGBoost model using either OneHotEncoder or TargetEncoder for categorical encoding. The model can be fine-tuned using GridSearch or trained with pre-selected best parameters obtained by GridSearch A REFORMULER ?(that give satisfactory results without excessive training time). <br /> 
      - *app.py* : A gradio application, that gives an estimation of the price you should pay considering the following input parameters the user choose : A COMPLETER The prediction is done with our best ? model (XGBoost and TargetEncoding) BEST = XGBoost + OneHot .<br />
      - *analysis.ipynb* :  An analysis of our data set and an interpretability of the model <br /> 
      - *MSE.ipynb* : (TO ADD) An analysis of errors as a function of inputs, using the Mean Square Error <br /> 
      
## How to set up the project
To use the scripts for training the model or launching the gradio application from within a Docker container, follow these steps:

1. Install Docker on your machine if it is not already installed. You can find installation instructions for Docker at https://docs.docker.com/get-docker/.
2. Open a terminal or command prompt and navigate to the directory where you want to clone the repository.
3. Clone the repository using the following command: **git clone https://github.com/TryoenJ/Defi-IA.git**
4. Navigate to the directory containing the repository: **cd Defi-IA/GIT-Final**
5. Build the Docker image using the following command: **docker build -t [image_name] .**
6. Start a new container based on the image_name image, and open a command shell within the container: <br /> 
**docker run -it --name [container_name] -v [absolute_path_to_the_folder_of_the_cloned_repository]:/workspace/[folder_name_in_the_container] [image_name]** <br />
7. Navigate in the container in the directory :
**cd workspace/[folder_name_in_the_container]** <br />
8. Download the model: <br />

option 1 (Lauriane link): <br />
**wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1NKbJqeZBIKEukFCsMITM0-L4wEN14wvZ' -O XGB_Target_model_saved.joblib** <br />

option 2 (Julie link): <br />
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xGZ6AUsLssx2ka8jQDuUY-webCxK5Xad' -O XGB_Target_model_saved.joblib  <br />

option 3 avec curl (marche pas non plus): <br />
curl -L -o XGB_Target_model_saved.joblib https://drive.google.com/uc?export=download&id=1NKbJqeZBIKEukFCsMITM0-L4wEN14wvZ  <br />
option 4 faire tourner train.py, save the model, et faire tourner app.py avec ca: <br />

8. Train the model with: **python train.py** <br /> 
9. Launch the gradio application with: **python app.py**
   
## Results
MSE, score ...
