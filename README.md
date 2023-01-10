# Defi-IA : 1001 nights
Please consider the final codes and files that are in **GIT-Final** folder.
## Overview of the project 
This project is for the kaggle competition (https://www.kaggle.com/competitions/defi-ia-2023).  <br /> 
The aim of the project is to accurately predict the price of a hotel room for a given night, taking into account various factors such as the location of the hotel, the language spoken by the person requesting the room, or the date of the request.


To explain our work for this project, you will find different scripts: <br /> 
      - *train.py* : This script is used to train a XGBoost model using either OneHotEncoder or Classical TargetEncoder or Smooth TargetEncoder for categorical encoding. The model can be fine-tuned using GridSearch or trained with pre-selected best parameters obtained by GridSearch. Two plots are generated in ./results_store directory, which represent espectively observed prices versus predicted prices and residuals versus predicted prices. <br /> 
      - *app.py* : A gradio application, that gives an estimation of the price you should pay considering the following input parameters the user chooses : date, stock, language, hotel_id, mobile. The prediction is done with our Target Encoding XGBoost model with 11 input features. <br />
      - *analysis.ipynb* :  A notebook containing an analysis of our dataset and an interpretability of the model <br /> 
      - *MSE.ipynb* : A notebook containing an analysis of Mean Square Errors as a function of some input features, with a Target Encoding XGBoost model with 7 input features <br /> 
      - *MSE.py* : Same as above with python. Plots saved in ./results_store directory <br /> 
      
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

8. Unzip the final model: <br />

**unzip XGB11_Target_model_saved_Final.zip**

9. Launch the gradio application with: **python app.py**

10. If you want to train the model, use the command : **python train.py** <br /> 

   
## Results
TO BE COMPLETED

XGB 11 inputs

RMSE= 1.0491885291687986

R2= 0.999842147804718

XGB 7 inputs

RMSE= 1.533388621745516

R2= 0.9996628306946584
