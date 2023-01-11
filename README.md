# Defi-IA : 1001 nights
Please consider the final codes and files that are in **GIT-Final** folder.
## Overview of the project 
This project is for the kaggle competition (https://www.kaggle.com/competitions/defi-ia-2023).  <br /> 
The aim of the project is to accurately predict the price of a hotel room for a given night, taking into account various factors such as the location of the hotel, the language spoken by the person requesting the room, or the date of the request.


To explain our work for this project, you will find different scripts: <br />

* *train.py* : This script is used to train a XGBoost model using either OneHotEncoder or Classical TargetEncoder or Smooth TargetEncoder for categorical encoding and provides the computational time of the process. TargetEncoder function of category_encoders was used for Classical Target Encoding, where two parameters (min_samples_leaf and smoothing) can be tuned to add smoothing. A TargetEncoderSmooth class was written for TargetEncoder with smoothing weight=10, based on Max Halford idea.
The model can be fine-tuned using GridSearch or trained with pre-selected best parameters obtained with GridSearch. Two plots are generated in ./results_store directory, which represent respectively observed prices versus predicted prices and residuals versus predicted prices. <br />

* *app.py* : A gradio application, that gives an estimation of the price you should pay considering the following input parameters the user chooses : date, stock, language, hotel_id, mobile. The prediction is done with our Classical Target Encoding XGBoost model with 11 input features. <br />
      
* *analysis.ipynb* :  A notebook containing an analysis of our dataset and an interpretability of the model <br />
      
* *MSE-11.ipynb* : A notebook containing an analysis of Mean Square Errors as a function of some input features, with a Target Encoding XGBoost model with 11 input features, illustrated with boxplots for K folds <br />
      
* *MSE-7.ipynb* : A notebook containing an analysis of Mean Square Errors as a function of some input features, with a Target Encoding XGBoost model with 7 input features, illustrated with boxplots for K folds <br />
      
* *MSE.py* : Same as above with python. Number of features to be chosen. Plots are saved in ./results_store directory. <br /> 
      
## How to set up the project
To use the scripts for training the model or launching the gradio application from within a Docker container, follow these steps:

1. Install Docker on your machine if it is not already installed. You can find installation instructions for Docker at https://docs.docker.com/get-docker/.

2. Open a terminal or command prompt and navigate to the directory where you want to clone the repository.

3. Clone the repository using the following command: **git clone https://github.com/TryoenJ/Defi-IA.git**

4. Navigate to the directory containing the repository: **cd Defi-IA/GIT-Final**

5. Build the Docker image, with the name of your choice, using the following command: **docker build -t [image_name] .**

6. Start a new container, with the name of your choice, based on the image_name image, and open a command shell within the container: <br /> 
**docker run -it --name [container_name] -v [absolute_path_to_the_folder_of_the_cloned_repository]:/workspace/[folder_name_in_the_container] [image_name]** <br />

7. Navigate in the container in the directory :
**cd workspace/[folder_name_in_the_container]** <br />

8. Unzip the final model: **unzip XGB11_Target_model_saved_Final.zip**

9. Launch the gradio application with: **python app.py**

10. If you want to train the model, use the command : **python train.py** <br /> . XGBoost model is used with a Category Encoder that can be chosen. GridSearch can be used to find the best parameters of the XGBoost model (n_estimators and max_depth) and the best parameters of the Classical Target Encoding (min_samples_leaf and smoothing) if chosen for category encoding. 

11. If you want to compute and plot the MSE as a function of some input features, use the command : **python MSE.py** <br /> 

   
## Results
We trained the model for all inputs except nb_requetes and for the 11 or 7 most influent inputs, playing with GridSearch and Categorical Encoding. Parameters n_estimators=3000 and max_depth=10 were retained for the XGBoost model. Concerning Classical Target Encoding, it was difficult to find the best parameters ; consequently no smoothing was added. For Smoothing Target, smoothing weight parameter m was fixed to 10 and 100 but it should have been tuned with GridSearch.

XGB with Target Encoding seems to be better than with OneHot Encoding on our dataset (cf. R2 score and RMSE). However OneHot Encoding did best on the Defi dataset.

Below are different RMSE and R2 quantities for different models, as well as the computational time. Training with OneHot Encoding is faster in our case. It seems strange because OneHot implies more input parameters. Maybe the Target Encoder used is not optimal.  <br /> 

* XGB O 11 inputs <br />
computational time: 1685 seconds <br />
RMSE= 1.10379 <br />
R2= 0.99983 <br />

* XGB O 7 inputs <br />
computational time: 1516 seconds <br />
RMSE= 1.57410 <br />
R2= 0.99964 <br />

* XGB CT 11 inputs <br />
computational time: 2148 seconds <br />
RMSE= 1.04919 <br />
R2= 0.99984 <br />

* XGB CT 7 inputs <br />
computational time: 1652 seconds <br />
RMSE= 1.53339 <br />
R2= 0.99966 <br />

* XGB ST 10 11 inputs  <br /> 
computational time: 1930 seconds  <br /> 
RMSE= 1.00960  <br /> 
R2= 0.99985  <br /> 

* XGB ST 100 11 inputs <br />
computational time: 2031 seconds <br />
RMSE= 1.01197 <br />
R2= 0.99985 <br />