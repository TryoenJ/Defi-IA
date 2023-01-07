# Defi-IA
Commands to run for training our model or launching the gradio application, from a Docker container:

1. Install Docker on your machine if it is not already installed. You can find installation instructions for Docker at https://docs.docker.com/get-docker/.
2. Open a terminal or command prompt and navigate to the directory where you want to clone the repository.
3. Clone the repository using the following command: **git clone https://github.com/TryoenJ/Defi-IA.git**
4. Navigate to the directory containing the repository: **cd Defi-IA/GIT-Final**
5. Build the Docker image using the following command: **docker build -t [imagename]**.
6. Start a new container based on the imagename image, and open a command shell within the container: <br />  **docker run -it [imagename] /bin/bash**
7. Train the model with: **python train.py** <br /> 
   Launch the gradio application with: **python app.py**
