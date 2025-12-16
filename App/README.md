1. Open a Terminal

Navigate to the project folder where the Dockerfile is located.

2. Download the model file (.pth)

https://drive.google.com/file/d/1ViorieU5F3Jmr1SoYkhNy1qIJZyRaGFF

2. Build the Docker Image

docker build -t brain-tumor-app .

3. Run the Container

docker run -p 8501:8501 brain-tumor-app