Brain Tumor Detection & Analysis Platform


        Project Overview
        
This project is a comprehensive Deep Learning solution designed to assist in the early detection and classification of brain tumors from MRI scans. It bridges the gap between research and deployment by providing both a robust training pipeline and a user-friendly web application for real-time inference.

The system is capable of classifying MRI images into four distinct categories:

  Glioma
  Meningioma
  Pituitary Tumor
  No Tumor




  

        Repository Structure
        
The project is organized into two main modules:

App/: Contains the deployment code. This is a containerized web application built with Streamlit that serves the trained model to end-users. It includes features for image upload, real-time prediction, and a locator for nearby medical facilities.

Training/: Dedicated to the research phase. It includes the source code for data preprocessing, the Convolutional Neural Network (CNN) architecture definition, and the training loops using PyTorch.





        Application Deployment
        
The application is fully containerized using Docker, ensuring consistency across different environments.

1. Prerequisites

Docker installed on your machine.

Trained Model Weights: The application requires a pre-trained .pth file.

2. Setup Instructions

Before building the container, you must download the model weights and place them in the correct directory.

Download Model: Get the model.pth file from the following Google Drive link: 👉 Download Model (.pth)

Place File: Move the downloaded file into the App/ directory. It should sit alongside app.py and Dockerfile.

3. Build & Run

Open a terminal in the App/ directory and run the following commands:

Bash
# 1. Build the Docker image
docker build -t brain-tumor-app .

# 2. Run the container (mapping port 8501)
docker run -p 8501:8501 brain-tumor-app
Access the web interface at: http://localhost:8501

App Features

AI Diagnosis: Instant classification of uploaded MRI scans with confidence scores.

Interactive Map: A feature powered by Folium that locates nearby oncologists and hospitals for immediate assistance.








         Model Training & Development
         
If you wish to retrain the model or experiment with the architecture, follow these steps in the Training/ directory.

1. Dataset

The model is trained on the Brain Tumor Classification (MRI) dataset available on Kaggle.

Source: https://www.kaggle.com/datasets/akrashnoor/brain-tumor

2. Data Preparation

Download the dataset and organize it within the Training/ folder to match the following structure required by the data loader:

Plaintext
Training/
└── kaggle/
    └── input/
        └── brain-tumor/
            ├── 2 classes/
            └── 4 classes/  <-- Used for multi-class training
            
3. Technical Implementation

Framework: PyTorch

Architecture: A custom CNN featuring 3 convolutional blocks (Conv2d + ReLU + MaxPool) followed by a fully connected classifier.

Preprocessing: Images are resized to 224x224 and normalized. Data augmentation (RandomHorizontalFlip, RandomRotation) is applied during training to improve generalization.

⚠️ Disclaimer
For Research Use Only. This tool is a prototype developed for educational and research purposes. The results provided by the AI model are based on statistical probabilities and do not constitute a certified medical diagnosis. It should never replace professional medical advice or examination by a qualified doctor.
