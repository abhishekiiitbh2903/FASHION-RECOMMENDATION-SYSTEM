# Fashion Recommender System

## Overview
This project implements a Fashion Recommender System using TensorFlow and Keras. 
The system utilizes a pre-trained ResNet50 model for feature extraction and employs transfer learning to adapt it for fashion recommendation tasks.

## Dataset
The dataset comprises over 44,000 fashion images. Each image is scaled to a size of 224x224 pixels with RGB channels.
The images are preprocessed and fed into the ResNet50 model to generate a feature vector of size 2048 for each image. 

## Model Architecture
The ResNet50 model is pre-trained on the ImageNet competition dataset, which enables it to capture rich features from images.
By utilizing transfer learning, the pre-trained model is fine-tuned for fashion recommendation, leveraging the knowledge gained from ImageNet.

## Recommendation Algorithm
The recommendation algorithm works by generating a 2048-dimensional feature vector for a query image and finding the 5 closest vectors in the feature space of the dataset.
AnnoyIndex is used to efficiently search for the nearest neighbors based on the euclidean distance of feature vectors.

## Repository Contents
- `app.py`: Main Flask application file responsible for running the recommendation system.
- `requirements.txt`: Contains the required Python packages to run the application.
- `README.md`: Provides information about the project, its setup, and usage.
- `embeddings.pkl`: Pickle file storing the 2048-dimensional feature vectors for each image.
- `filenames.pkl`: Pickle file storing the filenames or paths of the images.

## Setup
To run the app on your local machine:
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run `app.py` to generate the required pickle files and start the Flask application.

## Mathematics
1. To understand the mathematical concepts underlying the model building process, please refer to the detailed explanation [here](link/to/mathematics_file.md).

## Author
This project is developed by Abhishek Singh. For any queries or feedback, feel free to contact me.
