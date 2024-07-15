
# IITGuwahati_HackHerSquad
Instagram Post Recommendations based on User Interactions
## Overview

This project aims to generates personalized recommendations for Instagram posts based on user interactions, leveraging latent space data and associated comments. Recommendations are computed using cluster-based cosine similarity to suggest similar posts that users may find relevant.

## Methodology

Our project involves several stages:

1. **Data Collection:** Data is gathered from Instagram posts using the Apify platform. The data includes the post's image, the number of likes and comments, and a sample of 20 comments per post.

2. **Image Segmentation:** The images from the posts are processed using YOLO and SAM to isolate the clothes from the background. The result is a collection of segmented images where only the clothes are visible against a black background.

3. **Feature Extraction and Dimensionality Reduction:** The segmented images are then passed through an autoencoder to extract a compact representation (latent space) of the clothing items. PCA is further applied to reduce the dimensionality of the latent space.

4. **Clustering:** The PCA-transformed latent spaces are then clustered using the K-Means algorithm. Each cluster represents a distinct fashion style.

5. **Sentiment Analysis:** The comments associated with each post are analyzed using the AdaBoost algorithm to gauge public sentiment towards the fashion styles represented in the posts.

6. **Data Visualization:** Finally, a dashboard presents statistics for each cluster, including the number of images, corresponding likes and comments, and the distribution of sentiment (positive, negative).

## Dependencies
Ensure you have the following Python libraries installed:
pandas
numpy
scikit-learn

You can install these dependencies using pip:
pip install pandas numpy scikit-learn

## Setup Instructions
Dataset Preparation:

Place your latent space data in HDF5 format (latent_spaces.h5) and comments data in CSV format (instagram_posts_comments_freakinsindia_20240715_022736.csv) in the data/ directory.


## Repository Structure

The repository contains several directories:

- **data:** This directory contains all the datasets, including the raw and cleaned versions.
- **models:** This directory contains the machine learning models' weights, excluding the large SAM model which is not included in the repo due to its size.
- **images:** This directory contains two subdirectories - "original_images" and "segmented_images" which hold the original and segmented images respectively.
- **python scripts:** These are a series of Python files responsible for different steps in the pipeline. The order of execution is as follows:
    1. `data_preprocessing.py`
    2. `download_images.py`
    3. `image_segmentation.py`
    4. `latent_space_creator.py`
    5. `latent_space_clustering.py`
    6.  'app.py'



## Open script_name.py and modify the following variables under main() according to your needs:

USER_INTERACTION_IDS = [1, 2, 3]  # Replace with actual userinteraction post IDs
PROFILE_USERNAME = 'freakinsindia'
Ensure USER_INTERACTION_IDS contains actual IDs of user-interacted posts.
Run the Script:

Execute the script using Python:
python script_name.py

Output
Upon running the script:
A CSV file named recommendations_profile_username_timestamp.csv will be saved in the data/ directory.
This file contains recommended posts along with associated comments for the specified PROFILE_USERNAME.
Example Usage:

You can load and utilize the recommendations in other applications or analysis pipelines.

## Technologies Used

- **Python:** The project is implemented in Python, a powerful and versatile programming language that is widely used in data science and machine learning.
- **YOLO & SAM:** These algorithms are used for image segmentation.
- **PCA & KMeans:** These techniques are used for dimensionality reduction and clustering.
- **AdaBoost:** This machine learning algorithm is used for sentiment analysis.

