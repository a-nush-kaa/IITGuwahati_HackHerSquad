# IITGuwahati_HackHerSquad
Instagram Post Recommendations based on User Interactions
This script generates personalized recommendations for Instagram posts based on user interactions, leveraging latent space data and associated comments. Recommendations are computed using cluster-based cosine similarity to suggest similar posts that users may find relevant.

Dependencies
Ensure you have the following Python libraries installed:

pandas
numpy
scikit-learn
You can install these dependencies using pip:

bash
Copy code
pip install pandas numpy scikit-learn
Setup Instructions
Dataset Preparation:

Place your latent space data in HDF5 format (latent_spaces.h5) and comments data in CSV format (instagram_posts_comments_freakinsindia_20240715_022736.csv) in the data/ directory.
File Structure:

The directory structure should look like this:
kotlin
Copy code
project/
├── script_name.py
├── data/
│   ├── latent_spaces.h5
│   ├── instagram_posts_comments_freakinsindia_20240715_022736.csv
│   └── ... (other data files)
├── models/
│   └── ... (if any model files are used)
└── README.md
Execution Instructions
Edit Script Parameters:

Open script_name.py and modify the following variables under main() according to your needs:
python
Copy code
USER_INTERACTION_IDS = [1, 2, 3]  # Replace with actual user interaction post IDs
PROFILE_USERNAME = 'freakinsindia'
Ensure USER_INTERACTION_IDS contains actual IDs of user-interacted posts.
Run the Script:

Execute the script using Python:
bash
Copy code
python script_name.py
Output
Upon running the script:

Recommendations File:

A CSV file named recommendations_profile_username_timestamp.csv will be saved in the data/ directory.
This file contains recommended posts along with associated comments for the specified PROFILE_USERNAME.
Example Usage:

You can load and utilize the recommendations in other applications or analysis pipelines.

