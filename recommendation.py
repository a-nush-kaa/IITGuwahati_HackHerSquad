import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

def load_data(latent_space_file, comments_file):
    latent_df = pd.read_hdf(latent_space_file, key='df_items')
    comments_df = pd.read_csv(comments_file, sep=';')
    return latent_df, comments_df

def get_cluster_recommendations(latent_df, user_interaction_ids, top_n=5):
    user_interactions = latent_df[latent_df['path'].isin(user_interaction_ids)]
    if user_interactions.empty:
        print("No user interactions found in the latent space data.")
        return pd.DataFrame()

    # Get the cluster IDs of user interacted items
    clusters = user_interactions['cluster'].unique()

    # Get items in the same clusters as user interactions
    cluster_items = latent_df[latent_df['cluster'].isin(clusters)]

    # Compute cosine similarity between user interactions and cluster items
    user_latent_spaces = np.stack(user_interactions['latent_space'].apply(np.array))
    cluster_latent_spaces = np.stack(cluster_items['latent_space'].apply(np.array))
    
    similarity_matrix = cosine_similarity(user_latent_spaces, cluster_latent_spaces)

    # Get top N similar items for each user interaction
    recommendations = []
    for user_idx in range(similarity_matrix.shape[0]):
        similarity_scores = similarity_matrix[user_idx]
        similar_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommended_items = cluster_items.iloc[similar_indices]
        recommendations.append(recommended_items)

    # Combine all recommendations and remove duplicates
    recommendations_df = pd.concat(recommendations).drop_duplicates(subset='path')
    
    return recommendations_df

def merge_with_comments(recommendations_df, comments_df):
    comments_df['comments'] = comments_df['comments'].fillna('')
    merged_df = recommendations_df.merge(comments_df, left_on='path', right_on='id', how='left')
    return merged_df

def save_recommendations(recommendations_df, profile_username):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recommendations_filename = f"data/recommendations_{profile_username}_{timestamp}.csv"
    recommendations_df.to_csv(recommendations_filename, index=False)
    print(f"Recommendations saved to {recommendations_filename}")

def main():
    LATENT_SPACE_FILE = 'data/latent_spaces.h5'
    COMMENTS_FILE = 'data/instagram_posts_comments_freakinsindia_20240715_022736.csv'
    USER_INTERACTION_IDS = [1, 2, 3]  # Replace with actual user interaction post IDs
    PROFILE_USERNAME = 'freakinsindia'
    
    latent_df, comments_df = load_data(LATENT_SPACE_FILE, COMMENTS_FILE)
    latent_df['latent_space'] = latent_df['latent_space'].apply(tuple)  # Convert lists to tuples
    recommendations_df = get_cluster_recommendations(latent_df, USER_INTERACTION_IDS)
    if recommendations_df.empty:
        print("No recommendations generated.")
        return
    
    recommendations_df = merge_with_comments(recommendations_df, comments_df)
    save_recommendations(recommendations_df, PROFILE_USERNAME)

    print("Recommendation process completed.")

if __name__ == "__main__":
    main()
