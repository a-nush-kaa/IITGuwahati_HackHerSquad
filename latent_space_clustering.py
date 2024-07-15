from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def inspect_hdf5(file_path):
    """
    Function to read and display the contents of the HDF5 file.
    """
    df = pd.read_hdf(file_path, key='df_items')
    print("Contents of the HDF5 file:")
    print(df.head())  # Display the first few rows for brevity

    # If you want to save it to a CSV for easier inspection
    csv_path = file_path.replace('.h5', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nThe contents have also been saved to {csv_path}")

def main():
    """
    Main function to perform clustering on latent spaces.
    """
    FILE_PATH = 'data/latent_spaces.h5'
    
    # Inspect the HDF5 file before clustering
    inspect_hdf5(FILE_PATH)
    
    df = pd.read_hdf(FILE_PATH, key='df_items')
    
    # Ensure latent spaces are properly formatted as numpy arrays
    latent_space = np.stack(df['latent_space'].apply(np.array))
    
    n_samples = latent_space.shape[0]
    NUMBER_OF_CLUSTERS = min(10, n_samples)  # Ensure NUMBER_OF_CLUSTERS <= n_samples
    
    if n_samples < 2:
        print("Not enough samples for clustering.")
        return
    
    # Creating a KMeans model with the chosen number of clusters
    kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=0, n_init=10)
    
    # Fitting the model to your data
    kmeans.fit(latent_space)
    
    # Getting the cluster assignments for each image
    df['cluster'] = kmeans.labels_
    
    # Saving the updated data back to the HDF5 file
    df.to_hdf(FILE_PATH, key='df_items', mode='w')
    
    print(f"Cluster assignments have been saved to {FILE_PATH}")

if __name__ == "__main__":
    main()
