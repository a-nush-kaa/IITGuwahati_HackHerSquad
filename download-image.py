import pandas as pd
import requests
import cv2
import os 
import time
import numpy as np

def download_images(df, batch_size, delay):
    """
    Download images in a batch with a delay between batches.

    Args:
        df (pd.DataFrame): Dataframe with image URLs and corresponding IDs.
        batch_size (int): The number of images to be downloaded in a batch.
        delay (int): The delay time in seconds between batches.

    """
    # Make sure the images directory exists
    os.makedirs("images/original_images", exist_ok=True)

    # Iterate over the DataFrame rows with batch control
    for idx, row in df.iterrows():
        image_url = row["image"]
        id = row["id"]
        file_path = f"images/original_images/{id}.jpg"

        # Check if the file already exists
        if os.path.exists(file_path):
            continue

        # Try to request and save image, skip to the next one if there is an issue
        try:
            # Send a HTTP request to the URL of the image
            response = requests.get(image_url, stream=True)

            # Check if the request is successful
            if response.status_code == 200:
                # Convert bytes to numpy array
                nparr = np.frombuffer(response.content, np.uint8)
                # Decode numpy array into image
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Resize image to 256x256
                img = cv2.resize(img, (256, 256))
                
                # Save the image under the "images" directory and name it with the id
                cv2.imwrite(file_path, img)
                
                # Print download checkpoint
                print(f"Downloaded image {id}")
        
        except Exception as e:
            print(f"Error for image {id}: {str(e)}")

        # If we've reached the batch limit, sleep for a while
        if (idx + 1) % batch_size == 0:
            print(f"Batch {idx + 1} completed, sleeping for {delay} seconds.")
            time.sleep(delay)

def main():
    df = pd.read_csv(r"C:\Users\ANUSHKA NEGI\Pictures\amazon\Future-Fashion-Trends-Forecasting-main\data\instagram_posts_freakinsindia_20240715_022736.csv")

    download_images(df=df, batch_size=20, delay=10)
    
if __name__ == "__main__":
    main()
