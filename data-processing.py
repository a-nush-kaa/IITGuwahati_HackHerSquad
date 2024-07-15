import instaloader
import pandas as pd
import numpy as np
import emot
from datetime import datetime

def collect_instagram_posts(username, password, profile_username, num_posts=20):
    L = instaloader.Instaloader()
    
    try:
        # Login to Instagram
        L.login(username, password)
    except instaloader.exceptions.ConnectionException as e:
        print(f"Login failed: {e}")
        return pd.DataFrame()

    try:
        profile = instaloader.Profile.from_username(L.context, profile_username)
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile {profile_username} does not exist.")
        return pd.DataFrame()
    
    posts = []
    for post in profile.get_posts():
        if len(posts) >= num_posts:
            break
        try:
            post_data = {
                'id': post.shortcode,
                'type': 'Image' if not post.is_video else 'Video',
                'date': post.date_utc.strftime('%Y-%m-%d %H:%M:%S'),
                'caption': post.caption,
                'likesCount': post.likes,
                'commentsCount': post.comments,
                'latestComments': [{'text': comment.text} for comment in post.get_comments()],
                'images': [post.url],
                'is_video': post.is_video
            }
            posts.append(post_data)
        except Exception as e:
            print(f"Error processing post: {e}")

    return pd.DataFrame(posts)

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    return df 

def replace_emojis_with_text(text: str) -> str:
    emot_obj = emot.emot()
    try:
        emoji_info = emot_obj.emoji(text)
        num_emojis = len(emoji_info["value"])
        for i in range(num_emojis):
            text = text.replace(emoji_info["value"][i], emoji_info["mean"][i])
    except Exception as e:
        print(f"An error occurred while processing the text: {text}. The error is as follows: {e}")
    return text

def process_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df = df[["id", "type", "commentsCount", "likesCount", "latestComments", "images"]]
    
    new_columns = {"id": "id", "commentsCount": "n_comments", "likesCount": "n_likes", "latestComments": "comments", "images": "image"}
    df = df.rename(columns=new_columns)
    
    df = df[(df["type"] != "Video")]
    df = df[df["n_likes"] != -1.0]
    df = df[df["image"].notna()]
    df = df[df["image"].apply(len) > 0]
    df["image"] = df["image"].apply(lambda x: x[0])
    df["comments"] = df["comments"].apply(lambda x: [i["text"] for i in x if "text" in i])
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index + 1
    
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) and x else np.nan)
    df_comments = df.explode("comments")[["id", "comments"]]
    df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)
    df = df.drop("comments", axis=1)
    
    return df, df_comments

def save_data(df: pd.DataFrame, df_comments: pd.DataFrame, profile_username: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    posts_filename = f"data/instagram_posts_{profile_username}_{timestamp}.csv"
    comments_filename = f"data/instagram_posts_comments_{profile_username}_{timestamp}.csv"
    
    df.to_csv(posts_filename, index=False)
    df_comments.to_csv(comments_filename, index=False, sep=';')
    
    print(f"Posts data saved to {posts_filename}")
    print(f"Comments data saved to {comments_filename}")

def main():
    insta_username = 'me_nushki'  # replace with your Instagram username
    insta_password = 'ILOVEMYPARENTS@003'  # replace with your Instagram password
    profile_username = 'freakinsindia'  # replace with the Instagram profile username you want to scrape
    num_posts = 10
    
    df_posts = collect_instagram_posts(insta_username, insta_password, profile_username, num_posts)
    if df_posts.empty:
        print("No posts collected. Exiting.")
        return
    
    # Process and save the data with comments
    df_posts, df_comments = process_data(df_posts)
    save_data(df_posts, df_comments, profile_username)

    print("Data collection and processing completed.")

if __name__ == "__main__":
    main()
