import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(filepath):
    try:
        df = pd.read_csv(filepath, sep=';')
    except Exception as e:
        print(f"Error loading CSV file '{filepath}': {e}")
        return None
    return df


def clean_data(df):
    df = df.drop_duplicates(subset='comments', keep='first')
    if '608' in df.columns:
        df = df.drop(columns=['608'], axis=1)
    df['comments'] = df['comments'].fillna('')  # Fill NaN values with an empty string
    return df


def tokenize(text):
    if text is None:
        return []
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
    return tokens


def process_comments(df):
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])

    X_transformed = pipeline.fit_transform(df['comments'])
    return X_transformed, pipeline


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)


def transform_new_comments(comments, model):
    X_transformed = model.transform(comments)
    return X_transformed


def main():
    df = load_data(r'data\instagram_posts_comments_freakinsindia_20240715_022736.csv')
    if df is None:
        return
    
    df = clean_data(df)
    X_transformed, pipeline = process_comments(df)
    save_model(pipeline, 'models/comment_processing_pipeline.joblib')

    # Save transformed data to a CSV for inspection
    df_transformed = pd.DataFrame(X_transformed.toarray(), columns=pipeline.named_steps['vect'].get_feature_names_out())
    df_transformed.to_csv('data/transformed_comments.csv', index=False)

    # Example: Loading and using the model on new comments
    loaded_pipeline = load_model('models/comment_processing_pipeline.joblib')
    new_comments = ["This is a new comment.", "Another example of a comment."]
    transformed_new_comments = transform_new_comments(new_comments, loaded_pipeline)
    
    # Convert the transformed new comments to a DataFrame for a readable output
    df_new_comments_transformed = pd.DataFrame(transformed_new_comments.toarray(), columns=loaded_pipeline.named_steps['vect'].get_feature_names_out())
    print(df_new_comments_transformed)


if __name__ == "__main__":
    main()
