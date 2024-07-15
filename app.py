import pandas as pd
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Load and clean recommendation data
recommendation_df = pd.read_csv(r'data\recommendations_freakinsindia_20240715_202953.csv')
image_url_df = pd.read_csv(r'data\image_url.csv')

recommendation_df.drop_duplicates(subset='id', inplace=True)

# Load and clean image_url data
image_url_df = pd.read_csv('data\image_url.csv')
image_url_df.drop_duplicates(subset='id', inplace=True)

# Merge recommendation and image_url data
merged_df = pd.merge(recommendation_df, image_url_df, on='id', how='left')

# Fill missing image URLs
merged_df['image_url'].fillna('default_image_url.jpg', inplace=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        recommendations_data = merged_df.to_dict(orient='records')

        # Debug print for recommendations_data
        print(recommendations_data)

        return jsonify(recommendations_data)

    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
