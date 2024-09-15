import os
import pandas as pd
import pytesseract
from PIL import Image
import requests
from io import BytesIO
from src.utils import download_images

# Load the dataset
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Function to download image from URL
def download_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

# Function to perform OCR on image and extract text
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Process the training data
def process_train_data(train_df):
    for index, row in train_df.iterrows():
        img_url = row['image_link']
        img = download_image(img_url)

        if img:
            text = extract_text_from_image(img)
            print(f"OCR text for {row['index']}: {text}")

            # You can now parse the OCR output to extract entity values such as weight, volume, etc.
            # Extract key information based on entity_name
            if row['entity_name'] == 'item_weight':
                # Example of entity extraction logic for item weight
                weight = extract_weight_from_text(text)
                train_df.at[index, 'extracted_value'] = weight

    return train_df

# Example of extracting weight from text
def extract_weight_from_text(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in ['gram', 'kg', 'kilogram', 'ounce', 'lb', 'pound']:
            try:
                return f"{float(words[i-1])} {word}"
            except ValueError:
                continue
    return ""

# Process the test data
def process_test_data(test_df):
    predictions = []
    for index, row in test_df.iterrows():
        img_url = row['image_link']
        img = download_image(img_url)

        if img:
            text = extract_text_from_image(img)
            print(f"OCR text for {row['index']}: {text}")

            # Example of prediction logic: extract entity from text
            prediction = extract_weight_from_text(text) if row['entity_name'] == 'item_weight' else ""
            predictions.append({'index': row['index'], 'prediction': prediction})

    return pd.DataFrame(predictions)

# Run the pipeline
train_df_processed = process_train_data(train_df)
test_predictions = process_test_data(test_df)

# Save the test predictions to CSV
test_predictions.to_csv('test_predictions.csv', index=False)

# Run sanity check on the output
os.system('python src/sanity.py --input_file test_predictions.csv')
