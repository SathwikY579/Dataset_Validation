import streamlit as st
import os
import json
import zipfile
from io import BytesIO
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BertTokenizer, BertModel
import torch
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Clear cache
@st.cache_data
def clear_cache():
    if os.path.exists("uploaded_images"):
        for file in os.listdir("uploaded_images"):
            file_path = os.path.join("uploaded_images", file)
            os.remove(file_path)

# Load pre-trained model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model():
    # Load image captioning model and tokenizer
    image_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Load BERT model for semantic similarity
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_model = BertModel.from_pretrained("bert-base-uncased")
    
    return image_model, feature_extractor, image_tokenizer, text_tokenizer, text_model

image_model, feature_extractor, image_tokenizer, text_tokenizer, text_model = load_model()

max_length = 16
num_beams = 4

# Function to generate captions
def generate_caption(image):
    try:
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        output_ids = image_model.generate(pixel_values, max_length=max_length, num_beams=num_beams, early_stopping=True)
        caption = image_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None

# Function to compute sentence embeddings using BERT
def get_embedding(sentence, tokenizer, model):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().flatten()
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

# Streamlit UI
st.title("Image Captioning and Analysis")
st.write("Upload your dataset as a ZIP file, define your objectives, and specify the number of images to analyze.")

# Clear cache and remove previously extracted files
clear_cache()

# File uploader for ZIP file
uploaded_file = st.file_uploader("Upload a ZIP file containing image files", type=["zip"])
objective = st.text_area("Enter your objective:")
num_images = st.number_input("Enter the number of images to analyze:", min_value=1, step=1, value=1)

# Similarity threshold
similarity_threshold = st.slider("Set similarity threshold (0.0 to 1.0):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Generate Captions and Analyze"):
    if not uploaded_file:
        st.error("Please upload a ZIP file to proceed.")
    else:
        # Extract images from ZIP file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall("uploaded_images")

        # Debug: List extracted files
        st.write("Extracted files:")
        extracted_files = os.listdir("uploaded_images")
        st.write(extracted_files)

        # Generate captions for the specified number of images
        captions = {}
        image_files = [f for f in extracted_files if f.endswith((".jpg", ".jpeg", ".png"))]
        image_files = image_files[:num_images]

        for image_name in image_files:
            image_path = os.path.join("uploaded_images", image_name)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image {image_name}: {str(e)}")
                continue

            # Debug: Display image
            st.image(image, caption=f"Image: {image_name}", use_column_width=True)
            
            caption = generate_caption(image)
            if caption:
                captions[image_name] = caption
                st.write(f"Image: {image_name} - Caption: {caption}")
            else:
                st.error(f"Failed to generate caption for image: {image_name}")

        # Save captions to JSON
        with open("captions.json", "w") as f:
            json.dump(captions, f)

        # Text analysis
        all_captions = " ".join(captions.values())
        word_counts = Counter(all_captions.split())

        # Display word frequency analysis
        common_words = word_counts.most_common(10)
        st.write("Most common words in captions:")
        for word, count in common_words:
            st.write(f"{word}: {count}")

        # Plot word frequency
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.title("Word Frequency in Captions")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        st.pyplot(plt)

        # Comparison with user objectives
        objective_words = objective.split()
        match_count = sum(word_counts[word] for word in objective_words if word in word_counts)
        st.write(f"Number of objective words found in captions: {match_count}")
        st.write("Objective words found in captions:")
        for word in objective_words:
            if word in word_counts:
                st.write(f"{word}: {word_counts[word]}")
            else:
                st.write(f"{word}: Not found")

        # Semantic similarity analysis
        objective_embedding = get_embedding(objective, text_tokenizer, text_model)
        if objective_embedding is not None:
            caption_embeddings = [get_embedding(caption, text_tokenizer, text_model) for caption in captions.values()]

            # Calculate cosine similarity
            similarities = [cosine_similarity([objective_embedding], [embedding])[0][0] for embedding in caption_embeddings if embedding is not None]

            st.write("Semantic similarity of captions with the objective:")
            for image_name, similarity in zip(captions.keys(), similarities):
                st.write(f"Image: {image_name} - Similarity: {similarity:.4f}")

            # Overall similarity score
            average_similarity = np.mean(similarities)
            st.write(f"Overall Similarity Score: {average_similarity:.4f}")

            # Judging satisfaction
            if average_similarity >= similarity_threshold:
                st.success("The captions align well with the objective!")
            else:
                st.error("The captions do not sufficiently meet the objective.")
        else:
            st.error("Failed to compute objective embedding. Cannot proceed with semantic similarity analysis.")

# To run this app, save it as `app.py` and use the command `streamlit run app.py`
