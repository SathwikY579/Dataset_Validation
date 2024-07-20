import streamlit as st
import os
import json
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_model()

max_length = 16
num_beams = 4

# Function to generate captions
def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Streamlit UI
st.title("Image Captioning and Analysis")
st.write("Upload your dataset, define your objectives, and specify the number of images to analyze.")

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
objective = st.text_area("Enter your objective:")
num_images = st.number_input("Enter the number of images to analyze:", min_value=1, step=1, value=1)

if st.button("Generate Captions and Analyze"):
    if not uploaded_files:
        st.error("Please upload image files to proceed.")
    else:
        # Generate captions for the specified number of images
        captions = {}
        image_files = uploaded_files[:num_images]

        for uploaded_file in image_files:
            image = Image.open(uploaded_file).convert("RGB")
            caption = generate_caption(image)
            captions[uploaded_file.name] = caption
            st.write(f"Image: {uploaded_file.name} - Caption: {caption}")

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

# To run this app, save it as `app.py` and use the command `streamlit run app.py`
