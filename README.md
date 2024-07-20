# Image Captioning and Analysis

This project is a Streamlit application that generates captions for images in a dataset, analyzes the dataset based on these captions, and compares the results with the user's objectives. The application provides features for uploading a ZIP file containing images, defining objectives, specifying the number of images to analyze, calculating semantic similarity scores, and judging whether the captions meet the objectives based on a user-defined threshold.

## Features

- Upload a ZIP file containing image files
- Define your objectives
- Specify the number of images to analyze
- Generate captions for images
- Perform word frequency analysis on the generated captions
- Compare captions with user objectives using semantic similarity
- Calculate overall similarity score
- Judge whether captions meet the objectives based on similarity threshold

## Technologies and Models Used

This application leverages state-of-the-art models from the Hugging Face Transformers library to achieve image captioning and semantic similarity analysis:

### Image Captioning

For generating captions for the images, the application uses the following models and tokenizers:

- **VisionEncoderDecoderModel**: This model, `nlpconnect/vit-gpt2-image-captioning`, is a Vision-to-Language model that uses a Vision Transformer (ViT) for encoding images and GPT-2 for generating text captions.
- **ViTFeatureExtractor**: This feature extractor processes the images into the format required by the Vision Transformer.
- **AutoTokenizer**: The tokenizer converts the generated text into token format and back, using the same model, `nlpconnect/vit-gpt2-image-captioning`.

### Semantic Similarity

For comparing the generated captions with the user's objectives, the application uses the following models and tokenizers:

- **BertTokenizer**: This tokenizer is used for converting text into tokens suitable for BERT model processing.
- **BertModel**: The BERT model, `bert-base-uncased`, is used to generate embeddings for the text, which can be compared to measure semantic similarity.

The combination of these models allows the application to generate meaningful captions for the images and validate the dataset against user-defined objectives.

## How to Run

To run this application, you need to have Streamlit installed. Save the following code as `app.py` and run it using Streamlit:
