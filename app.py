import streamlit as st
import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

model = keras.models.load_model("FinalModel.keras")
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

stop_words = set(stopwords.words('english'))
custom_remove = ["''", '``', "rt", "https", "’", "“", "”", "\u200b", "--", "n't", "'s", "...", "//t.c"]

def clean_comment(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    clean_words = [word for word in words if word not in stop_words and word not in custom_remove]
    return " ".join(clean_words)

def embed_text(texts):
    embeddings = []
    for text in texts:
        emb = use([text])
        emb_np = tf.reshape(emb, [-1]).numpy()
        embeddings.append(emb_np)
    return np.array(embeddings)

def predict_bullying(comment_text):
    cleaned_comment = clean_comment(comment_text)

    embedded_comment = embed_text([cleaned_comment])
    predictions = model.predict(embedded_comment, verbose=0) 
    
    predicted_class_index = np.argmax(predictions)
    
    if predicted_class_index == 0:
        label = "nonbullying"
    else:
        label = "bullying"
        
    return label, predictions[0]

#Streamlit UI
st.set_page_config(page_title="YouTube Comment Bullying Detector", layout="wide")

st.title("YouTube Comment Bullying Detector")
st.markdown("Enter a YouTube video link to analyze comments for bullying.")

video_url = st.text_input("YouTube Video URL", "https://www.youtube.com/watch?v=dQw4w9WgXcQ") # Example URL

if st.button("Analyze Comments"):
    if video_url:
        st.info("Fetching and analyzing comments. This may take a moment...")
    
        results = []
        downloader = YoutubeCommentDownloader()
    
        comments_generator = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_RECENT)
    
        fetched_count = 0
        comments_list = []
    
        for comment in comments_generator:
            if 'text' in comment:
                comments_list.append(comment['text'])
                fetched_count += 1
            if fetched_count >= 10:
                break
    
        # Process and display comments
        if comments_list:
            for i, comment_text in enumerate(comments_list):
                predicted_label, probabilities = predict_bullying(comment_text)
            
                results.append({
                    "Comment": comment_text,
                    "Predicted Label": predicted_label,
                    "Prob_Nonbullying": f"{probabilities[0]:.4f}",
                    "Prob_Bullying": f"{probabilities[1]:.4f}"
                })
        
            df_results = pd.DataFrame(results)
            st.subheader("Analysis Results (Top 10 Recent Comments)")
            st.dataframe(df_results, hide_index=True)
        else:
            st.warning("No comments found or able to be analyzed for this video.")
    else:
        st.warning("Please provide a link of the YouTube video you want to analyze.")