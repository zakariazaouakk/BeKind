# 💬 YouTube Comment Bullying Detector

Here you can find the code and files for our Streamlit web application designed to detect cyberbullying in YouTube comments.

## Introduction

Our project aims to tackle online bullying by identifying negative text with a deep learning model. We built this web app as a proof-of-concept to show how to analyze YouTube comments for potential bullying.

## Project Description

Users input a YouTube video URL. The app gets recent comments, and our model classifies them as "bullying" or "nonbullying." Results (sentiment and confidence) are then shown on the website.

## Getting Started

### Prerequisites

* Python 3.8+
* All required Python packages (see `requirements.txt`)
* Internet connection

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/zakariazaouakk/BeKind.git](https://github.com/zakariazaouakk/BeKind.git)
    cd BeKind
    ```
2.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # For Windows
    ```
3.  **Install Packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK Data:**
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```
5.  **Run the Project:**
    ```bash
    streamlit run app.py
    ```

## 🚧 Limitations & Future Work

Our current work is binary (bullying vs. non-bullying). Cyberbullying is complex (sexism, racism, hate). Future work could be multi-classification.

Classifying bullying is subjective, depending on culture, age, etc. This means human-labeled data can have errors (false positives/negatives), which limits our model.

Sarcasm is hard to detect. Our model doesn't handle it, so some "bullying" flags were actually normal comments, especially if they lacked obvious vulgar words.
