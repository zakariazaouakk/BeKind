## Introduction

As someone who loves watching Twitch streamers and YouTubers, I often go through comments to see what people think. However, I usually struggle to identify its actual intent, sometimes due to sarcasm or cultural differences.

## Project Description

This web application lets users input a YouTube video URL. It then fetches and extracts the recent comments directly from that video. These extracted comments are then processed by our deep learning model, which classifies them as "bullying" or "nonbullying".

## Getting Started

### Prerequisites

* Python 3.10
* All required Python packages (see `requirements.txt`)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/zakariazaouakk/BeKind.git](https://github.com/zakariazaouakk/BeKind.git)
    cd BeKind
    ```
2.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  
    ```
3.  **Install Packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Project:**
    ```bash
    streamlit run app.py
    ```

## Notes

Classifying bullying is subjective, depending on culture, age, etc. This means human labeled data can have errors which limits the model.

Sarcasm is hard to detect. The model doesn't handle it, so some "bullying" flags were actually normal comments, especially if they lacked obvious vulgar words.
