# EpicureGlow Sentiment Analysis

EpicureGlow Sentiment Analysis is a Streamlit-based web application designed for restaurant owners to analyze and visualize customer sentiments based on reviews. The application employs Natural Language Processing (NLP) techniques to determine sentiments and provides insightful visualizations.

## Overview

This project comprises a Streamlit-based web application, consisting of authentication functionality for restaurant owners, sentiment analysis, and visualization of customer reviews. The sentiment analysis is powered by a Bag of Words model and a classifier trained on customer reviews.

## Features

- User authentication for restaurant owners.
- Sentiment analysis of customer reviews.
- Visualizations of sentiment distribution and rating trends.
- Multi-line and Excel file review import options.

## Installation

To run the EpicureGlow Sentiment Analysis application, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/Srish0218/EpicureGlowSentiment.git
cd DineVibe-Sentiment-Analysis
```

### 2. Create and Activate Virtual Environment

```bash
python -m myvenv venv
source venv/bin/activate  # On Windows, use: myvenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run auth.py
```

## How to Add Reviews and Tips

### Single-Line/Multi-Line Reviews

1. Navigate to the "Line Reviews" section.
2. Choose the review type (single-line or multi-line) from the dropdown.
3. Enter your reviews in the text area, with each review on a new line.
4. Optionally, show instructions for adding reviews.
5. Click the "Predict Sentiments" button to perform sentiment analysis.

### Excel File Import

1. Navigate to the "Import Excel File" section.
2. Upload an Excel file containing "Review" and "Rating" columns.
3. Ensure the Excel file structure has two columns named "Review" and "Rating."
4. The app will process the reviews and provide sentiment predictions.

## Usage

1. **Authentication**: The app begins with a login/signup page for restaurant owners. If you're a new user, sign up with your username, password, and restaurant name.

2. **Line Reviews**: Analyze sentiments of one-line or multi-line customer reviews. Enter reviews in the provided text area and click the "Predict Sentiments" button.

3. **Import Excel File**: Alternatively, import an Excel file with customer reviews and ratings. The app processes the data and provides sentiment predictions.

4. **Visualizations**: Explore visualizations such as rating distribution, sentiment distribution, and sentiment distribution in bar chart format.

## File Structure

The project structure is organized as follows:

```plaintext
DineVibe-Sentiment-Analysis/
│
├── auth.py
├── Book1.xlsx (Sample Excel file for importing reviews)
├── c1_BoW_Sentiment_Model.pkl (Bag of Words Sentiment Model)
├── c2_Classifier_Sentiment_Model (Classifier Sentiment Model)
├── database.py
├── login.py
├── main_app.py
├── restaurant_owners.db (SQLite database file)
├── requirements.tx
└── signup.py
```

## Visualization

### Rating Distribution

- Navigate to the "Import Excel File" section.
- Use the sliders to adjust chart width and height.
- The line chart displays the distribution of ratings.

### Sentiment Distribution

- Explore sentiment distribution in two different formats:
  - **Pie Chart**: Navigate to the "Import Excel File" section and adjust chart width and height sliders.
  - **Bar Chart**: Navigate to the "Line Reviews" or "Import Excel File" sections and adjust chart width and height sliders.

## Demo

For a live demo, visit the [EpicureGlow Sentiment Analysis](https://srish-EpicureGlowSentiment.streamlit.app/) website.

---

Feel free to contribute to this project or report any issues by opening an [issue](https://github.com/Srish0218/EpicureGlowSentiment/issues) on GitHub. We welcome your feedback and contributions!

## License

This project is licensed under the [MIT License](LICENSE.md).

**Enjoy analyzing the sentiments of your restaurant's reviews with DineVibe!** 🍽️😋# EmoEpicure
