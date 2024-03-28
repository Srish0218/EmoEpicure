# main_app.py
import pickle
import re
import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk import PorterStemmer
from nltk.corpus import stopwords

from database import create_table, get_owner_by_username

pd.set_option('mode.use_inf_as_na', True)

ps = PorterStemmer()
st.set_option('deprecation.showPyplotGlobalUse', False)

nltk.download('stopwords')

# Load BoW model
cv_file = 'c1_BoW_Sentiment_Model.pkl'
cv = pickle.load(open(cv_file, "rb"))

# Load classifier model
classifier_file = 'c2_Classifier_Sentiment_Model'
classifier = joblib.load(classifier_file)

# Load English stopwords
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')  # Remove 'not' from stopwords


# Function to preprocess and predict sentiment
def analyze_sentiment(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)

    # Transform using BoW model
    input_transformed = cv.transform([review]).toarray()

    # Make prediction
    return classifier.predict(input_transformed)[0]

# FUNCTION TO VISUALIZE RATING DISTRIBUTION USING LINE CHART
def visualize_rating_distribution(df):
    width1 = st.slider("Adjust Chart Width:", min_value=4, max_value=12, value=8)
    height1 = st.slider("Adjust Chart Height:", min_value=4, max_value=12, value=6)
    fig, ax = plt.subplots(figsize=(width1, height1))
    sns.lineplot(x='Rating', y='Count', data=df.groupby('Rating').size().reset_index(name='Count'), ax=ax)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    st.pyplot(fig)


# Function to visualize sentiment distribution using pie charts
def visualize_sentiment_distribution(df):
    width_key = "chart_width2"
    height_key = "chart_height2"
    width2 = st.slider("Adjust Chart Width:", min_value=4, max_value=12, value=8, key=width_key)
    height2 = st.slider("Adjust Chart Height:", min_value=4, max_value=12, value=6, key=height_key)
    fig1, ax1 = plt.subplots(figsize=(width2, height2))
    positive_count = df[df['Sentiment'] == 'Positive'].shape[0]
    negative_count = df[df['Sentiment'] == 'Negative'].shape[0]
    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    explode = (0.1, 0)  # explode 1st slice
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)


# Function to visualize sentiment distribution using bar chart
def visualize_sentiment_distribution_bar(df):
    chart_width_key = "bar_chart_width"
    chart_height_key = "bar_chart_height"
    chart_width = st.slider("Adjust Chart Width:", min_value=4, max_value=12, value=8, key=chart_width_key)
    chart_height = st.slider("Adjust Chart Height:", min_value=4, max_value=12, value=6, key=chart_height_key)
    fig, ax = plt.subplots(figsize=(chart_width, chart_height))
    sns.countplot(x='Sentiment', data=df, ax=ax)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    st.pyplot(fig)
def main_app():
    st.title("EpicureGlow Sentiment Analysis 🍽️😋")
    st.markdown("---")
    # Call create_table() to ensure the table exists
    create_table()

    # Main app
    if st.session_state.user_authenticated:
        section = st.selectbox("Select Section:", ["Line Reviews", "Import Excel File"])
        if section == "Line Reviews":
            # Use st.selectbox for selecting one-line or multi-line reviews
            review_type = st.selectbox("Select Review Type:", ["One-Line Reviews", "Multi-Line Reviews"])

            # Add a textarea for user input based on the selected review type
            user_input = st.text_area(f"Enter {review_type.lower()} (each on a new line):")
            show_instructions = st.checkbox("Show Instructions for Adding Reviews")

            # Show instructions if checkbox is selected
            if show_instructions:
                st.info("1. Be specific about your experience.\n"
                        "2. Mention key details like food quality, service, and ambiance.\n"
                        "3. Avoid using excessive capitalization or punctuation.")

            # Perform sentiment analysis on user input
            if st.button("Predict Sentiments"):

                if user_input:
                    st.success("Sentiment Analysis Result:")
                    # Split reviews if multi-line tab is selected
                    reviews = user_input.split('\n') if review_type == "Multi-Line Reviews" else [user_input]

                    # Process each review
                    results = {'Review': [], 'Sentiment': []}
                    for i, review in enumerate(reviews):
                        # Preprocess input
                        original_review = review
                        sentiment = 'Positive' if analyze_sentiment(review) == 1 else 'Negative'

                        # Display result
                        st.write(f"Review {i + 1}: {original_review}")
                        st.write(
                            f"Sentiment: {sentiment} :yum:" if sentiment == 'Positive' else f"Sentiment: {sentiment} :broken_heart:")

                        # Save results for visualization
                        results['Review'].append(original_review)
                        results['Sentiment'].append(sentiment)

                    # Visualize sentiment distribution
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.expander("Sentiment Distribution"):
                            st.title("Sentiment Distribution")
                            # Assuming visualize_sentiment_distribution is a function you've defined
                            visualize_sentiment_distribution(pd.DataFrame(results))
                    with col2:
                        with st.expander("Sentiment Distribution (Bar Chart)"):
                            st.title("Sentiment Distribution (Bar Chart)")
                            visualize_sentiment_distribution_bar(pd.DataFrame(results))
                else:
                    st.warning("Please enter a review.")
        elif section == "Import Excel File":
            st.subheader("Import Excel File")
            excel_file = st.file_uploader("Choose a file", type=["xlsx", "xls"])
            show_instructions = st.checkbox("Show Instructions for Adding Reviews using Excel Import")

            # Show instructions if checkbox is selected
            if show_instructions:
                st.info(
                    "Adding Reviews using excel import \n1. Select 'Import Excel File' in the app.\n2. Upload an Excel file "
                    "containing the 'Review' and 'Rating' columns.\n3. Ensure the Excel file structure has two columns named "
                    "'Review' and 'Rating'.\n4. The app will process the reviews and provide sentiment predictions.")

            # Process the loaded Excel file
            if excel_file is not None:
                try:
                    # Read the Excel file
                    df = pd.read_excel(excel_file)

                    # Display the loaded data
                    st.write("Loaded Data:")
                    st.dataframe(df)

                    # Process reviews from the loaded data
                    st.success("Sentiment Analysis Result:")
                    results = {'Review': [], 'Sentiment': []}
                    for i, row in df.iterrows():
                        review = row['Review'] if 'Review' in row else ''
                        rating = row['Rating'] if 'Rating' in row else 0

                        # Perform sentiment analysis based on rating
                        if rating >= 4:
                            sentiment = 'Positive'
                        else:
                            sentiment = 'Negative'

                        # Display result
                        st.write(f"Review {i + 1}: {review}")
                        st.write(
                            f"Sentiment: {sentiment} :yum: | Rating: {rating}/5" if sentiment == 'Positive' else f"Sentiment: {sentiment} :broken_heart: | Rating: {rating}/5")

                        # Save results for visualization
                        results['Review'].append(review)
                        results['Sentiment'].append(sentiment)

                    # Create three columns for visualizations
                    col1, col2, col3 = st.columns(3)

                    # Visualize rating distribution in the first column
                    with col1:
                        with st.expander("Rating Distribution"):
                            visualize_rating_distribution(df)
                    with col2:
                        with st.expander("Sentiment Distribution"):
                            visualize_sentiment_distribution(pd.DataFrame(results))
                    with col3:
                        with st.expander("Sentiment Distribution (Bar Chart)"):
                            visualize_sentiment_distribution_bar(pd.DataFrame(results))

                except Exception as e:
                    st.error(f"Error: {e}")

# Add your main app logic and functions here

if __name__ == "__main__":
    main_app()
