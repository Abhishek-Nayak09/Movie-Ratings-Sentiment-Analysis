import streamlit as st
import string
import pickle

tfidf_vectorizer = pickle.load(open('TfidfVectorizer.pkl', 'rb'))
model4 = pickle.load(open('text_classification.pkl', 'rb'))

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def preprocess_text(text):
    text = remove_punct(text)
    text = text.lower()
    return text

def map_to_star_rating(probability):
    # Map probability to a star rating between 1 and 5
    return round(1 + 4 * probability)

def stars_rating_html(stars):
    star_icon = "&#9733;"  # Star unicode character
    stars_html = f'<span style="font-size: 36px;">{star_icon * stars}</span>'
    return stars_html

def main():
    st.title('Movie Ratings Sentiment Analysis')
    st.subheader("Problem Statement:")
    st.write("Develop a sentiment analysis model for movie ratings that can accurately predict whether a"
             " given review is 1 to 5 stars. The model should take a text review of a movie as input and classify it into 1 to 5 stars, based on the sentiment expressed in the review.")
    st.write("By addressing these problem statements and objectives, you can develop a comprehensive"
             " sentiment analysis model for movie ratings that accurately predicts the sentiment of given"
             " reviews, ultimately enhancing the user experience and decision-making in the movie industry.")

    # Create a container for the layout
    main_container = st.container()
    
    # Add columns to the container for image and text
    col1, col2 = main_container.columns([2, 3])

    # Add image to the left column
    movie_image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2DUJF7t_ScHZn-03NRxYQG2rob_t-vjTjA0i9hni2dL-oMlLKv0HX9Ujx0FZmWkKaINU&usqp=CAU" # Replace with the actual path to your image
    col1.image(movie_image, use_column_width=True)



    # Add text box to the right column
    with col2:
        input_text = st.text_area("Enter Movie reviews", height=245)
        if st.button('Predict'):
            if input_text:
                preprocessed_text = preprocess_text(input_text)
                tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
            
            # Get probability of positive sentiment
                prediction_prob_positive = model4.predict_proba(tfidf_vector)[0][1]
            
            # Map probability to star rating
                star_rating = map_to_star_rating(prediction_prob_positive)
            
                st.write("Predicted Movie Ratings:", stars_rating_html(star_rating), unsafe_allow_html=True)

    st.subheader("Disclaimer:")
    st.write("The sentiment analysis model provided here, based on a Naive Bayes algorithm, "
            "is designed to predict the sentiment (positive/negative) of movie reviews. However, "
            "it is important to note the following:")
    st.write("1. Predictive Nature: The model's predictions are based on patterns learned from training     data and may not always accurately reflect the sentiment of a given review.")        
    st.write("2. Inherent Uncertainty: Sentiment analysis is inherently uncertain due to the complexity"
             "of human language and the subjectivity of sentiment interpretation.")
    st.write("3. Limited Understanding: The model may not capture subtle nuances, emotions, or "
            "context that can significantly impact the sentiment of a review.")
    st.write("4. User Interpretation: Users should interpret the model's predictions with caution and consider them as one of several factors when forming opinions or making decisions.")
    st.write("5. No Guarantee of Accuracy: While efforts have been made to develop an accurate sentiment analysis model, there is no guarantee that the predictions will always be correct.")
    st.write("6. Continual Improvement: The model and its accuracy can be further improved over time with additional data, fine-tuning, and advancements in sentiment analysis techniques.")
    st.write("Users are encouraged to use the predictions provided by the model as a reference or starting point, but personal judgment and critical thinking should always be applied when assessing the sentiment of movie reviews. The creators of the model do not assume liability for any decisions or actions taken based on its predictions.")
if __name__ == '__main__':
    main()
