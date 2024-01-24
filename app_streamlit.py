# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:15:28 2024

@author: Baptiste
"""

import streamlit as st
import pandas as pd
from keras.models import load_model
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_fr = set(stopwords.words('french'))
from nltk.tokenize import word_tokenize
from numpy import mean

from numpy import dot
from numpy.linalg import norm
from transformers import pipeline

from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

df = pd.read_csv("nlp_assurance_v2.csv")



st.set_page_config(
    page_title="Projet 2 NLP ",
    page_icon="💻",
)

st.title("Projet 2 NLP")
app_en_cour = st.sidebar.radio("Page selector",['Main page',"Prediction",'Summary','Explanation','Information retrievial','Chat bot'])

html_content = """
<style>
    .sidebar {
        background-color: #f4f4f4;
        padding: 20px;
        height: 100vh; /* Full height */
    }
    .main-content {
        padding: 20px;
    }
    h1 {
        color: #333;
    }
    p {
        color: #666;
    }
</style>

<div class="main-content">
    <p>Ce projet est réalisé par Vincent LEBOULENGER et Baptiste MIGUEL.</p>
    <p>Veuillez selectionner une application sur la gauche.</p>
</div>
"""

def semantic_search(query, docs, top=10):
    model = Word2Vec.load(r"C:\\Users\\utilisateur\\Downloads\\word2vec_reviews.model")
    flattened_docs = [word for doc in docs for word in doc if word in model.wv]
    tfidf = TfidfVectorizer(vocabulary=model.wv.key_to_index)
    tfidf.fit([" ".join(flattened_docs)])
    tfidf_scores = {word: tfidf.idf_[i] for word, i in tfidf.vocabulary_.items()}
    query_tokens = word_tokenize(query.lower())
    query_vectors = [model.wv[word] * tfidf_scores.get(word, 1) for word in query_tokens if word in model.wv]

    if not query_vectors:
        return []

    query_vector = mean(query_vectors, axis=0)
    scores = []
    for doc, doc_tokens in zip(docs, docs):
        doc_vectors = [model.wv[word] * tfidf_scores.get(word, 1) for word in doc_tokens if word in model.wv]
        if doc_vectors:
            doc_vector = mean(doc_vectors, axis=0)
            cosine_similarity = dot(query_vector, doc_vector) / (norm(query_vector) * norm(doc_vector))
            scores.append((doc, cosine_similarity))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top]


def creatoken(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in stopwords_fr ]
        tokens = [word for word in tokens if len(word) > 2]
        return tokens
    else:
        return []

df['token'] = df['Contenu du Texte'].apply(lambda x : creatoken(x))


def page_QA(css):
    st.title("Chatbot")
    question = st.text_input("Ask your question", key="question_input")
    qa_pipeline = pipeline("question-answering")
    if st.button("Send", key="send_button"):
        if question:
            sitee = {" ".join(row['token']): row['Assurance'] for _, row in df.iterrows()}
            reviews = df['token'].apply(lambda x: " ".join(x)).tolist()
            relevant_reviews = semantic_search(question, reviews, top=15)
            relevant_reviews_text = ' '.join([f"{sitee[review]}: {review}" for review, _ in relevant_reviews])
            answer = qa_pipeline({'question': question, 'context': relevant_reviews_text})
            st.session_state.conversation.append({
                'question': question,
                'answer': answer['answer']
            })
            question = ""
            st.experimental_rerun()
        else:
            st.warning("Please ask a question.")
    st.markdown(css_chatbot, unsafe_allow_html=True)
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        
    for exchange in st.session_state.conversation:
        user_question = exchange["question"]
        assistant_answer = exchange["answer"]
        
        user_bubble_style = "user-bubble"
        assistant_bubble_style = "assistant-bubble"
        
        if exchange.get("role") == "user":
            user_bubble_style += " user-label"
        else:
            assistant_bubble_style += " assistant-label"
        
        st.markdown(
            f'<div class="chat-container">'
            f'    <div class="chat-bubble {user_bubble_style}">{user_question}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        st.markdown(
            f'<div class="chat-container">'
            f'    <div class="chat-bubble {assistant_bubble_style}">{assistant_answer}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if st.button("Reset", key="reset-button"):
        st.session_state.conversation = []


css_chatbot = """
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            margin-bottom: 1rem;
        }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 8px;
            margin: 5px;
            display: inline-block;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #dcf8c6;
            align-self: flex-end;
            color: #000;
        }
        .assistant-bubble {
            background-color: #e6e6ea;
            align-self: flex-start;
            color: #000;
        }
        .label {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 2px;
            color: #7c7c7c;
        }
        .user-label {
            align-self: flex-end;
            text-align: right;
        }
        .assistant-label {
            align-self: flex-start;
            text-align: left;
        }
        .input-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
        }
        .input-field {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-right: 10px;
            outline: none;
        }
        .send-button, .reset-button {
            background-color: #25d366;
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
        }
    </style>
    """

#----------------------------------------------------------------------------------------------------------------------
def main_page():
    st.markdown(html_content, unsafe_allow_html=True)

     
def Information_retrieval():

    st.title("Information retrieval")
    user_input = st.text_input("Sur quel terme voulez-vous faire la recherche de commentaires ?")
    if st.button("Search"):
        if user_input:
            reviews = df['token'].tolist()
            search_results = semantic_search(user_input, reviews)
            st.markdown("<h2 style='font-size: 18px;'>Search Results :</h2>", unsafe_allow_html=True)
            for i, (review, similarity) in enumerate(search_results, start=1):
                st.markdown(f"Review {i} (Similarity Score : {similarity:.2f}):")
                review_html ='note :'+ str(df.iloc[i-1]["Nombre d'Étoiles Actives"])
                content_text = df.iloc[i-1]["Contenu du Texte"]  # Access 'Contenu du Texte' column
                st.markdown(review_html, unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center;">{content_text}</p><hr>', unsafe_allow_html=True)

def Prediction():
    
    st.title("Prediction")
    question = st.text_input("Ecrire une review :", key="question_input")
    if st.button("Predict"):
        if question:

            # Load the tokenizer
            with open('tokenizer.json', 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                tokenizer = tokenizer_from_json(tokenizer_data)

            # Load your trained model (ensure the path is correct)
            model = load_model('new_model.h5')

            # Prepare the input text
            sequences = tokenizer.texts_to_sequences(question)
            data_pad = pad_sequences(sequences, maxlen=100)

            # Make a prediction
            prediction = model.predict(data_pad)
            prediction = list(prediction[0])
            max_index = prediction.index(max(prediction))
            pred_final = "Avis positif" if (max_index == 0) else ("Avis mitigé" if (max_index == 1) else "Avis négatif")
            st.markdown(f'<p style="text-align: center;">{pred_final}</p><hr>', unsafe_allow_html=True)

#-----------------------------------------------------------------------------

if app_en_cour == 'Main page' : main_page()
if app_en_cour == 'Information retrievial' : Information_retrieval()
if app_en_cour == "Chat bot": page_QA(css_chatbot)
if app_en_cour == "Prediction": Prediction()


