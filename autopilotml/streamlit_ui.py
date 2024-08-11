import streamlit as st
import pandas as pd
import sys
import os

# Add the root directory of your project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autopilotml import load_data
from tokenizer.word2vec import Word2VecTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")



def main():
    # set the logo
    st.image("https://raw.githubusercontent.com/shyam1326/autopilotml/main/images/autopilotml_1.jpeg", use_column_width=True)

    

    task = st.sidebar.selectbox("Select the Task", ["Machine Learning", "Tokenizer", "AI Agents", "RAG", "Chatbot"])

    if task == "Tokenizer":
        model = st.sidebar.selectbox("Select the Tokenizer Model", ["Word2Vec", "FastText", "Glove", "Bert"])
        subtask =st.sidebar.radio(label = "Subtask", options=["Inference Similar Words", "Finetune Tokenizer"])

        if model == "Word2Vec" and subtask == "Inference Similar Words":
            @st.cache_resource
            def load_word2vec_model():
                return Word2VecTokenizer()
        
        
            st.title("Tokenizer")
            
            text = st.text_input("Enter the text")

            if text:
                tokenizer = load_word2vec_model()
                token = pd.DataFrame(tokenizer.similar(text), columns=["Word", "Similarity Score"])
                st.dataframe(token)


if __name__ == "__main__":
    main()






