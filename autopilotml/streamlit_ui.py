import streamlit as st
import sys
import os

# Add the root directory of your project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autopilotml import load_data


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

def main():
    # set the logo
    st.image("images/autopilotml_1.jpeg", use_column_width=True)

    data = st.file_uploader("Upload a file", type=["csv"])
    if data:
        df = load_data(path = data , csv=True, header=0)
        st.dataframe(df.head())


if __name__ == "__main__":
    main()






