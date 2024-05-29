import streamlit as st
import pandas as pd
import numpy as np

st. set_page_config(layout="wide")

v = st.write(""" <h2> <b style="color:red"> Enhanced Product Search using LLMs</b> </h2>""", unsafe_allow_html=True)
st.write("###")
st.write(""" <p> Hi, welcome to <b style="color:red">LLM Search</b> this free product search engine evaluating different retrieval models </p>""",unsafe_allow_html=True)
st.write("##")
query = st.text_input("Type in your search query...")
options = st.multiselect(
    "What Algorithms do you want to evaluate",
    ["BM25", "BM25+RNN", "Two-Tower"],
    ["BM25"])

if st.button("Search"):
    st.text("Here are few Recommendations..")
    st.write("#")
    names, movie_ids = (["Dummy1", "Dummy2", "Dummy3", "Dummy4", "Dummy4"],[1,2,3,4,5])
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i in range(0, 5):
        with cols[i]:
            st.write(f' <b style="color:#E50914"> {names[i]} </b>', unsafe_allow_html=True)
            # st.write("#")
            id = movie_ids[i]
            st.write("________")
