import streamlit as st

hw1 = st.Page("streamlit_app_hw1.py", title="HW 1", icon=":material/add_circle:")
hw2 = st.Page("streamlit_app_hw2.py", title="HW 2", icon=":material/add_circle:")
hw3 = st.Page("streamlit_app_hw3.py", title="HW 3", icon=":material/add_circle:")

pg = st.navigation([hw1, hw2, hw3])
st.set_page_config(page_title="688HW", page_icon=":material/edit:")
pg.run()
