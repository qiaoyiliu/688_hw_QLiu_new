import streamlit as st

lab1 = st.Page("streamlit_app_lab1.py", title="Lab 1", icon=":material/add_circle:")
lab2 = st.Page("streamlit_app_lab2.py", title="Lab 2", icon=":material/add_circle:")
pg = st.navigation([lab1, lab2])
st.set_page_config(page_title="688Labs", page_icon=":material/edit:")
pg.run()