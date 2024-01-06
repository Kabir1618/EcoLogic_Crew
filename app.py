import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from dotenv import load_dotenv
load_dotenv()
### Function to get response from the llama model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
# Load the CSV file into a DataFrame
data = pd.read_csv('datasets\AI_EarthHack_Dataset.csv', encoding='latin-1')

# Input text for similarity comparison
input_text = "I'm interested in sustainable solutions for construction consumption."

# Combine 'problem' and 'solution' columns into a single column for analysis
data['combined_text'] = data['problem'].fillna('') + " " + data['solution'].fillna('')

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the combined text data
tfidf_matrix = vectorizer.fit_transform(data['combined_text'].values.tolist() + [input_text])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# Get the most similar text indices
similar_indices = cosine_similarities.argsort()[0][::-1]

# Define a threshold for similarity
threshold = 0.5

# Retrieve similar rows based on the threshold
similar_texts = []
for index in similar_indices:
    if cosine_similarities[0][index] > threshold:
        similar_texts.append(data.iloc[index])

sim_texts = ""
# Print the similar rows
if similar_texts:
    for text in similar_texts:
        sim_texts += text
else:
    sim_texts = "N/A"


def getopenAIresponse(input_text, no_words):



    #llama Model
    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.9)

    #Prompt Template
    if sim_texts == "N/A":
        template="""
            I have the following business problem: {input_text}, I need {no_words} innovative ideas in one line to 
            make my business enviornmental friendly through "CIRCULAR ECONOMY". Along with each Idea be a devil's advocate and
            and give loopholes in the ideas.
            """   
    else:
        template="""
            I have the following business problem: {input_text}, I need {no_words} innovative ideas in one line to 
            make my business enviornmental friendly through "CIRCULAR ECONOMY". Along with each Idea be a devil's advocate and
            and give loopholes in the ideas. Some examples of ideas are:
            {sim_texts}
            """
    
    prompt = PromptTemplate(input_variables=['input_text', 'no_words'], template=template)

    #Generate the response
    response = llm(prompt.format(input_text=input_text, no_words=no_words))
    print(response)
    return response
    


st.set_page_config(
    page_title="Circular-Economy Idea Generation",
    page_icon="🌳", 
    layout="centered", 
    initial_sidebar_state="collapsed")

st.header("Make your business enviornment friendly with Circular Economy")

input_text = st.text_input("Enter the business problem you want to integrate Circular Economy in?") 

col_1, col_2 = st.columns([5,5])
with col_1:
    num_words = st.number_input("Enter the number of ideas ", min_value=1, max_value=10, value=10)


submit = st.button("Generate Ideas")

if submit:
    st.write(getopenAIresponse(input_text, num_words))