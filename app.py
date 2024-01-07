import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from dotenv import load_dotenv
load_dotenv()
### Function to get response from the llama model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
# Load the CSV file into a DataFrame
df = pd.read_csv('datasets\AI_EarthHack_Dataset.csv', encoding='latin-1')
null_indices = df[df['solution'].isnull()].index.tolist()

problems = [problem for problem in df['problem']]
solutions = [solution for solution in df['solution']]

for each_index in null_indices:
    problems.pop(each_index)
    solutions.pop(each_index)

# Combine problems and solutions for vectorization
corpus = problems + solutions

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus
tfidf_matrix = vectorizer.fit_transform(corpus)

def find_similar_problem(new_problem):
    # Transform the new problem using the fitted vectorizer
    new_problem_vector = vectorizer.transform([new_problem])

    # Calculate cosine similarity between the new problem and each problem in the dataset
    similarities = cosine_similarity(new_problem_vector, tfidf_matrix)

    # Get the index of the most similar problem
    most_similar_index = np.argmax(similarities)

    # Return the most similar problem and its corresponding solution
    most_similar_problem = problems[most_similar_index]
    most_similar_solution = solutions[most_similar_index]
    stri = "Problem : " + most_similar_problem + " | " +"Solution : " + most_similar_solution

    return stri


def getopenAIresponse(input_text, no_words):



    #llama Model
    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.9)
    sim_texts = find_similar_problem(input_text)
    print(sim_texts)
    

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
    
    prompt = PromptTemplate(input_variables=['input_text', 'no_words', "sim_texts"], template=template)

    #Generate the response
    response = llm(prompt.format(input_text=input_text, no_words=no_words, sim_texts=sim_texts))
    print(response)
    return response
    

    


st.set_page_config(
    page_title="Circular-Economy Idea Generation",
    page_icon="🌳", 
    layout="centered", 
    initial_sidebar_state="collapsed")

st.header("Make your business enviornment friendly with Circular Economy 🌏🌳")

input_text = st.text_input("Enter the business problem you want to integrate Circular Economy in?") 

col_1, col_2 = st.columns([5,5])
with col_1:
    num_words = st.number_input("Enter the number of ideas ", min_value=1, max_value=3, value=2)


submit = st.button("Generate Ideas")

if submit:
    st.write(getopenAIresponse(input_text, num_words))