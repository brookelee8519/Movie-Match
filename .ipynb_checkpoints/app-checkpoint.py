import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Match")

@st.cache_data
def load_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    credits.columns = ['movie_id', 'title', 'cast', 'crew']

    df = movies.merge(credits, left_on='id', right_on='movie_id')
    df = df.drop(columns=['title_y', 'movie_id'])
    df = df.rename(columns={'title_x': 'title'})

    def get_list(x):
        L = ast.literal_eval(x)
        return [i['name'] for i in L]

    df['genres_list'] = df['genres'].apply(get_list)
    df['keywords_list'] = df['keywords'].apply(get_list)
    df['cast_list'] = df['cast'].apply(get_list)
    df['cast_list'] = df['cast_list'].apply(lambda x: x[:3])

    def get_director(x):
        crew = ast.literal_eval(x)
        for person in crew:
            if person['job'] == 'Director':
                return person['name']
        return ''

    df['director'] = df['crew'].apply(get_director)

    def clean_names(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        return ''

    for col in ['genres_list', 'keywords_list', 'cast_list']:
        df[col] = df[col].apply(clean_names)

    df['director_clean'] = df['director'].apply(lambda x: str.lower(x.replace(" ","")) if isinstance(x, str) else '')
    df['tags'] = df['genres_list'] + df['keywords_list'] + df['cast_list']
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x)) + ' ' + df['director_clean']

    return df

df = load_data()

tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(df['tags'])
sim = cosine_similarity(matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, n=5):
    idx = indices[title]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]
    movies = []
    for i, score in scores:
        movies.append(df.iloc[i]['title'])
    return movies

st.title("Movie Match")
st.write("pick a movie and get some similar recs")

st.divider()

movie_list = df['title'].sort_values().tolist()
choice = st.selectbox("Movie:", movie_list)

if st.button("get recs"):
    results = recommend(choice)
    st.write("movies like " + choice + ":")
    cols = st.columns(len(results))
    for col, m in zip(cols, results):
        col.write(m)
