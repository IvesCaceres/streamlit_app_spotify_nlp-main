from pickle import load
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


model = load(open('nn_6_auto_cosine.model', 'rb'))
df = pd.read_excel('../src/datos_merged_1986_2023.xlsx')
df['year_s'] = df['year'].astype(str)
df['duration_ms_s'] = df['duration_ms'].astype(str)
df['popularity_s'] = df['popularity'].astype(str)

df['tags'] = df['track_name'] + " " + df['popularity_s']+ " "+ df['duration_ms_s'] \
    + " " + df['artist_genres']+ " " + df['year_s']
df['tags'] = df['tags'].apply(lambda x: str(x).replace(";"," "))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['tags'])

def lista_canciones(cancion):
    indice_cancion = df[df['track_name'] == cancion].index[0]
    distancia, indices = model.kneighbors(tfidf_matrix[indice_cancion])
    canciones_similares = [(df['track_name'][i], distancia[0][j]) for j, i in enumerate(indices[0])]
    return canciones_similares[1:]

def str_canciones_recomendadas(cancion_input):
    recomendaciones = lista_canciones(cancion_input)
    resultado = "Recomendaciones para " + cancion_input + "<br />"
    for i, (cancion, distancia) in enumerate(recomendaciones, start=1):
        resultado += f"{i}. Canción: {cancion}<br />"
    return resultado



# Crear la interfaz de Streamlit
st.title("Recomendación de Canciones")
cancion_input = st.text_input("Introduce el nombre de una canción:", "")
if st.button("Obtener Recomendaciones"):
    recomendaciones = str_canciones_recomendadas(cancion_input)
    st.markdown(recomendaciones, unsafe_allow_html=True)

