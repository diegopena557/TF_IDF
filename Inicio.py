import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import random
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="TF-IDF con Humor 😄", layout="centered")

st.title("💬 Demo de TF-IDF con Humor y Memoria")

st.write("""
Esta app analiza tus documentos en inglés con **TF-IDF** y responde preguntas según la similitud de texto.  
Además, crea **frases humorísticas** para ayudarte a **recordar las palabras clave**.  
""")

# 📚 Documentos de ejemplo
text_input = st.text_area(
    "📄 Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("❓ Escribe una pregunta (en inglés):", "Who is playing?")

# Stemmer en inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    """Convierte texto en stems normalizados (minúsculas, sin símbolos)."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# 🎭 Función para generar frases divertidas
def generate_funny_sentence(word):
    """Genera una frase humorística para una palabra clave."""
    templates = [
        f"When you forget '{word}', imagine a llama trying to spell it with sunglasses 😎.",
        f"'{word}' once tried to join a rock band, but forgot the lyrics!",
        f"Never trust a cat that whispers '{word}' at midnight 🐱‍👤.",
        f"If '{word}' were a snack, it would definitely be extra crunchy 🍪.",
        f"Remember '{word}' like your Wi-Fi password — you never know when you’ll need it!",
        f"'{word}' sounds like a superhero that only fights grammar mistakes 🦸‍♂️.",
        f"Picture '{word}' dancing salsa in your brain every time you study 💃."
    ]
    return random.choice(templates)

if st.button("🚀 Calcular TF-IDF y Buscar Respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        # Vectorizador con stemming
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)

        # 🧮 Mostrar matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        with st.expander("📊 Ver matriz TF-IDF"):
            st.dataframe(df_tfidf.round(3))

        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más relevante
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### 🎯 Resultado")
        st.success(f"**Documento más relevante:** {best_doc}")
        st.info(f"📈 Similitud: {best_score:.3f}")

        # Mostrar palabras clave relevantes
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

        st.write("### 🔑 Palabras clave encontradas")
        if matched:
            st.write(", ".join(matched))
        else:
            st.write("No se encontraron coincidencias directas en los documentos.")

        # 🎉 Generar frases humorísticas
        if matched:
            st.write("### 😂 Frases para recordar las palabras clave:")
            for m in matched:
                st.write(f"— {generate_funny_sentence(m)}")
        else:
            st.write("Intenta con una pregunta que comparta palabras con tus documentos.")

# 📘 Sección lateral
st.sidebar.title("ℹ️ Acerca de la app")
st.sidebar.write("""
Esta app demuestra cómo funciona **TF-IDF** (Term Frequency - Inverse Document Frequency)
para encontrar la relevancia de documentos frente a una pregunta.

La sección de frases humorísticas busca fomentar la **recordación emocional y divertida** 
de los términos clave, usando asociaciones graciosas y visuales 😄.
""")


