import streamlit as st
from PIL import Image
import io
from utils import load_generation_model, generate_image, load_classification_model, classify_image

# Cargar los modelos de HuggingFace
generation_model = load_generation_model()
classification_model = load_classification_model()

# Configurar la interfaz de Streamlit
st.title("Generación y Clasificación de Imágenes con Hugging Face")

col1, col2 = st.columns(2)

# Sección 1: Generación de Imágenes
with col1:
    st.header("Generación de Imágenes")
    prompt = st.text_input("Escribe una solicitud para generar una imagen")
    if st.button("Generar Imagen"):
        if prompt:
            generated_image = generate_image(generation_model, prompt)
            st.image(generated_image, caption="Imagen generada", use_column_width=True)
            # Guardar la imagen generada en una variable de sesión
            st.session_state["generated_image"] = generated_image
        else:
            st.warning("Por favor, ingrese una solicitud para generar una imagen.")

# Sección 2: Clasificación de Imágenes
with col2:
    st.header("Clasificación de Imágenes")
    
    # Subir imagen para clasificar
    uploaded_image = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        # Clasificar imagen subida
        if st.button("Clasificar Imagen Subida"):
            classification_results = classify_image(classification_model, image)
            st.write("Resultados de la clasificación:")
            for result in classification_results:
                st.write(f"{result['label']}: {result['score']:.4f}")
    
    # Clasificar la imagen generada en la Sección 1
    if "generated_image" in st.session_state:
        st.write("---")
        st.subheader("Clasificar la imagen generada")
        if st.button("Clasificar Imagen Generada"):
            generated_image = st.session_state["generated_image"]
            classification_results = classify_image(classification_model, generated_image)
            st.write("Resultados de la clasificación de la imagen generada:")
            for result in classification_results:
                st.write(f"{result['label']}: {result['score']:.4f}")