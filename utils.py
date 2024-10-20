from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Función para cargar el modelo de generación de imágenes
def load_generation_model():
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    return model

# Función para generar una imagen a partir de un texto
def generate_image(model, prompt):
    image = model(prompt).images[0]
    return image

# Función para cargar el modelo de clasificación de imágenes
def load_classification_model():
    classifier = pipeline("image-classification", model="microsoft/resnet-50")
    return classifier

# Función para clasificar una imagen
def classify_image(classifier, image):
    results = classifier(image)
    return results
