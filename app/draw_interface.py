import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

st.set_page_config(page_title="MNIST Draw & Predict", page_icon="✏️")
st.title("Dessinez un chiffre manuscrit (0-9)")

canvas_result = st_canvas(
    fill_color="black",       
    stroke_width=10,          
    stroke_color="white",     
    background_color="black", 
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Voir l'image du canvas") and canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype(np.uint8))
    st.image(img, caption="Votre dessin", width=140)