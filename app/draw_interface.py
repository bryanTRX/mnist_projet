import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from src.utils import preprocess_canvas_image, get_preprocessed_pil
from src.predict import predict
import pandas as pd

st.set_page_config(page_title="MNIST Draw & Predict", page_icon="✏️")
st.title("Dessinez un chiffre manuscrit (0-9)")

canvas_result = st_canvas(
    fill_color="black",          
    stroke_width=15,             
    stroke_color="white",        
    background_color="black",   
    height=280,                 
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Prédire") and canvas_result.image_data is not None:
    tensor_img = preprocess_canvas_image(canvas_result.image_data)
    img_display = get_preprocessed_pil(canvas_result.image_data)

    pred, conf, all_probs = predict(tensor_img)

    st.image(img_display, caption="Votre dessin (prétraité)", width=140)
    st.success(f"Prédiction : {pred} (Confiance : {conf*100:.2f}%)")
    
    df_probs = pd.DataFrame({
        "Chiffre": list(range(10)),
        "Probabilité (%)": [p*100 for p in all_probs]
    }).sort_values(by="Probabilité (%)", ascending=False)
    
    st.subheader("Probabilités pour chaque chiffre")
    st.table(df_probs)

    st.bar_chart(pd.DataFrame(all_probs, index=range(10), columns=["Probabilité"]))
