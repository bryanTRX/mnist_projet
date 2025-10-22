import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
from PIL import Image
from src.utils import preprocess_canvas_image, get_preprocessed_pil
from src.predict import predict
import time

st.set_page_config(
    page_title="Reconnaissance de Chiffres MNIST",
    page_icon="✏️",
    layout="wide",
)

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1E90FF;
    color: white;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    border: none;
    font-size: 16px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #104E8B;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
    <h1>🧠 Reconnaissance de Chiffres Manuscrits</h1>
    <p style='font-size: 18px; color: #666;'>Dessinez un chiffre (0-9) et laissez le modèle prédire le résultat.</p>
    <hr style="width: 60%; margin: auto; margin-bottom: 20px;">
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

if "reset_canvas" not in st.session_state:
    st.session_state.reset_canvas = False

with col1:
    st.subheader("Dessinez ici")
    
    if st.button("Effacer le canvas"):
        st.session_state.reset_canvas = True

    canvas_container = st.empty()
    with canvas_container.container():
        key = "canvas" if not st.session_state.reset_canvas else "canvas_new"
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=key,
        )

    if st.session_state.reset_canvas:
        st.session_state.reset_canvas = False

with col2:
    st.subheader("Résultat de la Prédiction")

    if st.button("Lancer la Prédiction"):
        if canvas_result.image_data is None:
            st.warning("Dessinez un chiffre avant de lancer la prédiction.")
        else:
            drawn_pixels = np.sum(canvas_result.image_data[:, :, 0] > 0)
            if drawn_pixels < 50:
                st.error("Vous devez dessiner un chiffre plus clairement.")
            else:
                with st.spinner("🧮 Le modèle réfléchit..."):
                    time.sleep(1)
                    tensor_img = preprocess_canvas_image(canvas_result.image_data)
                    img_display = get_preprocessed_pil(canvas_result.image_data)
                    pred, conf, all_probs = predict(tensor_img)

                    st.image(img_display, caption="Image prétraitée", width=140)

                    st.markdown(f"""
                    <div style='text-align: center; font-size: 22px;'>
                         <b>Chiffre prédit :</b> <span style='color:#1E90FF;'>{pred}</span><br>
                         <b>Confiance :</b> {conf*100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

                    df_probs = pd.DataFrame({
                        "Chiffre": list(range(10)),
                        "Probabilité (%)": [p * 100 for p in all_probs]
                    }).sort_values(by="Probabilité (%)", ascending=False)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader(" Distribution des Probabilités")
                    st.bar_chart(df_probs.set_index("Chiffre"), height=300)
    else:
        st.info(" Dessinez un chiffre à gauche puis cliquez sur **Lancer la Prédiction** pour voir le résultat.")

st.markdown("""
<hr>
<div style='text-align: center; color: #888;'>
    Développé avec ❤️ en Python, PyTorch et Streamlit
</div>
""", unsafe_allow_html=True)
