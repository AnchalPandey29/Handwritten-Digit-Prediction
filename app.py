import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

/* Header */
.header {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

/* Prediction */
.prediction {
    font-size: 60px;
    font-weight: bold;
    color: #4f46e5;
    text-align: center;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

model = load_model()

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>🤖 AI Digit Recognition System</h1>
    <p>Draw or upload a digit and let AI predict it instantly</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

drawing_mode = st.sidebar.radio(
    "Input Method",
    ["✍️ Draw Digit", "📤 Upload Image"]
)

stroke_width = st.sidebar.slider("Brush Size", 5, 25, 12)

st.sidebar.markdown(" replete with --- ")
st.sidebar.info("""
💡 Tips:
- Draw clearly in center
- Use thick strokes
- Avoid multiple digits
""")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1])

# ---------------- DRAW / UPLOAD ----------------
image_data = None

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if drawing_mode == "✍️ Draw Digit":
        st.subheader("Draw Digit Below")

        canvas = st_canvas(
            fill_color="black",
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas.image_data is not None:
            image_data = canvas.image_data

    else:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=200)
            image_data = np.array(image)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = Image.fromarray(img.astype('uint8')).convert("L")
    img = img.resize((28, 28))

    img = np.array(img)
    img = 255 - img  # invert

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# ---------------- PREDICTION ----------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if image_data is not None:
        processed = preprocess(image_data)

        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # BIG RESULT
        st.markdown(f'<div class="prediction">{digit}</div>', unsafe_allow_html=True)

        st.progress(float(confidence))
        st.caption(f"Confidence: {confidence*100:.2f}%")

        # ---------------- CHART ----------------
        probs = prediction.flatten()

        fig = px.bar(
            x=list(range(10)),
            y=probs,
            labels={'x': 'Digit', 'y': 'Probability'},
            title="Prediction Confidence"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Draw or upload an image to see prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("HandWritten Digit Prediction | Build by Anchal Pandey")
