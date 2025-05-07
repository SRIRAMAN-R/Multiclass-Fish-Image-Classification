import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fish_classifier import load_data
import plotly.express as px

# Load best model
MODEL_PATH = r"D:\Mainboot Project\project 5\models\mobilenetv2_fish_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
train_data, _, _ = load_data()
class_names = list(train_data.class_indices.keys())
st.set_page_config(page_title="🐟 Fish Species Classifier", layout="centered", page_icon="🐟")

# App title and subtitle
st.title("🐠 Fish Species Classifier")
st.markdown("Upload a fish image and identify its species using a deep learning model!")

# File uploader
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader("Choose a .jpg/.jpeg/.png image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction with spinner
    with st.spinner("🔍 Analyzing image..."):
        predictions = model.predict(img_array)[0]

    # Format class name
    def format_class_name(label):
        return label.replace("fish_sea_food_", "").replace("_", " ").title()

    # Get prediction results
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index]

    formatted_label = format_class_name(predicted_label)

    # Show prediction
    st.markdown("### 🎯 **Prediction Result**")
    st.success(f"✅ **Predicted Species:** {formatted_label}")
    st.metric("📈 Confidence", f"{confidence:.2%}")

    # Bar chart with shades of blue
    st.markdown("### 📊 Confidence Across All Species")
    fig = px.bar(
        x=predictions,
        y=class_names,
        orientation='h',
        labels={'x': 'Confidence', 'y': 'Species'},
        color=predictions,
        color_continuous_scale='blues',  
        text=[f"{p:.2%}" for p in predictions]
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=450,
        margin=dict(l=80, r=40, t=40, b=40),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("💡 Upload a clear fish image to get predictions.")

# Footer
st.markdown("---")
