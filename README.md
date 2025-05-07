# 🐟 Multiclass Fish Image Classification
A deep learning-based image classification system that identifies fish species from images using Convolutional Neural Networks (CNNs) and Transfer Learning. The project includes model evaluation, comparison, and deployment via a Streamlit web application.

# 📌 Features
- CNN model built from scratch
- Fine-tuned models using:
- VGG16
- ResNet50
- MobileNetV2
- InceptionV3
- EfficientNetB0
- Data augmentation for generalization
- Model evaluation with classification report & confusion matrix
- Streamlit app for real-time fish species prediction


# 🏃‍♂️ Usage
Train Models and Evaluate
```
python fish_classifier.py
```
- Trains one CNN and five transfer learning models
- Saves .h5 models in the models/ folder
- Prints evaluation metrics and confusion matrices

# Launch Streamlit App
```
streamlit run app.py
```
- Upload a fish image
- Get species prediction and model confidence
- View confidence scores for all classes via a bar chart

# 📚 Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Scikit-learn
- Streamlit
- Plotly
- PIL


