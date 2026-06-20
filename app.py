import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load model
model = tf.keras.models.load_model("mnist_ann.keras")

# Page Config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="centered"
)

# Title
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit between 0 and 9 and click Predict.")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("""
This project uses an Artificial Neural Network (ANN)
trained on the MNIST dataset to recognize handwritten digits.
""")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction Button
if st.button("Predict Digit"):

    if canvas_result.image_data is not None:

        # Convert canvas to image
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Convert to grayscale
        img = img.convert("L")

        # Resize to MNIST dimensions
        img = img.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(img)

        # Invert colors
        # MNIST = white digit on black background
        img_array = 255 - img_array

        # Normalize
        img_array = img_array / 255.0

        # IMPORTANT:
        # Your model expects (28,28)
        # Flatten layer inside model converts it to 784
        img_array = img_array.reshape(1, 28, 28)

        # Predict
        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Show processed image
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img,
                caption="Processed Image",
                width=150
            )

        with col2:
            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        # Probabilities
        st.subheader("Prediction Probabilities")

        prob_dict = {
            str(i): float(prediction[0][i])
            for i in range(10)
        }

        st.bar_chart(prob_dict)

    else:
        st.warning("Please draw a digit first.")