import streamlit as st
from ultralytics import YOLO
from PIL import Image

# -----------------------------------
# Load your trained YOLO model
# -----------------------------------
model = YOLO("best.pt")

# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------------
# Title
# -----------------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and let the trained YOLO model detect the disease.")

# -----------------------------------
# Upload image
# -----------------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # -----------------------------------
    # Detection button
    # -----------------------------------
    if st.button("🔍 Detect Disease"):

        with st.spinner("Analyzing the leaf..."):

            # Run YOUR trained model
            results = model(image, conf=0.25)

        result = results[0]

        # -----------------------------------
        # Display detection image
        # -----------------------------------
        st.subheader("Detection Result")

        annotated_image = result.plot()

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

        # -----------------------------------
        # Display detections
        # -----------------------------------
        if len(result.boxes) == 0:

            st.warning("No disease detected.")

        else:

            st.subheader("Detected Diseases")

            for box in result.boxes:

                # Class ID
                class_id = int(box.cls[0])

                # Confidence
                confidence = float(box.conf[0])

                # Automatically get class name
                disease_name = model.names[class_id]

                st.success(
                    f"🌱 {disease_name} — "
                    f"{confidence * 100:.2f}% confidence"
                )