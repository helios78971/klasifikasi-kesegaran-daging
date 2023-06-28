from keras.models import load_model
import keras.utils as image
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import base64

# Siapkan model
model = load_model('model\model_resnet.h5')

# Ukuran gambar
input_size=(128, 128, 3)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    
    [data-testid="stHeader"]{{
        background-color: rgba(0,0,0,0);}}
    
    .stTabs [class="st-c7"] {{
        background-color: teal;
    }}
    
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True
    )

def main():
    add_bg_from_local('assets\kyle-mackie-Xedxbjx7MFg-unsplash.jpg')
    st.title("Klasifikasi Kesegaran Daging Sapi")

    st.subheader("Metode")
    tab1, tab2 = st.tabs(["Upload", "Kamera"])

    with tab1:
        uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"], accept_multiple_files=False, label_visibility='collapsed')
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar Input")
            if uploaded_file:
                display = Image.open(uploaded_file)
                st.image(display)

            if st.button("Prediksi", key=1):
                if uploaded_file is not None:
                    # Preprocess gambar untuk uji model
                    img_path = uploaded_file
                    img = image.load_img(img_path, target_size=input_size[:2])
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    # Mendefinisikan tf.function untuk prediksi agar performa baik
                    @tf.function
                    def predict(image_array):
                        return model(image_array)

                    # Buat prediksi dari gambar
                    predictions = predict(tf.convert_to_tensor(img_array))

                    # Ambil indeks hasil prediksi
                    predicted_class_index = np.argmax(predictions)

                    # Nama kelas
                    class_names = ['Busuk', 'Segar', 'Setengah Segar']

                    # Ambil kelas berdasarkan indeks
                    predicted_class_name = class_names[predicted_class_index]

                    # Ambil probabilitas untuk kelas hasil prediksi
                    predicted_probability = predictions[0][predicted_class_index]

                    predicted_accuracy = predicted_probability * 100

                    with col2:
                        st.subheader("Prediksi")    
                        st.write(f"<h3>{predicted_class_name}</h3>", unsafe_allow_html=True)
                        st.write("<h1>{:.2f}%</h1>".format(predicted_accuracy), unsafe_allow_html=True)
    
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ambil Gambar")
                if 'Gambar' not in st.session_state.keys():
                    st.session_state['Gambar'] = None

                captured_image = st.camera_input("Test", label_visibility='collapsed')

            with col2:
                if captured_image is not None:
                    img = Image.open(captured_image)
                    img = img.resize(input_size[:2])
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    # Mendefinisikan tf.function untuk prediksi agar performa baik
                    @tf.function
                    def predict(image_array):
                        return model(image_array)

                    # Buat prediksi dari gambar
                    predictions = predict(tf.convert_to_tensor(img_array))

                    # Ambil indeks hasil prediksi
                    predicted_class_index = np.argmax(predictions)

                    # Nama kelas
                    class_names = ['Busuk', 'Segar', 'Setengah Segar']

                    # Ambil kelas berdasarkan indeks
                    predicted_class_name = class_names[predicted_class_index]

                    # Ambil probabilitas untuk kelas hasil prediksi
                    predicted_probability = predictions[0][predicted_class_index]

                    predicted_accuracy = predicted_probability * 100

                    st.subheader("Prediksi")
                    st.write(f"<h3>{predicted_class_name}</h3>", unsafe_allow_html=True)
                    st.write("<h1>{:.2f}%</h1>".format(predicted_accuracy), unsafe_allow_html=True)

if __name__ == '__main__':
    main()