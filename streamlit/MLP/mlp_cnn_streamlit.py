import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
from streamlit_option_menu import option_menu
import cv2

# to ignore the warning when uploading a file
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache_resource  # this will remember our model and will load it only once
def cnn_model():
    model = tf.keras.models.load_model('CNN_model.hdf5')
    return model


@st.cache_resource
def mlp_model():
    model = tf.keras.models.load_model('MLP_model.hdf5')
    return model


def predict_cnn(input_image, model):
    size = (100, 100)
    image = ImageOps.fit(input_image, size)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


def predict_mlp(input_image, model):
    img = cv2.imdecode(np.fromstring(input_image.read(), np.uint8), 1)
    image = cv2.imread(img)
    resize_image = cv2.resize(image, (100, 100))
    image_flat = resize_image.flatten()
    reshaped_img = np.vstack(image_flat)
    prediction = model.predict(reshaped_img)
    return prediction


def main():

    with st.sidebar:
        selected = option_menu('Multiple Model For Prediction', [
                               'Multilayer Perception Model', 'Convolution Neural Network'], default_index=0)

    st.write("""# Eyes Classification""")

    st.write("""We have two options on how we want to predict the images. Either by uploading an image file or by using an URL of an Image.""")
    st.markdown(
        "------------------------------------------------------------------------")

    if (selected == 'Multilayer Perception Model'):

        with st.spinner('Loading the model into Memory......'):
            mlp = mlp_model()

        class_name_mlp = ['Female', 'Male']

        # from the uploaded file
        file = st.file_uploader("""Please upload an image""",
                                type=['jpeg', 'jpg', 'png'])

        if file is not None:
            image = Image.open(file)
            st.image(image, use_column_width=True)

        class_btn = st.button("Classify")
        if class_btn:
            if file is None:
                st.text("Invalid Command !!! Please upload the file")
            else:
                with st.spinner('Predicting.....'):
                    prediction = predict_mlp(file, mlp)
                    output = f'This image is of {class_name_mlp[np.argmax(prediction)]}.'
                    st.success(output)

        st.markdown(
            "------------------------------------------------------------------------")

        # from the url
        st.text("URL of the Eye Image")
        path = st.text_input(
            "Enter the URL of the Eye Image to Classify", 'https://images.unsplash.com/photo-1494869042583-f6c911f04b4c?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80')
        if path is None:
            st.text("Please Enter a file path/URL")
        else:
            file_image = requests.get(path).content
            image = Image.open(BytesIO(file_image))
            st.image(image, use_column_width=True)
        class_btn_pred = st.button("Predict")
        if class_btn_pred:
            with st.spinner('Predicting.....'):
                prediction = predict_mlp(file, mlp)
                output = f'This image is of {class_name_mlp[np.argmax(prediction)]}.'
                st.success(output)
#############################################################################################################################
    if (selected == 'Convolution Neural Network'):
        with st.spinner('Loading the model into Memory......'):
            cnn = cnn_model()

        class_name_cnn = ['Female', 'Male']

        # from the uploaded file
        file = st.file_uploader("""# Please upload an image""",
                                type=['jpeg', 'jpg', 'png'])

        if file is not None:
            image = Image.open(file)
            st.image(image, use_column_width=True)

        class_btn = st.button("Classify")
        if class_btn:
            if file is None:
                st.text("Invalid Command !!! Please upload the file")
            else:
                with st.spinner('Predicting.....'):
                    prediction = predict_cnn(image, cnn)
                    output = f'This image is of {class_name_cnn[np.argmax(prediction)]}.'
                    st.success(output)

        st.markdown(
            "------------------------------------------------------------------------")

        # from the url
        st.text("URL of the Eye Image")
        path = st.text_input(
            "Enter the URL of the Eye Image to Classify", 'https://images.unsplash.com/photo-1494869042583-f6c911f04b4c?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80')
        if path is None:
            st.text("Please Enter a file path/URL")
        else:
            file_image = requests.get(path).content
            image = Image.open(BytesIO(file_image))
            st.image(image, use_column_width=True)
        class_btn_pred = st.button("Predict")
        if class_btn_pred:
            with st.spinner('Predicting.....'):
                prediction = predict_cnn(image, cnn)
                output = f'This image is of {class_name_cnn[np.argmax(prediction)]}.'
                st.success(output)


if __name__ == '__main__':
    main()
