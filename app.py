# import the Package
import os
from tkinter import font
import numpy as np 
from PIL import Image , ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf



# Create a title of web App
st.title("Image Classification with Cifar10 Dataset")


# writing description
st.write("""The CIFAR-10 are labeled tiny images [dataset](http://www.cs.toronto.edu/~kriz/cifar.html). They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. There are total 10 classes as shown below. We will use neural networks for classification.
""",font=('Arial', 16))

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write(' ')

with col2:
    st.image("cifar-10.jpg",width=600)
with col3:
    st.write(' ')

#adding test image
#subheader
st.subheader("Sample Test Image")
st.image('testing_images.png')
#adding train image'
st.subheader("Sample Train Image")
st.image('training_images.png')

#class distribution
st.subheader("Class Distribution of Cifar10 TrainDataset")
st.image('per_class_distrubution.png')

#class distribution of test
st.subheader("Class Distribution of Cifar10 Test Dataset")
st.image('per_class_distrubution_test.png')

# showing heatmap
st.subheader("Heatmap of Cifar10 Train Dataset")
st.image('best_heatmap.png')

#alogrithm selection
st.sidebar.header("Select Algorithm")
algorithm = st.sidebar.selectbox("Algorithm", ("ANN", "Sequential-CNN", "CNN-Functional-API","CNN-Functional API trained with augmented images"))
#writng the selected algorithm
st.header('Your selected algorithm is: ' + algorithm)



class_name = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

# Create a function to load my saved model
@st.cache(allow_output_mutation=True)
def load_my_model():
    if algorithm == "ANN":
        model = tf.keras.models.load_model("ann_v1.h5")
    elif algorithm == "Sequential-CNN":
        model = tf.keras.models.load_model("cnn_v1.h5")
    elif algorithm == "CNN-Functional-API":
        model = tf.keras.models.load_model("fapi_v1.h5")
    elif algorithm == "CNN-Functional API trained with augmented images":
        model = tf.keras.models.load_model("fapi_aug_v1.h5")
    return model

model = load_my_model()





st.header("Please Upload images related to this things...")
st.text(class_name)

# create a file uploader and take a image as an jpg or png
file = st.file_uploader("Upload the image" , type=["jpg" , "png"])

# Create a function to take and image and predict the class
def import_and_predict(image_data , model):
    size = (32 ,32)
    image = ImageOps.fit(image_data , size , Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if st.button("Predict"):
    image = Image.open(file)
    st.image(image , use_column_width=True)
    predictions = import_and_predict(image , model)

    class_name = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

    string = "Image mostly same as :-" + class_name[np.argmax(predictions)]
    st.success(string)

#showing accuracy graphs of the model
st.header("Graphs of selected model")
if algorithm == "ANN":
    st.image(["ann_accuracy.png","ann_loss.png"])
elif algorithm == "Sequential-CNN":
    st.image(["cnn_accuracy.png","cnn_loss.png"])
elif algorithm == "CNN-Functional-API":
    st.image(["cnn_fun_accuracy.png","cnn_fun_loss.png"])


#algorithm == "CNN-Functional API trained with augmented images":
st.header("Best Model")
st.image(["best_accuracy.png","best_loss.png"])
    #st.image("fapi_aug_v1.png", caption="CNN-Functional API trained with augmented images Accuracy Graph",width=500)




    
