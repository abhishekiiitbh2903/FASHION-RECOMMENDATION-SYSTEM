import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from annoy import AnnoyIndex
# feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
# filenames = pickle.load(open('filenames.pkl','rb'))

embedding_path = os.path.join(os.getcwd(), 'embeddings.pkl') # Absolute Path
filename_path = os.path.join(os.getcwd(), 'filenames.pkl')

# Load data using absolute paths
feature_list = pickle.load(open(embedding_path, 'rb'))
filenames = pickle.load(open(filename_path, 'rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.markdown(
    """
    <style>
        @keyframes slidein {
            0% {
                transform: translateY(-50%);
                opacity: 0;
            }
            50% {
                transform: translateY(0%);
                opacity: 1;
            }
            100% {
                transform: translateY(-50%);
                opacity: 0;
            }
        }
        .animated-title {
            animation: slidein 2s linear infinite;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 class='animated-title'>Fashion Recommender System</h1>",
    unsafe_allow_html=True
)
st.write("""
    <style>
    .center {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# st.image('fashion.png', height=200, caption='Fashion Image',use_column_width=True)
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    # neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    # neighbors.fit(feature_list)

    # distances, indices = neighbors.kneighbors([features])

    # return indices[0]

    dimensions = len(features) 
    annoy_index = AnnoyIndex(dimensions, metric='manhattan') 

    for i, feature in enumerate(feature_list):  
        annoy_index.add_item(i, feature)

    annoy_index.build(n_trees=10) 
    indices = annoy_index.get_nns_by_vector(features, 6)  # Get indices of 6 nearest neighbors

    return indices



# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image,width=100)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        st.text(indices)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0]],width=120)
        with col2:
            st.image(filenames[indices[1]],width=120)
        with col3:
            st.image(filenames[indices[2]],width=120)
        with col4:
            st.image(filenames[indices[3]],width=120)
        with col5:
            st.image(filenames[indices[4]],width=120)
    else:
        st.header("Some error occured in file upload")