import streamlit as st
from streamlit_webrtc import webrtc_streamer
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import av
import cv2

##########
##### Set up sidebar.
##########

# Add in location to select image.

page_names = ['Capture', 'Upload']

page = st.radio('Select', page_names)

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)

image = Image.open('./images/roboflow_logo.png')
st.sidebar.image(image,
                 use_column_width=True)

image = Image.open('./images/streamlit_logo.png')
st.sidebar.image(image,
                 use_column_width=True)

image = Image.open('./images/NoliAlonsoPathLabSystemsLogo.png')
st.sidebar.image(image,
                 use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write('# Peripheral Smear: White Blood Cell Identifier')

if page = 'Capture':
    st.subheader("Take a picture with your camera:")
    img_file_buffer = st.camera_input("Image capture:")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        #st.write(type(cv2_img))

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        st.write(cv2_img.shape)

if page = 'Upload':
    st.subheader('Select an image to upload.')
    uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    ## Pull in default image or user-selected image.
    if uploaded_file is None:
        if img_file_buffer is None:
            # Default image.
            url = 'https://github.com/NoliAlonso/streamlit-bccd/blob/master/BCCD_sample_images/im_0000_20230601_124318.jpg?raw=true'
            image = Image.open(requests.get(url, stream=True).raw)
        
        else:
            image = Image.open(cv2_img)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)


## Subtitle.
st.write('### Inferenced Image')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=95, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')

## Construct the URL to retrieve image.
upload_url = ''.join([
    'https://detect.roboflow.com/peripheralbloodsmear/12',
    '?api_key=oDkrH1XmBTgm5SIRerW7',
    '&format=image',
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&stroke=2',
    '&labels=True'
])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})

image = Image.open(BytesIO(r.content))

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Display image.
st.image(image,
         use_column_width=True)

## Construct the URL to retrieve JSON.
upload_url = ''.join([
    'https://detect.roboflow.com/peripheralbloodsmear/12',
    '?api_key=oDkrH1XmBTgm5SIRerW7'
])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})

## Save the JSON.
output_dict = r.json()

## Generate list of confidences.
confidences = [box['confidence'] for box in output_dict['predictions']]

## Summary statistics section in main app.
st.write('### Summary Statistics')
st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

## Histogram in main app.
st.write('### Histogram of Confidence Levels')
fig, ax = plt.subplots()
ax.hist(confidences, bins=10, range=(0.0,1.0))
st.pyplot(fig)

## Display the JSON in main app.
st.write('### JSON Output')
st.write(r.json())
