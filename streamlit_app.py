import streamlit as st
from streamlit_webrtc import webrtc_streamer
from camera_input_live import camera_input_live
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

page_names = ['Take picture', 'Upload picture', 'Real-Time']

page = st.sidebar.radio('Choose image source', page_names)

###

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold:', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)

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
st.write('# White Blood Cell Identifier & Counter')

img_str = None  # Initialize img_str variable
label_counts = {}

if page == 'Take picture':
    img_file_buffer = st.camera_input("Take a picture:")

    if img_file_buffer is not None:
        # To read image file buffer with PIL:
        image = Image.open(img_file_buffer)
        cv2_img = np.array(image.convert("RGB"))

        # Convert to JPEG Buffer.
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

else:
    if page == 'Upload picture':
        uploaded_file = st.file_uploader('Select an image:', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

        ## Pull in default image or user-selected image.
        if uploaded_file is None:
            # Default image.
            url = 'https://github.com/NoliAlonso/streamlit-bccd/blob/master/BCCD_sample_images/im_0000_20230601_124318.jpg?raw=true'
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))
        else:
            # User-selected image.
            image = Image.open(uploaded_file)

        # Convert to JPEG Buffer.
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

    else:
        if page == 'Real-Time':
            image = camera_input_live()

            if image is not None:
                st.image(image)
                bytes_data = image.getvalue()
                pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                cv2_img = np.array(pil_image)

                if cv2_img.size > 0:  # Check if the image is not empty
                    # Display the "Infer" button
                    if st.button("Infer"):
                        # Perform calculations or operations on cv2_img
                        mean_value = np.mean(cv2_img)

                        if np.isnan(mean_value):  # Check if the mean value is NaN (invalid)
                            # Handle the case of invalid value
                            mean_value = 0.0  # Set a default value or perform a different action

                        # Convert to JPEG Buffer.
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format='JPEG')
                        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
                        # Further processing with img_str and mean_value if needed
                        ...
                        
                else:
                    # Handle the case of an empty image
                    img_str = ""
                    mean_value = 0.0

if img_str is not None:  # Check if img_str is defined

    ## Subtitle.
    st.write('### Inferenced Image')

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

    if r.ok:
        try:
            image = Image.open(io.BytesIO(r.content))

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

            ## Display the JSON in main app.
            st.write('### JSON Output')
            st.write(r.json())

            # Update label counts using JSON output
            for box in output_dict['predictions']:
                label = box['class']
                label_counts[label] = label_counts.get(label, 0) + 1

            # Display label counts in a table
            st.write('### Label Counts')
            table_data = [[label, count] for label, count in label_counts.items()]
            st.table(table_data)
            
        except IOError:
            st.write("Error: Failed to open the image from the API response.")
    else:
        st.write("Error: API request failed.")
