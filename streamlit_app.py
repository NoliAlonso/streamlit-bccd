# load config
from curses.ascii import SI
import json
with open('Roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

import streamlit as st
from streamlit_webrtc import webrtc_streamer
from camera_input_live import camera_input_live
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import av
import cv2
import datetime
import pandas as pd

##########
##### Set up sidebar.
##########
titlemessage = '# WBC Identifier & Counter'
st.sidebar.write(titlemessage)
st.sidebar.write('Developed by Dr. Alonso')

##########
st.sidebar.divider()

# Initialize the class_counts dictionary as an empty dictionary in the session state
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = {}
    st.session_state.last_updated = datetime.time(0,0)

def SubmitJSONdataframe():
    # Add the dataframe data to the class_counts dictionary
    for index, row in df_grouped.iterrows():
        # Use get method to handle cases where the class name is not already in the dictionary
        st.session_state.class_counts[row['class']] = st.session_state.class_counts.get(row['class'], 0) + row['count']
    st.session_state.last_updated = datetime.datetime.now().ctime()
    
# Define a function to increment a cell count by 1
def increment_count(cell):
    st.session_state.class_counts[cell] += 1
    st.session_state.last_updated = datetime.datetime.now().ctime()

# Define a function to decrement a cell count by 1
def decrement_count(cell):
     # Check if the cell count is greater than zero
    if st.session_state.class_counts[cell] > 0:
        # Decrement the count by 1
        st.session_state.class_counts[cell] -= 1
    st.session_state.last_updated = datetime.datetime.now().ctime()

def UpdateTheTime():
    st.session_state.last_updated = datetime.datetime.now().ctime()

def ResetAll():
    st.session_state.class_counts = {}
    st.session_state.last_updated = datetime.datetime.now().ctime()


#########

# Create a dataframe from the class counts dictionary
dfCount = pd.DataFrame(list(st.session_state.class_counts.items()), columns=['class', 'count'])
dfCount.columns = ['Cell', 'Count']

# Check if the class counts dictionary is empty
if st.session_state.class_counts:
    # Display the updated dataframe
    st.sidebar.dataframe(dfCount, use_container_width=True, hide_index=True)
    
    # Compute the total of all counts
    DiffCountTotal = dfCount['Count'].sum()

    colu1, colu2 = st.sidebar.columns([0.7, 0.3])
    with colu1:
        st.write(' ')
    with colu2:
        # Display the total in the sidebar
        st.write('Total = ', DiffCountTotal)

    # Loop through each row of the dataframe and add buttons
    for i in range(len(dfCount)):
        cell = dfCount.iloc[i, 0] # Get the cell name
        # Pass a unique key argument to each button widget
        with st.sidebar.container():
            col1, col2, col3 = st.columns([0.6,0.2,0.2])

            with col1:
                st.write(cell) # Display the cell name
            # Pass a unique key argument to each button widget
            with col2:
                st.button(':heavy_plus_sign:', on_click=increment_count, args=(cell,), key=f"increment_{i}") # Add an increment button
            with col3:
                st.button(':heavy_minus_sign:', on_click=decrement_count, args=(cell,), key=f"decrement_{i}") # Add a decrement button

    st.sidebar.divider()

    st.sidebar.write('Count last updated at ', st.session_state.last_updated)
else:
    st.sidebar.write('Add cell types or inference an image to begin.');

st.sidebar.divider()

st.sidebar.write('Add cell to count:')
#add classes, classname - button that adds 1 

cell_names = ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil', 'NRBC','Blast']

col_cn1, col_cn2 = st.sidebar.columns([0.2, 0.8])

with col_cn1:
    imageLogo = Image.open('./images/Neutrophil.png')
    st.image(imageLogo)

    imageLogo = Image.open('./images/Lymphocyte.png')
    st.image(imageLogo)

    imageLogo = Image.open('./images/Monocyte.png')
    st.image(imageLogo)

    imageLogo = Image.open('./images/Eosinophil.png')
    st.image(imageLogo)

    imageLogo = Image.open('./images/Basophil.png')
    st.image(imageLogo)


with col_cn2:
    with st.sidebar.form(key='UpdateTheTime'):
    # Loop through each cell name and create a button
        for cell_name in cell_names:
            if st.sidebar.button(label=cell_name, on_click=UpdateTheTime):
                # Check if the button is clicked
                if cell_name not in st.session_state.class_counts:
                    # Add the cell name to the session state dictionary with a count of 1
                    st.session_state.class_counts[cell_name] = 1 

st.sidebar.divider()

with st.sidebar.form(key='resetform'):
    reset_button = st.form_submit_button(label='Reset', on_click=ResetAll)

st.sidebar.divider()

########## Logos

imageLogo = Image.open('./images/roboflow_logo.png')
st.sidebar.image(imageLogo, use_column_width=True)

imageLogo = Image.open('./images/streamlit_logo.png')
st.sidebar.image(imageLogo, use_column_width=True)

imageLogo = Image.open('./images/NoliAlonsoPathLabSystemsLogo.png')
st.sidebar.image(imageLogo, use_column_width=True)

st.sidebar.write('Disclaimer, as is, for research purposes only.')

##########
##### Set up main app.
##########

## Title.
st.write(titlemessage)

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        # Add in location to select image.
        page_names = ['Take picture', 'Upload picture']
        page = st.radio('Select image source:', page_names)
    with col2:
        confidence_threshold = st.slider('Confidence threshold:', 0.0, 1.0, 0.5, 0.01)        
    with col3:
        overlap_threshold = st.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)

st.divider()

img_str = None  # Initialize img_str variable
mean_value = 0.0

#########

if page == 'Take picture':
    img_file_buffer = st.camera_input("Take a picture:")

    if img_file_buffer is not None:
        # To read image file buffer with PIL:
        image1 = Image.open(img_file_buffer)

        #crop edges

        # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
        image1size = np.array(image1)
        height, width, channels = image1size.shape
        scale = ROBOFLOW_SIZE / max(height, width)
        image1 = cv2.resize(image1size, (round(scale * width), round(scale * height)))

        # Convert numpy array to PIL.Image
        image1 = Image.fromarray(image1)
        # Save image as JPEG buffer
        buffered = io.BytesIO()
        image1.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

else:
    if page == 'Upload picture':
        uploaded_file = st.file_uploader('Select an image:', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

        ## Pull in default image or user-selected image.
        if uploaded_file is None:
            if st.checkbox("Use test images", value=False):
                option = st.selectbox(
                    'Select an image:',
                    ('im_0000_20230601_124318.jpg', 'im_0001_20230601_124844.jpg', 'im_0002_20230601_124933.jpg', 'im_0003_20230601_125012.jpg', 'im_0004_20230601_125124.jpg'))

                ## Construct the URL 
                url = ''.join([
                    'https://github.com/NoliAlonso/streamlit-bccd/blob/master/BCCD_sample_images/',
                    option,
                    '?raw=true'
                ])
            
                response = requests.get(url)
                image2 = Image.open(io.BytesIO(response.content))
                
                # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
                image2size = np.array(image2)
                height, width, channels = image2size.shape
                scale = ROBOFLOW_SIZE / max(height, width)
                image2 = cv2.resize(image2size, (round(scale * width), round(scale * height)))

                # Convert numpy array to PIL.Image
                image2 = Image.fromarray(image2)
                # Save image as JPEG buffer
                buffered = io.BytesIO()
                image2.save(buffered, format='JPEG')
                img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

                # Or use cv2.imencode to encode image as base64 string
                #img_str = base64.b64encode(cv2.imencode('.jpg', image2)[1]).decode('ascii')
                
        else:
            # User-selected image.
            image2 = Image.open(uploaded_file)

            # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
            image2size = np.array(image2)
            height, width, channels = image2size.shape
            scale = ROBOFLOW_SIZE / max(height, width)
            image2 = cv2.resize(image2size, (round(scale * width), round(scale * height)))

            # Convert numpy array to PIL.Image
            image2 = Image.fromarray(image2)
            # Save image as JPEG buffer
            buffered = io.BytesIO()
            image2.save(buffered, format='JPEG')
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')


st.divider()

if img_str is not None:  # Check if img_str is defined

    ## Subtitle.
    st.write('### Inferenced Image')

    ## Construct the URL to retrieve image.
    # (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
    upload_url = ''.join([
        'https://detect.roboflow.com/',
        ROBOFLOW_MODEL,
        '?api_key=',
        ROBOFLOW_API_KEY,
        '&format=image',
        f'&overlap={overlap_threshold * 100}',
        f'&confidence={confidence_threshold * 100}',
        '&stroke=0',
        '&labels=False'
    ])
    
    ## POST to the API.
    r = requests.post(upload_url,
                      data=img_str,
                      headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })

    if r.ok:
        try:
            image4 = Image.open(io.BytesIO(r.content))

            # Convert to JPEG Buffer.
            buffered = io.BytesIO()
            image4.save(buffered, quality=95, format='JPEG')

            # Display image.
            #st.image(image4, use_column_width=True)

            ## Construct the URL to retrieve JSON.
            upload_url = ''.join([
                'https://detect.roboflow.com/',
                ROBOFLOW_MODEL,
                '?api_key=',
                ROBOFLOW_API_KEY
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
            #st.write('### JSON Output')
            #st.write(r.json())

            draw = ImageDraw.Draw(image4)
            #font = ImageFont.load_default()

            font_path = "Roboto-Regular.ttf"  # Replace with the path to your desired font file
            font_size = 20  # Set the desired font size

            # Load the custom font with the desired size
            font = ImageFont.truetype(font_path, font_size)

            for prediction in output_dict['predictions']:
                color = "#4892EA"
                x1 = prediction['x'] - prediction['width'] / 2
                x2 = prediction['x'] + prediction['width'] / 2
                y1 = prediction['y'] - prediction['height'] / 2
                y2 = prediction['y'] + prediction['height'] / 2

                draw.rectangle([
                    x1, y1, x2, y2
                ], outline=color, width=5)

                if True:
                    text = f"{prediction['class']} ({prediction['confidence']*100:.1f}%)"
                    # Get the bounding box of the text using the loaded font
                    text_bbox = font.getbbox(text)

                    # Calculate the width and height of the text
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # Create a new button image with a larger size
                    button_size = (text_width + 5, text_height + 5)
                    button_img = Image.new('RGBA', button_size, color)

                    # Create a new button_draw object with the larger font
                    button_draw = ImageDraw.Draw(button_img)
                    button_draw.text((1, 1), text, font=font, fill=(255, 255, 255, 255))

                    # put button on source image in position (0, 0)
                    image4.paste(button_img, (int(x1 - 10), int(y1 - 20)))
            st.image(image4, use_column_width=True)

            ###

            # Create a dataframe from the JSON output of the image inference
            df = pd.json_normalize(output_dict['predictions'])            

            if df.empty:
                st.write('# Nothing detected.')

            else:
                # Group by 'class' and get their counts
                df_grouped = df.groupby('class').size().reset_index(name='count')

                # Display the dataframe
                st.dataframe(df_grouped, use_container_width=True, hide_index=True)

                with st.form(key='my_form'):
                    submit_button = st.form_submit_button(label='Add to diff count', on_click=SubmitJSONdataframe)

        except IOError:
            st.write("Error: Failed to open the image from the API response.")
    else:
        st.write("Error: API request failed.")
