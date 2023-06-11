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

def AddNeutrophil():
    if 'Neutrophil' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['Neutrophil'] = 1 

def AddLymphocyte():
    # Check if the button is clicked
    if 'Lymphocyte' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['Lymphocyte'] = 1 

def AddMonocyte():
    # Check if the button is clicked
    if 'Monocyte' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['Monocyte'] = 1 

def AddEosinophil():
    if 'Eosinophil' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['Eosinophil'] = 1 

def AddBasophil():
    if 'Basophil' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['Basophil'] = 1 

def AddNRBC():
    if 'NRBC' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['NRBC'] = 1 

def AddBlast():
    if 'Blast' not in st.session_state.class_counts:
        # Add the cell name to the session state dictionary with a count of 1
        st.session_state.class_counts['Blast'] = 1 

def ResetAll():
    st.session_state.class_counts = {}
    st.session_state.last_updated = datetime.datetime.now().ctime()

#########

#def process_image(image):
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    #image_size = np.array(image)
    #height, width, channels = image_size.shape
    #scale = ROBOFLOW_SIZE / max(height, width)
    #resized_image = cv2.resize(image_size, (round(scale * width), round(scale * height)))

    ## Convert numpy array to PIL.Image
    #processed_image = Image.fromarray(resized_image)
    #return processed_image

def process_image(image):
    # Convert PIL image to OpenCV format (numpy array)
    cv_image = np.array(image)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to obtain a binary image
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the main object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding rectangle coordinates
    cropped_image = cv_image[y:y+h, x:x+w]

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, _ = cropped_image.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    resized_image = cv2.resize(cropped_image, (round(scale * width), round(scale * height)))

    # Convert OpenCV image to PIL format
    processed_image = Image.fromarray(resized_image)

    return processed_image

def encode_image(image):
    # Save image as JPEG buffer
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
    return img_str

def take_picture():
    img_file_buffer = st.camera_input("Take a picture:")

    if img_file_buffer is not None:
        # To read image file buffer with PIL:
        image = Image.open(img_file_buffer)
        processed_image = process_image(image)
        img_str = encode_image(processed_image)
        return img_str

def upload_picture():
    uploaded_file = st.file_uploader('Select an image:', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    if uploaded_file is not None:
        # User-selected image.
        image = Image.open(uploaded_file)
        processed_image = process_image(image)
        img_str = encode_image(processed_image)
        return img_str

    if st.checkbox("Use test images", value=False):
        option = st.selectbox(
            'Select a sample image:',
            ('im_0000_20230601_124318.jpg', 'im_0001_20230601_124844.jpg', 'im_0002_20230601_124933.jpg', 'im_0003_20230601_125012.jpg', 'im_0004_20230601_125124.jpg', 'IMG_20200206_062929.jpg')
        )

        # Construct the URL 
        url = f'https://github.com/NoliAlonso/streamlit-bccd/blob/master/BCCD_sample_images/{option}?raw=true'
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        processed_image = process_image(image)
        img_str = encode_image(processed_image)
        return img_str


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
    NRBC_count = 0

    colu1, colu2 = st.sidebar.columns([0.2, 0.8])
    with colu1:
        st.write(' ')
    with colu2:
        # Display the total in the sidebar
        st.write('Total = ', DiffCountTotal)

        # Subtract the count of 'NRBC' from the total count
        if 'NRBC' in dfCount['Cell'].values:
            NRBC_count = dfCount.loc[dfCount['Cell'] == 'NRBC', 'Count'].values[0]

        if NRBC_count > 0:
            st.write('NRBCs = ', NRBC_count)
            DiffCountTotal -= NRBC_count

            if DiffCountTotal >= 100:
                st.write('Please enter uncorrected WBC Count:')
                uncorrected_wbc_count = st.number_input('Uncorrected WBC Count', min_value=0)
                submit_button = st.button('Submit')

                if submit_button:
                    corrected_wbc_count = uncorrected_wbc_count / (NRBC_count / 100)
                    st.write('Corrected WBC Count = ', corrected_wbc_count)


        st.write('WBCs = ', DiffCountTotal)        

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

    st.sidebar.write('Count last updated at ', st.session_state.last_updated)
else:
    st.sidebar.write('Add cell types or inference an image to begin.');

with st.sidebar.form(key='resetform'):
    reset_button = st.form_submit_button(label='Reset', on_click=ResetAll)

########## Logos

imageLogo = Image.open('./images/roboflow_logo.png')
st.sidebar.image(imageLogo, use_column_width=True)

imageLogo = Image.open('./images/streamlit_logo.png')
st.sidebar.image(imageLogo, use_column_width=True)

imageLogo = Image.open('./images/NoliAlonsoPathLabSystemsLogo.png')
st.sidebar.image(imageLogo, use_column_width=True)

st.sidebar.write('Disclaimer: As is, for research purposes only.')

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
    img_str = take_picture()
elif page == 'Upload picture':
    img_str = upload_picture()

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

                st.divider()

        except IOError:
            st.write("Error: Failed to open the image from the API response.")
    else:
        st.write("Error: API request failed.")

else:
    st.divider()

st.write('Add a cell to the list:')
#add classes, classname - button that adds 1 

colc1, colc2, colc3, colc4, colc5, = st.columns(5)

with colc1:
    imageLogo = Image.open('./images/Neutrophil.png')
    st.image(imageLogo)    
    st.button(label='Neutrophil', on_click=AddNeutrophil)  

with colc2:
    imageLogo = Image.open('./images/Lymphocyte.png')
    st.image(imageLogo)
    st.button(label='Lymphocyte', on_click=AddLymphocyte)

with colc3:
    imageLogo = Image.open('./images/Monocyte.png')
    st.image(imageLogo)
    st.button(label='Monocyte', on_click=AddMonocyte)

with colc4:
    imageLogo = Image.open('./images/Eosinophil.png')
    st.image(imageLogo)
    st.button(label='Eosinophil', on_click=AddEosinophil)

with colc5:
    imageLogo = Image.open('./images/Basophil.png')
    st.image(imageLogo)
    st.button(label='Basophil', on_click=AddBasophil)

colc6, colc7 = st.columns(2)

with colc6:
    imageLogo = Image.open('./images/NRBC.png')
    st.image(imageLogo)
    st.button(label='NRBC', on_click=AddNRBC)

with colc7:
    imageLogo = Image.open('./images/Blast.png')
    st.image(imageLogo)
    st.button(label='Blast', on_click=AddBlast)