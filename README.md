# Web app for White Blood Cell Identification and Counting using an Object Detection AI Model

Steps to install:
  - Install python version 3.8 or up to 3.12, as required by Streamlit
  - Install streamlit with pip:
    ```
    pip install streamlit
    ```
  - Install the prerequisites:
    ```
    pip install -r requirements.txt
    ```
  - Run the app with:
    ```
     streamlit run streamlit_app.py
    ```
  
This app should launch in the browser and allow you to capture images or upload images then run inference them via Roboflow. 
Unfortunately there are a couple of problems:
  - It is only useful if run as localhost since all modern browsers all camera access only on https.
  - It doesn't work with Raspberry Pi cameras or any CSI camera, but works instantly once a USB camera is plugged in.

Roboflow has deployment options, including local on a Jetson Nano with internet access.

This app was created using [Roboflow](https://roboflow.com) and [Streamlit](https://streamlit.io/).
