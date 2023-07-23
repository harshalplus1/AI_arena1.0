# app.py

import streamlit as st
import subprocess
import os
from datetime import datetime


def install_requirements(requirements_file):
    with open(requirements_file, "r") as file:
        for line in file:
            package = line.strip()
            if package and not package.startswith("#"):
                os.system(f"pip install {package}")


def main():
    st.title("FitCV - Your Personal AI Trainer")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 24px; margin: 0;'> <b> Enter your name: </b></p>",
        unsafe_allow_html=True,
    )
    # name of the user is stored in the user_name variable
    flag = 0
    user_name = st.text_input("", key="user_name")
    flag = 1
    st.markdown(
        "<h4> <br> Click the button below to start running the application:</h4>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Launch The App"):
        st.spinner("Launching...")
        # Create a loading text placeholder
        loading_text = st.empty()

        # Update the loading text
        loading_text.text("Installing the packages, Please wait...")
        install_requirements(requirements_file)

        if user_name != "" and flag == 1:
            try:
                with open("output.txt", "a+") as file:
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    file.write(
                        f"===============================================" + "\n"
                    )
                    file.write(f"============={dt_string}===============" + "\n")
                    file.write(f"Name: '{user_name}'" + "\n")
            except Exception as e:
                print(f"error : {e}")
        process = subprocess.Popen(
            ["python", "recognition.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        loading_text.text("Installed the Packages! Launching, Please wait...")

        # Read the output from the subprocess
        output, error = process.communicate()

        if process.returncode == 0:
            st.success("Finished Successfully!")
        else:
            st.error(f"Error launching the script: {error.decode()}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h6>Notes: <br> <br> <ul> <li> Press enter after typing your name in the above text field to save the changes. </li> <br> <li> After clicking on the button, u will get a pop up window, please wait for some time since the packages required to run the program are being installed. </li> <br> <li> After selecting a label, the current window closes and a new window opens with the functionality of the label selected so please wait for a while since the screens take time to load. </li> <br> <li> In case of any error during registration, make sure that your face is clearly visible with sufficient lighting. </li> <br> <li> After using the app, you can check the information about your session in output.txt (inside the FitCV folder), please make sure to open this file using a text editor. </li> <br> <li> Navigate through the screen by hovering over to the lables using your index finger and while choosing a label make sure not to overlap your finger with any other labels so as to ensure error-free working of the program. </li> <br> <li> For jumping jacks and squats, make sure your whole body is visible to get better accuracy. </li> <br> <li> You can end the program by selecting the end label or pressing the 'q' key. </li> </h6>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    requirements_file = "requirements.txt"
    main()
