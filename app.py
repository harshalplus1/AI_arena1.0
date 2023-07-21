# app.py

import streamlit as st
import subprocess
import os


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
    user_name = st.text_input("", key="user_name")
    try:
        with open('output.txt', 'a+') as file:
            file.write(f"Name: '{user_name}'" + "\n")
    except Exception as e:
        print(f"error : {e}")
    st.markdown(
        "<h4> <br> Click the button below to start running the application:</h4>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Launch The App"):
        st.spinner("Launching...")
        install_requirements(requirements_file)
        process = subprocess.Popen(
            ["python", "recognition.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Read the output from the subprocess
        output, error = process.communicate()

        if process.returncode == 0:
            st.success("Finished Successfully!")
        else:
            st.error(f"Error launching the script: {error.decode()}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h6>Note: <br> <br> <ul> <li> After clicking on the button, u will get a pop up window, please wait for some time since the program takes time to load. </li> <br> <li> Navigate through the screen by hovering over to the lables using your index finger and while choosing a label make sure not to overlap your finger with any other labels so as to ensure error-free working. </li> <br> <li> For jumping jacks and squats, make sure your whole body is visible to get better accuracy. </li> <br> <li> You can end the program by choosing the end label or pressing the 'q' key. </li> <br> <li> Please wait after selecting a label as the screens take time to load. </li> </h6>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    requirements_file = "requirements.txt"
    main()
