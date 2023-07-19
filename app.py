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
    st.title("FitWiz - Your Personal AI Trainer")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h4>Click the button below to start running the application:</h4>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Launch The App"):
        st.spinner("Launching...")
        process = subprocess.Popen(
            ["python", "menu.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Read the output from the subprocess
        output, error = process.communicate()

        if process.returncode == 0:
            st.success("Finished Successfully!")
        else:
            st.error(f"Error launching the script: {error.decode()}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h6>Note: <br> <br> <ul>  <li> After clicking on the button, u will get a pop up window in a short while, please wait for some time if it doesn't appear immediately. </li> <br> <li> Point your right hand index finger to the window u want to open on the screen, the current window will close and another pop up window will appear having the functionality of that screen, please wait for some time if it doesn't happen immediately. </li> <br> <li> To close the window and stop the application press the 'q' key. </li> </h6>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    requirements_file = "requirements.txt"
    install_requirements(requirements_file)
    main()
