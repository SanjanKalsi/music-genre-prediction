# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Music Genre Prediction",
        page_icon="🎶",
    )

    st.write("# Music Genre Prediction App")

    st.sidebar.success("Select an option above.")

    st.markdown(  
        """
        In this Aapplication we'll predict the genre of a song. Using the file uploaded we must ensure 
        that the correct genre of the song/audio is predicted using a machine learning model.


        **👈 Select 'Upload MP3 File' tab from the sidebar**
        to upload music and predict its genre.


        ### Business/Real-world impact of solving the problem & its importance
        - Classification and prediction of a genre of a song can be helpful in creating recommender systems for music that is hosted on an app available to the end user.
        - This would help in creating logic which can be used by a streaming app to create a personalized playlist for anyone using the app.
        - The objective of automating this process of music classification is to select songs quicker and easier, rather than doing it manually. One has to listen to the whole song if one has to classify the song and then select a genre.
        - This is a highly tedious task and takes a toll on the time spent in the process, and a whole lot of difficulty as well. Thus, automating the task of music genre classification can help find other invaluable data such as popular genres, trends and popular artists.
    """
    )


if __name__ == "__main__":
    run()
