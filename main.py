import easyocr
import keras_ocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from test_model import easyocr_df

"""
idea is to create a function X
def X(img_name, coordinate, expected_text)
    detect text in coordinate
    if text in coordinate match with expected test:
        result == pass // maybe can also use draw annotation
        * can draw annotion box change color?
            if can pass use green, fail use red.

Once this finnish, try implement at live video(webcam)

"""
