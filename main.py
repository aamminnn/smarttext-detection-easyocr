import easyocr
import keras_ocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm

"""
idea is to create a function X
def X(img_name, coordinate, expected_text)
    detect text in coordinate
    if text in coordinate match with expected test:
        result == pass // maybe can also use draw annotation

# NOTE: 
    can draw annotion box change color?
    if can pass use green, fail use red.

# UPDATE: 
    [1] drawAnnotatios function does not support line_color 
    [2] coordinate is not used at the moment
    [3] Live video is not created yet.
# Improvisation:
    [1] save annotated image as result_image. wont do it yet to save space.
    [2] implement live video

"""
image_filename = glob('real_images/*')

def smarttext(image_filename, expected_text:list):
    for image_filename in tqdm(image_filename):
        reader = easyocr.Reader(['en'], gpu = True)
        dfs = []
        result = reader.readtext(image_filename)
        image_id = os.path.splitext(os.path.basename(image_filename))[0]
        image_filename_df = pd.DataFrame(result, columns=['bbox','text','conf'])
        image_filename_df['image_id'] = image_id
        dfs.append(image_filename_df)
        easyocr_df = pd.concat(dfs)

    results = easyocr_df.query('image_id == @image_id')[['text','bbox']].values.tolist()
    print('list type = ',results)
    
    # results = [(x[0], np.array(x[1])) for x in results] #this is required if want to draw box and detected text
    for items in expected_text:
        found_match = False
        for i, result in enumerate(results):
            if items == result[0]:
                results[i] = [(result[0], np.array(result[1]))] #this is required if want to draw box and detected text
                found_match = True
                print(f'match found for {items} in results. PASS')
                keras_ocr.tools.drawAnnotations(plt.imread(image_filename), results[i])
                plt.show()
        if not found_match:
            print(f'No match found for {items} in results. FAIL')
        

smarttext(
    image_filename = image_filename, 
    expected_text = ['BRAVIA VU31','Android TV OS version']
)
