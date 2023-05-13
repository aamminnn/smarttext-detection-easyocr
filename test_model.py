from tqdm import tqdm
import easyocr
import cv2
import pandas as pd
import os
from glob import glob
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    image_filename = glob('images/*')

    # easyocr
    reader = easyocr.Reader(['en'], gpu = True)

    # Read text from an image file
    # result = reader.readtext('images/000a533ef1b9cacf.jpg')
    # # Print the detected text
    # for detection in result:
    #     print(detection[1])

    dfs = []
    for image_filename in tqdm(image_filename[:5]):
        result = reader.readtext(image_filename)
        image_id = os.path.splitext(os.path.basename(image_filename))[0]
        # image_filename_id = image_filename.split('/')[-1].split('.')[0]
        image_filename_df = pd.DataFrame(result, columns=['bbox','text','conf'])
        image_filename_df['image_id'] = image_id
        dfs.append(image_filename_df)
    easyocr_df = pd.concat(dfs)
    print(dfs)
    print(" above is dfs")
    print(easyocr_df)

    def plot_result(image_filename_fn, easyocr_df):
        easy_results = easyocr_df.query('image_id == @image_id')[['text','bbox']].values.tolist()
        print('list type = ',easy_results)
        easy_results = [(x[0], np.array(x[1])) for x in easy_results]
        keras_ocr.tools.drawAnnotations(plt.imread(image_filename_fn), 
                                        easy_results)
        print("tuple type = ",easy_results)
        plt.show()

    plot_result(image_filename, easyocr_df)