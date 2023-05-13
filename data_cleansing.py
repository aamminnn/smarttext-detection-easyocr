import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import os

image_filename = glob('images/*')

# plot example image
plt.style.use('ggplot')
img1 = plt.imread(image_filename[0])
plt.imshow(img1) #show first image
img_id = os.path.splitext(os.path.basename(image_filename[0]))[0]
print(img_id)
fig, ax = plt.subplots(figsize=(10,10)) #make image bigger
ax.imshow(plt.imread(image_filename[0]))
plt.show()

# display first 25 images
fig, axs = plt.subplots(1,5,figsize=(20,20))
axs = axs.flatten()
for i in range(len(image_filename)):
    axs[i].imshow(plt.imread(image_filename[i]))
    axs[i].axis('off')
    image_id = os.path.splitext(os.path.basename(image_filename[i]))[0]
    # n_annot = len(annot.query('image_id == @image_id'))
    axs[i].set_title(f'{image_id}')
plt.show()