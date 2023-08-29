import numpy as np 
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, restoration
import tqdm
imgs = io.ImageCollection('cleaned\K1\*.png')[160:170]

# get the entropy profile

all_entropy_imgs = []
for each in tqdm.tqdm(imgs):
    all_entropy_imgs.append(filters.rank.entropy(each, morphology.disk(3)))
all_entropy_imgs = np.array(all_entropy_imgs)
median_entropy = np.median(all_entropy_imgs, axis=0)

#%%
bgsubtr = all_entropy_imgs - median_entropy

idx = 167 - 160
plt.figure()
plt.subplot(131)
plt.imshow(bgsubtr[idx])
plt.subplot(132)
plt.imshow(imgs[idx])

from skimage.filters import threshold_yen

thresh = threshold_yen(bgsubtr[idx])
plt.subplot(133)
plt.imshow(bgsubtr[idx]>thresh)
