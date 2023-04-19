# -*- coding: utf-8 -*-
"""
Classifying blobs into bats or not
----------------------------------
Here I'll run a binary regression on each detected blob to 

@author: theja
"""
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LogisticRegression
import skimage 
from skimage import measure
from skimage.filters import threshold_yen
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))
#%%
folder = 'blob_regression/images/'
image_paths = glob.glob(folder+'*.png')
images = [skimage.io.imread(each) for each in image_paths]
#images = skimage.io.imread_collection(folder+'*.png')

#%%

def remove_vert_horiz_lines(image):
    '''Specifically designed only for 640 x 512 images!!
    Thanks to roadrunner66 for the SO answer https://stackoverflow.com/a/37447304/4955732
    '''
    if image.shape != (512,640):
        raise ValueError(f'Only 512x640 images permitted. Current shape is {image.shape}')
    twod_fft = np.fft.fft2(image)
    shifted = np.fft.fftshift(twod_fft)
    
    clean_fft = shifted.copy()
    clean_fft[256,:] = 0
    clean_fft[:,320] = 0
    clean_img = np.abs(np.fft.ifft2(clean_fft))
    clean_img = np.array(clean_img, dtype=image.dtype)
    return clean_img
cleaned_images = []
print('Cleaning fixed pattern noise ...\n')
for each in tqdm.tqdm(images):
    cleaned_images.append(remove_vert_horiz_lines(each[:,:,2]))
    
#%%

print('Finding blobs...')
all_contours = []
all_regions = []
for i, tgt in tqdm.tqdm(enumerate((cleaned_images))):
    thresh = threshold_yen(tgt, nbins=256)
    binzd = tgt >thresh*0.8
    regions = measure.label(np.uint8(binzd),)
    #contours = measure.find_contours(np.float32(binzd), 0.9)
    #all_contours.append(contours)
    all_regions.append(regions)

msmts = []
for each in all_regions:
    msmts.append(measure.regionprops(each))

#%%
def plot_raw_blobs(image, blob_msmts, ioff=False):
    plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    ax.imshow(image)    
    for i,region in enumerate(blob_msmts):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red',
                                  linewidth=0.5)
        ax.add_patch(rect)
        plt.text(minc, minr,str(i),
                     fontsize=10, color='w')
    ax2 = plt.subplot(122, sharex=ax, sharey=ax)
    ax2.imshow(image)

# if not os.path.exists('.\blobbed'):
#     os.mkdir('blobbed')

# plt.ioff()
# for i, each in tqdm.tqdm(enumerate(images)):
#     plot_raw_blobs(each, msmts[i])
#     plt.savefig(f'blobbed\{i}_marked_blobs')
# plt.ion()
    
    


#%%
# Convert all the measurements into a dataframe

#%%
# Run a logistic regression OR some other kind of classifier
# Circularity, eccentricity and area are likely to play a big role - and be
# independent of how the image is exported. 
# Also mean region pixel value (sum pixels/num pixels) - rel the mean of the whole
# image.(use msmt.coords to get all region pixels)

def make_input_df():
    input_data = pd.DataFrame(data={'file_name':[],
                                'blob_label':[],
                                'unique_label':[],
                                'area':[],
                                'mean_pixel':[],
                                'mean_re_image':[],
                                'perimeter':[],
                                'ecc':[]})
    return input_data

all_frame_msmts = []
for i, input_image in tqdm.tqdm(enumerate(cleaned_images)):
    one_frame_blobs = []
    for props in msmts[i]:
        props_df = pd.DataFrame()
        props_df['file_name'] = [os.path.split(image_paths[i])[-1]]
        props_df['blob_label'] = [props.label]
        props_df['area'] = [props.area]
        majoraxis = props.axis_major_length
        minoraxis = props.axis_minor_length
        props_df['majoraxis'] = [majoraxis]
        props_df['minoraxis'] = [minoraxis]
        try:
            ratio = majoraxis/minoraxis
        except:
            ratio = 0
        props_df['majorminor'] = [ratio]
        xs, ys = props.coords[:,0], props.coords[:,1]
        mean_intensity = np.mean(input_image[xs, ys])
        props_df['mean_pixel'] = [mean_intensity]
        props_df['sd_pixel'] = np.std(input_image[xs, ys])
        props_df['sum_pix'] = np.sum(input_image[xs, ys])
        props_df['mean_re_image'] = [mean_intensity/np.median(input_image)]
        props_df['perimeter'] = [props.perimeter]
        props_df['ecc'] = [props.eccentricity]
        props_df['feret'] = [props.feret_diameter_max]
        props_df['max'] = [np.max(input_image[xs,ys])]
        props_df['min'] = [np.min(input_image[xs,ys])]

        one_frame_blobs.append(props_df)
    one_frame = pd.concat(one_frame_blobs).reset_index(drop=True)
    all_frame_msmts.append(one_frame)
all_blobs = pd.concat(all_frame_msmts).reset_index(drop=True)
all_blobs['logsum_pix'] = np.log10(all_blobs['sum_pix'])
all_blobs['bat'] = np.tile(0,all_blobs.shape[0])
#%%
# Manually mark out the valid ids for each frame as a dict
# If two blobs are on the same bat - and both of them are equal size-ish, 
# I'm choosing both.
onlyfile = [os.path.split(each)[-1] for each in image_paths]
bat_blobs = {}
bat_blobs[onlyfile[0]] = [0, 6]
bat_blobs[onlyfile[1]] = [0, 1, 2, 6]
bat_blobs[onlyfile[2]] = [1,2,3,4,5,6,7]
bat_blobs[onlyfile[3]] = [54, 55, 56, 58, 59, 60, 61, 67, 65]
bat_blobs[onlyfile[4]] = [63, 67, 68, 69, 70, 64, 73, 74, ]
bat_blobs[onlyfile[5]] = [0,1,5,2,6,7,13,4,8,12,9,10,11]
bat_blobs[onlyfile[6]] = [64, 63, 65, 68, 73, 67, 76, 79, 80, 84, 71]
bat_blobs[onlyfile[7]] = [50, 54, 53, 52, 55, 57]
bat_blobs[onlyfile[8]] = [43, 46, 45, 47, 41, 44]
bat_blobs[onlyfile[9]] = [37, 38, 43, 42, 40, 41, 44, 45, 39, 46]
bat_blobs[onlyfile[10]] = [48, 54, 51, 53, 49, 52]
bat_blobs[onlyfile[11]] = [60, 55, 65, 47, 48, 49]


for file, bat_ids in bat_blobs.items():
    rows_w_filename = all_blobs['file_name'] == file
    for each in bat_ids:
        rows_w_bloblabel = all_blobs['blob_label'] == each
        valid_row = np.logical_and(rows_w_filename, rows_w_bloblabel)
        all_blobs.loc[valid_row,'bat'] = 1


#%%
# Now let's perform some logistic regression to classify bat and non-bat
# blobs
xx = all_blobs.loc[:,['area','mean_re_image','logsum_pix','sd_pixel','feret']].to_numpy()
yy = all_blobs['bat'].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(xx, yy,
                                                    test_size=0.2,
                                                    random_state=10)

#%%
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Z = logreg.predict(X_test)

match = jaccard_score(Y_test, Z)
print(f'match: {match}')

#%%
wbat = all_blobs[all_blobs['bat']==1]
wobat = all_blobs[all_blobs['bat']==0]


plt.figure()
colname = 'logsum_pix'
plt.boxplot([wbat[colname], wobat[colname]])


