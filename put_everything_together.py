"""
The model is tested against some 
"""


from skimage.io import imread
from skimage.color import rgb2gray
from os import listdir
import numpy as np
import pickle
from dice_processing import crop_dice, center_and_clean, threshold_a_die, show_image


#%%
clf = pickle.load(open( "model_with_rotations_-25to25.pkl", "rb" ))# trained in classify dice
tests_folder="tests"
tests=listdir(tests_folder)
for file in tests:
    image=imread(tests_folder+"/"+file)
    cropped_dice=crop_dice(image,degree_rotation=0)
    results=[]
    for image in cropped_dice:
        im=threshold_a_die(rgb2gray(image))
        #show_image(im)
        im=center_and_clean(im)
        #show_image(im)
        results.append(clf.predict(im.reshape(1,-1)))
    results=np.stack(results).astype(int)
    print(f"the dice in {file} are {results.tolist()}, the total is {results.sum()}")        
        
#%%

clf = pickle.load(open( "model_rf.pkl", "rb" ))# trained in classify dice
tests_folder="tests"
tests=listdir(tests_folder)
for file in tests:
    image=imread(tests_folder+"/"+file)
    cropped_dice=crop_dice(image,degree_rotation=0)
    results=[]
    for image in cropped_dice:
        im=threshold_a_die(rgb2gray(image))
        #show_image(im)
        im=center_and_clean(im)
        #show_image(im)
        results.append(clf.predict(im.reshape(1,-1)))
    results=np.stack(results).astype(int)
    print(f"the dice in {file} are {results.tolist()}, the total is {results.sum()}")        
        
#%%