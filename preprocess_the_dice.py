"""
This runs the various stages of my program  
"""
from os import listdir
from skimage.io import imread, imsave
from dice_processing import center_and_clean, threshold_a_die, crop_dice

#%%
for folder in "123456":
    break#I didn't include these images
    files = [ folder + "/raw/"+file for file in listdir( folder + "/raw")]
    for i,file in enumerate(files):
        im = imread(file)
        dice = crop_dice(im)
        for j,die in enumerate(dice):
            imsave( folder + "/results2/" + "{}.png".format(i+j), die)

#%% threshold a dice to get a better view of the number
for folder in "123456":
    files = [folder+"/results/"+file for file in listdir(folder+"/results")]
    for file in files:
        im = imread(file,as_grey=False)
        im = threshold_a_die(im)
        imsave( folder+"/threshold/"+file.split("/")[-1],im)
        
#%% clean the threshold image
for folder in "123456":    
    files = [folder+"/threshold/"+file for file in listdir(folder+"/threshold")]
    for file in files:
        im = imread(file)
        im = center_and_clean(im)
        imsave(folder+"/better_tests/"+file.split("/")[-1],im)
        
        
