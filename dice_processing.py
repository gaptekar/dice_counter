"""
The functions that handle the various steps of the image processing
"""
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.transform import resize, rescale, rotate
from skimage.filters import gaussian
from skimage.color import rgb2gray
import numpy as np
from skimage import img_as_ubyte, img_as_float

def show_image(image):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()

def crop_dice(image_color,degree_rotation=0):
    results=[]
    
    image=rgb2gray(image_color)    
    thresh = threshold_otsu(image)
    bw = closing(image < thresh, square(3))
    
    # remove border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    
    regions = [region for region in regionprops(label_image) if region.area >= 100]
    regions = sorted(regions,key= lambda x: x.area, reverse=True)
    
    #ensure that the images size are close to eachother
    regions = [region for region in regions if region.area>regions[1].area*.8 ]
    
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        die=image_color[minr:maxr,minc:maxc]
        if degree_rotation:
            die=rotate(die,degree_rotation,resize=True)
        results.append(np.copy(die))
        
    return results

def threshold_a_die(image):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(5))
    h,w = image.shape
    border=int((h+w)/2*(1/6))
    clear=clear_border(bw,border)
    return img_as_ubyte(clear)

def center_largest_object(image):
    "find the largest object in an image and put it in it's own image"

    label_image=label(image)
    region=max( regionprops(label_image), key=lambda x: x.area)
    minr, minc, maxr, maxc = region.bbox
    h , w = maxr-minr, maxc-minc
    new_size=int(max(h,w)*1.2)
    clean_image=np.zeros((new_size,new_size)).astype(bool)
    r_offset=(new_size-h)//2
    c_offset=(new_size-w)//2
    clean_image[(r_offset):(maxr-minr+r_offset),(c_offset):(maxc-minc+c_offset)]=image[minr:maxr,minc:maxc].astype(bool)
    return clean_image   

def center_and_clean(image,guess_orientation=False,new_size=50):
    clean_image=center_largest_object(image)
    
    if guess_orientation:
        print("why here")
        #roate based on moment
        props=max(regionprops(label(clean_image)), key= lambda x: x.area)
        row,col=props.centroid
        radian=props.orientation
        clean_image=rotate(clean_image,-1*radian*180/3.14159,center=(int(row),int(col)))
        
        
        #try to rotate the image based on its centroid
        props=max(regionprops(label(clean_image)), key= lambda x: x.area)
        row,col=props.centroid
        minr, minc, maxr, maxc = props.bbox
        
        lr_sym=col-((maxc-minc)/2+minc)
        ud_sym=row-((maxr-minr)/2+minr)
        
        if  lr_sym >1.6:
            clean_image=rotate(clean_image,180)
        elif  ud_sym < -1.6:
            clean_image=rotate(clean_image,180)
            
    final_image=center_largest_object(clean_image)
    final_image=resize(final_image/final_image.max(),(new_size,new_size), mode="constant")
    return img_as_float(final_image)
 
if __name__=="__main__":
    image=imread("tests/IMG_4780.jpg")
    for cropped_dice in crop_dice(image,degree_rotation=0)[1:]:
        im=threshold_a_die(rgb2gray(cropped_dice))
        show_image(im)
        im=center_and_clean(im)
        show_image(im)
        im=img_as_float(im)
