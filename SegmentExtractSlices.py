import numpy as np
import pandas as pd 
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
import random
from PIL import Image
from lungmask import mask
import SimpleITK as sitk

INPUT_FOLDER = '/media/pkao/Dataset/COVIDDataset/Others'
OUTPUT_FOLDER = '/media/pkao/Dataset/COVIDDataset/SegmentedOthers231/'

patients = os.listdir(INPUT_FOLDER)
print(len(patients))

def load_scan(path):
    series = [pydicom.filereader.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices = []
#     print(len(series))
#     print(os.listdir(path))
#     print(slices)
    for i in range(len(series)):
        try:
            if(series[i].ImagePositionPatient[2]):
                slices.append(series[i])
        except:
            continue
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

MIN_BOUND = -1250.0
MAX_BOUND = 250
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

for i in range(len(patients)):
    in_filename = str(INPUT_FOLDER+patients[i])
    out_filename = str(OUTPUT_FOLDER+patients[i]+str("_mask.nii"))
    !lungmask $in_filename $out_filename --batchsize 1 --modelname R231CovidWeb

#To get other lung disorder segmented slices
image_path = '/media/pkao/Dataset/COVIDDataset/Others/'
segmented_image_path = '/media/pkao/Dataset/COVIDDataset/SegmentedOthers231/'
masked_image_path = '/media/pkao/Dataset/COVIDDataset/MaskedOthersSlices/'
patients = os.listdir(image_path)
for i in range(len(patients)):
    patient = load_scan(image_path + patients[i])
    patient_pixels = get_pixels_hu(patient)
    patient_pixels = normalize(patient_pixels)
    print(patient_pixels.shape)
    seg_patient = nib.load(segmented_image_path + patients[i]+"_mask.nii")
    seg_patient_pixels = seg_patient.get_fdata()
    print(seg_patient_pixels.shape)
    patient_pixels_trans = np.zeros(np.shape(seg_patient_pixels))
    for l in range(patient_pixels.shape[0]):
        for m in range(patient_pixels.shape[1]):
            for n in range(patient_pixels.shape[2]):
                patient_pixels_trans[m][n][l] = patient_pixels[l][n][m]
    print(patient_pixels_trans.shape)
    masked_image = np.where(seg_patient_pixels!=0, patient_pixels_trans, 0)
    print(masked_image.shape)
    for j in range(masked_image.shape[2]):
        if(np.any(masked_image[:,:,j])):
#             im = Image.fromarray(masked_image[:,:,j])
#             im.save(str(masked_image_path)+ patients[i]+'_slice_'+str(j)+'.nii')
            img = nib.Nifti1Image(np.flip(masked_image[:,:,j]), np.eye(4))
            nib.save(img, os.path.join(masked_image_path, patients[i]+'_slice_'+str(j)+'.nii'))
            if(patients[i] in x_train):
                list_train.append(str(patients[i]+'_slice_'+str(j)+'.nii'))
            elif(patients[i] in x_valid):   
                list_val.append(str(patients[i]+'_slice_'+str(j)+'.nii') )               
            else:
                list_test.append(str(patients[i]+'_slice_'+str(j)+'.nii'))
