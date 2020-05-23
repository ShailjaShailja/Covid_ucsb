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

# Some constants 
INPUT_FOLDER = '/media/pkao/Dataset/COVID Dataset/Non-covid PNA/'
OUTPUT_FOLDER = '/media/pkao/Dataset/COVID Dataset/test_sample/'

# INPUT_FOLDER = '/media/pkao/Dataset/COVID Dataset/COVID/'
# OUTPUT_FOLDER = '/media/pkao/Dataset/COVID Dataset/ProcessedCOVID/'

# INPUT_FOLDER = '/media/pkao/Dataset/COVID Dataset/test/'
# OUTPUT_FOLDER = '/media/pkao/Dataset/COVID Dataset/ProcessedOthers/'

patients = os.listdir(INPUT_FOLDER)
print(patients)

# Load the scans in given folder path
def load_scan(path):
    series = [pydicom.filereader.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices = []
#     print(os.listdir(path))
#     print(slices)
    for i in range(len(series)):
        try:
            if(series[i].ImagePositionPatient[2]):
                slices.append(series[i])
        except:
            print("")
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

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing =np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, norm, val  = measure.marching_cubes_lewiner(p, threshold, step_size=1, allow_degenerate=True)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
     #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    # plt.imshow(binary_image[20], cmap=plt.cm.gray)
    # plt.show()
    # plt.imshow(ndimage.binary_dilation(binary_image[20]).astype(binary_image.dtype), cmap=plt.cm.gray)
    # plt.show()
    return ndimage.binary_dilation(binary_image).astype(binary_image.dtype)
    # return binary_image
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

for i in range(len(patients)):
    first_patient = load_scan(INPUT_FOLDER + patients[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    # plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()

    # Show some slice in the middle
    # plt.imshow(first_patient_pixels[20], cmap=plt.cm.gray)
    # plt.show()

    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    # img = normalize(pix_resampled)
    # img = zero_center(img)
    img = np.where(segmented_lungs_fill!=0, pix_resampled, 0)
    # img = np.where(segmented_lungs_fill!=0, img, 0)
    print(np.flip(img.T, 1).shape)
    img = nib.Nifti1Image(np.flip(img.T, 1), np.eye(4))
    os.mkdir(OUTPUT_FOLDER + patients[i])
    nib.save(img, os.path.join(OUTPUT_FOLDER + patients[i],'masked_ct.nii'))
    mask = nib.Nifti1Image(np.flip(segmented_lungs_fill.T, 1), np.eye(4))
    nib.save(mask, os.path.join(OUTPUT_FOLDER + patients[i],'mask.nii.gz'))
