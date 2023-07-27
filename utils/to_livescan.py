import argparse
import cv2
import os
import math
import numpy as np
from tqdm import tqdm
import fingerprint_enhancer
from histogram_equalization import dhe, Ying_2017_CAIP, he
import matplotlib.pyplot as plt

def coherence_filter(img, sigma = 11, str_sigma = 11, blend = 0.5, iter_n = 4):
    '''
    from https://github.com/opencv/opencv/blob/3.2.0/samples/python/coherence.py
    sigma is dim of kernel used in sobel operator
    str_sigma is dim of neighborhood to calculate the covariation matrix of derivatives
    '''
    h, w = img.shape[:2]

    for i in range(iter_n):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]

        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img*(1.0 - blend) + img1*blend)
    return img


# Utility functions to generate gabor filter kernels
# from https://colab.research.google.com/drive/1u5X8Vg9nXWPEDFFtUwbkdbQxBh4hba_M

_sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)
# sigma is adjusted according to the ridge period, so that the filter does not contain more than three effective peaks 
def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period

def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)

def gabor_kernel(period, orientation):
    f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
    f /= f.sum()
    f -= f.mean()
    return f

def from_amt_to_scan(image):
    """heavily from https://colab.research.google.com/drive/1u5X8Vg9nXWPEDFFtUwbkdbQxBh4hba_M"""
    
    fingerprint = image

    # Calculate the local gradient (using Sobel filters)
    gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)

    # Calculate the magnitude of the gradient for each pixel
    gx2, gy2 = gx**2, gy**2

    W = (29, 29) # (23, 23)
    gxx = cv2.boxFilter(gx2, -1, W, normalize = False)
    gyy = cv2.boxFilter(gy2, -1, W, normalize = False)
    gxy = cv2.boxFilter(gx * gy, -1, W, normalize = False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction

    # _region = fingerprint[10:90,80:130]
    center_h = fingerprint.shape[0]//3
    center_w = fingerprint.shape[1]//2
    #_region = fingerprint[center_h-40:center_h+40, center_w-25:center_w+25]
    region = fingerprint[center_h-40:center_h+40, center_w+10:center_w+60]

    # before computing the x-signature, the region is smoothed to reduce noise
    smoothed = cv2.blur(region, (5,5), -1)
    xs = np.sum(smoothed, 1) # the x-signature of the region

    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    # Calculate all the distances between consecutive peaks
    distances = local_maxima[1:] - local_maxima[:-1]

    # Estimate the ridge line period as the average of the above distances
    ridge_period = np.average(distances)

    # Create the filter bank
    or_count = 8
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

    # Filter the whole image with each filter
    # Note that the negative image is actually used, to have white ridges on a black background as a result!!
    nf = 255-fingerprint
    all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)
    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    # Convert to gray scale and apply the mask
    enhanced = np.clip(filtered, 0, 255).astype(np.uint8)

    return enhanced


if __name__ == "__main__":

    # accept and read 2 command line arguments: input path and output path
    parser = argparse.ArgumentParser(
        description="Script to transform fingertips from segmented finger photos into livescan-like images",
        add_help=True,
    )
    parser.add_argument("-i", "--input", help="input path", required=True, type=str)
    parser.add_argument("-o", "--output", help="output path", required=True, type=str)
    args = parser.parse_args()

    inputdir = args.input  # input path
    outputdir = args.output  # output path

    for subdir, dirs, files in os.walk(inputdir):
        # subdir is the current folder, dirs are the subfolders, files are the files in the folder
        for file in tqdm(files):
            
            # create parent path if it doesn't exist
            if not os.path.isdir(outputdir + subdir.split("/", 1)[1]):
                os.makedirs(outputdir + subdir.split("/", 1)[1])

            name = os.path.join(subdir, file)
            image = cv2.imread(name)

            # sharpening 1
            # kernel = np.array([[0, -1, 0],
            #                     [-1, 5,-1],
            #                     [0, -1, 0]])
            # image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()

            # sharpening 2
            # image = cv2.addWeighted(image, 4, cv2.blur(image, (30, 30)), -4, 128)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()

            # enhancement
            # image = Ying_2017_CAIP(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()

            # image = coherence_filter(image, sigma=21, str_sigma=21, iter_n=1)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # gamma correction: mitigates shadow, makes image brighter and improves the visibility of details in darker areas
            image = np.power((image / 255.0), 0.7) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
            # plt.imshow(image, cmap = 'gray')
            # plt.show()

            # hyperparameters too hard to tune, can enhance some block patterns
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16)) # tileGridSize defines the number of equally sized tiles in row and column
            # image = clahe.apply(image)
            # plt.imshow(image, cmap = 'gray')
            # plt.show()

            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 0)
            # plt.imshow(image, cmap = 'gray')
            # plt.show()

            # (247, 410) = average (image.shape[1]//3, image.shape[0]//3)
            image = cv2.resize(image, (247, 410), interpolation=cv2.INTER_AREA) # from_amt_to_scan works on smaller images

            image = from_amt_to_scan(image)
            #plt.imshow(image, cmap = 'gray')
            #plt.show()

            image = fingerprint_enhancer.enhance_Fingerprint(image)
            #plt.imshow(image, cmap = 'gray')
            #plt.show()

            cv2.imwrite(outputdir + subdir.split("/", 1)[1] + "/" + file, image)
    
    print("correctly finished!")

# base approach:
# fingertip -> gray -> amt

# best for us:
# fingertip -> gray -> (gamma) -> amt[21,31] -> crop -> notebook code -> enhance_Fingerprint