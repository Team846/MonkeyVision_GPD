import cv2
import numpy as np
#import pipeline.htmlserver
#from pipeline.visionmain import VisionMain
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg
import os
import scipy.misc as sm
from scipy.ndimage import convolve
#import skimage



# MULTI SCALE RETINEX
# so that the image is in grayscale
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

def get_ksize(sigma):
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8)/0.15) + 2.0)

def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

# def ssr(img, sigma):
#     # Single-scale retinex of an image
#     # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
#     # F = surrounding function, here Gaussian

#     return np.log10(img + 0.00000000000000000001) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)

def ssr(img, sigma):
    # Ensure img is in float format to prevent overflow
    img = img.astype(np.float32)
    
    # Add 1.0 to prevent log(0)
    return np.log10(1.0 + img) - np.log10(1.0 + get_gaussian_blur(img, ksize=0, sigma=sigma))

def msr(img, sigma_scales = [15, 80, 250], low_per = 1, high_per = 1): #multiscale retinex
    # Multi-scale retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)

    # divide MSR by weights of each scale
    # here we use equal weights
    msr = msr / len(sigma_scales)

    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    msr = cv2.normalize(msr, None, 0, 200, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return msr


#CANNY FILL 


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)
    

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    #pi_4 = np.pi / 4
    #pi_2 = np.pi / 2
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                #theta = D[i,j] #* 180 / np.pi #angle in degrees
                #theta_mod = theta % np.pi
                q = 255
                r = 255
                #alpha = None
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                """
                if (0 <= theta_mod < pi_4):
                    alpha = np.abs(np.tan(theta_mod))
                    q = (alpha * img[i + 1, j + 1]) + ((1 - alpha) * img[i, j + 1])
                    r = (alpha * img[i - 1, j - 1]) + ((1 - alpha) * img[i, j - 1]) 
                    
                elif (pi_4 <= theta_mod < pi_2):
                    alpha = np.abs(1./np.tan(theta_mod))
                    q = (alpha * img[i + 1, j + 1]) + ((1 - alpha) * img[i + 1, j])
                    r = (alpha * img[i - 1, j - 1]) + ((1 - alpha) * img[i - 1, j])
                    
                elif (pi_2 <= theta_mod < (3*pi_4)):
                    alpha = np.abs(1./np.tan(theta_mod))
                    q = (alpha * img[i + 1, j - 1]) + ((1 - alpha) * img[i + 1, j])
                    r = (alpha * img[i - 1, j + 1]) + ((1 - alpha) * img[i - 1, j])
                
                elif ((3*pi_4) <= theta_mod < np.pi):
                    alpha = np.abs(np.tan(theta_mod))
                    q = (alpha * img[i + 1, j - 1]) + ((1 - alpha) * img[i, j - 1])
                    r = (alpha * img[i - 1, j + 1]) + ((1 - alpha) * img[i, j + 1])
                """
                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
                

            except IndexError as e:
                pass
    
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    
    M, N = img.shape  
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    
    return img



def Canny_detector(img):
    """ Your implementation instead of skimage """
    
    img_filtered = convolve(img, gaussian_kernel(5, sigma=15.0))
    grad, theta = sobel_filters(img_filtered)
    img_nms = non_max_suppression(grad, theta)
    img_thresh, weak, strong = threshold(img_nms, lowThresholdRatio=0.07, highThresholdRatio=0.19)
    img_final = hysteresis(img_thresh, weak, strong=strong)
   
    return img_final
canny_imgs = []
canny_img = Canny_detector(image)
canny_imgs.append(canny_img)
    

if __name__ == "__main__":
    # vision_main = VisionMain()
    ## server = pipeline.htmlserver.HTMLServer(vision_main)
    # vision_main.execute()
    result = msr(image)
    result = Canny_detector(image)
    cv2.imwrite("result.jpeg", result)
