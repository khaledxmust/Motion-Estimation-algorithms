import cv2
import numpy as np
from PIL import Image
from copy import deepcopy 
from resizeimage import resizeimage
from matplotlib import pyplot as plt

#Pre-Processing..
path1 = 'hi1.png'
path2 = 'hi2.png'

img1, img2 = Image.open(path1), Image.open(path2)
img1 = resizeimage.resize_cover(img1, [720, 432], validate=False) # Image Resizing
img2 = resizeimage.resize_cover(img2, [720, 432], validate=False) # Image Resizing
rgb1, rgb2 = np.array(img1), np.array(img2)
rgb1 = cv2.copyMakeBorder(rgb1, 56, 56, 56, 56, cv2.BORDER_CONSTANT, value=0) # Border
rgb2 = cv2.copyMakeBorder(rgb2, 56, 56, 56, 56, cv2.BORDER_CONSTANT, value=0) # Border

#%% Converting Colorspace and Spliting Channels

def RGB2YUV( rgb ): #RGB 2 YUV
    
    m = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                  [-0.14714119, -0.28886916,  0.43601035 ],
                  [ 0.61497538, -0.51496512, -0.10001026 ]])

    yuv = np.dot(rgb, m.T)
    return yuv
Y1, Y2 = RGB2YUV( rgb1 ), RGB2YUV( rgb2 )
x1, x2, x3 = Y1[:,:,0], Y1[:,:,1], Y1[:,:,2] # Splitting Channels Image 1
y1, y2, y3 = Y2[:,:,0], Y2[:,:,1], Y2[:,:,2] # Splitting Channels Image 2
Diff = Y1 - Y2

#plt.imshow(diff.astype(int))
#plt.show()
#%% Split Image to 8x8 Blocks and Computing TSS Motion Vector

def MotionEstimation(Ch1,Ch2):
    Mv = [] # Motion Vector
    for x in range(56, np.shape(Y2)[0]-56, 8):
        for y in range(56, np.shape(Y2)[1]-56, 8):
            SAD = 100000
            xi , xj = 0, 0
            block1 = Ch1[x:x+8,y:y+8]
            Step1 = Ch2[x:x+8,y:y+8]
            Sad = np.sum(np.abs(block1-Step1))
            Psnr = PSNR(block1, Step1)
            if (Psnr > 20):
                SAD = Sad
                xi, xj = x, y
            Mv.append([xi,xj])
    return Mv

Mv = MotionEstimation(x1,y1)

#%% Motion Compensation algorithm
                
P1 = deepcopy(np.zeros(x1.shape))
P2 = deepcopy(np.zeros(x2.shape))
P3 = deepcopy(np.zeros(x3.shape))

def MotionCompensation(pChannel,Channel,Mvec):
    z=0
    for x in range(56, np.shape(Y2)[0]-56, 8):
        for y in range(56, np.shape(Y2)[1]-56, 8):
            pChannel[Mvec[z][0]:(Mvec[z][0])+8,Mvec[z][1]:(Mvec[z][1])+8] = Channel[x:x+8,y:y+8]
            z = z + 1
    return pChannel

#%% Reconstructing the Image & Converting to RGB

P1 = MotionCompensation(P1,x1,Mv)
P2 = MotionCompensation(P2,x2,Mv)
P3 = MotionCompensation(P3,x3,Mv)

for x in range(56, np.shape(Y2)[0]-56, 8):
       for y in range(56, np.shape(Y2)[1]-56, 8):
           a = P1[x:x+8,y:y+8]
           A = P1[x-8:x+16,y-8:y+16]
           B = P2[x-8:x+16,y-8:y+16]
           C = P3[x-8:x+16,y-8:y+16]
           if (np.sum(np.abs(a-np.zeros(a.shape))) == 0 ): 
               A = np.where(A==0, np.mean(np.where(A[np.nonzero(A)])), A)
               A = A.reshape(-1, 9).mean(axis=1).reshape(8,8)
               B = np.where(B==0, np.mean(np.where(B[np.nonzero(B)])), B)
               B = B.reshape(-1, 9).mean(axis=1).reshape(8,8)
               C = np.where(C==0, np.mean(np.where(C[np.nonzero(C)])), C)
               C = C.reshape(-1, 9).mean(axis=1).reshape(8,8)
               P1[x:x+8,y:y+8] = A
               P2[x:x+8,y:y+8] = B
               P3[x:x+8,y:y+8] = C

reim = np.stack((P1, P2, P3))
reim = np.rollaxis(reim, 0, 3)

def YUV2RGB( yuv ): #Returning RGB'
     
    m = np.array([[ 1     ,  0      ,  1.13983 ],
                  [ 1     , -0.39465, -0.58060 ],
                  [ 1     ,  2.03211,  0       ]])
     
    xrgb = np.dot(yuv,m.T)
    return xrgb
xrgb = YUV2RGB( reim )

#%% Computing PSNR 

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

#%% Plotting and Visualization

#Blocks Without Background..
Px = MotionCompensation(np.zeros(x1.shape),x1,Mv) 
Px = YUV2RGB(np.rollaxis(np.stack((Px, P2, P3)), 0, 3))

titles = ['Image 1', 'Image 2', 'Diffrence', 'Blocks']
output = [rgb1, rgb2, Diff, Px]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.title(titles[i])
    if i == 0:
        plt.imshow(output[i].astype(int))
    if i == 1:
        plt.imshow(output[i].astype(int))
    if i == 2:
        plt.imshow(output[i].astype(int))
    if i == 3:
        plt.imshow(output[i].astype(int))
plt.figure(figsize=(6,12))
plt.imshow(xrgb[56:-56,56:-56].astype(int))
plt.title('Predicted Image')
plt.xticks([])
plt.yticks([])
plt.show()

print('  PSNR between Image 2 and Predicted: %.2f ' %PSNR(rgb2,xrgb))