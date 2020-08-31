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
            Step1 = [Ch2[x:x+8,y:y+8], Ch2[x:x+8,y+(4*8):y+(4*8)+8], Ch2[x+(2*8):x+(2*8)+8,y+(2*8):y+(2*8)+8], Ch2[x+(4*8):x+(4*8)+8,y:y+8], Ch2[x+(2*8):x+(2*8)+8,y-(2*8):y-(2*8)+8], Ch2[x:x+8,y-(4*8):y-(4*8)+8], Ch2[x-(2*8):x-(2*8)+8,y-(2*8):y-(2*8)+8], Ch2[x-(4*8):x-(4*8)+8,y:y+8], Ch2[x-(2*8):x-(2*8)+8,y+(2*8):y+(2*8)+8]]
            for i in range(0,5): # Using a grid of (9) indices & 4 steps block search
                Sad = np.sum(np.abs(block1-Step1[i]))
                if (Sad<SAD):
                    SAD = Sad
                    if i == 0: xi, xj = x, y
                    if i == 1: xi, xj = x, y+(4*8)
                    if i == 2: xi, xj = x+(2*8), y+(2*8)
                    if i == 3: xi, xj = x+(4*8), y
                    if i == 4: xi, xj = x+(2*8), y-(2*8)
                    if i == 5: xi, xj = x, y-(4*8)
                    if i == 6: xi, xj = x-(2*8), y-(2*8)
                    if i == 7: xi, xj = x-(4*8), y
                    if i == 8: xi, xj = x-(2*8), y+(2*8)
                    Step2 = [Ch2[x:x+8,y:y+8], Ch2[x:x+8,y+(2*8):y+(2*8)+8], Ch2[x+(8):x+(8)+8,y+(8):y+(8)+8], Ch2[x+(2*8):x+(2*8)+8,y:y+8], Ch2[x+(8):x+(8)+8,y-(8):y-(8)+8], Ch2[x:x+8,y-(2*8):y-(2*8)+8], Ch2[x-(8):x-(8)+8,y-(8):y-(8)+8], Ch2[x-(2*8):x-(2*8)+8,y:y+8], Ch2[x-(8):x-(8)+8,y+(8):y+(8)+8]]
                    for j in range(0,5): # Using a grid of (9) indices & 4 steps block search
                        Sad = np.sum(np.abs(block1-Step2[j]))
                        if (Sad<SAD):
                            SAD = Sad
                            if j == 0: xi, xj = xi, xj
                            if j == 1: xi, xj = xi, xj+(2*8)
                            if j == 2: xi, xj = xi+(8), xj+(8)
                            if j == 3: xi, xj = xi+(2*8), xj
                            if j == 4: xi, xj = xi+(8), xj-(8)
                            if j == 5: xi, xj = xi, xj-(2*8)
                            if j == 6: xi, xj = xi-(8), xj-(8)
                            if j == 7: xi, xj = xi-(2*8), xj
                            if j == 8: xi, xj = xi-(8), xj+(8)
                            Step3 = [Ch2[x:x+8,y:y+8], Ch2[x:x+8,y+(2*8):y+(2*8)+8], Ch2[x+(8):x+(8)+8,y+(8):y+(8)+8], Ch2[x+(2*8):x+(2*8)+8,y:y+8], Ch2[x+(8):x+(8)+8,y-(8):y-(8)+8], Ch2[x:x+8,y-(2*8):y-(2*8)+8], Ch2[x-(8):x-(8)+8,y-(8):y-(8)+8], Ch2[x-(2*8):x-(2*8)+8,y:y+8], Ch2[x-(8):x-(8)+8,y+(8):y+(8)+8]]
                            for k in range(0,5): # Using a grid of (9) indices & 4 steps block search
                                Sad = np.sum(np.abs(block1-Step3[k]))
                                if (Sad<SAD):
                                    SAD = Sad
                                    if k == 0: xi, xj = xi, xj
                                    if k == 1: xi, xj = xi, xj+(8)
                                    if k == 2: xi, xj = xi, xj
                                    if k == 3: xi, xj = xi+(8), xj
                                    if k == 4: xi, xj = xi, xj
                                    if k == 5: xi, xj = xi, xj-(8)
                                    if k == 6: xi, xj = xi, xj
                                    if k == 7: xi, xj = xi-(8), xj
                                    if k == 8: xi, xj = xi, xj
            Mv.append([xi,xj])
    return Mv

Mv = MotionEstimation(x1,y1)

#%% Motion Compensation algorithm
                
P1 = deepcopy(x1)
P2 = deepcopy(x2)
P3 = deepcopy(x3)

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