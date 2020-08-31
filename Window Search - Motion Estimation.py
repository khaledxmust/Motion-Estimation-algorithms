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
#%% Split Image to 8x8 Blocks and 72x72 Windows then Computing Motion Vector

def MotionEstimation(Ch1,Ch2):
    Mv = [] # Motion Vector
    for x in range(0, (np.shape(Y2)[0]), 72):                                   # Window loops
        for y in range(0, (np.shape(Y2)[1]), 72):
            for m in range(x, x+72, 8):                                         # Block loops
                for n in range(y, y+72, 8):
                    SAD = 100000
                    xi , xj = 0, 0
                    block1 = Ch1[m:m+8,n:n+8]
                    for i in range(x, (x+72)-7):                                # Search loops
                        for j in range(y, (y+72)-7):
                            block2 = Ch2[i:i+8,j:j+8]
                            Sad = np.sum(np.abs(block1-block2))
                            if (Sad < SAD):                                     # Minimum SAD
                                SAD = Sad
                                xi , xj = i, j                                  # Coordinates
                    Mv.append([xi,xj])
    return Mv

Mv = MotionEstimation(x1,y1)

#%% Motion Compensation algorithm
                
P1 = deepcopy(x1)
P2 = deepcopy(x2)
P3 = deepcopy(x3)

def MotionCompensation(pChannel,Channel,Mvec):
    z=0
    for x in range(0, (np.shape(Y1)[0]), 72):
        for y in range(0, (np.shape(Y1)[1]), 72):
            for i in range(x, (x+72)-7,8):
                for j in range(y, (y+72)-7,8):
                    pChannel[Mvec[z][0]:(Mvec[z][0])+8,Mvec[z][1]:(Mvec[z][1])+8] = Channel[i:i+8,j:j+8]
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
plt.imshow(xrgb.astype(int))
plt.title('Predicted Image')
plt.xticks([])
plt.yticks([])
plt.show()

print('  PSNR between Image 2 and Predicted: %.2f ' %PSNR(rgb2,xrgb))