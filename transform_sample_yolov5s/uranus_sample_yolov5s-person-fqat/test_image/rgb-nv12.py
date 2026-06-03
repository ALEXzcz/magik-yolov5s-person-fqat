import numpy as np
import os
import cv2
import sys

argvs = (sys.argv)
if len(sys.argv) != 4:
    print ("img_path w h")
    exit(0)
img_path = argvs[1]
w = int(argvs[2])
h = int(argvs[3])

img = cv2.imread(img_path)
img = cv2.resize(img, (w, h))
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

def mergeUV(u, v):
    if u.shape == v.shape:
        uv = np.zeros(shape=(u.shape[0], u.shape[1]*2))
        for i in range(0, u.shape[0]):
            for j in range(0, u.shape[1]):
                uv[i, 2*j] = u[i, j]
                uv[i, 2*j+1] = v[i, j]
        return uv
    else:
        print("size of Channel U is different with Channel V")

def rgb2nv12(image):
    if image.ndim == 3:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        y = (0.299*r+0.587*g+0.114*b)
        u = (-0.169*r-0.331*g+0.5*b+128)[::2, ::2]
        v = (0.5*r-0.419*g-0.081*b+128)[::2, ::2]
        uv = mergeUV(u, v)
        yuv = np.vstack((y, uv))
        return yuv.astype(np.uint8)
    else:
        print("image is not BGR format")

base_name = os.path.splitext(os.path.basename(img_path))[0]
output_file_jpg = f"{base_name}_{w}x{h}_bgr.jpg"
output_file_nv12 = f"{base_name}_{w}x{h}_nv12.yuv"

cv2.imwrite(output_file_jpg, img)
yuv = rgb2nv12(rgb_img)
yuv.tofile(output_file_nv12)

# python rgb-nv12.py bus.jpg 640 640
