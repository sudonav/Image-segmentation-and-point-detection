
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def generate_laplace_filter():
    laplace_filter = [[-1,-1,-1],
                     [-1,8,-1],
                     [-1,-1,-1]]
    return np.asarray(laplace_filter)


# In[3]:


def convolve(patch):
    convolved_patch = [patch[j][i] for j in range(len(patch[0])-1,-1,-1) for i in range(len(patch)-1,-1,-1)]
    return np.asarray([convolved_patch[i:i+len(patch)] for i in range(0, len(convolved_patch), len(patch))])


# In[4]:


def detect_point(porous_image):
    padded_image = np.pad(porous_image, ((1,1),(1,1)), 'edge')
    laplace_image = np.zeros(porous_image.shape)

    laplace_filter = generate_laplace_filter()
    convolved_laplace_filter = convolve(laplace_filter)
    structuring_element_shape = convolved_laplace_filter.shape

    for i in range(len(porous_image)):
        for j in range(len(porous_image[0])):
            patch = padded_image[i:i+structuring_element_shape[0],j:j+structuring_element_shape[1]]
            output = sum([sum([patch[x][y] * convolved_laplace_filter[x][y] for y in range(len(convolved_laplace_filter))]) for x in range(len(convolved_laplace_filter))])
            laplace_image[i][j] = np.absolute(output)  
    return laplace_image


# In[7]:


porous_image = cv.imread("point.jpg",0)
laplace_image = detect_point(porous_image)

maximum = np.max(laplace_image)
T = 0.9 * maximum

output_image = np.zeros(porous_image.shape)

for i in range(len(output_image)):
    for j in range(len(output_image[0])):
        if(laplace_image[i][j] > T):
            output_image[i][j] = 255
            print(i,j)
            cv.circle(porous_image, (j,i), 20, (255,0,0), 3)
        else:
            output_image[i][j] = 0   


# In[9]:


cv.imwrite("laplace_point.jpg",laplace_image)
cv.imwrite("mask_point.jpg",output_image)
cv.imwrite("detect_point.jpg",porous_image)
plt.figure(figsize=(10,8))
plt.imshow(porous_image)


# In[28]:


segment = cv.imread("segment.jpg",0)
segment_color = cv.imread("segment.jpg")
pixels = np.ravel(segment).astype(int)
pixels = pixels[np.where(pixels > 0)]
intensity = np.arange(256)
pixel_intensity = np.zeros(256).astype(int)
for every_pixel in pixels:
    pixel_intensity[every_pixel] += 1


# In[29]:


plt.figure(figsize=(20,8))
plt.xlabel("Gray Scale Intensity")
plt.ylabel("Number of pixels")
plt.bar(intensity,pixel_intensity)
plt.show()


# In[30]:


#Based on the observation the threshold of the object T = 203
T = 203
colors = ['C3' if i == T else 'C0' for i in range(len(intensity))]
plt.figure(figsize=(20,8))
plt.xlabel("Gray Scale Intensity")
plt.ylabel("Number of pixels")
plt.bar(intensity,pixel_intensity, color = colors)
plt.show()


# In[31]:


final_image = ((segment > T).astype(int)*255)
plt.figure(figsize=(20,8))
plt.imshow(final_image)


# In[32]:


min_x = 0
max_x = 0
min_y = 0
max_y = 0
for y in range(len(final_image)):
    for x in range(len(final_image[0])):
        if(final_image[y][x] == 255):
            if(min_x == 0 or x < min_x):
                min_x = x               
            if(max_x == 0 or x > max_x):
                max_x = x
            if(min_y == 0 or y < min_y):
                min_y = y
            if(max_y == 0 or y > max_y):
                max_y = y
                
print(min_x,min_y,max_x,max_y)


# In[34]:


segment_boundary = np.copy(segment_color)
cv.rectangle(segment_boundary, (min_x,min_y-5), (max_x,max_y+5), (255,0,0), 2)
cv.imwrite("segment_detect_obj.jpg",segment_boundary)
plt.figure(figsize=(20,8))
plt.imshow(segment_boundary)

