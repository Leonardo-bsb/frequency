import cv2
import numpy as np
import matplotlib.pyplot as plt
grey_level=255
height,width=200,200
blank_image=grey_level* np.ones((height,width),dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50,100)
fontScale = 3
color = (0)
thickness = 5
image = cv2.putText(blank_image, 'a', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

print( "Image")
unique0, counts0 = np.unique(image, return_counts=True)
print(unique0,counts0)
# plt.hist(image,bins=30)
# plt.show()
threshold=150
for i in range(height):
     for j in range(width):
        if(blank_image[i,j]>threshold):
               blank_image[i,j]=255
        else:
             blank_image[i,j]=0

# plt.hist(blank_image,bins=30)
# plt.show()

# cv2.imshow('Image',blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import networkx as nx
G = nx.Graph()
for i in range(height):
     for j in range(width):
            G.add_nodes_from([((i,j), {"greylevel": blank_image[i,j]})])

for i in range (height-1):
    for j in range(width-1):
      if(blank_image[i,j]< threshold and blank_image[i,j+1]<threshold):
            G.add_edge((i,j),(i,j+1))
      if(blank_image[i,j] < threshold and blank_image[i+1,j] <threshold):
            G.add_edge((i,j),(i+1,j))
      if(blank_image[i,j] < threshold and blank_image[i+1,j+1] < threshold):      
            G.add_edge((i,j),(i+1,j+1))

i=height-1
for j in range(width-1):
       if(blank_image[i,j] < threshold and blank_image[i,j+1]< threshold):    
            G.add_edge((i,j),(i,j+1))

print(G.number_of_nodes())
print(G.number_of_edges())

#print(list(G.edges))

number_connected_components=nx.number_connected_components(G)
print(number_connected_components)
connected=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
# print(connected)

blank_image_1=grey_level* np.ones((height,width),dtype=np.uint8)

for edge in connected[0]:
    blank_image_1[edge[0],edge[1]]=0


horizontal_image=np.concatenate((blank_image,blank_image_1),axis=1)
cv2.imshow('Horizontal Image',horizontal_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

unique, counts = np.unique(blank_image, return_counts=True)
print( "Blank Image")
print(unique,counts)

print( "Blank Image 1")
unique1, counts1 = np.unique(blank_image_1, return_counts=True)
print(unique1,counts1)
