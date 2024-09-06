import cv2
import numpy as np
import matplotlib.pyplot as plt
grey_level=255
height,width=200,250
blank_image=grey_level* np.ones((height,width),dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50,100)
fontScale = 3
color = (0)
thickness = 5
image = cv2.putText(blank_image, 'a b', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

# cv2.imshow('Image',blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print( "Image")
# unique0, counts0 = np.unique(image, return_counts=True)
# print(unique0,counts0)


# plt.hist(blank_image,bins=30)
# plt.show()


import networkx as nx




G = nx.Graph()
def distance(edge):
    return G.edges[edge[0],edge[1]]['distance']


for i in range(height):
     for j in range(width):
            G.add_nodes_from([((i,j), {"greylevel": image[i,j]})])


for i in range (height-1):
    for j in range(1,width-1):
        G.add_edge((i,j),(i,j+1),distance=abs(np.subtract(image[i,j],image[i,j+1])))
        G.add_edge((i,j),(i+1,j),distance=abs(np.subtract(image[i,j],image[i+1,j])))
        G.add_edge((i,j),(i+1,j+1),distance=abs(np.subtract(image[i,j],image[i+1,j+1])))
        G.add_edge((i,j),(i+1,j-1),distance=abs(np.subtract(image[i,j],image[i+1,j-1])))


print(G.number_of_nodes())
print(G.number_of_edges())

#print(list(G.edges))

number_connected_components=nx.number_connected_components(G)
print(number_connected_components)
threshold=20
for edge in G.edges:
     if(distance(edge)> threshold):
          G.remove_edge(edge[0],edge[1])
print('After edge removal')

number_connected_components=nx.number_connected_components(G)
print(number_connected_components)

connected=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
# print(connected)

# blank_image_1=grey_level* np.ones((height,width),dtype=np.uint8)

# for edge in connected[0]:
#     blank_image_1[edge[0],edge[1]]=0

# for edge in connected[1]:
#     blank_image_1[edge[0],edge[1]]=255


# horizontal_image=np.concatenate((blank_image,blank_image_1),axis=1)
# cv2.imshow('Horizontal Image',horizontal_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# unique, counts = np.unique(blank_image, return_counts=True)
# print( "Blank Image")
# print(unique,counts)

# print( "Blank Image 1")
# unique1, counts1 = np.unique(blank_image_1, return_counts=True)
# print(unique1,counts1)
blank_image_BGR = np.zeros((height,width,3), np.uint8)
for node in connected[0]:
    blank_image_BGR[node[0],node[1]]=(255,255,255)

for node in connected[1]:
    blank_image_BGR[node[0],node[1]]=(255,000,0)

for n in range(2,80):
    for node in connected[n]:
        blank_image_BGR[node[0],node[1]]=(0,0,255)



cv2.imshow('Color Image',blank_image_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
