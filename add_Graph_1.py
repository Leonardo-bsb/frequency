import cv2
import numpy as np
grey_level=255
height,width=10,10
blank_image=grey_level* np.ones((height,width),dtype=np.uint8)
blank_image[0,0]=100
blank_image[5,5]=100
blank_image[5,6]=100
blank_image[6,5]=100
blank_image[7,7]=100
blank_image[8,8]=100
print(blank_image)

import networkx as nx
G = nx.Graph()
for i in range(height):
     for j in range(width):
            G.add_nodes_from([((i,j), {"greylevel": blank_image[i,j]})])
            
for i in range (height-1):
    for j in range(width-1):
      if(blank_image[i,j]!= 255 and blank_image[i,j+1]!=255):
            G.add_edge((i,j),(i,j+1))
      if(blank_image[i,j]!= 255 and blank_image[i+1,j]!=255):
            G.add_edge((i,j),(i+1,j))
      if(blank_image[i,j]!= 255 and blank_image[i+1,j+1]!=255):      
            G.add_edge((i,j),(i+1,j+1))

i=height-1
for j in range(width-1):
       if(blank_image[i,j]!= 255 and blank_image[i,j+1]!=255):    
            G.add_edge((i,j),(i,j+1))

print(G.number_of_nodes())
print(G.number_of_edges())

#print(list(G.edges))
connected=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print(connected)
            


