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
threshold=0
for edge in G.edges:
     if(distance(edge)> threshold):
          G.remove_edge(edge[0],edge[1])
print('After edge removal')

number_connected_components=nx.number_connected_components(G)
print(number_connected_components)

connected=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

for i in range(100):
     print('Connected ', i, ':', len(connected[i]))


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


for node in connected[2]:
    blank_image_BGR[node[0],node[1]]=(0,0,255)

for node in connected[3]:
    blank_image_BGR[node[0],node[1]]=(0,255,0)

for node in connected[4]:
    blank_image_BGR[node[0],node[1]]=(0,255,255)





cv2.imshow('Color Image',blank_image_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

rows=set()
colummns=set()
for node in connected[1]:
    rows.add(node[0])
    colummns.add(node[1])
y1=min(rows)
print("min row: ", y1)
y2=max(rows)
print("max row: ", y2)
x1=min(colummns)
print("min col: ", x1)
x2=max(colummns)
print("max col: ", x2)


def roi(image,x1,y1,x2,y2): #x1 row_origin, y1 col origin, x2 row dest,y2,col dest
    for i in range(y1,y2): 
        image[i,x1]=(0,0,255) #first col
        image[i,x2]=(0,0,255) #last col
    for i in range(x1,x2):
        image[y1,i]=(0,0,255) #first row
        image[y2,i]=(0,0,255) #last row

    return image

image_roi= roi(blank_image_BGR,x1,y1,x2,y2)

cv2.imshow('Image with Roi',image_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def generate_object_graph(image,nodes):
#     rows=set()
#     cols=set()
#     step=8
#     for node in nodes:
#         rows.add(node[0])
#         cols.add(node[1])
#     x1=min(cols)
#     x2=max(cols)
#     y1=min(rows)
#     y2=max(rows)
#     
#     vertices=np.zeros(shape=(step,step),dtype=np.uint8)
#     i=0;j=0
#     for row in y_points:
#         for col in x_points:
#             vertices[i,j]=image[row,col]
#             j=j+1
#         i=i+1
#         j=0
#     return vertices

def generate_object_graph(image,nodes):
    points_width=16
    points_height=16
    vertices=np.ones(shape=(points_height,points_width),dtype=np.uint8)
    rows=set()
    cols=set()
    
    for node in nodes:
        rows.add(node[0])
        cols.add(node[1])
    x1=min(cols)
    x2=max(cols)
    y1=min(rows)
    y2=max(rows)
    width=x2-x1+1
    height=y2-y1+1
    object_matriz=np.ones(shape=(height,width),dtype=np.uint8)
    for node in nodes:
        object_matriz[node[0]-y1,node[1]-x1]=image[node[0],node[1]]
        
    
    # x_points=np.floor(np.linspace(x1,x2,step,endpoint=True)).astype(np.uint8)
    # y_points=np.floor(np.linspace(y1,y2,step,endpoint=True)).astype(np.uint8)
    x_points=np.floor(np.linspace(0,width-1,points_width,endpoint=True)).astype(np.uint8)
    y_points=np.floor(np.linspace(0,height-1,points_height,endpoint=True)).astype(np.uint8)
    print('x_points size',np.size(x_points))
    print('y_points size',np.size(y_points))

    # x_step=np.floor((x2-x1)/step).astype(np.uint8)
    # y_step=np.floor((y2-y1)/step).astype(np.uint8)
    for i in range(points_height):
        for j in range(points_width):
            vertices[i,j]=object_matriz[y_points[i],x_points[j]]
    vertices=np.where(vertices==0,1,0)
    return vertices
    

   
        
        
    # return vertices

vertices=generate_object_graph(image,connected[1])
print(vertices)
import pandas as pd

df=pd.DataFrame(vertices)
df.to_csv('out.csv',index=False)
    