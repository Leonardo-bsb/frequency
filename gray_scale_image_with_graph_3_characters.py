import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_text_image(height,width,text):
    grey_level=255
    # height,width=200,250
    blank_image=grey_level* np.ones((height,width),dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50,100)
    fontScale = 3
    color = (0)
    thickness = 5
    image = cv2.putText(blank_image, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    return image

import networkx as nx

height,width=200,250
text='a b'
image=generate_text_image(height,width,text)

def distance(edge):
            return G.edges[edge[0],edge[1]]['distance']

def generate_graph_from_image(image):

    height,width=np.shape(image)
    G = nx.Graph()
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

    return G

G=generate_graph_from_image(image)

number_connected_components=nx.number_connected_components(G)
print(number_connected_components)



def remove_edges(G,threshold):

    for edge in G.edges:
        if(distance(edge)> threshold):
            G.remove_edge(edge[0],edge[1])
    print('After edge removal')
    return G

number_connected_components=nx.number_connected_components(G)
print(number_connected_components)
def generate_image_connected_components(G):
    connected=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    # for i in range(100):
    #     print('Connected ', i, ':', len(connected[i]))

    image_BGR = np.zeros((height,width,3), np.uint8)
    for node in connected[0]:
        image_BGR[node[0],node[1]]=(255,255,255)

    for node in connected[1]:
        image_BGR[node[0],node[1]]=(255,000,0)


    for node in connected[2]:
        image_BGR[node[0],node[1]]=(0,0,255)

    for node in connected[3]:
        image_BGR[node[0],node[1]]=(0,255,0)

    for node in connected[4]:
        image_BGR[node[0],node[1]]=(0,255,255)

    return image_BGR

threshold=0
G=remove_edges(G,threshold)

image_BGR=generate_image_connected_components(G)

cv2.imshow('Color Image',image_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()


connected=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
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

image_roi= roi(image_BGR,x1,y1,x2,y2)

cv2.imshow('Image with Roi',image_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


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
        
 
    x_points=np.floor(np.linspace(0,width-1,points_width,endpoint=True)).astype(np.uint8)
    y_points=np.floor(np.linspace(0,height-1,points_height,endpoint=True)).astype(np.uint8)
    print('x_points size',np.size(x_points))
    print('y_points size',np.size(y_points))

    for i in range(points_height):
        for j in range(points_width):
            vertices[i,j]=object_matriz[y_points[i],x_points[j]]
    vertices=np.where(vertices==0,1,0)
    return vertices
    


vertices=generate_object_graph(image,connected[1])
print(vertices)
import pandas as pd

df=pd.DataFrame(vertices)
df.to_csv('out.csv',index=False)
    