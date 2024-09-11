import networkx as nx

def generate_graph_from_image(image):

    height,width=np.shape(image)
    G = nx.Graph()
    for i in range(height):
        for j in range(width):
                G.add_nodes_from([((i,j), {"greylevel": image[i,j]})])


    for i in range (height-1):
        for j in range(1,width-1):
            G.add_edge((i,j),(i,j+1),distance=abs(np.subtract(image[i,j],image[i,j+1])))
            dist=abs(np.subtract(image[i,j],image[i,j+1]))
            G.add_edge((i,j),(i+1,j),distance=abs(np.subtract(image[i,j],image[i+1,j])))
            dist=abs(np.subtract(image[i,j],image[i,j+1]))
            G.add_edge((i,j),(i+1,j+1),distance=abs(np.subtract(image[i,j],image[i+1,j+1])))
            dist=abs(np.subtract(image[i,j],image[i,j+1]))
            G.add_edge((i,j),(i+1,j-1),distance=abs(np.subtract(image[i,j],image[i+1,j-1])))
            dist=abs(np.subtract(image[i,j],image[i,j+1]))

    print('After generating graph')
    print(G.number_of_nodes())
    print(G.number_of_edges())

    return G



def distance(G,edge):
            return G.edges[edge[0],edge[1]]['distance']



def remove_edges(G,threshold):


    number_connected_components=nx.number_connected_components(G)
    print('connected components before edge removal',number_connected_components)

    for edge in G.edges:
        if(distance(G,edge)> threshold):
            G.remove_edge(edge[0],edge[1])
    
    # print('Connected components after edge removal',number_connected_components)
    print('After removing')
    print(G.number_of_nodes())
    print(G.number_of_edges())
    return G




import numpy as np
height,width=10,20
blank_image=np.zeros((height,width),dtype=np.uint8)
blank_image[5,5]=1
blank_image[5,6]=1
H=generate_graph_from_image(blank_image)

threshold=0
G=remove_edges(H,threshold)