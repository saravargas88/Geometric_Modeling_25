
import numpy as np
import igl
import meshplot as mp
from scipy.spatial.transform import Rotation
import scipy
import ipywidgets as iw
import time
import scipy.sparse as sp
from sksparse.cholmod import cholesky


class cholesky_mesh_deformer:
    'To make the mesh deforming easy for the 4 different meshes I make a mesh deformer class. '
    'In order to make the meshes '
    def __init__(self, v, f, labels):
        self.v=v #vertex points from input mesh
        self.f= f #faces from input mesh 
        self.handle_indices= np.where(labels != 0)[0]
        self.free_indices= np.where(labels == 0)[0]
        self.A= None #to compute once and store here
        self.neighbor_edge= np.zeros((v.shape[0],1))  #will be using the same edges in apply detail 


    #Note: capital V is the input vertices that might be deformed
    def extract_smooth_surface(self, V):  
        'minimize the thin plate energy' 
        ' d/dv[ vTLwM^-1Lwv ] = 0 given v constraints' 
        L = igl.cotmatrix(self.v, self.f)           
        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)     
        M_inv = sp.diags(1 / M.diagonal())
        self.A = 2* L @ M_inv @ L              
        
        V_constrained = V[self.handle_indices, :]               
        V_free = np.zeros((self.free_indices.shape[0], 3))

        'submatrices splitting free and constrained'
        Aff = self.A[self.free_indices, :][:, self.free_indices]          
        Afc = self.A[self.free_indices, :][:, self.handle_indices]             
        
        'solve the system: Aff* v_free = -Afc *v_constrained'
        rhs = - Afc * V_constrained 

        'CHOLESKY FACTORIZATION'
        factor = cholesky(Aff)
        
        for i in range(3):
            V_free[:,i]= factor(rhs[:,i])

        'stack the smoothed V with free vertex solutions and constraints'
        V_smoothed = np.zeros_like(V)
        V_smoothed[self.free_indices, :] = V_free
        V_smoothed[self.handle_indices, :] = V_constrained
    
        return V_smoothed

    def encode_details(self, S,B):
        '''
        1. Define the displacement from S to B 
        2. Create orthogonal frame 
            2.1 normal 
            2.2 pick longest edge and project it to tangent plane
            2.3 cross product tangent and normal to make 2nd tangent
        3. Use orthogonal frame to solve for coefficients

        '''
        displacement= S - B
    
        #Define orthogonal reference
        normal= igl.per_vertex_normals(B, self.f)
       
    
        #to get the tangent we look at the adjacent edges 
        #consistently pick the longest edge
        tangent1 = np.zeros_like(normal)
        
        
        adjacency_list= igl.adjacency_list( self.f) 
        current_max_edge=np.zeros((self.v.shape[0], 1))
        # per vertex we pick largest outgoing edge then project on the tangent plane
        for i in range(len(adjacency_list)):
            for j in adjacency_list[i]:
                edge = B[i] - B[j]  # edge vector
                normal_i = normal[i, :]  # normal at vertex i
                
                # Projection: subtract normal component to isolate tangent component
                projection = edge - np.dot(edge, normal_i ) * normal_i
                norm_projection = np.linalg.norm(projection) 
                
                if norm_projection > current_max_edge[i]:
                    tangent1[i, :] = projection / norm_projection  
                    current_max_edge[i] = norm_projection
                    self.neighbor_edge[i] = j #keep track of edge
    
        # other orthogonal tangent vector via cross product
        tangent2 = np.cross( normal,tangent1,  axis=1)  

        #solve for a1, a2, a3: 
        n_vertices = self.v.shape[0]
        coefficients = np.zeros((n_vertices, 3))

        for i in range(n_vertices):
            frame_i = np.stack([normal[i], tangent1[i], tangent2[i]], axis=1)  # (3,3)
            coefficients[i] = np.linalg.solve(frame_i, displacement[i])  # (3,)

    
        return coefficients[:, 0], coefficients[:, 1], coefficients[:, 2]

    #apply details to smoothed bprime surface making a local frame bprime
    def apply_details(self, Bprime, a1,a2,a3 ):
    
        # orthonormal frame for B' : 
        normal_prime = igl.per_vertex_normals(Bprime, self.f)
        tangent1_prime = np.zeros_like(normal_prime)
    
        for i in range(Bprime.shape[0]):
            #same edge picked in extracting detail
        
            neighbor = int(self.neighbor_edge[i])
            
            edge = Bprime[i] - Bprime[neighbor]
            projection = edge - np.dot(edge, normal_prime[i]) * normal_prime[i]
            
            norm_proj = np.linalg.norm(projection)
            
            tangent1_prime[i] = projection / norm_proj
      
        tangent2_prime = np.cross(normal_prime, tangent1_prime, axis=1)
    
    
        n_vertices = Bprime.shape[0]
        details = np.zeros_like(Bprime)

        for i in range(n_vertices):
            frame_i = np.stack([normal_prime[i], tangent1_prime[i], tangent2_prime[i]], axis=1)  # (3,3)
            coeffs_i = np.array([a1[i], a2[i], a3[i]])  
            details[i] = frame_i @ coeffs_i  

        return Bprime + details
    
    




