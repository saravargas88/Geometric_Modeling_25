import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import igl
import triangle as tr
import ipywidgets as iw
import time
import meshplot as mp

class HarmonicDeformer:
    #class to interactively move a ed object by moving a boundary cage!
    #we precompute harmonic weights of the cage points before defomrs
    #apply new cage points to defomr
    def __init__(self, cage_v, cage_f, cage_before_triangulation):
        """
        Parameters:
        - cage_v: array of triangulated cage vertex positions (boundary + interior)
        - cage_f: array of triangle face indices to cage v
        - cage_before_triangulation: (K, 2) original cage (before triangulation)

        Storing: 
        - cage handles (boundary vs)
        - number of handle vertices
        - harmonic_basis: to once computed store harmonic weights
        """
        self.cage_v = cage_v
        self.cage_f = cage_f
        self.cage_handle_indices = np.arange(cage_before_triangulation.shape[0])
        
        self.num_vertices = cage_v.shape[0]
        self.cage_indicis_total= np.arange(self.num_vertices)
        
        
        self.num_boundary = len(self.cage_handle_indices)
        self.harmonic_basis = None 

    def precompute_weights(self):
        '''
        COMPUTES THE HARMONIC BASIS FUNCTION for the cage

        
        We want to solve a Laplace equation: Δphi= 0 over the cage where phi is a scalar function. 
        This is the harmonic functions and we discretize them into a linear system using: 
        - L : cotan weight laplacian matrix 
        - M: mass matrix

        we then solve A = M_inv @ L --> A @ (PHI_MATRIX) = 0

        - partition A into constrained (cage boundary) points and free (interior cage) points

        Boils down to : Afc @ phi_constrained + Aff @ phi_free =0
        FINALLY HARMONIC BASIS: 

        solve for phi_free!  Aff @ phi_f = - Afc @ phi_c  solv
                                   |----|
                                     x 
        Then stack phi_constrained and phi_interior to get the Harmonic basis for the cage points. 
        '''
        boundary_idx = self.cage_handle_indices
        interior_idx = np.setdiff1d(self.cage_indicis_total, boundary_idx)
        
        L = igl.cotmatrix(self.cage_v, self.cage_f)
        M = igl.massmatrix(self.cage_v, self.cage_f, igl.MASSMATRIX_TYPE_VORONOI)
        Minv = sp.diags(1.0 / M.diagonal())
        A = Minv @ L

        Aff = A[interior_idx, :][:, interior_idx]
        Afc = A[interior_idx, :][:, boundary_idx]

        phi_c = np.eye(self.num_boundary)
        rhs = -Afc @ phi_c
        phi_f = sp.linalg.spsolve(Aff, rhs)


        PHI = np.zeros((self.num_vertices, self.num_boundary))
        PHI[boundary_idx, :] = phi_c
        PHI[interior_idx, :] = phi_f

        self.harmonic_basis = PHI
        '''HARMONIC BASIS IS SIZE: VERTICES BY BOUNDARY VERTICES: 
        - EACH ROW CORRESPONDS TO A VERTEX IN THE TRIANGULATED CAGE
        - EACH COLUMN CORRECPONDS TO HARMONIC BASIS FUNCTION 
                - 1 AT BOUNDARY Vi  
                - 0 AT ALL OTHER BOUNDARY Vs 
                - SMOOTH ELSEWHERE           '''

        print(" Harmonic basis computed. Shape:", PHI.shape)

    def deform(self, new_cage_positions):
        if self.harmonic_basis is None:
            raise RuntimeError("Call precompute_weights() first.")
        return self.harmonic_basis @ new_cage_positions

    def compute_weights_for_mesh(self, mesh_vertices_2d):
        num_mesh_v = mesh_vertices_2d.shape[0]
        num_basis = self.harmonic_basis.shape[1]
        print('num of basis' , num_basis)
        
        weights = np.zeros((num_mesh_v, num_basis))

        for i in range(num_mesh_v): #for each mesh vertex: 
            
            p = mesh_vertices_2d[i]
            #express the mesh vertex by a barycentric coordinate of a cage triangle. 
            #loop through all the triangles in the cage and find the relationship
            
            for tri in self.cage_f:
                #tri is a fave in the cage 
                #a,b,c are the vertices in that face 
                a= self.cage_v[tri[0], :2]
                b= self.cage_v[tri[1], :2]
                c= self.cage_v[tri[2], :2]
                #we calculate the barycentr
                bary = barycentric_coordinates(p, a, b, c)
                
                if bary is not None:
                    u, v, w = bary
                    phi_a = self.harmonic_basis[tri[0], :]
                    phi_b = self.harmonic_basis[tri[1], :]
                    phi_c = self.harmonic_basis[tri[2], :]
                    weights[i, :] = u * phi_a + v * phi_b + w * phi_c
                    break
        return weights
