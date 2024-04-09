import numpy as np
from numpy import linalg as LA
import math as mt
from scipy.linalg import polar
from reduction import process_fiber_network, count_boundary_nodes
from tensor import compute_tensor
import trial_m as mn 

#deformed_co_for_reduced.txt
# Using the first module to compute the reduced stiffness matrix
process_fiber_network('rearranged_mesh_tr.txt', 'deformed_co_for_reduced.txt', 'rearranged_mesh_tr.txt.params')
n_boundary_nodes = count_boundary_nodes('rearranged_mesh_tr.txt', boundary_value=0.5, rtol=1e-5, atol=1e-8)
n_dof = 3

boundary_nodes_file = 'boundary_nodes_coordinates.txt'  # Update with actual path
X = np.loadtxt(boundary_nodes_file)

def compute_RVE_volume_from_coordinates(coordinates):
    """
    Compute the volume of the RVE based on the boundary nodes' coordinates.

    Parameters:
    coordinates - An array of shape (n, 3) where each row is the x, y, z coordinates of a node.

    Returns:
    The computed volume of the rectangular cuboid that bounds the RVE.
    """
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    z_coords = coordinates[:, 2]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

V0 = compute_RVE_volume_from_coordinates(X)
print("Computed RVE Volume V0:", V0)

F = np.array([[2, 1, 0], [0.7, 0.5, 0.1], [0, -0.2, 1]])   # Deformation gradient
#F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 


R, U = polar(F)

# Identity matrix for Kronecker delta representation
identity = np.eye(U.shape[0])

#B = np.zeros((3, 3, 3, 3))

B1 = np.einsum('ks,ql->klsq', identity, U)
B2 = np.einsum('lq,ks->klsq', identity, U)
B3 = np.einsum('ls,qk->klsq', identity, U)
B4 = np.einsum('kq,ls->klsq', identity, U)

B = 0.25*(B1 + B2 + B3 + B4)

# Calculate Mklpq using the provided formula
'''for k in range(3):
    for l in range(3):
        for s in range(3):
            for q in range(3):
                B[k, l, s, q] = 0.25 * (identity[k, s] * U[q, l] + identity[l, q] * U[k, s] + identity[l, s] * U[q, k] + identity[k, q] * U[l, s])

#print("Calculated Mandel 4th order tensor Mklpq:")'''
#print(B)
B_mandel = mn.to_mandel(B)
B_inv = LA.inv(B_mandel)
M = mn.from_mandel(B_inv)
#print(b_det)
#print(B_mandel)
#print(B_inv)
#print(B_inv)


# Using the second module to compute and save the tensor
C_M = compute_tensor('Global_reduced_stiffness_matrix.mtx', 'boundary_nodes_coordinates.txt', n_boundary_nodes, n_dof, V0)
#print("Calculated Material Stiffness Tensor C_M")
#print(C_M)

#save_tensor(C_M, 'C_M_output.txt', 'C_M_tensor.npy', n_dof)
#print(C_M)
#C_M_Matrix = mn.to_mandel(C_M)
#print(C_M_Matrix)
def compute_Aijkl(C_M, F, M, R):
    """
    Compute the tensor Aijkl based on the given equation using Einstein summation.
    """
    F_inv = LA.inv(F)
    term1 = np.einsum('im,jn->ijnm', F_inv, np.eye(F_inv.shape[1]))
    term2 = np.einsum('jm,in->ijnm', F_inv, np.eye(F_inv.shape[1]))

    # Add the terms together
    combined_terms = 0.5*(term1 + term2)
    #M_inv = LA.inv(M)
    # Compute Aijkl using Einstein summation
    # Note: np.einsum automatically handles the Kronecker delta (δ_jn) by repeating indices
    Aijkl = np.einsum('mnpq,ps,klsq,ijnm->ijkl', C_M, R, M, combined_terms)
    return Aijkl


# Compute Aijkl
A = compute_Aijkl(C_M, F, M, R)

def validate_minor_symmetries(A):
    """
    Check if the given fourth-order tensor satisfies the minor symmetries.

    """
    # Checking the first minor symmetry M[i, j, k, l] == M[j, i, k, l]
    first_symmetry = np.allclose(A, A.transpose(1, 0, 2, 3))
    
    # Checking the second minor symmetry M[i, j, k, l] == M[i, j, l, k]
    second_symmetry = np.allclose(A, A.transpose(0, 1, 3, 2))
    
    if first_symmetry and second_symmetry:
        return "Both minor symmetries are satisfied."
    elif first_symmetry:
        return "Only the first minor symmetry is satisfied."
    elif second_symmetry:
        return "Only the second minor symmetry is satisfied."
    else:
        return "None of the minor symmetries are satisfied."
    
validation_result_1 = validate_minor_symmetries(M)
print("symmetry for M")
print(validation_result_1)
validation_result_2 = validate_minor_symmetries(A)
print("symmetry for A")
print(validation_result_2)

A = mn.to_mandel(A)

print(A)
