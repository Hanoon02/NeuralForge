import math
# Eigenvalues, eigenvectors, and SVD: foundations of PCA, spectral methods, and matrix factorizations.
# TODO: power_iteration(A, max_iterations, tolerance) - Estimate the dominant eigenvalue and eigenvector.
# TODO: eigen_decomposition(A) - Calculate eigenvalues and eigenvectors of a symmetric matrix.
# TODO: singular_value_decomposition(A) - Decompose a matrix into U, Σ, and Vᵀ.
# TODO: reconstruct_from_svd(U, S, VT) - Reconstruct a matrix from its SVD factors.
# TODO: truncate_svd(U, S, VT, k) - Keep the top-k singular components (low-rank / PCA truncation).
# TODO: explained_variance_ratio(singular_values) - Fraction of variance captured by each component.

'''
Fundamentals:

1. Eigenvector, is a vector whose direction does not change when a matrix transformation is applied to it.
2. Eigenvalue, is a scalar that scales the eigenvector.
'''


'''
How to compute eigenvalue of a matrix?
1. The characteristic equation is det(A-λI) = 0
2, Solving this gives you two values of λ, which are the eigenvalues of the matrix.

Being code, and in python, 
'''

def eigenvalues_2x2(A):
    
    # Unpacking the values for simplicity
    a, b = A[0]
    c, d = A[1]
    
    # Solving for det(A-λI) = 0 is same as solving for det(A) = λ^2 - (a+d)λ + (ad-bc) = 0
    # Plugging this into the quadratic formula, we get:
    # λ = (a+d) ± sqrt((a+d)^2 - 4(ad-bc)) / 2
    # Now a+d is and will be called as trace and ad-bc will be called as determinant
    # Lets define discriminant as (a+d)^2 - 4(ad-bc), which becomes trace^2 - 4*determinant
    # Finally the solution becomes:
    # λ = trace ± sqrt(discriminant) / 2, splitting it, gives us two solutions,
    # λ1 = trace + sqrt(discriminant)) / 2
    # λ2 = trace - sqrt(discriminant)) / 2
    
    
    trace = a + d
    determinant = a*d - b*c
    discriminant = trace**2 - 4*determinant
    lambda1 = (trace + math.sqrt(discriminant)) / 2
    lambda2 = (trace - math.sqrt(discriminant)) / 2
    return lambda1, lambda2
    
    


'''
To get eigenvectors, its not the same having a finite set of results, as was in case of eigenvalues.

1. For each eigenvalue, solve the system of linear equations (A-λI)v = 0
2. v for a 2x2 is [x, y]
3. When solving this, we always take Y=1 and then compute the X value, which is x = -b/a or x = -d/c
4. Finally, we get the eigenvector as [X, 1]
5. There is an eigenvector of that type for all equations.

Consider this as a base set of eigenvalues, thats it.

There has to be specific handling for division by zero issue, we can either use the non zero equation result, or use [1, 0] as a fallback.

'''

def eigenvector(A, eigenvalues):
    a, b = A[0]
    c, d = A[1]
    
    eigenvectors = []
    for eigenvalue in eigenvalues:
        temp_a = a - eigenvalue
        temp_d = d - eigenvalue
        temp_b, temp_c = b, c
        y_1 = 1
        y_2 = 1

        
        if temp_a != 0 and temp_c != 0:
            x_1 = - temp_b / temp_a
            x_2 = - temp_d / temp_c
            eigenvectors.extend([[x_1, y_1], [x_2, y_2]])
        elif temp_a != 0:
            x_1 = -temp_b / temp_a
            eigenvectors.append([x_1, y_1])
        elif temp_c != 0:
            x_2 = -temp_d / temp_c
            eigenvectors.append([x_2, y_2])
        else:
            eigenvectors.append([1, 0])
        
    return eigenvectors



'''
QR Decomposition:
We can represent any square matrix as a product of an orthogonal matrix and an upper triangular matrix.

ie, A = Q * R, where Q is orthogonal and R is upper triangular.

'''

def dot_product(vector_a, vector_b): # Used repetitively for QR decomposition
    return sum(a * b for a, b in zip(vector_a, vector_b))
def vector_norm(vector): # Used repetitively for QR decomposition
    return math.sqrt(dot_product(vector, vector))

def qr_decomposition(A):
    row_count = len(A)
    column_count = len(A[0])
    
    orthonormal_vectors = [] # The matrix formed from this will be the Q matrix
    
    # Orthonormal vectors are computed using Gram-Schmidt process
    # Its done column by column
    # Actual method involves creating orthonormal for first column
    # Then subtracting its projection from the second column and normalizing the result
    # This process is repeated for each column
    # Finally, the orthonormal vectors are normalized to form the Q matrix
    
    # In reality, this can be compressed to a single formula, which is:
    # u_j = v_j - sum(dot_product(u_i, v_j) * u_i for i in range(j))
    # Where u_j is the orthonormal vector and v_j is the original vector
    
    for column_index in range(column_count):
        original_column = [A[row_index][column_index] for row_index in range(row_count)]
        orthonormal_column = original_column.copy()
        
        for previous_orthonormal_vector in orthonormal_vectors: # Running the summarized equation
            projection_coefficient = dot_product(previous_orthonormal_vector, original_column)
            orthonormal_column = [orthonormal_column[i] - projection_coefficient * previous_orthonormal_vector[i] for i in range(row_count)]
            
        orthogonal_normalized_column = vector_norm(orthonormal_column)
        orthonormal_vector = [column / orthogonal_normalized_column for column in orthonormal_column] # Normalizing the result
        orthonormal_vectors.append(orthonormal_vector) # Adding the result to the list
        
    Q_matrix = [
        [orthonormal_vectors[column_index][row_index] for column_index in range(column_count)]
        for row_index in range(row_count)
    ]
    
    R_matrix = [[0.0] * column_count for _ in range(column_count)] # The matrix will be the upper triangular matrix, with correct values updated

    # Mathematically, R_matrix = Q_matrix(transpose) * A
    # But each entry of R_matrix, ie r_ij = dot_product(Q_matrix[i], A[i])
    
    for i in range(column_count):
        for j in range(column_count):
            q_vector = [Q_matrix[row_index][i] for row_index in range(row_count)]
            a_vector = [A[row_index][j] for row_index in range(row_count)]
            R_matrix[i][j] = dot_product(q_vector, a_vector)
 
    
    return Q_matrix, R_matrix
    