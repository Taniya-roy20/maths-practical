 # practical 1: find cofactors,determinant,adjoint and inverse of a matrix
 # Taking Input of Matrix from user (User Define matrix Input)
# NR: Number of Rows
# NC: Number of Column
import numpy as np

NR = int(input("Enter the number of rows: "))
NC = int(input("Enter the number of Columns: "))

print("Enter the entries in a single line (seperated by space): ")

# User input of entries in a 
# single line seperated by space
entries = list(map(int, input().split()))

# For printing the matrix 
A = np.array(entries).reshape(NR, NC)
print("Matrix A is as follows:", "\n", A)

# For finding the inverse of a matrix
A_Inverse = np.linalg.inv(A)
print("Inverse of A is", "\n", A_Inverse)

# For finding the transpose of a matrix
Transpose_of_A_Inverse = np.transpose(A_Inverse)

print("Transpose of A Inverse is", "\n", Transpose_of_A_Inverse)

# For finding the determinant of a matrix
Determinant_of_A = np.linalg.det(A)
print("Determinant of A is", "\n", Determinant_of_A)

# For finding the cofactor of a Matrix
Cofactor_of_A = np.dot(Transpose_of_A_Inverse, Determinant_of_A)
print("The Cofactor of a Matrix is:", "\n", Cofactor_of_A)

# For finding the Adjoint of a Matrix
Adjoint_of_A = np.transpose(Cofactor_of_A)
print("The Adjoint of a Matrix is:", "\n",Adjoint_of_A)
![OUTPUT 1](https://github.com/user-attachments/assets/d4a429be-5be0-494b-b53a-378cfd99232b)


# practical 2: convert the matrix into echelon form and find its matrix 
import numpy as np
NR=int(input("Enter the number of rows:"))
NC=int(input("Enter the number of columns:"))
print("Enter the enteries in a single line(separated by space")

# user input of enteries in a single line separated by space
enteries=list(map(int,input().split()))

# for printing the matrix
A=np.array(enteries).reshape(NR,NC)
print("Matrix X is as follows:",'\n',A)
print("The Rank of a Matrix:",np.linalg.matrix_rank(A))
![PR-2 OUTPUT](https://github.com/user-attachments/assets/d9a68b78-0a4a-404b-a7ce-914100602636)

# Practical 3: solve the system of equation using gauss elimination method.
import numpy as np
NR=int(input("Enter the number of rows:"))
NC=int(input("Enter the number of columns:"))
print("Enter the elements of coefficient matrix(A)in a single line(separated by space):")
coefficient_enteries=list(map(float,input().split()))
coefficient_matrix=np.array(coefficient_enteries).reshape(NR,NC)
print("coefficient matrix(A) is as follows:",'\n',coefficient_matrix,"\n")

# Column matrix (B) elements
print("Enter the elements of column matrix(B)in a single line(separated by space):")
column_enteries=list(map(float,input().split()))
column_matrix=np.array(column_enteries).reshape(NR,1)
print("column matrix(B) is as follows:",'\n',column_matrix,"\n")

**solution_of_the_system_of_equations=np.linalg.solve(coefficient_matrix,column_matrix)**
print("solution of the system of equaton using Gauss elimination method")
print(solution_of_the_system_of_equations)
![PR-3 OUTPUT](https://github.com/user-attachments/assets/bfd29c04-96e1-4df6-a865-00dc7230e04b)

# Practical 4: solve a system of homogeneous equation using gauss jordon method:
import numpy as np
from numpy import linalg 
# Input for the coefficient matrix (A)
print("Enter the dimension of coefficients matrix (A):")
rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))

print("Enter the elements of coefficients matrix (A) in a single line (separated by space):")
coefficients_entries = list(map(float, input().split()))

# Create the coefficient matrix
coefficient_matrix = np.array(coefficients_entries).reshape(rows, cols)
print("Coefficient Matrix (A) is as follows:", '\n', coefficient_matrix, "\n")

# Input for the column matrix (B)
print("Enter the elements of column matrix (B) in a single line (separated by space):")
column_entries = list(map(float, input().split()))

# Create the column matrix
column_matrix = np.array(column_entries).reshape(rows, 1)
print("Column Matrix (B) is as follows:", '\n', column_matrix, "\n")

# Solution of the system of equations using the inverse of the coefficient matrix
**try:**
    inv_of_coefficient_matrix = linalg.inv(coefficient_matrix)
    solution_of_the_system_of_equations = np.matmul(inv_of_coefficient_matrix, column_matrix)

    print("Solution of the system of equations using the GAUSS JORDAN mathod:")
    print(solution_of_the_system_of_equations)
**except linalg.LinAlgError:**
    print("The coefficient matrix is singular and cannot be inverted.")
**output same as pactical 3**

# practical 5: verify the linear dependence of vectors.
# Verify the linear dependence of vectors.
# Generate a linear combination of given vectors of
# Rn/ matrices of the same size.
import numpy as np

def is_linearly_dependent(vectors):
    # Convert the list of vectors to a numpy matrix (each vector is a column)
    A = np.column_stack(vectors)

    #Perform row reduction (Gaussian elimination)
    rank = np.linalg.matrix_rank(A)

    # If the rank of the matrix is less than the number of vectors, they are linearly dependent
    if rank< A.shape[1]:
        print("The vectors are linearly dependent.")
        # Find a non-trivial solution to the equation A * C = 0
        # Where C is the vector of coefficients
        # Solve A * C = 0 (using least squares)
        C = np.linalg.lstsq(A, np.zeros(A.shape[0]), rcond=None)[0]
        print("Non-trivial linear combination (coefficients):")
        print(C)
    else:
        print("The vectors are linearly independent.")

def generate_linear_combination(vectors, coefficients):
    # Linear combination of vectors based on given coefficients
    result = np.zeros_like(vectors[0])
    for i, vec in enumerate(vectors):
        result += coefficients[i] * vec
    return result

# Example: 3 vectors in R^3 (3D space)
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([3, 6, 9])
vectors = [v1, v2, v3]

# Check if the vectors are linearly dependent
is_linearly_dependent(vectors)

# Example: Generating a linear combination
coefficients = np.array([2, -1, 1]) # coefficients for the linear combination
linear_combination = generate_linear_combination(vectors, coefficients)
print("\nGenerated linear combination:")
print(linear_combination)
![image](https://github.com/user-attachments/assets/a5724800-d672-471a-9666-91f68133afc9)

# PRACTICAL 6:Check the diagonalizable property of matrices and find the corresponding eigenvalue and verify the Cayley- Hamilton theorem
# Check the diagonalizable property of matrices and
# find the corresponding eigenvalue and verify the Cayley-Hamilton theorem.
import numpy as np
from scipy.linalg import eig
from sympy import Matrix, symbols, det

# Function to check diagonalizability
def is_diagonalizable(A):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)

    # Check the geometric multipicity (number of independent eigenvectors)
    # If the number of independent eihgenvectors matches the size of the matrix, it's diagonalizable
    rank = np.linalg.matrix_rank(eigenvectors)
    if rank == A.shape[0]:
        return True, eigenvalues
    else:
        return False, eigenvalues

# Function to verify thr Cayley-Hamilton theorem
def verify_cayley_hamilton(A):
    # Convert matrix to sympy Matrix
    A_sympy= Matrix(A)

    # Compute the characteristics polynomial
    lambda_symbol = symbols('lambda')
    char_poly = A_sympy.charpoly(lambda_symbol)

    #The characteristics polynomail as a sympy expression
    characteristic_polynomial = char_poly.as_expr()

    # Replace lambda with the matrix A in the characteristic polynomial
    A_substitution = characteristic_polynomial.subs(lambda_symbol, A_sympy)

    # Check if the result is the zero matrix (Cayley-Hamilton should hold)
    return A_substitution.is_zero
#Example matrix
A = np.array([[4, -1, 1],
              [-1, 4,-2],
              [1, -2, 3]])

# Step 1: Check if the matrix is diagonalizable
diagonalizable, eigenvalues = is_diagonalizable(A)
print("Is the matrix diagonalizable?", diagonalizable)
print("Eigenvalues of the matrix:", eigenvalues)

# Step 2:Verify Cayley-Hamilton theorem
is_cayley_hamilton_true = verify_cayley_hamilton(A)
print("Does the matrix satisfy the Cayley-Hamilton theorem?", is_cayley_hamilton_true)
![image](https://github.com/user-attachments/assets/fd4c56ce-6996-41a8-8610-33161e65305c)

# practical 7: Compute Gradient of a scalar field, Divergence and Curl of a vector field.
# Compute Gradient of a scalar field, Divergence and curl of a vector field.
import sympy as sp

# Define symbols (coordinates)
x, y, z = sp.symbols('x y z')

# Example Scalar Field f(x, y, z)
f = x**2 + y**2 + z**2

# Example Vector Field A(x, y, z)
A_x = x * y
A_y = y * z
A_z = z * x
A = sp.Matrix([A_x, A_y, A_z])

# 1. Compute the Gradient of the scalar field f
gradient_f = sp.Matrix([sp.diff(f, var) for var in (x, y, z)])
print("Gradient of f(x, y, z):")
sp.pprint(gradient_f)

# 2. Compute the Divergence of the vector field A
divergence_A = sp.diff(A_x, x) + sp.diff(A_y, y) + sp.diff(A_z, z)
print("\nDivergence of f(x, y, z):")
sp.pprint(divergence_A)

# 3. Compute the curl of the vector field A
curl_A = sp.Matrix([
    sp.diff(A_z, y) - sp.diff(A_y, z),  # i-component
    sp.diff(A_x, z) - sp.diff(A_z, x),  # j-component
    sp.diff(A_y, x) - sp.diff(A_x, y)   # k-component
])
print("\nCurl of A(x, y, z):")
sp.pprint(curl_A)
![image](https://github.com/user-attachments/assets/16d74791-60d5-49dd-aef3-ba465c6a6003)

      

      
   









