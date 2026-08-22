def validate_operation(matrix_first, matrix_second, operation):
    height_check = len(matrix_first) == len(matrix_second)
    width_check = len(matrix_first[0]) == len(matrix_second[0])
    if operation == "add" or operation == "subtract":
        return height_check and width_check
    elif operation == "multiply":
        return width_check
    
def add_matrices(matrix_first, matrix_second):
    if not validate_operation(matrix_first, matrix_second, "add"):
        raise ValueError("Matrices are not the same size")
    result_matrix = []
    height = len(matrix_first)
    width = len(matrix_first[0])
    for i in range(height):
        row = []
        for j in range(width):
            row.append(matrix_first[i][j] + matrix_second[i][j])
        result_matrix.append(row)
    return result_matrix

def subtract_matrices(matrix_first, matrix_second):
    if not validate_operation(matrix_first, matrix_second, "subtract"):
        raise ValueError("Matrices are not the same size")
    result_matrix = []
    height = len(matrix_first)
    width = len(matrix_first[0])
    for i in range(height):
        row = []
        for j in range(width):
            row.append(matrix_first[i][j] - matrix_second[i][j])
        result_matrix.append(row)
    return result_matrix
    
def multiply_matrices(matrix_first, matrix_second):
    if not validate_operation(matrix_first, matrix_second, "multiply"):
        raise ValueError("Matrices are not the same size")
    result_matrix = []
    height = len(matrix_first)
    width = len(matrix_second[0])
    for i in range(height):
        row = []
        for j in range(width):
            row.append(matrix_first[i][j] * matrix_second[i][j])
        result_matrix.append(row)
    return result_matrix