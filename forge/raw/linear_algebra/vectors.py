import math

# Pure-Python vector ops: geometry and norms behind distances, attention scores, and projections.
# TODO: vector_add(a, b) - Add two vectors element-wise.
# TODO: vector_subtract(a, b) - Subtract one vector from another.
# TODO: scalar_multiply(vector, scalar) - Multiply every element by a scalar.
# TODO: elementwise_multiply(a, b) - Multiply corresponding vector elements (Hadamard product).
# TODO: outer_product(a, b) - Calculate the outer product of two vectors.
# TODO: normalize(vector) - Scale a vector to unit L2 magnitude.
# TODO: cosine_similarity(a, b) - Calculate cosine similarity between vectors.
# TODO: euclidean_distance(a, b) - Calculate Euclidean (L2) distance between vectors.
# TODO: manhattan_distance(a, b) - Calculate Manhattan (L1) distance between vectors.
# TODO: project_vector(a, b) - Project vector a onto vector b.
# TODO: are_orthogonal(a, b) - Check whether two vectors are orthogonal.

def dot_product(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))

def vector_norm(vector, p):
    # L1 norm
    if p == 1:
        return sum(abs(x) for x in vector)
    # L2 norm
    elif p == 2:
        return math.sqrt(sum(x**2 for x in vector))
    else:
        raise ValueError(f"Unsupported norm: {p}")