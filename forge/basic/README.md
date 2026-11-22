# Basic

The Basic section essentially focuses on using standard packages like NumPy, Pandas, and Matplotlib to address the requirements of ML/DL applications.

## When to use NumPy?

- When an operation needs to be applied independently to every element of an array (element-wise operations).
- When performing mathematical or linear algebra operations.
- To avoid writing nested loops manually.

## When to use Pandas?

- When working with tabular data.
- For data manipulation, filtering, aggregation, and analysis.

## Numpy Notes

1. np.asarray(input) converts input to a numpy array, if it is not already.
    - Regular np.array will copy the input.

2. Numpy's promotion system is used to determine the data type of the output array.
    - Priority of data types is: bool, int, float, complex
    - Largest bit size is used.

3. np.astype(dtype, copy=False) is used to convert the input to the desired data type.
    - copy=False means the input is modified in place.

4. np.reshape(-1) is used to flatten the matrix to a vector.
