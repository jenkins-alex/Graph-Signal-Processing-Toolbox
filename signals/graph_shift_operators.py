import numpy as np

class GraphShiftOperator:
    """Class for a graph shift operator used in graph signal processing
    """
    def __init__(self, shift_matrix):
        """initialise object with a shift matrix, e.g. graph adjacency matrix,
        graph Laplacian matrix, etc.

        Args:
            shift_matrix (np.ndarray): shift matrix
        """
        self.shift_matrix = shift_matrix
    
    def get_shift_matrix(self):
        """get shift matrix back

        Returns:
            np.ndarray: shift matrix
        """
        return self.shift_matrix
    
    def power(self, p):
        """calculate the power of the graph shift matrix

        Args:
            p (int): power (must be greater than 1 in current implementation)

        Returns:
            np.ndarray: shift matrix to the power of p
        """
        if p < 1:
            raise ValueError("p must be greater than 1.")
        if type(p) is not int:
            raise ValueError("power must be an integer.")

        new_matrix = self._recursive_matmul(self.shift_matrix, self.shift_matrix, p)
        return new_matrix
       
    def _recursive_matmul(self, left_m, right_m, p):
        """multiply two matrices together (from left to right) recursively using p
        as a counter 

        Args:
            left_m (np.ndarray): left matrix for multiplication
            right_m (np.ndarray): right matrix for multiplication
            p (int): counter for number of multiplications 

        Returns:
            np.ndarray: mutliplication of left and right matrices p times
        """

        # if counter is 1 mutliplications are completed
        if p == 1:
            return left_m

        # perform matrix multiplication
        result_m = np.matmul(left_m, right_m)
        result = self._recursive_matmul(result_m, right_m, p-1)
        return result