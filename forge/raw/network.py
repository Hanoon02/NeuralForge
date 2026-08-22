"""
A network has multiple connection matrices, which are the weights between two layers.
Lets say the hidden layer count is h, then the number of connection matrix is h+1


In a weight matrix, the left, or i index is for the start part, and the top, or j index is for the end part.
You basically end up with (input_size X output_size) matrix for that connection part
"""


def generate_empty_weight_matrix(input_size, output_size, hidden_layers):
    weight_matrix = []
    count_of_hidden_layers = len(hidden_layers)
    if count_of_hidden_layers == 0:
        connection_matrix = [[0 for j in range(output_size)] for i in range(input_size)]
        weight_matrix.append(connection_matrix)
        return weight_matrix
    else:
        input_to_first_hidden_layer_connection_matrix = [
            [0 for j in range(hidden_layers[0])] for i in range(input_size)
        ]
        weight_matrix.append(input_to_first_hidden_layer_connection_matrix)
        for hidden_layer in range(1, count_of_hidden_layers):
            connection_matrix = [
                [0 for j in range(hidden_layers[hidden_layer])]
                for i in range(hidden_layers[hidden_layer - 1])
            ]
            weight_matrix.append(connection_matrix)
        last_hidden_layer_to_output_connection_matrix = [
            [0 for j in range(output_size)]
            for i in range(hidden_layers[count_of_hidden_layers - 1])
        ]
        weight_matrix.append(last_hidden_layer_to_output_connection_matrix)
        return weight_matrix
