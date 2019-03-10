def calculate_output_dims(layer_type, input_dims, hyperparameters):
    """Calculates the output dimensions of a given layer

    Keyword Arguments:
    layer_type -- string specifying layer type
    input_dims -- list containing the input dimensions, e.g. [W_in, H_in, D_in]
    hyperparameters -- list containing the hyperparameters such as kernel number
    """

    if layer_type == 'Conv':
        return conv_output_dims(input_dims, hyperparameters)
    elif layer_type == 'Pool':
        return pool_output_dims(input_dims, hyperparameters)
    else:
        print('Unrecognized layer type')


def conv_output_dims(input_dims, hyperparameters):
    """Calculates the output dimensions of a given convolutional layer

    Keyword Arguments:
    input_dims -- list in format [w_in, h_in, d_in]
        respectively corresponding to input width, height and depth
    hyperparameters -- list in format [k, f, s, p]
        respectively corresponding to number of kernels, kernel size, stride
        and padding
    """
    # parse input dimensions
    w_in, h_in, d_in = input_dims
    # parse hyperparameters
    k, f, s, p = hyperparameters
    # calculate output width
    w_out = int((w_in - f + 2 * p) / s + 1)
    # calculate output height
    h_out = int((h_in - f + 2 * p) / s + 1)
    # calculate output depth
    d_out = int(k)
    # return list of output dimensions
    return [w_out, h_out, d_out]


def pool_output_dims(input_dims, hyperparameters):
    """Calculates the output dimensions of a given pooling layer

    Keyword Arguments:
    input_dims -- list in format [w_in, h_in, d_in]
        respectively corresponding to input width, height and depth
    hyperparameters -- list in format [f, s]
        respectively corresponding to pooling extent and  stride
    """
    # parse input dimensions
    w_in, h_in, d_in = input_dims
    # parse hyperparameters
    f, s = hyperparameters
    # calculate output width
    w_out = int((w_in - f) / s + 1)
    # calculate output height
    h_out = int((h_in - f) / s + 1)
    # calculate output depth
    d_out = int(d_in)
    # return list of output dimensions
    return [w_out, h_out, d_out]


while True:
    print("Welcome to the NN layer dimensionality analyzer")
    # get layer type
    while True:
        requested_type = input(
            "Which layer type are you interested in? [Conv/Pool]: "
        )
        if requested_type in ['Conv', 'Pool']:
            break
        else:
            print("Invalid layer type")

    # get inputs dimensions
    requested_input = input(
        "Please enter the input dimensions separated by a space: "
    ).split()
    # convert to integers
    requested_input = list(map(int, requested_input))

    # get hyperparameters
    requested_hyperparameters = input(
        "Please enter the hyperparameters separated by a space: "
    ).split()
    # convert to integers
    requested_hyperparameters = list(map(int, requested_hyperparameters))

    # calculate output dimensions
    output_dims = calculate_output_dims(
        requested_type, requested_input, requested_hyperparameters
    )

    print("The output dimensions for your requested layer are {}.".format(output_dims))
