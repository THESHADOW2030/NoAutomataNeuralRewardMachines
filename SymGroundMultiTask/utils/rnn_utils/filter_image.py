import torchvision


def make_filter_image(layer, use_color=True, scale_each=True):
    """Build an image of the weights of the filters in a given convolutional layer."""
    weights = layer.weight.data.to("cpu")
    if not use_color:
        n_input_channels = weights.size()[1]
        weights = weights.view([weights.size()[0], 1, weights.size()[1]*weights.size()[2], weights.size()[3]])
    img = torchvision.utils.make_grid(weights, normalize=True, scale_each=scale_each)
    return img