import torch.nn as nn


def get_shapes(model, input):
    """Get a dictionary {module: (in_shape, out_shape), ...} for modules in `model`.

    Currently only processes linear and Conv2d modules. Gets the input and output
    shapes by performing trace of the network (setting hooks, etc).

    Args:
        input_size (tuple): the shape of the model input
        REMOVE: batch_size (int)
        device: (str): 'cuda' or 'cpu'

    Returns:
        Dictionary {`nn.Module`: tuple(in_shape, out_shape)}
    """
    VALID_MODULES = [
        nn.Linear, 
        nn.Conv2d 
    ]

    shapes = {}
    hooks = []
    
    def register_hook(module):
        def hook(module, input, output):
            print(module)
            print(input)
            print(output)
            shapes[module] = (tuple(input[0].shape), tuple(output.shape))
        if type(module) in VALID_MODULES:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model(input)
    for hook in hooks:
        hook.remove()
    return shapes

