from collections.abc import Mapping


def is_namedtuple(data):
    """
    Checks if `x` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    data_type = type(data)
    bases = data_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(data_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(member, str) for member in fields)


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor)
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for k, t in tensor.items()
            }
        )
    elif hasattr(tensor, "to"):
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device)
    else:
        return tensor
