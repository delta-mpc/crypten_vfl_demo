import typing

import crypten


def crypten_collate(batch):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, crypten.CrypTensor):
        return crypten.stack(list(batch), dim=0)

    elif isinstance(elem, typing.Sequence):
        size = len(elem)
        assert all(len(b) == size for b in batch), "each element in list of batch should be of equal size"
        transposed = zip(*batch)
        return [crypten_collate(samples) for samples in transposed]

    elif isinstance(elem, typing.Mapping):
        return {key: crypten_collate([b[key] for b in batch]) for key in elem}

    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(crypten_collate(samples) for samples in zip(*batch)))

    return "crypten_collate: batch must contain CrypTensor, dicts or lists; found {}".format(elem_type)
