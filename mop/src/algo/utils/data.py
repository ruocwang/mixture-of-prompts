import random

SEED=0


def subsample_data(data, subsample_size, generator=None, seed=None):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    assert not (generator is not None and seed is not None), 'Must provide either a generator or a seed'

    # random.seed()
    if seed is not None:
        original_seed = random.getstate()
        random.seed(seed)

    inputs, outputs = data
    subsample_size = min(subsample_size, len(inputs))

    assert len(inputs) == len(outputs)
    if generator is None:
        indices = random.sample(range(len(inputs)), subsample_size)
    else:
        indices = generator.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]

    if seed is not None:
        random.setstate(original_seed) # reverse back to original seed

    return inputs, outputs


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    random.seed(SEED)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
