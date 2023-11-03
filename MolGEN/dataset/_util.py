from typing import List


def one_hot_encoding(value: int, choices: List) -> List:
    r"""One-hot encoding of a value given a list of possible choices.
    
    Args:
        value: The value to encode.
        choices: A list of possible choices.
    
    Returns:
        A list of length ``len(choices) + 1`` with all zeros except for a one
        at the index of the value in the choices list.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding