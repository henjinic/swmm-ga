import random


def choices(sequence, weights=None):
    if weights is None:
        weights = [1] * len(sequence)
    return sequence[_choices_index(weights)]

def randpop(sequence, weights=None):
    if weights is None:
        weights = [1] * len(sequence)
    return sequence.pop(_choices_index(weights))

def _choices_index(weights):
    normalized_weights = [weight / sum(weights) for weight in weights]
    accumulated_weights = [sum(normalized_weights[:i + 1]) for i in range(len(weights))]

    offset = random.random()
    for i, weight in enumerate(accumulated_weights):
        if offset <= weight:
            return i


def main():
    pass

if __name__ == "__main__":
    main()