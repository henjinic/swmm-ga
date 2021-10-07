# -*- coding: utf-8 -*-
import random


def choices(sequence, weights=None):
    if weights is None:
        return sequence[random.randrange(len(sequence))]
    return sequence[_choices_index(weights)]

def randpop(sequence, weights=None):
    if weights is None:
        return sequence.pop(random.randrange(len(sequence)))
    return sequence.pop(_choices_index(weights))

def _choices_index(weights):
    weights = [0 if weight < 0 else weight for weight in weights]

    if sum(weights) <= 0:
        return random.randrange(len(weights))

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