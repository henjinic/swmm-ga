# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import chain


def grid_sum(grid):
    return sum(chain.from_iterable(grid))


def labeled_sum(label_grid, value_grid):
    result = defaultdict(float)
    for label_row, value_row in zip(label_grid, value_grid):
        for label, value in zip(label_row, value_row):
            result[label] += value
    return result


def main():
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ]
    print(grid_sum(grid))


if __name__ == "__main__":
    main()
