# -*- coding: utf-8 -*-
from collections import defaultdict
from copy import deepcopy
from itertools import chain


def grid_sum(grid):
    return sum(chain.from_iterable(grid))


def labeled_sum(label_grid, value_grid):
    result = defaultdict(float)
    for label_row, value_row in zip(label_grid, value_grid):
        for label, value in zip(label_row, value_row):
            result[label] += value
    return result


def get_diff_coords(grid1, grid2):
    """
    ```
    # example
    arg1: [[1, 2], | arg2: [[1, 1],
           [1, 1]] |        [2, 3]]

    return: defaultdict {
        frozenset {1, 2}: [(0, 1), (1, 0)]
        frozenset {1, 3}: [(1, 1)]
    }
    ```
    """
    result = defaultdict(list)
    for r, (row1, row2) in enumerate(zip(grid1, grid2)):
        for c, (e1, e2) in enumerate(zip(row1, row2)):
            if e1 != e2:
                result[frozenset((e1, e2))].append((r, c))
    return result


def is_in(grid, r, c):
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
        return True
    return False


def count_neighbor(grid, r, c, targets):
    vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    coords = [(r + dr, c + dc) for dr, dc in vectors]
    return sum(1 for r, c in coords if is_in(grid, r, c) and grid[r][c] in targets)


def analyze_cluster(grid):
    result = defaultdict(list)
    check_grid = deepcopy(grid)
    for r, row in enumerate(check_grid):
        for c, e in enumerate(row):
            if e == 0:
                continue
            result[e].append(_fill_zeros_in_cluster(check_grid, r, c, e, grid))
    return result


def _fill_zeros_in_cluster(check_grid, r, c, target, grid):
    target_coords = {(r, c)}
    count = 0
    neighbor_coords = set()
    while target_coords:
        r, c = target_coords.pop()
        check_grid[r][c] = 0
        target_coords |= _get_neighbor_coords_include(check_grid, r, c, [target])
        count += 1
        neighbor_coords |= _get_neighbor_coords_exclude(grid, r, c, [target])
    return {"count": count, "neighbor_coords": list(neighbor_coords)}


def _get_neighbor_coords_exclude(grid, r, c, exclusions):
    vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    coords = [(r + dr, c + dc) for dr, dc in vectors]
    return {(r, c) for r, c in coords if is_in(grid, r, c) and grid[r][c] not in exclusions}


def _get_neighbor_coords_include(grid, r, c, inclusions):
    vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    coords = [(r + dr, c + dc) for dr, dc in vectors]
    return {(r, c) for r, c in coords if is_in(grid, r, c) and grid[r][c] in inclusions}


def main():
    from pprint import pp
    grid = [
        [0, 2, 2],
        [1, 3, 1],
        [2, 3, 2]
        ]
    pp(analyze_cluster(grid))


if __name__ == "__main__":
    main()
