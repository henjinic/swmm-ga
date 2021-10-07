# -*- coding: utf-8 -*-
from collections import defaultdict
from copy import deepcopy


def lerp(start, end, size):
    """## Linear Interpolation"""
    gap = (end - start) / (size - 1)

    for i in range(size - 1):
        yield start + i * gap
    yield end


class Grid:

    def __init__(self, raw_data=None, height=None, width=None, value=0, direction_masks=None):
        """
        `__init__(raw_data)`\n
        `__init__(height, width, value=0)`\n
        """

        if raw_data is not None:
            self._raw_grid = raw_data
        else:
            self._raw_grid = [[value] * width for _ in range(height)]

        self._direction_masks = direction_masks

        if direction_masks is not None:
            self._unit_vector_cache = defaultdict(list)

            for r in range(self.height):
                for c in range(self.width):

                    if self._direction_masks["up"][r][c]:
                        self._unit_vector_cache[r, c].append((-1, 0))

                    if self._direction_masks["down"][r][c]:
                        self._unit_vector_cache[r, c].append((1, 0))

                    if self._direction_masks["left"][r][c]:
                        self._unit_vector_cache[r, c].append((0, -1))

                    if self._direction_masks["right"][r][c]:
                        self._unit_vector_cache[r, c].append((0, 1))

    def __getitem__(self, coord):
        return self._raw_grid[coord[0]][coord[1]]

    def __setitem__(self, coord, value):
        self._raw_grid[coord[0]][coord[1]] = value

    def __str__(self):
        lines = ["[" + " ".join(map(str, line)) + "]" for line in self._raw_grid]
        aligned_lines = [" " + lines[i] if i else "[" + lines[i] for i in range(len(lines))]

        return "\n".join(aligned_lines) + "]"

    @property
    def raw(self):
        return self._raw_grid

    @property
    def height(self):
        return len(self._raw_grid)

    @property
    def width(self):
        return len(self._raw_grid[0])

    def copy(self):
        return Grid(raw_data=deepcopy(self._raw_grid), direction_masks=self._direction_masks)

    def get_coords(self, filter):
        return [(r, c) for r in range(self.height) for c in range(self.width) if filter((r, c))]

    def analyze_cluster(self, masks=None):
        """
        ```
        return: (
            defaultdict {code: [size of each cluster]},
            the number of clusters
        )
        ```
        """
        if masks is None:
            masks = {}
            result = defaultdict(lambda : dict(sizes=[], neighbors=[]))
        else:
            result = defaultdict(lambda : dict(sizes=[], neighbors=[], is_near_mask=[]))

        check_grid = self.copy()

        for r in range(self.height):
            for c in range(self.width):
                if check_grid[r, c] == 0:
                    continue

                code, count, neighbors, is_near_mask = check_grid._fill_zeros_in_cluster(r, c, self, masks)
                result[code]["sizes"].append(count)
                result[code]["neighbors"].append(neighbors)
                if code in masks:
                    result[code]["is_near_mask"].append(is_near_mask)

        return result, sum(len(result[code]["sizes"]) for code in result)

    def _fill_zeros_in_cluster(self, r, c, original_map, masks):
        target_code = self[r, c]
        target_coords = [(r, c)]
        count = 0
        neighbors = []
        if target_code in masks:
            is_near_mask = [False] * len(masks[target_code])
        else:
            is_near_mask = None

        while target_coords:
            r, c = target_coords.pop(0)
            neighbors += self.traverse_neighbor(r, c,
                                                lambda x: original_map[x],
                                                lambda x: original_map[x] != 0
                                                          and original_map[x] != target_code)

            if target_code in masks:
                for i, mask in enumerate(masks[target_code]):
                    if is_near_mask[i]:
                        continue

                    is_near_mask[i] = bool(mask.count_neighbor(r, c, [1]))

            self[r, c] = 0
            count += 1

            target_coords += self.traverse_neighbor(r, c, lambda x: x, lambda x: self[x] == target_code and x not in target_coords)

        return target_code, count, list(set(neighbors)), is_near_mask

    def count_neighbor(self, r, c, targets):
        return sum(self.traverse_neighbor(r, c, lambda x: 1, lambda x: self[x] in targets))

    def traverse_neighbor(self, r, c, action, filter=lambda: True):
        if self._direction_masks is None:
            vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        else:
            vectors = self._unit_vector_cache[r, c]

        return [action((r + dr, c + dc)) for dr, dc in vectors
                                         if 0 <= r + dr < self.height
                                            and 0 <= c + dc < self.width
                                            and filter((r + dr, c + dc))]

    def get_diff_coords(self, partner):
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

        for r in range(self.height):
            for c in range(self.width):
                if self[r, c] != partner[r, c]:
                    result[frozenset((self[r, c], partner[r, c]))].append((r, c))

        return result

    def sum(self):
        return sum(self.traverse(lambda x: self[x], lambda x: True))

    def traverse(self, action, filter):
        return [action((r, c)) for r in range(self.height) for c in range(self.width) if filter((r, c))]

    def each_sum(self, reference_grid):
        reference_grid = Grid(reference_grid)
        result = defaultdict(int)

        for r in range(self.height):
            for c in range(self.width):
                if self[r, c] == 0:
                    continue

                result[self[r, c]] += reference_grid[r, c]

        return result


def main():
    for i in lerp(0, 10, 5):
        print(i)


if __name__ == "__main__":
    main()
