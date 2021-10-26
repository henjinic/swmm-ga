# -*- coding: utf-8 -*-


def lerp(start, end, size):
    """## Linear Interpolation"""
    gap = (end - start) / (size - 1)

    for i in range(size - 1):
        yield start + i * gap
    yield end


# class Grid:

#     def __init__(self, raw_data=None, height=None, width=None, value=0, direction_masks=None):
#         """
#         `__init__(raw_data)`\n
#         `__init__(height, width, value=0)`\n
#         """

#         if raw_data is not None:
#             if isinstance(raw_data, Grid):
#                 self._raw_grid = deepcopy(raw_data.raw)
#             else:
#                 self._raw_grid = deepcopy(raw_data)
#         else:
#             self._raw_grid = [[value] * width for _ in range(height)]

#         if direction_masks is None and isinstance(raw_data, Grid):
#             self._direction_masks = raw_data._direction_masks
#         else:
#             self._direction_masks = direction_masks

#         if direction_masks is not None:
#             self._unit_vector_cache = defaultdict(list)

#             for r in range(self.height):
#                 for c in range(self.width):
#                     if self._direction_masks["up"][r][c]:
#                         self._unit_vector_cache[r, c].append((-1, 0))
#                     if self._direction_masks["down"][r][c]:
#                         self._unit_vector_cache[r, c].append((1, 0))
#                     if self._direction_masks["left"][r][c]:
#                         self._unit_vector_cache[r, c].append((0, -1))
#                     if self._direction_masks["right"][r][c]:
#                         self._unit_vector_cache[r, c].append((0, 1))


def main():
    print([x for x in lerp(0, 10, 5)])
    print([x for x in lerp(10, 0, 5)])

if __name__ == "__main__":
    main()
