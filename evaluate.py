"""
An implementation of the pattern recognition

@author: Blaise Wang
"""

import numpy as np


def has_winner(chess, x: int, y: int) -> bool:
    diff = y - x
    rot = x - (len(chess) - 1) + y
    bias = max(0, diff)
    rot_bias = max(0, rot)

    for array in [chess[x, y - 4: y + 5], chess[x - 4: x + 5, y],
                  np.diagonal(chess, diff)[y - bias - 4:y - bias + 5],
                  np.diagonal(np.rot90(chess), rot)[x - rot_bias - 4:x - rot_bias + 5]]:
        for i in range(5):
            similarity = 0
            for j in range(5):
                if array[i + j] != chess[x, y]:
                    break
                similarity += 1
            if similarity == 5:
                return True
    return False
