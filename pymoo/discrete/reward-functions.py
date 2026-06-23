import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt



##### Hyper volume function
def  hypervolume_function(hv_block_1, hv_block_2):
        # A_max =  np.argmax(hv_block_1)
        hv_block_1 = hv_block_1[np.argsort(hv_block_1[:, 0])]

        hv = 0.0
        prev_f1 = hv_block_2[0]

        for f1, f2 in reversed(hv_block_1):
            width = prev_f1 - f1
            height = hv_block_2[1] - f2
            hv += width * height
            prev_f1 = f1

        return hv # return teh  hyper volume for teh  two points


def hausdorf_distance(hd_1, hd_rf): #rf fro refernce
    def directed(point_a, point_b):
        return max(
            min(np.linalg.norm(a - b) for b in point_b)
            for a in point_a
        )


    return max(directed(hd_1, hd_rf), directed(hd_rf, hd_1))


#### Crow-space Function
def  crowspace(front_set):
    n_points, n_obj = front_set.shape
    distance = np.zeros(n_points)

    for m in range(n_obj):
        sorted_indices = np.argsort(front_set[:, m])
        sorted_front = front_set[sorted_indices]

        distance[sorted_indices[0]] = np.inf
        distance[sorted_indices[-1]] = np.inf

        f_min = sorted_front[0, m]
        f_max = sorted_front[-1, m]

        if f_max - f_min == 0:
            continue

        for i in range(1, n_points - 1):
            distance[sorted_indices[i]] += (
                    (sorted_front[i + 1, m] - sorted_front[i - 1, m])
                    / (f_max - f_min)
            )

    return distance





##### Multi Objective function