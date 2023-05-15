import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from itertools import combinations
import pandas as pd
import poisson_disc as poi
import mosek
# from mathutils.geometry import intersect_point_line
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

"""
Created on Tue May 19 14:39:25 2020

@author: ilia10000
"""

def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


class SoftKNN:
    def __init__(self, k=None):
        self.x = []
        self.y = []
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def calc_dists(self, point):
        dists = []
        for prototype in self.x:
            dist = np.linalg.norm(point - prototype)
            dists.append(dist)
        return dists

    def calc_lab(self, dists):
        label = np.zeros_like(self.y[0])
        tups = zip(self.y, dists)
        if self.k is None:
            for prototype, dist in tups:
                label += prototype / dist
        else:
            tups = list(tups)
            res = sorted(tups, key=lambda x: x[1])[:self.k]
            for prototype, dist in res:
                label += prototype / dist
        return label

    def predict(self, point):
        dists = self.calc_dists(point)
        label = self.calc_lab(dists)
        pred = np.argmax(label)
        return pred

    def probabilities(self, points):
        preds = []
        for point in points:
            dists = self.calc_dists(point)
            label = self.calc_lab(dists)
            preds.append(label)
        return np.array(preds)


"""
start patch #4
blender's mathutil library provides a default function to provide the closest point on line with respect to the given point, 
but the method is unstable for higher dimensions and prone to return NANs. 
https://github.com/dfelinto/blender/blob/master/source/blender/python/mathutils/mathutils_geometry.c#L774
Writing our custom intersect_point_line() function using the above for inspiration.
@author - avyav_kumar.singh@kcl.ac.uk    
2022-02-12T22:10:11Z
"""


def intersect_point_line(point, line_point1, line_point2):
    distance_point1_point = point - line_point1
    distance_point2_point1 = line_point2 - line_point1

    magnitude_point1_point2 = np.sum(distance_point2_point1 ** 2)
    dot_points = np.dot(distance_point1_point, distance_point2_point1)
    distance = dot_points / magnitude_point1_point2 if magnitude_point1_point2 != 0 else 0

    if distance < 0:
        return line_point1
    elif distance > 1:
        return line_point2
    else:
        return line_point1 + distance_point2_point1 * distance


"""
end patch #4
"""


def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def get_projections(centroids, endpoints=[0, -1]):
    # intersects = [list(intersect_point_line(centroid, centroids[endpoints[0]], centroids[endpoints[1]])[0]) for centroid in centroids]
    intersects = [list(intersect_point_line(centroid, centroids[endpoints[0]], centroids[endpoints[1]])) for centroid in
                  centroids]
    return np.array(intersects)


def dist(a, b):
    return np.linalg.norm(a - b)


def get_dists(intersects, active_classes):
    inter = intersects[active_classes]
    dists = [dist(inter[0], inter[i]) for i in range(len(inter))]
    return dists


def mid(a, b):
    return (a + b) / 2


def get_mids(intersects, active_classes):
    inter = intersects[active_classes]
    mids = [mid(inter[i], inter[i + 1]) for i in range(len(inter) - 1)]
    return mids


def get_mid_dists(intersects, active_classes):
    inter = intersects[active_classes]
    mids = [mid(inter[i], inter[i + 1]) for i in range(len(inter) - 1)]
    mid_dists = [dist(inter[0], mids[i]) for i in range(len(mids))]
    return mid_dists


def create_system_V4(n, mid_dists=None, tot_dist=None, endpoints=None, dists=None, var=None):
    A = []
    epsilon = 1e-3
    assert (len(mid_dists) == n - 1)
    constraints = []
    for i in range(len(dists)):

        vector = np.zeros(n * 2)
        vector[i] += 1. / (dists[i] + epsilon - endpoints[0])
        vector[n + i] += 1. / (endpoints[1] - dists[i] + epsilon)
        q1 = var[i] / (dists[i] + epsilon - endpoints[0])
        q2 = var[n + i] / (endpoints[1] - dists[i] + epsilon)
        for j in range(len(dists)):
            if i != j:
                vector[j] -= 1. / (dists[i] + epsilon - endpoints[0])
                vector[n + j] -= 1. / (endpoints[1] - dists[i] + epsilon)
                q3 = var[j] / (dists[i] + epsilon - endpoints[0])
                q4 = var[n + j] / (endpoints[1] - dists[i] + epsilon)
                constraint = q1 + q2 >= q3 + q4 + epsilon * epsilon
                constraints.append(constraint)
        A.append(vector)

    for i in range(len(mid_dists)):
        q1 = var[i] / (mid_dists[i] - endpoints[0])
        q2 = var[n + i] / (endpoints[1] - mid_dists[i])

        q3 = var[i + 1] / (mid_dists[i] - endpoints[0])
        q4 = var[n + i + 1] / (endpoints[1] - mid_dists[i])
        constraint = q1 + q2 == q3 + q4
        constraints.append(constraint)
    constraints.append(var >= 0)
    constraints.append(var <= 1)
    constraints.append(sum(var[0:n]) == 1)
    constraints.append(sum(var[n:]) == 1)
    # A.append(np.append(np.ones(n),np.zeros(n)))
    # A.append(np.append(np.zeros(n),np.ones(n)))
    # A.append(np.zeros(n*2))
    # A[-1][n-1]=1
    # A.append(np.zeros(n*2))
    # A[-1][2*n-1]=1
    # b = np.ones(len(A))*101
    b = np.zeros(len(A))
    # b[-4]=1
    # b[-3]=1
    return A, b, constraints


def vis_line(line, num_classes, dat, centroids, colors, return_plt=False):
    cmap = "tab20"
    # print(colors)
    for i in range(num_classes):
        temp = dat[0][dat[1] == i]
        x = [t[0] for t in temp]
        y = [t[1] for t in temp]
        if i in line:  # or True:
            plt.scatter(x, y, label=i, color=colors[i], alpha=0.5)
            plt.scatter(np.mean(x), np.mean(y), c="black")
    plt.plot([centroids[line[0]][0], centroids[line[-1]][0]], [centroids[line[0]][1], centroids[line[-1]][1]])
    plt.legend()
    if return_plt:
        return plt
    plt.show()


def line_features(line, centroids):
    intersects = get_projections(centroids, [line[0], line[-1]])
    dists = get_dists(intersects, line)
    mids = get_mids(intersects, line)
    mid_dists = get_mid_dists(intersects, line)
    return intersects, dists, mids, mid_dists


def get_line_prototypes(line, centroids):
    n = len(line)
    x = cp.Variable(2 * n)
    # left = cp.Parameter(nonneg=True)
    # right = cp.Parameter(nonneg=True)
    # left.value=0
    # right.value = dists[-1]
    intersects, dists, mids, mid_dists = line_features(line, centroids)
    left = 0
    right = dists[-1]
    # print(mid_dists)
    #     print(dists)
    # print(right)
    s3 = create_system_V4(n, mid_dists=mid_dists, tot_dist=None, endpoints=[left, right], dists=dists, var=x)

    A, b, constraints = s3
    A = np.array(A)
    # print("dists are ", dists)
    # print("x is ", x)
    # print("constraints are ", constraints)
    objective = cp.Maximize(cp.sum(A @ x) + cp.sum_smallest(A @ x, 2))
    # constraints = [0 <= x, x <= 1, sum(x[0:5])==1, sum(x[5:10])==1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.MOSEK)
    distX = np.array([centroids[line[0]], centroids[line[-1]]])
    distY = np.zeros((2, len(centroids)))
    distY[0, line], distY[1, line] = x.value[0:n], x.value[n:]
    return distX, distY


def get_dist_to_line(point, line, centroids):
    inter = list(intersect_point_line(point, centroids[line[0]], centroids[line[-1]])[0])
    d = dist(inter, point)
    return d


def dist_to_line(x1, y1, x2, y2, x3, y3):  # x3,y3 is the point
    px = x2 - x1
    py = y2 - y1
    norm = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = (dx * dx + dy * dy) ** .5
    return dist


def lineseg_dist(p, a, b):
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


def dist_to_line_multiple(endpoints, x3, y3):  # x3,y3 is the point
    x1 = endpoints[:, 0][:, 0]
    x2 = endpoints[:, 1][:, 0]
    y1 = endpoints[:, 0][:, 1]
    y2 = endpoints[:, 1][:, 1]
    px = x2 - x1
    py = y2 - y1
    norm = 1. * px * px + 1. * py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / norm
    u[u > 1] = 1
    u[u < 0] = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = (dx * dx + dy * dy) ** .5
    return dist


def dist_to_line_multiple_V2(endpoints, points):  # x3,y3 is the point
    x1 = endpoints[:, 0][:, 0]
    x2 = endpoints[:, 1][:, 0]
    y1 = endpoints[:, 0][:, 1]
    y2 = endpoints[:, 1][:, 1]
    x3 = points[:, 0]
    y3 = points[:, 1]
    px = x2 - x1
    py = y2 - y1
    norm = 1. * px * px + 1. * py * py
    u = (np.subtract.outer(x3, x1) * px + np.subtract.outer(y3, y1) * py) / norm
    u[u > 1] = 1
    u[u < 0] = 0

    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3[:, np.newaxis]
    dy = y - y3[:, np.newaxis]
    dist = (dx * dx + dy * dy) ** .5
    return dist


def closest_line(lines, centroids, point):
    #     print(lines)
    dists = [dist_to_line(*centroids[line[0]], *centroids[line[1]], *point) for line in lines]
    mindex = np.argmin(dists)
    return lines[mindex], dists[mindex]


def closest_line_multiple(lines, centroids):
    lines = np.array(lines)
    centroids = np.array(centroids)
    # dists = [[dist_to_line(*centroids[line[0]], *centroids[line[1]], *point) for line in lines] for point in centroids]
    # dists = [dist_to_line_multiple(centroids[lines], *point) for point in centroids]
    dists = dist_to_line_multiple_V2(centroids[lines], centroids)
    dists = np.array(dists)
    mindex = np.argmin(dists, axis=1)
    """
    start patch #12
    np.choose is unstable for higher dimensionalities
    @author - avyav_kumar.singh@kcl.ac.uk    
    2022-02-26T19:59:01Z
    """
    required_array = dists[range(len(mindex)), mindex]
    return lines[mindex], np.sum(required_array)
    """
    end patch #12
    """


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def ccw_multi(A, B, C):
    return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])


# Return true if line segments AB and CD intersect
def intersecting(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def intersecting_multi(A, B, C, D):
    return np.any((ccw_multi(A, C, D) != ccw_multi(B, C, D)) * (ccw_multi(A, B, C) != ccw_multi(A, B, D)))


def any_intersect(lines, centroids, combos):
    if intersecting_multi(centroids[lines[combos[:, 0]]][:, 0], centroids[lines[combos[:, 0]]][:, 1],
                          centroids[lines[combos[:, 1]]][:, 0], centroids[lines[combos[:, 1]]][:, 1]):
        return True
    return False


def line_order(centroids, active_classes):
    intersects = get_projections(centroids, endpoints=[active_classes[0], active_classes[1]])
    dists = get_dists(intersects, active_classes)
    return active_classes[np.argsort(dists)]


def get_pairwise_dists(centroids, active_classes):
    active_locs = np.array(centroids)[np.array(active_classes)]
    dists = np.linalg.norm(active_locs[:, None, :] - active_locs[None, :, :], axis=-1)
    return dists


def line_order_no_endpoints(centroids, active_classes):
    dists = get_pairwise_dists(centroids, active_classes)
    endpoints = np.argmax(dists)
    endpoints = np.unravel_index(endpoints, dists.shape)
    endpoints = [active_classes[endpoints[0]], active_classes[endpoints[1]]]
    intersects = get_projections(centroids, endpoints=endpoints)
    start = intersects[endpoints[0]]
    intersects = intersects[active_classes]
    dists2 = [dist(start, intersects[i]) for i in range(len(intersects))]
    return active_classes[np.argsort(dists2)]


def find_lines_brute(centroids, k=3):
    num_classes = len(centroids)
    """
    start patch #2
    when two classes are required with only 1 line to be fitted, return the trivial line
    @author - avyav_kumar.singh@kcl.ac.uk    
    2022-01-28T12:50:19Z
    """
    if num_classes == 2 and k == 1:
        full_lines = [np.array([0, 1])]
        return full_lines
    """
    end patch #2
    """
    centroids = np.array(centroids)
    lines = list(combinations(range(num_classes), 2))
    lines = np.array(lines)
    triple_lines = np.array(list(combinations(range(len(lines)), k)))
    index_lines = np.array(lines[triple_lines])
    index_centroids = np.array(centroids)[index_lines]
    total_dists = []
    all_nearest = []
    lines_intersect = []
    combos = np.array(list(combinations(range(k), 2)))
    for triplex in range(len(triple_lines)):
        triple_line = index_centroids[triplex]  # [centroids[[line]] for line in index_lines[triplex]]
        nearest_array, total_dist = closest_line_multiple(index_lines[triplex], centroids)
        all_nearest.append(nearest_array)
        total_dists.append(total_dist)
        lines_intersect.append(any_intersect(index_lines[triplex], centroids, combos))
    lines_intersect = np.array(lines_intersect)
    triple_lines = np.array(triple_lines)
    total_dists = np.array(total_dists)
    all_nearest = np.array(all_nearest)
    triple_lines = triple_lines[lines_intersect == False]
    total_dists = total_dists[lines_intersect == False]
    all_nearest = all_nearest[lines_intersect == False]
    mindex = np.argmin(total_dists)
    top_lines_endpoints = lines[[triple_lines[mindex]]]
    full_lines = list(np.copy(top_lines_endpoints))
    for i in range(len(all_nearest[mindex])):
        for j in range(len(top_lines_endpoints)):
            if np.all(all_nearest[mindex][i] == top_lines_endpoints[j]):
                if i not in full_lines[j]:
                    full_lines[j] = list(full_lines[j]) + [i]
    return full_lines


# Find best line, then find best non-intersecting lines
def find_lines_greedy1(centroids, k=3):
    num_classes = len(centroids)
    """
    start patch #3
    when two classes are required with only 1 line to be fitted, return the trivial line
    @author - avyav_kumar.singh@kcl.ac.uk    
    2022-01-28T12:50:19Z
    """
    if num_classes == 2 and k == 1:
        full_lines = [np.array([0, 1])]
        return full_lines
    """
    end patch #3
    """
    centroids = np.array(centroids)
    lines = list(combinations(range(num_classes), 2))
    lines = np.array(lines)
    triple_lines = np.array(list(combinations(range(len(lines)), k)))
    index_lines = np.array(lines[triple_lines])
    index_centroids = np.array(centroids)[index_lines]
    total_dists = []
    all_nearest = []
    lines_intersect = []

    combos = np.array(list(combinations(range(k), 2)))
    for triplex in range(len(triple_lines)):
        triple_line = index_centroids[triplex]  # [centroids[[line]] for line in index_lines[triplex]]
        nearest_array, total_dist = closest_line_multiple(index_lines[triplex], centroids)
        all_nearest.append(nearest_array)
        total_dists.append(total_dist)
        lines_intersect.append(any_intersect(index_lines[triplex], centroids, combos))
    lines_intersect = np.array(lines_intersect)
    triple_lines = np.array(triple_lines)
    total_dists = np.array(total_dists)
    all_nearest = np.array(all_nearest)
    triple_lines = triple_lines[lines_intersect == False]
    total_dists = total_dists[lines_intersect == False]
    all_nearest = all_nearest[lines_intersect == False]
    mindex = np.argmin(total_dists)
    top_lines_endpoints = lines[[triple_lines[mindex]]]
    full_lines = list(np.copy(top_lines_endpoints))
    for i in range(len(all_nearest[mindex])):
        for j in range(len(top_lines_endpoints)):
            if np.all(all_nearest[mindex][i] == top_lines_endpoints[j]):
                if i not in full_lines[j]:
                    full_lines[j] = list(full_lines[j]) + [i]
    return full_lines


import pandas as pd

result1 = None


def point_on_line(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    # if you need the closest point belonging to the segment
    t = max(0, min(1, t))
    result = a + t * ab
    return result


def dist_to_line_multiD(point, A, B):
    if np.array_equal(A, B):
        return dist(point, A)
    proj = point_on_line(point, A, B)
    length = dist(A, B)
    #     distA = dist(proj,A)
    #     distB = dist(proj,B)
    #     if distA>length or distB>length:
    #         return min(distA,distB)
    return dist(proj, point)


def closest_line_multiD(lines, centroids):
    lines = np.array(lines)
    centroids = np.array(centroids)
    dists = [[dist_to_line_multiD(point, centroids[line[0]], centroids[line[1]]) for line in lines] for point in
             centroids]
    # dists = [dist_to_line_multiple(centroids[lines], *point) for point in centroids]
    # dists = dist_to_line_multiple_V2(centroids[lines], centroids)
    dists = np.array(dists)
    mindex = np.argmin(dists, axis=1)
    """
    start patch #13
    np.choose is unstable for higher dimensionalities
    @author - avyav_kumar.singh@kcl.ac.uk    
    2022-02-26T19:59:01Z
    """
    required_array = dists[range(len(mindex)), mindex]
    return lines[mindex], np.sum(required_array)
    """
    end patch #13
    """


"""
patch #3
fix input parameters to make the data and labels distinction clear
@author - avyav_kumar.singh@kcl.ac.uk
2022-02-02T20:09:43Z
"""


def find_lines_R_multiD(dat, labels, centroids, dims=2, k=5, max_diff=1e-5):
    cols = []
    for dim in range(dims):
        x = np.array(dat[:, dim])
        cols.append(x)
    df = pd.DataFrame(np.array([*cols, labels]).transpose())

    """
    start patch #1
    fix naming of columns in the data frame
    @author - avyav_kumar.singh@kcl.ac.uk
    reference - https://stackoverflow.com/a/62033656
    2022-01-27T17:25:50Z
    """
    df.columns = [*[str(i) for i in range(dims)], "My Hopes And Dreams"]
    df["My Hopes And Dreams"] = df["My Hopes And Dreams"].apply(np.int64)
    """
    end patch #1
    """

    # invoke R %R -i df -i k -i max_diff -i dims -o result1 result1 <- recursive_reg(as.matrix(df[,-(dims+1)]), df[,
    # dims+1]+1, k = k, max_diff = max_diff)
    result1 = invoke_R_wrapper(df, k, max_diff, dims)

    lines = [list(r) for r in result1]
    lines = np.array([[line[0], line[-1]] for line in lines]) - 1
    nearest_array, total_dist = closest_line_multiD(lines, centroids)
    full_lines = [list(line) for line in lines]
    for i in range(len(nearest_array)):
        for j in range(len(lines)):
            if np.all(nearest_array[i] == lines[j]):
                if i not in full_lines[j]:
                    full_lines[j] = list(full_lines[j]) + [i]
    return full_lines


def invoke_R_wrapper(df, k, max_diff, dims):
    r = ro.r
    r.source("/home/aksingh/Documents/meta-learned-lines/lines/lo_shot_definitions.R")
    data = df.iloc[:, :-1].to_numpy()
    labels = df.iloc[:, -1:].to_numpy() + 1

    nr, nc = data.shape
    data_r = ro.r.matrix(data, nrow=nr, ncol=nc)

    p = r.recursive_reg(data_r, labels, k=k, max_diff=max_diff, keep_all=True)
    return p