import math
import regex
import scipy.spatial
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from dataclasses import dataclass

# system = {}

# each entry in the array is a prem, and then an array of tuples of type (var, weight)
# equations = []

initPattern = "(?P<prem>[A-Z])\ \>\=\ \{((?:(?P<point>\([0-9]+.[0-9]*,[0-9]+.[0-9]*\)),)*(?P<point>\([0-9]+.[0-9]*,[0-9]+.[0-9]*\)))\}"
equationPattern = "(?P<prem>[A-Z])\ \>\=\ (?:(?P<element>(\([0-9]+.[0-9]*\))[A-Z])*\ \+\ )(?:(?P<element>\([0-9]+.[0-9]*\)[A-Z]))"


### TODO when dimension error is thrown, make sure to find the pair of points with the greatest distance and only keep those. ###
@dataclass
class EquationTerm:
    variable: str
    weight: float


@dataclass
class Equation:
    premise: str
    terms: List[EquationTerm]



def apply_system(prev_state: dict[str, np.ndarray], system: List[Equation]):
    new_state: dict[str, np.ndarray] = {}

    for equation in system:
        # print(equation)
        """Calculate RHS of expression, put it into the new state of that premise"""
        if equation.premise in new_state:
            new_state[equation.premise] = np.append(
                new_state[equation.premise], apply_eqn(equation, prev_state), axis=0)
        else:
            new_state[equation.premise] = apply_eqn(equation, prev_state)
    for premise in prev_state:
        """Make sure to include the old state, also only keep the convex vertices"""
        if premise in new_state:
            # print(prev_state)
            # print(new_state)
            new_state[premise] = convexify(
                np.append(new_state[premise], prev_state[premise], axis=0))
        else:
            new_state[premise] = prev_state[premise]
    return new_state


def apply_eqn(equation: Equation, prev_state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """We will assume two terms for now

    Args:
        equation (Equation): equation being applied
        prev_state (dict[str, np.ndarray]): state dict we are basing our iteration on

    Returns:
        np.ndarray: new state inclusion for the premise of the equation
    """
    retval: np.ndarray = convexify(normalizedMinkowskiSum(prev_state[equation.terms[0].variable], prev_state[equation.terms[1].variable], [
                                   equation.terms[0].weight, equation.terms[1].weight]))
    # weight = 0

    # for term in equation.terms:
    #     if retval == None:
    #         retval = prev_state[term.variable] * term.weight
    #     retval = convexify(np.append(retval, normalizedMinkowskiSum(
    #         retval, prev_state[term.variable], [weight, term.weight]), axis=0))
    return retval
    # X = np.append(X, normalizedMinkowskiSum(X, Y, [0.5, 0.5]), axis=0)


def convexify(hemi: np.array):
    if (len(hemi) > 2):
        try:
            hull = scipy.spatial.ConvexHull(hemi)
            vertices = []
            for idx in hull.vertices:
                vertices.append(hull.points[idx])
            return np.array(vertices)
        except scipy.spatial.QhullError:
            # TODO log dimensionality exception
            # Find furthest-apart pair of elements
            outex = None
            inex = None
            maxdist = 0
            for outer in hemi:
                for inner in hemi:
                    dist = distance(outer,inner)
                    if dist>=maxdist:
                        maxdist = dist
                        outex = outer
                        inex = inner
            return np.array([outex, inex])
            # return hemi

    else:
        return hemi


def parse_system(fname):
    system = {}
    equations = []
    with open(fname) as file:
        for line in file:
            if ((line.startswith("%"))):
                pass
            elif (m := regex.match(initPattern, line)):
                # print(m.capturesdict())
                premise = m.captures("prem")[0]
                old = system.get(m.captures("prem")[0])
                points = []
                for point in m.captures("point"):
                    stringfloats = point[1:-1] .split(',')
                    points.append(
                        [float(stringfloats[0]), float(stringfloats[1])])
                if (not premise in system):
                    new = points
                else:
                    new = np.append(old, points, axis=0)
                # print(premise)
                # print(new)
                system[premise] = new

            elif (m := regex.match(equationPattern, line)):
                # print(m.capturesdict())
                elements = []
                for element in m.captures("element"):
                    var = element[-1]
                    weight = float(element[1:-2])
                    elements.append(EquationTerm(var, weight))
                equations.append(Equation(m.captures("prem")[0], elements))
            else:
                print("ERROR PARSING:")
                print("    " + line)
    for premise in system:
        system[premise] = np.array(system[premise])
    return (system, equations)


def plot_hemi(hemi, **kwargs):
    """
    Plots a hemihedra.

    Args:
        hemi (np.ndarray[float, float]): list of hemihedra vertices.
        color (string): matplotlib args for color and such
    """
    color = kwargs.get("color")
    if (len(hemi) > 2):
        try:
            hull = scipy.spatial.ConvexHull(hemi)
            # surfacepoints = np.array([])
            # for vertexIndex in hull.vertices:
            #     surfacepoints.append(hull.points[vertexIndex])
            # plt.plot(surfacepoints[:,0], surfacepoints[:,1], 'o')
            plt.plot(hull.points[hull.vertices, 0],
                     hull.points[hull.vertices, 1], color + 'o', lw=2)

            for simplex in hull.simplices:

                plt.plot(hull.points[simplex, 0],
                         hull.points[simplex, 1], color + '-')
        except scipy.spatial.QhullError:
            # TODO log dimensionality exception
            plt.plot(hemi[:, 0], hemi[:, 1], color + 'o-')

    elif (len(hemi) == 2):
        plt.plot(hemi[:, 0], hemi[:, 1], color + 'o-')
    else:
        plt.plot(hemi[:, 0], hemi[:, 1], color + 'o')


def sort_vertices(polygon):
    """Sorts vertices by polar angles.

    Args:
        polygon (list[list[float, float]]): list of polygon vertices

    Returns:
        list[list[float, float]]: list of polygon vertices sorted
    """
    cx, cy = polygon.mean(0)  # center of mass
    x, y = polygon.T
    angles = np.arctan2(y-cy, x-cx)
    indices = np.argsort(angles)
    return polygon[indices]

def distance(p1, p2):
    """Distance of two vectors in 2R space.
    
    Args:
        p1 (list[float, float]): first vector
        p2 (list[float, float): second vector

    Returns:
        float: value of cross product
    """
    return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow(p1[0]-p2[0],2))
    # return np.linalg.norm(p1-p2)
    #TODO figure out why above not working


def crossprod(p1, p2):
    """Cross product of two vectors in 2R space.

    Args:
        p1 (list[float, float]): first vector
        p2 (list[float, float): second vector

    Returns:
        float: value of cross product
    """
    return p1[0] * p2[1] - p1[1] * p2[0]


def minkowskisum(pol1, pol2):
    """Calculate Minkowski sum of two convex polygons.

    Args:
        pol1 (np.ndarray[float, float]): first polygon
        pol2 (np.ndarray[float, float]): second polygon

    Returns:
        np.ndarray[np.ndarray[float, float]]: list of the Minkowski sum vertices
    """
    msum = []
    pol1 = sort_vertices(pol1)
    pol2 = sort_vertices(pol2)

    # sort vertices so that is starts with lowest y-value
    min1, min2 = np.argmin(pol1[:, 1]), np.argmin(
        pol2[:, 1])  # index of vertex with min y value
    pol1 = np.vstack((pol1[:min1], pol1[min1:]))
    pol2 = np.vstack((pol2[:min2], pol2[min2:]))

    i, j = 0, 0
    l1, l2 = len(pol1), len(pol2)
    # iterate through all the vertices
    while i < len(pol1) or j < len(pol2):
        msum.append(pol1[i % l1] + pol2[j % l2])
        cross = crossprod((pol1[(i+1) % l1] - pol1[i % l1]),
                          pol2[(j+1) % l2] - pol2[j % l2])
        # using right-hand rule choose the vector with the lower polar angle and iterate this polygon's vertex
        if cross >= 0:
            i += 1
        if cross <= 0:
            j += 1

    return np.array(msum)


def normalizedMinkowskiSum(pol1, pol2, weights):
    """Calculate Normalized Minkowski sum of two convex polygons.

    Args:
        pol1 (np.ndarray[float, float]): first polygon
        pol2 (np.ndarray[float, float]): second polygon
        weights (np.ndarray[float]): weights for polygons

    Returns:
        np.ndarray[np.ndarray[float, float]]: list of the Minkowski sum vertices
    """
    normp1 = pol1 * weights[0]
    normp2 = pol2 * weights[1]
    return minkowskisum(normp1, normp2)
