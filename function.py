import numpy as np


# function to find the center of the box

def getCenter(left, right, top, down):
    return ((right+left)/2, (down+top)/2)