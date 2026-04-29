import math

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def body_length(head, tail):
    return distance(head, tail)

def height(withers, ground):
    return distance(withers, ground)