import numpy as np


class Node:

    def __init__(self, antennas, pos, node_id):
        (p1, p2, p3) = pos
        self.pos = np.array([p1, p2, p3]).T
        (self.nx, self.ny, self.nz) = antennas
        self.antnum = self.nx * self.ny * self.nz
        self.node_id = node_id

    def get_antennas(self):
        return self.nx, self.ny, self.nz


class BS(Node):

    def __init__(self, antennas, pos, bs_id):
        Node.__init__(self, antennas, pos, bs_id)


class Vehicle(Node):

    def __init__(self, antennas, pos, vehicle_id, speed):
        Node.__init__(self, antennas, pos, vehicle_id)
        self.speed = speed


class IRS(Node):

    def __init__(self, antennas, pos, irs_id):
        Node.__init__(self, antennas, pos, irs_id)