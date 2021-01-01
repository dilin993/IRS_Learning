from numpy.core import overrides

from nodes.node import Node
from channelmodels.raytracing import RayTracingChannelModel, RayTracingLosChannelModel


class Links:

    def __init__(self):
        self.links = {}

    def add_link(self, node1, node2, channel):
        self.links[(node1.node_id, node2.node_id)] = channel

    def get_link(self, node1, node2):
        return self.links[(node1.node_id, node2.node_id)]


class RayTracingLinks(Links):

    def __init__(self, data_path):
        Links.__init__(self)
        self.data_path = data_path

    def add_link(self, node1, node2, point1, point2):
        Links.add_link(self, node1, node2, RayTracingChannelModel(node2.antnum, node1.antnum, node2.node_id,
                                                                  node1.node_id, point2, point1, self.data_path))

    def modify_link(self, node1, node2, point1, point2):
        self.links[(node1.node_id, node2.node_id)] = RayTracingChannelModel(node2.antnum, node1.antnum, node2.node_id,
                                                                            node1.node_id, point2, point1,
                                                                            self.data_path)


class RayTracingLosLinks(Links):

    def __init__(self, data_path, fc, delta):
        Links.__init__(self)
        self.data_path = data_path
        self.fc = fc
        self.delta = delta

    def add_link(self, node1, node2, point1, point2):
        Links.add_link(self, node1, node2, RayTracingLosChannelModel(node2, node1, point2, point1, self.data_path,
                                                                     self.fc, self.delta))

    def modify_link(self, node1, node2, point1, point2):
        self.links[(node1.node_id, node2.node_id)] = RayTracingLosChannelModel(node2, node1, point2, point1,
                                                                               self.data_path, self.fc, self.delta)
