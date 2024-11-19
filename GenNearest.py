
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        # TP-TODO
        distance = float('inf')
        index = 0
        for i in range(self.videoSkeletonTarget.skeCount()):
            empty = np.zeros((self.videoSkeletonTarget.ske_height_crop, self.videoSkeletonTarget.ske_width_crop, 3), dtype=np.uint8)
            d = ske.distance(self.videoSkeletonTarget.ske[i])
            if distance > d:
                distance = d
                index = i
        
        empty = np.zeros((self.videoSkeletonTarget.ske_height_crop, self.videoSkeletonTarget.ske_width_crop, 3), dtype=np.uint8)
        self.videoSkeletonTarget.ske[index].draw(empty)
        return empty




