import numpy as np
from PIL import Image
from math import ceil, floor

chars64 = "$@B%&WM#*Z0OQLCJUYXoahkbdpqwmzcvunxrjft/|1[]?-_+~<>i!lI;:,^\"`'. "
chars16 = "$@%&#?Ili*<>!;:,+~^\".'_- "
chars8 = "$%#/_-."
"""Sentence = ('@$%&','#', song.upper(), song.lower(), '?!', '\\/', '_.', ' ')"""

X, Y = 0, 1
CHARS, THRESHOLD = 0, 1
class ImageHandler():    
    def __init__(self, filename, width):
        with Image.open(filename) as image:
            height = round(image.height*(width/image.width)*0.5)
            self.dimensions = (width, height)
            image = image.convert("L").resize(self.dimensions)
            self.image = np.asarray([[image.getpixel((x,y)) for x in range(width)] for y in range(height)], dtype=np.uint8)
    
    def __getitem__(self, x, y):
        return self.image[y][x]
    
    def __iter__(self):
        return np.nditer(self.image)

    def rows(self):
        for y in range(self.dimensions[Y]):
            yield iter(self.image[y])

    def __len__(self):
        return self.dimensions

class Mapping():
    def __init__(self, chars, thresholds, image=None):
        self.chars = chars
        self.mappedThresholds = self.getMapping(thresholds)

    def getMapping(self, thresholds):
        thresholds[-1] = 256
        mapping = np.zeros(256,np.uint8)
        prevThreshold = 0
        for i, threshold in enumerate(thresholds):
            threshold = min(threshold, 256)
            mapping[prevThreshold:threshold] = [i]*(threshold - prevThreshold)
            prevThreshold = threshold
        return mapping

    def __getitem__(self,pixel):
        return self.chars[self.mappedThresholds[pixel]]
    
class SequenceMapping(Mapping):
    def __init__(self, chars, thresholds, image=None):
        super().__init__(chars, thresholds, image)
        self.i = -1
    
    def __getitem__(self, pixel):
        self.i += 1
        location = self.chars[self.mappedThresholds[pixel]]
        return location[self.i%len(location)]

class ASCII_Image():
    def __init__(self, filename, width=120, characters=chars8, thresholds=None, image:ImageHandler=ImageHandler,mapping:Mapping=Mapping, invert=False, adjustFactor=0, **kwargs):
        self.mappedImage = None
        self.distribution = None
        self.image = image(filename, width)
        self.chars = characters if not invert else characters[::-1]
        if not thresholds:
            i = ceil(256/len(self.chars))
            thresholds = [x for x in range(i,256+i,i)]
        if adjustFactor:
            adjusted = self.getColourDistributedThresholds(len(characters))
            thresholds = [floor(a*(1-adjustFactor)+b*adjustFactor) for a, b in zip(thresholds,adjusted)]
        self.mapping = mapping(self.chars, thresholds, self.image,**kwargs)

    def getColourDistributedThresholds(self, n):
        thresholds = [255]*n
        rSum = 0
        partitions = self.image.dimensions[0]*self.image.dimensions[1]/n
        tI = 0
        tN = partitions
        for i, nPixels in enumerate(self.colourDistribution()):
            rSum += nPixels
            if rSum >= tN:
                thresholds[tI] = i
                tN += partitions
                tI += 1
        a = thresholds[-1]
        step = min(4, a//n)
        for i in range(-2,-n,-1):
            if thresholds[i] >= a:
                a -= step
                thresholds[i] = a
            else: break
        return thresholds

    def colourDistribution(self):
        if not self.distribution:
            self.distribution = np.zeros(256,np.uint64)
            for pixel in self.image:
                self.distribution[pixel] += 1
        return self.distribution
    
    def map(self, fileName=None):
        if not self.mappedImage:
            self.mappedImage = "\n".join(["".join([self.mapping[pixel] for pixel in row]) for row in self.image.rows()])
        if fileName:
            with open(fileName, 'w') as f:
                f.write(self.map())
        return self.mappedImage