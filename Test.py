from ASCIIArt import *
from time import perf_counter
class Timing():
    def __init__(self, sinceLastOut=False, ms=False):
        self.s = perf_counter()
        self.last = self.s
        self.sinceLastOut = sinceLastOut
        self.unit = 1000 if ms else 1

    def out(self,label):
        print(f"{label}: {((e := perf_counter()) - self.last)*self.unit}")
        if self.sinceLastOut:
            self.last = e

    def totalOut(self, label=""):
        if not self.sinceLastOut:
            self.out(label)
        print(f"Total: {(perf_counter() - self.s)*self.unit}")

def timedTest(*args, **kwargs):
    t = Timing(True, True)
    asciiart = ASCII_Image(*args, **kwargs)
    t.out("Initialise")
    output = asciiart.map()
    t.out("Map")
    asciiart.map("output.txt")
    t.out("File Output")
    t.totalOut()

sentence = None
with open("Text/Never Gonna Give You Up.txt") as f:
    sentence = "".join([x for x in f.read() if x.isalnum() or x in "-.,"])
"""
Sentence sequence: ('&','#', sentence.upper(), sentence.lower(), '?', '<', '_-', ' ')
Binary Texture: ["CharTextures/1.txt", "CharTextures/0.txt", "CharTextures/0Negative.txt", "CharTextures/1Negative.txt"]
"""
timedTest("Images/Companion Cube.png", characters=["CharTextures/1.txt", "CharTextures/0.txt", "CharTextures/0Negative.txt", "CharTextures/1Negative.txt"], adjustFactor=1,mapping=TextureMapping, width=500)