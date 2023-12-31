import numpy as np
from ASCIIArt import *
from threading import Thread
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

    def tOut(self, label=""):
        if not self.sinceLastOut:
            self.out(label)
        print(f"Total: {(perf_counter() - self.s)*self.unit}")

sentence = None
with open("Text/Never Gonna Give You Up.txt") as f:
    sentence = "".join([x for x in f.read() if x.isalnum() or x in "-.,"])
t = Timing(True, True)
asciiart = ASCII_Image("Images/Companion Cube.png", characters=('@$%&','#', sentence.upper(), sentence.lower(), '?!', '\\/', '_.', ' '), adjustFactor=1,mapping=SequenceMapping, width=160, invert=True)
t.out("Initialise")
output = asciiart.map()
t.out("Map")
asciiart.map("output.txt")
t.out("File Output")
t.tOut()