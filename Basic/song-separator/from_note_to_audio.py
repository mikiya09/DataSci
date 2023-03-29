from music21 import *
from mingus.midi import fluidsynth
import time

fluidsynth.init("./soundFonts/Strings.sf2")
fluidsynth.play_Note(note("C-5"))