import miditoolkit

import chord_recognition
import utils


note_items, tempo_items = utils.read_items('./data/evaluation/000.midi')
note_items = utils.quantize_items(note_items)
chord_items = utils.extract_chords(note_items)

print(*chord_items, sep='\n')
