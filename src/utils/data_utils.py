import numpy as np

def crop_sequence(seq, start, T, sample_rate=1):
    return seq[start:start + T * sample_rate:sample_rate]