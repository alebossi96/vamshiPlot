import pandas as pd
import numpy as np
def hour_to_sec(df, axis):
    to_fill = 2*np.arange(0, len(df))
    df[axis][:] = to_fill #TODO da correggere
    return df
def el_to_sec(df, axis):
    to_fill = 1*np.arange(0, len(df))
    df[axis][:] = to_fill #TODO da correggere
    return df
def hour_to_sec10(df, axis):
    to_fill = 10*np.arange(0, len(df))
    df[axis][:] = to_fill #TODO da correggere
    return df
