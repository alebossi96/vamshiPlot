import pandas as pd
import numpy as np
def hour_to_sec(df, axis):
    to_fill = 2*np.arange(0, len(df))
    df[axis][:] = to_fill
    return df
