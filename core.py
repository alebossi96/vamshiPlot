import math
import pandas as pd
import numpy as np
import csv

class Context:
    def __init__(self):
        pass

def get_delimiter(file_path):
    sniffer = csv.Sniffer()
    with open(file_path, "r") as f:
        for line in f:
            return sniffer.sniff(line).delimiter
class SelectionFromInstruction:
    def __init__(self, i, column, scenario):
        self.select = None
        self.value = None
        self.range_select_min = None
        self.range_select_max = None
        self.subsample = None
        for el in column:
            #mettere qui che non voglio ''
            if "Select"+str(i+1) == el:
                self.select = scenario["Select"+str(i+1)]
            elif "Value"+str(i+1) == el:
                self.value = scenario["Value"+str(i+1)]
            elif "Range_min"+str(i+1) == el:
                self.range_select_min = scenario["Range_min"+str(i+1)]
            elif "Range_max"+str(i+1) == el:
                self.range_select_max = scenario["Range_max"+str(i+1)]
            elif "Subsample"+str(i+1) == el:
                self.subsample = scenario["Subsample"+str(i+1)]
class Selection:
    def __init__(self, sfi, idx):
        self.select = None
        self.value = None
        self.range_select_min = None
        self.range_select_max = None
        self.subsample = None
        if sfi.select is not None and sfi.select[idx] != '':
            self.select = sfi.select[idx]
        if sfi.value is not None and sfi.value[idx] != '':
            self.value = sfi.value[idx]
        if sfi.range_select_min is not None and sfi.range_select_min[idx] != '':
            self.range_select_min = sfi.range_select_min[idx]
        if sfi.range_select_max is not None and sfi.range_select_max[idx] != '':
            self.range_select_max = sfi.range_select_max[idx]
        if sfi.subsample is not None and sfi.subsample[idx]!= '':
            self.subsample = sfi.subsample[idx]
    def select_data(self, data):
        """
        Deve essere solo una delle 4?
        """
        if pd.isna(self.select):
            return data
        if self.subsample is not None :
            by_wl = data.groupby(self.select)
            index = pd.pivot_table(data, index = self.select).index
            list_group = []
            to_get = np.arange(0, len(index), self.subsample)
            index_to_get = index[to_get]
            new = []
            for idx, frame in by_wl:
                if idx in index_to_get:
                    new.append(frame)
            data = pd.concat(new,sort=False)
        if self.value is not None:
            data = data[data[self.select]==self.value]
        if self.range_select_min is not None and self.range_select_max is not None:
            data = data[(data[self.select]>self.range_select_min) & (data[self.select]<self.range_select_max)]
        if self.range_select_min is not None:
            data = data[data[self.select]>self.range_select_min]
        if self.range_select_max is not None:
            data =  data[data[self.select]<self.range_select_max]
        return data.sort_index().reset_index()
def invert_selection_instr(sfi):
    i_len = len(sfi)
    idx_len = len(sfi[0].select)
    sel = []
    for idx in range(idx_len):
        sel_el = []
        for i in range(i_len):
            if sfi[i].select[idx] is not None:
                sel_el.append(Selection(sfi[i], idx))
        sel.append(sel_el)
    return sel

