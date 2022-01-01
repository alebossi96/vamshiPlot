import copy
from  librerieTesi.diffuseRaman import core
from scipy.signal import peak_widths
import numpy as np

class RawData:
    def __init__(
            self,
            filename: str,
            n_banks: int,
            n_basis: int,
            n_meas: int = -1,
            compress: bool = True,
            tot_banks = None,
            background_idx = None):
        self.n_banks =n_banks
        self.n_basis = n_basis
        if n_meas == -1: # se non specificato è uguale a n_basis, (caso compressioni)
            self.n_meas = n_basis
        else:
            self.n_meas = n_meas
        (self.time,self.data) = core.readSdt(filename, tot_banks)
        self.n_points = len(self.time)
        self.compress = compress
        self.accumulate()
        if background_idx:
            self.remove_bkg(background_idx[0], background_idx[1])
    def accumulate(self) -> None:
        self.tot = np.zeros((self.__len__(),self.n_points ))
        i = 0
        for el in self.data:
            if self.compress:
                idx = i%self.n_basis
                self.tot[idx,:]+= el
            else:
                self.tot[i,:]= el
            i+=1
    def tot_counts(self):
        return sum(sum(self.tot))
    def tot_time_domain(self):
        return np.sum(self.tot, axis = 0)
    def remove_bkg(self, t_start, t_stop):
        idx_start = core.time_to_idx(self.time, t_start)
        idx_stop = core.time_to_idx(self.time, t_stop)
        length = idx_stop-idx_start
        bkg = np.sum(self.tot[:,idx_start:idx_stop], axis = 1)/length
        self.tot =self.tot - bkg[:,np.newaxis]
    def cut(self, t_start, t_stop):
        idx_start = core.time_to_idx(self.time, t_start)
        idx_stop = core.time_to_idx(self.time, t_stop)
        length = idx_stop- idx_start
        self.time = self.time[:length]
        self.tot = self.tot[:,idx_start:idx_stop]
        self.n_points = length
    def __len__(self) -> int:
        if self.compress:
            return self.n_basis
        return len(self.data)#ho copiato tutto, cambio solo tot
    def __call__(self,i0: int, iF: int):#TODO era copy
        new = copy.copy(self)
        new.n_meas = iF-i0
        new.data = self.data[i0:iF]
        new.tot =self.tot[i0:iF,:]
        return new
    def __add__(self,b):
        return self.tot+b.tot
    def __sub__(self,b):
        return self.tot-b.tot
