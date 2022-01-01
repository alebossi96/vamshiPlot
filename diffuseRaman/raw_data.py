import copy
from typing import List, Tuple
from  librerieTesi.diffuseRaman import core
from scipy.signal import peak_widths
import numpy as np
import  sdtfile
class RawData:
    def __init__(
            self,
            filename: str,
            n_banks: int,
            n_basis: int,
            n_meas: int = -1,
            compress: bool = True,
            tot_banks = None,
            background_time = None):
        self.n_banks =n_banks
        self.n_basis = n_basis
        if n_meas == -1: # se non specificato è uguale a n_basis, (caso compressioni)
            self.n_meas = n_basis
        else:
            self.n_meas = n_meas
        (self.time,self.data) = self.readSdt(filename, tot_banks)
        self.n_points = len(self.time)
        self.compress = compress
        self.accumulate()
        if background_time:
            self.remove_bkg(background_time[0], background_time[1])
    def readSdt(self, filename: str, tot_banks = None ) -> Tuple[np.array,List]:
        data = []
        for i in range(self.n_banks):
            if tot_banks:
                zeros = int(np.log10(tot_banks+1))
            else:
                zeros = int(np.log10(self.n_banks+1))
            fn = filename
            if self.n_banks == 1:
                sdt = sdtfile.SdtFile(fn+".sdt")
            elif i==0:
                sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
            elif i<10:
                sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
            elif i<100:
                sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-1)+str(i)+".sdt")
            elif i<1000:
                sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-2)+str(i)+".sdt")
            elif i<10000:
                sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-3)+str(i)+".sdt")
            time = sdt.times[0][:]
            batch = sdt.data[0].shape[0]
            for j in range(batch):
                data.append(sdt.data[0][ j,:])
        return (time,data)
    
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
    def t_gate_bin(self, num_t_gates):
        time = np.zeros((num_t_gates,))
        tot = np.zeros((self.__len__(),num_t_gates))
        step = self.n_points/num_t_gates
        for i in range(num_t_gates):
            time[i] = self.time[int(i*step)]
            tot[:,i] = np.sum(self.tot[:,int(i*step):int((i+1)*step)], axis = 1)
        self.time = time
        self.tot = tot
        self.n_points = num_t_gates
    def change_um_time(self, conv):
        self.time *= conv
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
