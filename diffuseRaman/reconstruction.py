import sys
import copy
from typing import Tuple
import numpy as np
from scipy.sparse.linalg import lsmr
from scipy.linalg import hadamard
from scipy.signal import peak_widths
import librerieTesi.hadamardOrdering as hadamardOrdering
import librerieTesi.timeLasso as tLs
from sklearn import linear_model
from librerieTesi.diffuseRaman import raw_data as rd
RAST = "rast"
#assert False, "Oh no! This assertion failed!"
HAD = "had"
"""
USARE scipy.optimize
class ridgeTime:
    def __init__(self,alpha):
        self.alpha = alpha
    def fit(self, matrix, data):
        self.recons = np.zeros((n_basis,self.nPoints ))
        lmd = 10;
        prev = np.zeros(self.recons[:,0].shape)
        for i in range(self.nPoints ):
            M = np.matmul(self.matrix.T,self.matrix)
            M += lmd *np.eye(M.shape[0])
            b = np.matmul(self.matrix.T, dataNp[:,i]) +lmd* prev
            self.recons[:,i] = lsmr(M,b)[0]
            prev = self.recons[:,i]
"""
#TODO fare velocemente il calcolo della risoluzione
#TODO importare modifiche fatte nel lab computer
#TODO data type f_tT_
#TODO mypy
#TODO dovrei anche selezionare solo un certo tGates
#TODO fare calibrazione direttamente con i wavenumber!
class Reconstruction:
    def __init__(self,
            data: rd.RawData,
            rast_had: str,
            lambda_0: int,
            method: str="lsmr",
            alpha: float=0,
            remove_first_line: bool = True,
            ref_cal_wl : Tuple[int, int] = None,
            ref_bas : int = 32,
            cake_cutting = False,
            normalize = False,
            filename_bkg = None,
            n_banks_bkg= None,
            ):
        self.data = data
        self.rast_had = rast_had
        self.lambda_0 = lambda_0
        self.method = method
        self.ref_cal_wl = ref_cal_wl
        if rast_had == RAST:
            self.remove_first_line = False
        else:
            self.remove_first_line = remove_first_line
        self.n_basis = self.data.n_basis
        self.cake_cutting = cake_cutting
        self.normalize = normalize
        if self.normalize:
            self.normalize_with_background(filename_bkg,n_banks_bkg)
        assert not(rast_had != RAST and rast_had != HAD), "It must be RAST or HAD"
        assert not (rast_had == RAST and not self.data.compress),"It cannot be raster and not compressed!"
        assert not(rast_had == RAST and self.remove_first_line), "It cannot be raster and remove first line!"
        #TODO da fare assert per normalize!
        if method == "lsmr":
            print("lsmr")
            self.reg = linear_model.LinearRegression()
        elif method == "Lasso":
            print("Lasso")
            self.reg = linear_model.Lasso(alpha=alpha)
        elif method == "Ridge":
            print("Ridge")
            self.reg = linear_model.Ridge(alpha=alpha)
        elif method == "tLs":
            print("tLs")
            self.reg = tLs.timeLasso(alpha=alpha)
        self.execute()
        if remove_first_line:
            ref_cal_wl[0] += 1
        self.axis(ref_cal_wl, ref_bas)
        self.start_range = 0
        self.stop_range = len(self.wavelength())
    def M(self) -> np.array:
        dim = self.data.getn_basis()
        if self.cake_cutting: 
            H = 0.5*(hadamardOrdering.cake_cutting(dim) + np.ones((dim,dim)))
        else:
            H = 0.5*(hadamard(dim) + np.ones((dim,dim)))
        if self.data.getIsCompress():
            return H
        matrix = np.zeros((len(self.data), self.n_basis))
        for i in range(len(self.data)):
            matrix[i,:]= H[i%self.n_basis,:]
        return matrix
    def execute(self) -> None:
        if self.rast_had == HAD:
            self.reconstruct_hadamard()
        elif self.rast_had == RAST:
            self.recons = self.data.getData()
        else:
            sys.exit("controlla se hai scritto RAST o HAD")
    def reconstruct_hadamard(self) -> None:
        self.reg.fit(self.M() ,self.data.getData())
        recons1 = self.reg.coef_
        self.recons= recons1
    def axis(self, ref_cal : Tuple[int, int] = None, ref_bas : int = 32) -> None:
        self.calibration_wl(ref_cal, ref_bas)
        self.calibration_wn()
    def calibration_wl(self, ref_cal_wl : Tuple[int, int] = None, ref_bas : int = 32) -> None:
        #ref is in a basis 32
        p = np.array([  1.51491966* 64/self.n_basis, 809.28844682])
        if ref_cal_wl is not None:
            p[1] = ref_cal_wl[1] - ref_cal_wl[0]*p[0]*self.n_basis/ref_bas          
        self.wl=np.zeros(self.n_basis)
        for i in range(self.n_basis):
            self.wl[i]=i*p[0]+p[1]
    def calibration_wn(self) -> None:
        self.wn=np.zeros(self.n_basis)
        for i in range(self.n_basis):
            self.wn[i]=(1/self.lambda_0-1/self.wl[i])*1e7
    def wl_with_time(self, time: int) -> np.array:
        idx = np.where(self.time == time)
        return self.recons[:,idx]
    def line(self, idx: int) -> np.array:
        return self.recons[idx,:]
    def wavenumber(self) -> np.array:
        if not self.normalize:
            if self.remove_first_line:
                return self.wn[1:]
            return self.wn
        return self.wn[self.start_range:self.stop_range]
    def wavelength(self) -> np.array:
        if not self.normalize:
            if self.remove_first_line:
                return self.wl[1:]
            return self.wl
        return self.wl[self.start_range:self.stop_range]
    def time(self, inNs: bool = True) -> np.array:
        if inNs:
            conv = 1e9
        else:
            conv = 1
        return self.data.time * conv
    def reconstruction(self) -> np.array:
        if self.rast_had == RAST:
            return self.recons
        recons = self.recons.T
        if not self.normalize:
            if self.remove_first_line:
                return recons[1:,:]
            return recons
        norm_spect = recons[ self.start_range:self.stop_range,:]#TODO/self.bkg_spectr[self.start_range:self.stop_range, np.newaxis]
        return norm_spect
    def find_maximum_idx(self):
        return np.where(self.spectrograph() == np.amax(self.spectrograph()))[0][0]
        #TODO useful for calibration but I will need to implement it
    def fwhm(self,axis: np.array,data: np.array) -> float:
        pos0 = np.where(data == np.amax(data))[0][0]
        fwhm_idx = int(peak_widths(data, [pos0])[0])
        fwhm =  (axis[pos0+fwhm_idx] - axis[pos0-fwhm_idx])
        return fwhm
    def t_gates(self, init, fin):
        #TODO dovrei fare nuovo oggetto?
        init = int(init)
        fin = int(fin)
        return np.sum(self.reconstruction()[:,init:fin],axis = 1)
    def spectrograph(self):
        return np.sum(self.reconstruction(), axis = 1)
    def toF(self):
        return np.sum(self.reconstruction(), axis = 0)
    def background(self, initial_pos, final_pos):
        tot_bkg_wn = np.sum(self.reconstruction()[:,initial_pos:final_pos],axis = 1)
        return tot_bkg_wn/(final_pos-initial_pos)
    def reconstruction_remove_background(self,initial_pos, final_pos):
        return self.reconstruction() - self.background(initial_pos, final_pos)[:, np.newaxis]
    def normalize_with_background(self, filename_bkg,n_banks):
        data = rd.RawData(filename = filename_bkg,
                n_banks = n_banks,#Banks[i],
                n_basis = self.n_basis,#B_measurements[i],
                compress = True,
                )
        rec = Reconstruction(data = data,
            rast_had = HAD,
            lambda_0 = self.lambda_0,
            method ="lsmr",
            remove_first_line = True,
            ref_cal_wl = self.ref_cal_wl,
            ref_bas = self.n_basis)
        self.bkg_spectr = rec.spectrograph()/np.max(rec.spectrograph())
        limit = 0.2
        self.start_range = np.argmax(self.bkg_spectr > limit)
        self.stop_range = np.argmin(self.bkg_spectr[(self.start_range+10):] > limit) +self.start_range+10
    def __len__(self):
        if self.remove_first_line:
            return self.n_basis -1
        return self.n_basis
    def __sub__(self, b):
        new = copy.copy(self)
        return new.recons-b.recons
    def __sum__(self):
        return sum(self.spectrograph())
