import numpy as np
from numba import *

@jit(nopython=True)
def compute_fourier_block(x, hc):
    l, nx = x.size, np.pi * x / 2
    sin, cos = np.sin(nx), np.cos(nx)
    rex, imx = cos.copy(), sin.copy()
    #tmp = np.empty(l, dtype=rex.dtype)
    reh = np.empty(hc, dtype=cos.dtype)
    imh = np.empty(hc, dtype=sin.dtype)

    reh[0], imh[0] = l, 0
    for h in range(1, hc):
        reh[h] = np.sum(rex, axis=0)
        imh[h] = np.sum(imx, axis=0)
        
        #Numba does not support 
        #Numpy copyto function
        #np.copyto(tmp, rex)
        tmp = rex.copy()
        rex = rex * cos - imx * sin
        imx = tmp * sin + imx * cos

    return (reh, imh)

@jit(nopython=True, parallel=True)
def compute_fourier(x, hc, bk = 4096):
    xl = np.asarray(x)
    lx, dx = xl.size, xl.dtype
    bc = lx // bk + bool(lx % bk)
    reh = np.empty((bc, hc), dtype=dx)
    imh = np.empty((bc, hc), dtype=dx)
    for b in prange(bc): 
        f = min(b * bk, lx)
        l = min(f + bk, lx)
        xi = xl[f:l]
        r, i = compute_fourier_block(xi, hc)
        reh[b, :], imh[b, :] = r, i
    return (np.sum(reh, axis=0), np.sum(imh, axis=0))

@jit(nopython=True)
def evaluate_fourier_block(x, re, im):
    hc = re.size
    assert hc == im.size
    xs, nx = x.shape, np.pi * x / 2
    sin, cos = np.sin(nx), np.cos(nx)
    rex, imx = cos.copy(), sin.copy()
    
    res = re[0] * np.ones(xs, dtype=x.dtype)
    for h in range(1, hc):
        res += re[h] * rex + im[h] * imx
        
        #Numba does ot support 
        #Numpy copyto function
        #np.copyto(tmp, rex)
        tmp = rex.copy()
        rex = rex * cos - imx * sin
        imx = tmp * sin + imx * cos

    return res
    
@jit(nopython=True, parallel=True)
def evaluate_fourier(x, re, im, bk = 512):
    xl = np.asarray(x)
    lx, dx = xl.size, xl.dtype
    bc = lx // bk + bool(lx % bk)
    res = np.empty(lx, dtype=dx)
    for b in prange(bc): 
        f = min(b * bk, lx)
        l = min(f + bk, lx)
        xi = xl[f:l]
        r = evaluate_fourier_block(xi, re, im)
        res[f:l] = r
    return res
