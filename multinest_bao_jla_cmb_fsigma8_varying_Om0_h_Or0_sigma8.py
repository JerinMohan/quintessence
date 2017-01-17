import json
from numpy import log, exp, pi
import scipy.stats, scipy
import pymultinest
import threading, subprocess 
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pl
import corner
from matplotlib.ticker import MaxNLocator
from emcee.utils import MPIPool
import sys
from fluids import *
from scalar_all_eqn import *

#####################################################
# JLA data
#####################################################
dataSN = np.loadtxt('jla_mub.txt')
zsn = dataSN[:,0]
mu_b = dataSN[:,1]

covdata = np.loadtxt('jla_mub_covmatrix.dat')
covarray = covdata[1:len(covdata)]

covmat = covarray.reshape(31,31)

def chijla(Om0,h,Or0,sigma8):
	ld = np.vectorize(LCDM(Om0,h,0.045,Or0*1e-5,0.96,sigma8).luminosity_distance_z)
	x = 5.0*np.log10(ld(zsn))+25.-mu_b
	covinv = np.linalg.inv(covmat)
	return np.dot(x,np.dot(covinv,x))

######################################################
# fsigma8 data
######################################################
datafs8 = np.loadtxt('fs8.dat')
zfs8 = datafs8[:,0]
fs8 = datafs8[:,1]
sgm_fs8 = datafs8[:,2]

def chifs8(Om0,h,Or0,sigma8):
	fs8_th = LCDM(Om0,h,0.045,Or0*1e-5,0.96,sigma8).fsigma8z(zfs8)
	return np.sum((fs8-fs8_th)**2./sgm_fs8**2)

######################################################
# BAO data
######################################################
covinv = [[0.48435, -0.101383, -0.164945, -0.0305703, -0.097874, -0.106738], [-0.101383, 3.2882, -2.45497, -0.0787898, -0.252254, -0.2751], [-0.164945, -2.454987, 9.55916, -0.128187, -0.410404, -0.447574], [-0.0305703, -0.0787898, -0.128187, 2.78728, -2.75632, 1.16437], [-0.097874, -0.252254, -0.410404, -2.75632, 14.9245, -7.32441], [-0.106738, -0.2751, -0.447574, 1.16437, -7.32441, 14.5022]]

def chibao(Om0,h,Or0,sigma8):
	lcdm = LCDM(Om0,h,0.045,Or0*1e-5,0.96,sigma8)
	Dh = lcdm.D_H()
	invdh = 1/Dh
	xx = [lcdm.comoving_distance_z(1091.)*invdh/((lcdm.comoving_distance_z(0.106)*invdh)**2*0.106/(lcdm.hubble_normalized_z(0.106)))**(1./3) - 30.95, lcdm.comoving_distance_z(1091.)*invdh/((lcdm.comoving_distance_z(0.2)*invdh)**2*0.2/(lcdm.hubble_normalized_z(0.2)))**(1./3) - 17.55, lcdm.comoving_distance_z(1091.)*invdh/((lcdm.comoving_distance_z(0.35)*invdh)**2*0.35/(lcdm.hubble_normalized_z(0.35)))**(1./3) - 10.11, lcdm.comoving_distance_z(1091.)*invdh/((lcdm.comoving_distance_z(0.44)*invdh)**2*0.44/(lcdm.hubble_normalized_z(0.44)))**(1./3) - 8.44, lcdm.comoving_distance_z(1091.)*invdh/((lcdm.comoving_distance_z(0.6)*invdh)**2*0.6/(lcdm.hubble_normalized_z(0.6)))**(1./3) - 6.69, lcdm.comoving_distance_z(1091.)*invdh/((lcdm.comoving_distance_z(0.73)*invdh)**2*0.73/(lcdm.hubble_normalized_z(0.73)))**(1./3) - 5.45]
	chibao1 = np.dot(xx,np.dot(covinv,xx))
	return chibao1

######################################################
# CMB Priors
#####################################################
def Rth(z,Om0,h,Or0):
	lcdm = LCDM(Om0,h,0.045,Or0*1e-5,0.96,sigma8)
	Dh = lcdm.D_H()
	return np.sqrt(lcdm.om_z(0.))*(1+z)*lcdm.ang_dis_z(z)/Dh

def chishift(Om0,h,Or0,sigma8):
	return (1.7499-Rth(z_star(Om0,h),Om0,h,Or0))**2/(0.0088**2)

corr_mat = np.array([[1.,0.5262,0.4708],[0.5262,1.0,0.8704],[0.4708,0.8704,1]])
err_mat = np.array([0.18,0.0088,0.53])
cov_cmb = []
for i in range(len(err_mat)):
	for j in range(len(err_mat)):
		cov_cmb = cov_cmb + [corr_mat[i,j]*err_mat[i]*err_mat[j]]
cov_cmb = np.array(cov_cmb)
cov_cmb = cov_cmb.reshape(3,3)
icovcmb = np.linalg.inv(cov_cmb)
def chicmb(Om0,h,Or0,sigma8):
	vec = np.array([acoustic_length(z_star(Om0,h),Om0,h,Or0)-301.18, Rth(z_star(Om0,h),Om0,h,Or0)-1.7499, z_star(Om0,h)-1090.41])
	return np.dot(vec,np.dot(icovcmb,vec))

corr_mat1 = np.array([[1.,0.54],[0.54,1.0]])
err_mat1 = np.array([0.14,0.0074])
cov_cmb1 = []
for i in range(len(err_mat1)):
	for j in range(len(err_mat1)):
		cov_cmb1 = cov_cmb1 + [corr_mat1[i,j]*err_mat1[i]*err_mat1[j]]
cov_cmb1 = np.array(cov_cmb1)
cov_cmb1 = cov_cmb1.reshape(2,2)
icovcmb1 = np.linalg.inv(cov_cmb1)
def chicmb1(Om0,h,Or0,sigma8):
	lcdm = LCDM(Om0,h,0.045,Or0*1e-5,0.96,sigma8)
	vec = np.array([lcdm.acoustic_length()-301.18, lcdm.cmb_shift_parameter()-1.7499])
	return np.dot(vec,np.dot(icovcmb1,vec))

#####################################################


def chi2(Om0,h,Or0,sigma8):
	return chibao(Om0,h,Or0,sigma8)+chijla(Om0,h,Or0,sigma8)+chicmb1(Om0,h,Or0,sigma8)+chifs8(Om0,h,Or0,sigma8)

def lik(Om0,h,Or0,sigma8):
	return np.exp(-chi2(Om0,h,Or0,sigma8)/2.)

"""
def lik(Om0,h,Or0,sigma8):
	ll= np.exp(-chi2(Ophi,li,Orad_i,h,sigma8)/2.)
        if ll <=0:
           return .000000000001
        return ll
"""
def lnprior(Om0,h,Or0,sigma8):
	if 0.1 < Om0 < 0.9 and 0.5 < h < 0.9 and 3. < Or0 < 9. and 0.6 < sigma8 < 1.:
		return 0.0
	return -np.inf

"""
def lnlike(Ophi,li,Orad_i,h,sigma8):
	return np.log(lik(Om0,h,Or0,sigma8))
"""

def lnlike(Om0,h,Or0,sigma8):

        return -chi2(Om0,h,Or0,sigma8)/2.

def lnprob(Om0,h,Or0,sigma8):
	lp = lnprior(Om0,h,Or0,sigma8)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(Om0,h,Or0,sigma8)


def prior(cube, ndim, nparams):
       cube[0] = cube[0] * 0.9
       cube[1] = cube[1] * 0.9
       cube[2] = cube[2] * 9.
       cube[3] = cube[3] * 1.
      


def loglike(cube, ndim, nparams):
       Om0 = cube[0]
       h   = cube[1]
       Or0= cube[2]
       sigma8 = cube[3]
       #print (len(lnprob(Ophi,li,Orad_i,h,sigma8)))
       return lnlike(Om0,h,Or0,sigma8)


parameters = ["Om0","h","Or0","sigma8"]
n_params = len(parameters)


pymultinest.run(loglike, prior, n_params,importance_nested_sampling = True,evidence_tolerance = 0.5,sampling_efficiency = 'model',n_iter_before_update = 10, outputfiles_basename='out/', resume = False, verbose = True, log_zero = -1e100, max_iter=0, init_MPI = False,n_live_points = 400)

json.dump(parameters, open('out/params.json', 'w')) # save parameter names 
