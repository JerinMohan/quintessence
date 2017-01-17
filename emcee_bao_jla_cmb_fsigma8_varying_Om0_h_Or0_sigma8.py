import sys
sys.path.insert(0, '../../../')
from fluids import *
import emcee
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pl
import corner
from matplotlib.ticker import MaxNLocator
from emcee.utils import MPIPool

#####################################################
# JLA data
#####################################################
dataSN = np.loadtxt('../../../data/jla_mub.txt')
zsn = dataSN[:,0]
mu_b = dataSN[:,1]

covdata = np.loadtxt('../../../data/jla_mub_covmatrix.dat')
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
datafs8 = np.loadtxt('../../../data/fs8.dat')
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

def lnprior(Om0,h,Or0,sigma8):
	if 0.1 < Om0 < 0.9 and 0.5 < h < 0.9 and 3. < Or0 < 9. and 0.6 < sigma8 < 1.:
		return 0.0
	return -np.inf

def lnlike(Om0,h,Or0,sigma8):
	return np.log(lik(Om0,h,Or0,sigma8))

def lnprob(Om0,h,Or0,sigma8):
	lp = lnprior(Om0,h,Or0,sigma8)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(Om0,h,Or0,sigma8)

def ff(theta):
	Om0,h,Or0,sigma8=theta
	return -2.*lnprob(Om0,h,Or0,sigma8)

def lnp(theta):
	Om0,h,Or0,sigma8=theta
	return lnprob(Om0,h,Or0,sigma8)


# Find the maximum likelihood value.

result = opt.minimize(ff, [0.3,0.67,5.,0.7])
Om0_ml,h_ml,Or0_ml, sigma8_ml = result['x']
print("""Maximum likelihood result:
    Om0 = {0} (truth: {1})
    h = {2} (truth: {3})
    Orad = {4} (truth: {5})
    sigma8 = {6} (truth: {7})
""".format(Om0_ml, 0.3, h_ml, 0.67, Or0_ml, 5, sigma8_ml, 0.7))

# Set up the sampler.
ndim, nwalkers = 4, 100
pos = [result['x'] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
#print pos
# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 2000)
print("Done.")


pl.clf()
fig, axes = pl.subplots(4, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(Om0_ml, color="#888888", lw=2)
axes[0].set_ylabel("$\Omega_{m0}$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(h_ml, color="#888888", lw=2)
axes[1].set_ylabel("$h$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(Or0_ml, color="#888888", lw=2)
axes[2].set_ylabel("$\Omega_{r0}$")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(sigma8_ml, color="#888888", lw=2)
axes[3].set_ylabel("$\sigma_8$")

fig.tight_layout(h_pad=0.0)
fig.savefig("plots/line-time_bao_jla_cmb_fsigma8_varying_Om0_h_Or0_sigma8.png")

# Make the triangle plot.
burnin = 200
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

length=len(samples[:,0])

np.savetxt('runs/LCDM_bao_jla_cmb_fsigma8_varying_Om0_h_Or0_sigma8.dat', np.array(samples), fmt='%f', header="Om0 h Or0 sigma8")


test1 = open("runs/LCDM_bao_jla_cmb_fsigma8_varying_Om0_h_Or0_sigma8","w")
test1.write('%s' % (Om0_ml),)
test1.write('%s' % (" "))
test1.write('%s' % (h_ml),)
test1.write('%s' % (" "))
test1.write('%s' % (Or0_ml),)
test1.write('%s' % (" "))
test1.write('%s\n' % (sigma8_ml),)


fig = corner.corner(samples, labels=["$\Omega_{m0}$", "$h$", "$\Omega_{r0}$", r"$\sigma_8$"])
fig.savefig("plots/line-triangle_LCDM_bao_jla_cmb_fsigma8_varying_Om0_h_Or0_sigma8.png")



