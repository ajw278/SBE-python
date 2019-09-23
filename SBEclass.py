from __future__ import print_function

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from pylab import cm
from scipy.ndimage import convolve1d
from matplotlib.colors import LogNorm
import time
import os
import scipy.special as special
from scipy.integrate import simps


ALPHA=5.4e-3

class SBEClass():
	
	initialize = False
	
	def __init__(self, sigma0, Q, omega0, mach=None, rho0=None, qvir = 1.3,tsn = 3., Mcrit=1e3, MMCmin=10.0, mumass=0.5, timeit=False, Lcrit=8e38, oname='sbeclass_instance'):
		self.initialize=True
		self.time=timeit

		pi = np.pi
		self.G = 6.67e-11 # units: m**3 kg**-1 s**-2
		self.MSun = 1.989e30 # units: kg
		self.pc = 3.086e16 # 1 pc in meters
		self.Myr = 3.156e13 # 1 Myr in seconds
		self.phi_p = 3.0 # constant to account for the gravity of stars (K12b)
		self.phifb = 1.6e-5 # feedback efficiency (units: m**2 s**-3) (K12b)
		self.phix = 1.12 # constant (KM05)
		self.phit = 1.91 # constant (FK12)
		self.theta = 0.35 # constant (PN2011)
		self.ycut = 0.1 # constant (HC2011)
		self.sigSB = 5.67e-8 # Stefan-Boltzmann constant
		self.c = 299792458.0 # speed of light
		self.kappa0 = 2.410e-5 # opacity constant
		self.psi = 0.3 # light-to-mass ratio
		self.phitrap = 0.2 # trapping ratio
		self.eta = 2.0*1.305*3.0*pi/64.0 # for Plummer
		self.g_close = 1.5 # close encounter correction
		self.phish = 2.8 # higher-order energy loss correction
		self.rh2r2av = 0.25 # for Plummer
		self.f = 0.7 # fraction of injected energy that is used for unbinding the region
		self.sfemax =0.5
		self.fbr = 8.0
		self.alpha = 2.55
		self.beta =0.42
		self.d1 = 2.9
		self.d2 = 4.0
		self.G0 = 1.6e-3
		self.vdisp = 3.0

		self.theta0 = 0.59
		self.theta1 = 0.70
		self.theta2 = 0.71
		self.theta3 = 3.9
		self.theta4 = 0.36

		self.Lcrit = Lcrit

		self.MOB = 99.
		self.sfe_th = 0.2
		self.sfe_eff= 0.1

		self.sigmaFf = 0.5

		self.oname = oname


		self.sigma0 = sigma0
		self.Q = Q
		self.omega0=omega0

		self.mumass = mumass
		self.qvir = qvir
		self.tsn = tsn

		self.Mcrit=Mcrit
		self.MMCmin = MMCmin


		if rho0==None:
			self.get_rho0()
		else:
			self.rho0=rho0

		if mach==None:
			self.get_mach()
		else:
			self.mach =mach

		self.get_Cext()

		#set limits on the max and min allowed masses of star forming regions
		self.Mcminmin=10.
		self.Mcminmax=1e5
		self.Mcmaxmax = 1e9
		self.get_Mc_max()
		self.get_Mc_min()

		
		self.F0f = max(self.F0f_func(),1e-6)

		return None

	def get_h0(self):
		sigma0=self.sigma0
		rho0 = self.get_rho0()

		self.h0 = sigma0/(2.*rho0)
		return self.h0


	#Calculate field flux in units of G0
	def F0f_func(self):

		#get scale height of galactic disc
		h0 =self.get_h0()

		#Calculate the mean cluster radius
		MMC_space = np.logspace(-5, 10, 100)
		sfe = self.tildesfe_func(MMC_space)

		#Define phispace by MMC_space
		phispace = MMC_space*sfe/self.Mcrit

		#Get ICMF (un-weight by phi)
		dpdphi = self.dpdphi_func(phispace)/phispace
		norm = np.trapz(dpdphi, phispace)
	
		RGMC = self.Rgmc_func(MMC_space)

		RGMC_mean = np.trapz(RGMC*dpdphi, phispace)/norm


		#Mean sep. between star forming regions		
		d0  = max(2*h0,2*RGMC_mean)
		d0_ = d0*self.pc*1e2
		
		#Redefine phispace 
		phispace = np.logspace(-3., 5,100)


		#Get ICMF (un-weight by phi)
		igrand =  self.dpdphi_func(phispace)/phispace
		norm = np.trapz(igrand, phispace)
		igrand/=norm

		xspace_all = np.logspace(-10, 6, 200)
		dpdx_all = self.dpdx_func(xspace_all)
		norm_all = np.trapz(dpdx_all, xspace_all)

		#Calculate factor due to extinction
		ext_factor = np.trapz(dpdx_all*np.exp(-xspace_all*(2.*d0/h0)*self.Cext), xspace_all)/norm_all
		phifactor = np.amax(np.array([np.ones(len(phispace)), phispace]), axis=0)

		meanF =np.trapz(igrand*ext_factor*phifactor*self.Lambda_phi_func(phispace)/(d0_**2.), phispace)*self.Lcrit

		self.F0f = meanF/self.G0

		return self.F0f		


	#Fraction of mass in GMCs
	def fgmc_func(self,sigma0=None):
		if sigma0==None:
			sigma0=self.sigma0
		surfg2 = sigma0/100. #surface density in units of 100 MSun/pc**2
		return 1.0/(1+0.025*surfg2**(-2)) #molecular gas fraction

	#Mach number 
	def get_mach(self,sigma0=None,Q=None,omega0=None):
		if sigma0==None:
			sigma0=self.sigma0
		if Q==None:
			Q = self.Q
		if omega0==None:
			omega0 = self.omega0
		
		surfg2 =sigma0/100.0
		phi = (10-8*self.fgmc_func(sigma0)) #ratio between cloud pressure and mid-plane pressure
		self.mach = 2.82*phi**0.125*Q/omega0*surfg2 # 1D Mach number

		return self.mach

	#ISM density
	def get_rho0(self, Q=None, sigma0=None, omega0=None):

		if sigma0==None:
			sigma0=self.sigma0
		
		if omega0==None:
			omega0 = self.omega0

		rho0 = 3.*(omega0/self.Myr)**2./(np.pi*self.G*self.Q**2.)
		self.rho0 = rho0*self.pc**3./self.MSun

		return self.rho0

	#2d free-fall time-scale
	def tff2d_func(self, omega0=None):
		if omega0==None:
			omega0=self.omega0

		return np.sqrt(np.pi)/omega0


	#Density dependent feedback time-scale
	def tfb_fcoll_func(self, x, tff, tsn, sigma0=None, omega0=None, mach=None, rho0=None, Q=None):
		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0= self.omega0
		if mach==None:
			mach= self.mach
		if rho0==None:
			rho0 = self.rho0
		if Q==None:
			Q =self.Q

		fMC = self.fgmc_func()
		fSigma = self.fSigma_func(fMC)
		surffb = max(sigma0*fSigma, sigma0)

		surfg = sigma0*self.MSun/self.pc**2.
		omega=omega0/self.Myr
		tsn_ = tsn*self.Myr
		ssfrff = self.ssfrff_func(mach)
		tff_ = tff*self.Myr

		frac = 4*np.pi**2*self.G**2*tff_*Q**2*surfg**2/(self.phifb*0.012*tsn_**2*omega**2*x)


		return tsn/2.*(1.+np.sqrt(1.+frac))


	#factor to account for the fraction mass that collapses before fb
	def fcoll_func(self, mach=None, omega0=None, rho0=None, rvals='fcoll'):
		if mach==None:
			mach =self.mach
		if omega0==None:
			omega0 = self.omega0
		if rho0==None:
			rho0=self.rho0
		
		tff = self.tff_func(rho0)
		ssfrff = self.ssfrff_func(mach)
		tsn = self.tsn
		mean_tfb = self.tfb_fcoll_func(1.0, tff, tsn)
		tff2d = self.tff2d_func(omega0)


		fcoll = (mean_tfb/tff2d)**4.

		if type(fcoll)==np.ndarray:
			fcoll = np.amin(np.array([fcoll, np.ones(len(fcoll))]), axis=0)
		else:
			fcoll =  min(fcoll,1.)


		if rvals=='fcoll':
			return fcoll
		if rvals=='all':
			return fcoll, tff2d, mean_tfb


	#Minimum mass of a star forming region
	def get_Mc_min(self):

		MMC_space = np.logspace(0, 10, 1000)
		sfe = self.tildesfe_func(MMC_space)

		

		#Check whether SFE has multiple regions above threshold
		if sfe[-1]>=self.sfe_th:
			dsfe = np.diff(sfe)
			asign = np.sign(dsfe)
			signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
			im=0
			for ich in signchange:
				im+=1
				if ich==1 and im>2:
					break
			
			

			MMC_space = MMC_space[:im]
			sfe = sfe[:im]

		#Take point where SFE exceeds threshold
		imin = np.argmin(np.absolute(sfe-self.sfe_th))

		self.Mcmin = MMC_space[imin]*self.sfe_th

		#Apply limits on minimum mass of star forming region		
		if self.Mcmin<self.Mcminmin:
			self.Mcmin = self.Mcminmin
		elif self.Mcmin>self.Mcminmax:
			self.Mcmin = self.Mcminmax
		
		if hasattr(self, 'Mcmax'):
			if self.Mcmax<self.Mcmin and self.Mcmax>self.Mcminmin:
				self.Mcmin = self.Mcmax
			elif self.Mcmax<self.Mcminmin:
				self.Mcmax = self.Mcminmin
				self.Mcmin = self.Mcmax
		else:
			print('Error: minimum cluster mass called before maximum')
			sys.exit()

		print('Minimum cluster mass:', self.Mcmin)
		print('Maximum cluster mass:', self.Mcmax)

		return self.Mcmin

	#GMC radius for given GMC mass
	def Rgmc_func(self, MMC, sigma0=None):
		if sigma0==None:
			sigma0=self.sigma0

		sigmaGMC = sigma0*self.fgmc_func()
		return np.sqrt(MMC/(np.pi*sigmaGMC))

	
	#GMC mass dependent feedback time-scale 
	def tildetfb_func(self, MMC,tsn, ssfrff, sigma0=None, omega0=None, rho0=None):
		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0= self.omega0
		if rho0==None:
			rho0 = self.rho0


		fMC = self.fgmc_func()
		fSigma = self.fSigma_func(fMC)

		MMC_ = MMC*self.MSun
		sigma_ = sigma0*self.MSun/(self.pc**2.)
		rho_ = rho0*self.MSun/(self.pc**3.)
		omega_=omega0/self.Myr

		tsn_ = tsn*self.Myr

		C_const = np.sqrt(np.sqrt(np.pi)/(8.*self.G))
		frac=  ( 8.*np.pi**0.5 * C_const *self.phi_p * self.G * sigma_**2 * MMC_**(3/4.) )/( 3. *self. phifb * ssfrff * tsn_**2 * (fSigma*sigma_)**(9/4.) )

		t_fb = (tsn/2.) * ( 1. + np.sqrt( 1. +frac ) )


		return tsn/2.*(1.+np.sqrt(1.+frac))
		
				 
	#GMC mass dependent SFE calculation (Trujillo-Gomez+2019)
	def tildesfe_func(self, MMC, sigma0=None, omega0=None, mach=None,rho0=None, ssfrff=None, Q=None, tsnap=10.0):

		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0= self.omega0
		if mach==None:
			mach= self.mach
		if rho0==None:
			rho0 = self.rho0
		if Q==None:
			Q =self.Q

		if ssfrff==None:
			ssfrff = self.ssfrff_func(mach)

		MMC = np.array(MMC)

		tff = self.tildetff_func(MMC)
		tsn = self.tildetsn_func(MMC, ssfrff)
		tfb = self.tildetfb_func(MMC, tsn, ssfrff)
		
		# SN feedback
		efb =np.array(ssfrff/tff*tfb)
		sfe = np.array([efb, np.ones(len(MMC))*self.sfemax])

		sfe = np.amin(sfe, axis=0)

		return sfe
		

	#Toomre mass
	def MT_func(self, sigma0=None, omega0=None):
		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0=self.omega0

		surfg = sigma0*self.MSun/(self.pc**2.)
		omega = omega0/self.Myr
		
		return np.pi**5.*self.G**2.*surfg**3./(omega**4*self.MSun)


	#Maximum mass of star forming region
	def get_Mc_max(self, sigma0=None, omega0=None, mach=None):

		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0=self.omega0
		if mach==None:
			mach = self.mach
	
		MT = self.MT_func(sigma0=sigma0, omega0=omega0)
		fcoll = self.fcoll_func(mach=mach, omega0=omega0)

		self.Mcmax = self.sfe_eff*MT*fcoll 

		if self.Mcmax>self.Mcmaxmax:
			self.Mcmax = self.Mcmaxmax

		
		if hasattr(self, 'Mcmin'):
			if self.Mcmax<self.Mcmin:
				self.Mcmin= self.Mcmax
		
		return self.Mcmax

	#These functions just define parameters in terms of other variables
	def Lambda_psi_func(self, psi, psif):
		return psi-psif


	def psi_Lambda_func(self,Lambda, psi0f):
		return Lambda + psi0f
	

	def Lambda_phi_func(self, phi):
		return (1.-np.exp(-(self.fbr*phi)**self.alpha))*np.log(1.+ phi)


	def dLambdadphi_func(self, phi, sfe, Mcrit=None, debug=False):
		if Mcrit==None:
			Mcrit = self.Mcrit

		A = np.power(self.fbr, self.alpha)
		dLdp =np.exp(-A*phi**self.alpha)*(A*self.alpha*phi**(self.alpha-1)*np.log(phi+1.))  + ((1.-np.exp(-A*phi**self.alpha)))/(phi + 1.)
				
		return dLdp


	def phi_psi_func(self, psi, psi0f, sfe):
		Lambda = self.Lambda_psi_func(psi,psi0f)
		phi = self.phi_Lambda_func(Lambda, sfe)
		return phi

	#Numerical inversion of Lambda(phi)... better way?
	def phi_Lambda_func(self, Lambda, sfe):
		MMax = 10.*self.Mcmax
		phi_space = np.logspace(-8, 10, 2000)
		Lambda_space =  self.Lambda_phi_func(phi_space)
		phi_Lambda_func_interp = interpolate.interp1d(Lambda_space, phi_space)
		Lambda[np.where(Lambda<0.0)] = 0.0
		Lambda[np.where(Lambda>np.amax(Lambda_space))]=np.amax(Lambda_space)
		return phi_Lambda_func_interp(Lambda)

	#Dispersion in gas density
	def sig_rho_func(self, mach=None, b=0.5):
		if mach==None:
			mach = self.mach
		
		sig = np.sqrt(np.log(1.+3.*(b*mach)**2.))
		return sig


	#Specific SFR per ff time... using 0.01	
	def ssfrff_func(self, mach=None,sig_rho=None, qvir=None):
		
		if mach==None:
			mach=self.mach

		if qvir==None:
			qvir = self.qvir

		if sig_rho==None:
			sig_rho = self.sig_rho_func(mach)

		"""ssfrff = 0.014*(qvir/1.3)**(-0.68)*(mach/100.)**(-0.32)
		erf_arg = (sig_rho**2 -np.log(0.68*qvir**2*mach**4))/(2.**1.5*sig_rho)
		ssfrff = 0.13/2.*(1.+special.erf(erf_arg))
		exit()"""

		ssfrff = 0.01

		return ssfrff


	def fSigma_func(self, fMC):
		return 3.92*np.sqrt((10.-8.*fMC)/2.)

	def tildetff_func(self, massMC):
		fMC = self.fgmc_func()
		fSigma = self.fSigma_func(fMC)
		MMC = massMC*self.MSun
		sigma0 = self.sigma0
		surfg = sigma0*self.MSun/(self.pc**2.)
		return 1./np.sqrt(8.*self.G)*(np.pi*MMC/(fSigma**3.*surfg**3.))**(0.25)/self.Myr
		

	def tildetsn_func(self,MMC, ssfrff):
		MOB_ = self.MOB*self.MSun
		MMC_ = MMC*self.MSun
		surfg = self.sigma0*self.MSun/(self.pc**2.)
		fMC = self.fgmc_func()
		fSigma = self.fSigma_func(fMC)
		deltatsn = np.sqrt(np.pi**0.5/8./self.G)*MOB_/ssfrff*(fSigma*surfg*MMC_)**-0.75/self.Myr
		return self.tsn+deltatsn

	#free fall time
	def tff_func(self, rhog):
		return np.sqrt(3*np.pi/(32.*self.G*(rhog*self.MSun/self.pc**3.)))/self.Myr


	#density dependent feedback time-scale
	def tfb_func(self, x, tff, tsn, sigma0=None, omega0=None, mach=None, rho0=None, Q=None):
		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0= self.omega0
		if mach==None:
			mach= self.mach
		if rho0==None:
			rho0 = self.rho0
		if Q==None:
			Q =self.Q


		fMC = self.fgmc_func()
		fSigma = self.fSigma_func(fMC)
		surffb = max(sigma0*fSigma, sigma0)

		surfg = sigma0*self.MSun/self.pc**2.
		omega=omega0/self.Myr
		tsn_ = tsn*self.Myr
		ssfrff = self.ssfrff_func(mach)
		tff_ = tff*self.Myr

		frac = 2*np.pi**2*self.G**2*tff_*Q**2*surfg**2/(self.phifb*ssfrff*tsn_**2*omega**2*x)

		
		tfb = tsn/2.*(1.0+np.sqrt(1.0+frac))

		return tfb
	
	#Density dependent SFE
	def sfe_func(self, x, sigma0=None, omega0=None, mach=None,rho0=None, ssfrff=None, Q=None, tsnap=10.0):

		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0= self.omega0
		if mach==None:
			mach= self.mach
		if rho0==None:
			rho0 = self.rho0
		if Q==None:
			Q =self.Q

		if ssfrff==None:
			ssfrff = self.ssfrff_func(mach)
		
		fMC = self.fgmc_func()
		fSigma = self.fSigma_func(fMC)
		surffb = sigma0*fSigma
		
		surfg = sigma0*self.MSun/self.pc**2.
		surffb = surffb*self.MSun/self.pc**2
		omega=omega0/self.Myr

		tff = self.tff_func(x*rho0)
		tfb = self.tfb_func(x, tff, self.tsn)

		efb = ssfrff*tfb/tff

		einc = ssfrff*tsnap/tff


		if type(x)==np.ndarray:
			sfe = 1./(1./efb+1./(np.ones(len(x))*self.sfemax)+ 1./einc)
		else:
			sfe = 1./(1./efb+1./self.sfemax+1./einc)
			return sfe

		return sfe

	#Extinction constant
	def get_Cext(self, sigma0=None):
		if sigma0==None:
			sigma0=self.sigma0

		self.Cext = sigma0/13.36
		return self.Cext

	#Lognormal density distribution
	def dpdx_func(self, x, sig=None, mulnx=None, mach=None):
		if mach==None:
			mach=self.mach
		
		if sig==None:
			sig = self.sig_rho_func(mach)
		if mulnx==None:
			mulnx = -0.5*sig**2.
		dpdx= 1./(np.sqrt(2.*np.pi)*sig*x)*np.exp(-((np.log(x)-mulnx)**2.)/(2.*sig**2.))

		return dpdx

	#Jeans mass
	def MJ_func(self, Q=None, sigma0=None, omega0=None):
		if Q==None:
			Q = self.Q
		if sigma0==None:
			sigma0=self.sigma0
		if omega0==None:
			omega0=self.omega0
		
		MJeans = 5e-4*Q**4*(sigma0**3)*(omega0**-4)
		return MJeans

	
	
	#Lambda = psi0 - psi0f
	def dLambdadpsi0_func(self, psi0, psi0f):
		#return 1./(1.-psi0+psi0f)
		return 1.

	
	#psi0 = exp(-chi*Cext) *Lambda + psi0f
	def dLambdadpsi0ext_func(self, psi0, chi, psi0f):
		return np.exp(chi*self.Cext)

	#ICMF, weighted by phi
	def dpdphi_func(self,phi, Q=None, sigma0=None, omega0=None,  Mcrit=None):
		
		if Mcrit==None:
			Mcrit =self.Mcrit

		phimax = self.Mcmax/self.Mcrit
		phimin = self.Mcmin/self.Mcrit
		
		#Define normalisation
		phi_space = np.logspace(np.log10(phimin/50.0), np.log10(phimax*50.0),200)
		

		def dpdphi_val(phi_d):
			dpdphi_v = np.exp(-phimin/phi_d)*np.exp(-phi_d/phimax)*np.power(phi_d,-1.0)
			return dpdphi_v
		
		norm = np.trapz(dpdphi_val(phi_space), phi_space)

		return dpdphi_val(phi)/norm

	#Relationship between stellar density and FUV flux
	def F0HM_func(self, rhost):
		FUV = 1000.*(rhost)**0.5
		return FUV
	

	#Define stellar IMF
	def IMF_func(self, mstar,mmin=0.08,mmax=100.0):
		S0 = 0.08
		S1 = 0.5

		P1 = -1.3
		P2 = -2.3


		F1 = 0.035
		F2 = F1*np.power(S1, P1)/np.power(S1, P2)

		def imf_val(m):
			imf = np.zeros(m.shape)

			imf[(m > S0) & (m <= S1)] = F1*np.power(m[(m > S0) & (m <= S1)], P1)
			imf[m > S1] = F2*np.power(m[m > S1], P2)
			imf[m>mmax] = 0.0
			return imf
		
		mspace = np.logspace(np.log10(mmin), np.log10(mmax),100)
		norm = np.trapz(imf_val(mspace), mspace)
		

		return imf_val(mstar)/norm

	
	#PDF for psi0 for give psi0f (and SFE)
	def dpdpsi0_func(self, psi0, psi0f, sfe):
		dpdpsi0 = np.zeros(len(psi0))
		ipsi0f= np.argmin(np.absolute(psi0-psi0f))
		dlogpsi0 = np.log10(psi0[1])-np.log10(psi0[0])
		
		isub = np.where(psi0>psi0f)[0]
		
		Lambda = self.Lambda_psi_func(psi0[isub],psi0f)
		dLambdadpsi0 = self.dLambdadpsi0_func(psi0,psi0f)

		phi = self.phi_psi_func(psi0[isub], psi0f, sfe)
		dLambdadphi = self.dLambdadphi_func(phi, sfe)
		dphidLambda = 1./dLambdadphi
		dpdphi = self.dpdphi_func(phi)

		
		dpdpsi0[isub] =np.absolute(dpdphi*dphidLambda*dLambdadpsi0) 

		norm = np.trapz(dpdpsi0, psi0)
		
		#If norm=0 its because psi0f is too large 
		#Use 'delta function' approx. -- (could do this better)
		if norm==0.:
			print('Norm = 0 ', psi0f)
			idelta = np.argmin(np.absolute(psi0-psi0f))
			dpdpsi0[:] =0.
			dpdpsi0[idelta-1] =1.
			dpdpsi0/=np.trapz(dpdpsi0,psi0)
			return dpdpsi0
		
		#Normalisation
		dpdpsi0 /= norm

		return dpdpsi0

	#The following functions are for extinction calculations
	#Derivative of central overdensity  as function of gamma
	#(gamma is a radius parameter)
	def dxcdgamma_func(self,gamma,x):
		return 5*x*gamma*np.power(1.+gamma**2., 3./2.)

	#central overdensity as a function of radius, overdensity
	def xc_gamma_func(self, gamma,x):
		return x*(1.+gamma**2.)**(5./2.)


	#Return chi (eff. surf. dens.) as a fn of Lambda, phi
	def chi_gamma_phi_func(self,phi, gamma,x, sfe):

		def indef_integral(z):
			return z*(2.*z*z+3.)/(1.+z*z)**(3./2.)

		gammaS = self.gammaS_func(phi, gamma, x,sfe)
		factor = (1.-sfe)*sfe**(-1./3.)/3.*np.power(3.*self.Mcrit*self.rho0**2./(4.*np.pi*self.sigma0**3), 1./3.)*x**(2./3.)
		chi_arr = factor*(1.+gamma)**(5./3.)*phi**(1./3.)

		chi_arr *= indef_integral(gamma)-indef_integral(gammaS)
		
		return chi_arr


	#Extincted psi0 as fn of phi, Lambda
	def psi0ext_phi_Lambda_func(self, phi, Lambda, x, sfe, psi0f):
		chi = self.chi_gamma_phi_func(phi, Lambda,x, sfe)
		Lambda = self.Lambda_phi_func(phi)

		psi0ext = np.exp(-self.Cext*chi)*Lambda +psi0f
		return psi0ext
		

	#Lambda as a function of psi0, eff. sd and psi0f
	def Lambda_psi_chi_func(self, psi0,chi, psi0f):
		Lambda =np.exp(self.Cext*chi)*(psi0-psi0f)
		return Lambda


	#Probability of (non-zero) chi value
	#This function is inefficient and should probably be rewritten...
	def dpdchi1_func(self, chi_d, x, sfe):
		
		#Step 1: define phi dummy array 
		phi_dd = np.logspace(-3, 4, 400)

		#Step 2: define gamma dummy array
		gamma_dd = np.logspace(-4, 4,500)
		
		#Step 3: get chi(gamma, phi) and also dchi/dphi at each Lambda
		phi_mg, gamma_mg = np.meshgrid(phi_dd, gamma_dd, indexing='ij')
		chi_dd  = self.chi_gamma_phi_func(phi_mg,gamma_mg, x,sfe)


		#Step 4: get dpdchi at each chi by integrating over dLambda between 
		#Lambda_1(phi) and infinity (got dpdLambda and dpdphi)  - get dchi/dphi 
		#numerically at each Lambda for the integral (from step 2) 
		dpdchi = np.zeros(len(chi_d))

		#EXTREMELY INEFFICIENT AND REQUIRES HIGH RES FOR ACCURACY
		#This should be redone...
		#This is where all the time is spent in the extinction calc.
		for ichi in range(len(chi_d)):
			igs = np.zeros(len(phi_dd), dtype=int)
			dpdLambda = np.zeros(len(phi_dd))
			igs = np.argmin(np.absolute(chi_d[ichi]-chi_dd), axis=1)
			gamma_d = gamma_dd[igs]
			igs[np.where(igs==len(gamma_dd)-1)] -=1
			iels = [np.arange(len(phi_dd)), igs]
			ielsp1 = [np.arange(len(phi_dd)), igs+1]
			deltagamma = np.absolute(chi_d[ichi]-chi_dd[iels])/chi_d[ichi]
			iout = np.where(deltagamma>0.1)[0]
			dgammadchi =  (gamma_dd[igs+1]- gamma_dd[igs])/(chi_dd[ielsp1]-chi_dd[iels])
			
			iout= np.array(iout)
			dpdphi = self.dpdphi_func(phi_dd)
			if len(iout)<len(phi_dd):

				gamma_d = gamma_dd[igs]
				dpdphi[iout] = 0.0

				xc_g = self.xc_gamma_func(gamma_d, x)
				dpdgamma = self.dpdx_func(xc_g)*self.dxcdgamma_func(gamma_d, x)*(1.+gamma_d**2.)**(5./2.)

				integrand = dpdphi*dpdgamma*dgammadchi
				integrand[iout] = 0.0
				integrand[np.where((np.isnan(integrand))|(np.isfinite(integrand)==False))] = 0.0

				dpdchi[ichi] = np.trapz(integrand, phi_dd)

		return dpdchi


	#Probability of a star being born inside the initial Stromgren radius
	def pSphi0_func(self, psi0,x,sfe, psi0f):

		phi_dd = np.logspace(-5,5,100)
		gamma_dd = np.logspace(-5,5,1000)

		phi_mg, gamma_mg = np.meshgrid(phi_dd, gamma_dd, indexing ='ij')
		
		
		psi0ext = self.psi0ext_phi_gamma_func(phi_mg, gamma_mg, x, sfe, psi0f)

		gamma_func = interpolate.interp2d(phi_mg, psi0ext, gamma_mg)

		pS = np.zeros(len(psi0))
		
		#Calc. each gamma (radius) for given phi, psi0
		gamma_d = gamma_func(phi_dd, psi0)

		#for each psi0, take each phi and calculate the central density and corresponding prob.
		for ipsi in range(len(psi0)):
			igs = np.zeros(len(phi_dd), dtype=int)
			dpdgamma = np.zeros(len(phi_dd))
			dphicphi = np.zeros(len(phi_dd))
			
			xc_g = self.xc_gamma_func(gamma_d[ipsi], x)
			dpdgamma = self.dpdx_func(xc_g)*self.dxcdgamma_func(gamma_d[ipsi], x)*(1.+gamma_d[ipsi]**2.)**(5./2.)
			dpdphi = self.dpdphi_func(phi_dd)
			integrand_1 = dpdphi*dpdgamma*self.pchi0_func(phi_dd, x,sfe)
			integrand_2 = dpdphi*dpdgamma
			pnorm =  np.trapz(integrand_2, phi_dd)
			if not pnorm==0.:
				#Need to normalise by integral over phi (assuming fixed gamma for psi0)
				pS[ipsi] = np.trapz(integrand_1, phi_dd)/np.trapz(integrand_2, phi_dd)
		
		return pS
	
	#PDF for extincted psi0
	#This function is not efficient and requires high resolution for accuracy
	#Should be imroved. Inversion methods? 
	def dpdpsi0ext_func(self, psi0, psi0f, x, sfe, convolve=True):

		dpdpsi0 = np.zeros(len(psi0))

		#Step 1: truncate PDF below psi0f
		ipsi0f= np.argmin(np.absolute(psi0-psi0f))
		dpsi0 = psi0f-psi0[ipsi0f-1]
		
		isub  = np.where(psi0>psi0f)[0]

		#Step 2: Establish a phi for each chi, Lambda
		#(Chi range a bit arbitrary, tested for pm 4 in solar nbhd and CMZ)
		#Use caution here!
		chi_d = np.logspace(np.log10(x)-4.0, np.log10(x)+4.0, 400)

		#Step 3: get dpdchi - int. between gamma_1(phi) and infinity 
		#(got dpdgamma and dpdphi)  - get dchi/dphi numerically 
		#THE MOST INEFFICIENT FUNCTION HERE...
		if self.time:
			tstart=time.time()
		dpdchi = self.dpdchi1_func(chi_d,x,sfe)
		if self.time:
			print('Time for dpdchi1_func:',time.time()-tstart)


		#Step 4: need to integrate over chi 
		dpdpsi0 = np.zeros(len(psi0))
		pSpsi0 = np.zeros(len(psi0))


		if self.time:
			tstart=time.time()

		for ipsi in isub:
			#Get phi as a function of chi, psi
			Lambda = self.Lambda_psi_chi_func(psi0[ipsi], chi_d, psi0f)
			inosol = np.where(np.isnan(Lambda))[0]
			isol = np.arange(len(Lambda))#np.where((np.isnan(Lambda)==False)&(np.isfinite(Lambda)))[0]
			if len(inosol)>=len(Lambda)-1:
				dpdpsi0[ipsi] = 0.0
			else:
				Lambda_sub = Lambda[isol]
				phi_sub = self.phi_Lambda_func(Lambda_sub, sfe)
				
				dLambdadphi = self.dLambdadphi_func(phi_sub, sfe)
				dphidLambda = 1./dLambdadphi
				dphidLambda[np.where(dLambdadphi==np.inf)] =0.0
				
				
				ilimit = np.argmin(np.absolute(chi_d+np.log((psi0[ipsi]-psi0f)/self.Cext)))

				
				#dphidLambda[ilimit:]=0.0

				dpdphi = self.dpdphi_func(phi_sub)
				inonzero = np.where(dphidLambda>0)[0]
				
				#dpsi0dLambda = np.exp(-self.Cext*chi_d[isol])
				dLambdadpsi0 = np.exp(self.Cext*chi_d[isol])

				dLambdadpsi0[np.where(np.isnan(dLambdadpsi0))] = 0.0
				
				integrand = dpdchi[isol]*dpdphi*dLambdadpsi0*dphidLambda

				integrand[np.where((np.isnan(integrand))|(np.isfinite(integrand)==False))] = 0.0
				inonzero = np.where(integrand>0.0)[0]
				if len(inonzero)>2:
					maxnz = np.amax(inonzero)
					minnz = np.amin(inonzero)
					integrand[maxnz-2:]=0.0
					integrand[:2+minnz]=0.0

					
					dpdpsi0[ipsi] = max(np.trapz(np.absolute(integrand), chi_d[isol]),0.0)

					
		if self.time:
			print('Time for psi0 loop:',time.time()-tstart)

		#Ignoring probability of being born in Stromgren radius (small)
		"""if self.time:
			tstart= time.time()
		pSpsi = self.pSphi0_func(psi0,x,sfe, psi0f)
		if self.time:
			print('Time for pSpsi_func:',time.time()-tstart)"""

		
		#Check we're not including divergent cells outside integration region
		#(Limitation or trapz -- could do more sophisticated things with 
		#adaptive cells.. this is fine though so long as res. is high enough)
		iout = np.argmin(np.absolute(1.-(psi0-psi0f)))
		if psi0[iout]-psi0f>1.:
			iout-=1

		imin = np.argmin(np.absolute(psi0-psi0f))
		if psi0[imin]<psi0f:
			imin+=1

		dpdpsi0[:imin+2]=0.0
		if imin+3>=iout-1:
			dpdpsi0[imin:imin+2] = 1.0
			dpdpsi0[imin+2:] =0.0 
			dpdpsi0[:imin] = 0.0



		#If norm ==0 then this is because psi0f >1 -- delta fn. approx. 
		#As with non-extincted, could do better..
		norm = np.trapz(dpdpsi0, psi0)
		if norm==0.0:
			idelta = np.argmin(np.absolute(psi0-psi0f))
			dpdpsi0[idelta-1:idelta+1] =1.

		dpdpsi0/=np.trapz(dpdpsi0,psi0)

		return dpdpsi0



	#Convert stellar density into gas density (for x-axis)
	def rhog_rhost_func(self, rhost):
		rhog_space = np.logspace(-20, 20, 2000)
		x_space = rhog_space/self.rho0
		rhost_space =  rhog_space*self.sfe_func(x_space)
		rhog_rhost_func_interp = interpolate.interp1d(np.log10(rhost_space), np.log10(rhog_space))
		
		return 10.**rhog_rhost_func_interp(np.log10(rhost))

	#Ionizing photons 
	def Theta_func(self,phi):
		comp1 = (1.-np.exp(-self.d1*phi))**self.d2
		comp2 = np.log(1.+self.d1*phi)
		return comp1*comp2

	#Stromgren radius
	def gammaS_func(self, phi, Lambda, x,sfe):
		fact = (self.rho0/1.6)**(-1./3.)
		Theta = self.Theta_func(phi)
		return fact*sfe**(1./3.)*x**(-1/3.)*phi**(-1/3.)*(1.+Lambda)**(-5./6.)*Theta**(1./3.)

	
	#Probability of star being inside the stromgren radius
	def pchi0_func(self, phi, x, sfe):
		phi = np.array(phi)
		gamma_space = np.logspace(-5, 5, 1000)

		dpdlngamma = gamma_space*5*x*gamma_space*(1+gamma_space**2.)**(3./2.)*self.dpdx_func(x*(1.+gamma_space)**(5./2.))
		dpdlngamma /= np.trapz(dpdlngamma, np.log(gamma_space))

		pgspace = np.meshgrid(phi, gamma_space, indexing='ij')
		gammaS_space = self.gammaS_func(pgspace[0], pgspace[1], x,sfe)

		inds = np.where(pgspace[1]<gammaS_space)
		pinds, ginds = inds[:]

		pchi0 = np.zeros(len(phi))
		for ip in range(len(phi)):
			ipinds = np.where(pinds==ip)[0]
			ginds_ = ginds[ipinds]
			if len(ginds_)>0:
				pchi0[ip] = np.trapz(dpdlngamma[ginds_], np.log(gamma_space[ginds_]))
			
		return pchi0


	#Get prob chi=0 over all x, phi -- also has plotting function
	def get_pchi0_func(self, xspace=None, phispace=None, plot=True, pmin=1e-3, pmax=1.0):

		if type(xspace)==type(None):	
			xspace = np.logspace(-2, 6, 100)
		if type(phispace)==type(None):		
			phispace = np.logspace(-2,4, 200)

		sfe = self.sfe_func(xspace)
		
		pchi0_space = np.zeros((len(xspace), len(phispace)))
		ix = 0

		for x in xspace:
			if ix != 0 and ix !=len(xspace)-1:
				pchi0_space[ix] = self.pchi0_func(phispace, x,sfe[ix])
				pchi0_space[ix][0] = 0.
				pchi0_space[ix][-1] = 0. 
				
			ix+=1

		
		if plot:
			dlnx = np.log(xspace[1])-np.log(xspace[0])
			dlnphi = np.log(phispace[1])-np.log(phispace[0])

			ext = [np.log10(np.amin(xspace))-dlnx/2., np.log10(np.amax(xspace))+dlnx/2, np.log10(np.amin(phispace))-dlnphi/2., np.log10(np.amax(phispace))+dlnphi/2.]
			plt.rc('text',usetex=True)
			plt.rc('font', family='serif')
			fig = plt.figure(figsize=(3.2,3.))
			ax = plt.gca()
			imax = ax.imshow(np.rot90(pchi0_space), aspect='auto',interpolation='none', cmap=cm.gray_r, norm=LogNorm(vmin=pmin, vmax=pmax),  extent=ext)
			
			
			xvs = np.arange(int(ext[0]), int(ext[1]+0.99), 2)
			xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
			
			yvs = np.arange(int(ext[2]), int(ext[3]+0.99))
			yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]
			plt.xticks(xvs, xls)
			plt.yticks(yvs, yls)
			plt.ylabel("$\phi$")
			plt.xlabel("$x$")
			plt.xlim([ext[0], ext[1]])
			plt.ylim([ext[2],ext[3]])


			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)

			cbar = plt.colorbar(imax, cax=cax, label='$p_\mathrm{S}(\phi, x)$')
			cblog = np.array(np.arange(int(np.log10(pmin)), int(np.log10(pmax)+1.0)), dtype='float')
			cbtick = np.power(10,cblog)
			cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]

			cbar.set_ticks(cbtick)
			cbar.set_ticklabels(cblabs)

			plt.savefig('paper_figure_pchi0_CMZ.pdf', bbox_inches='tight', format='pdf')

			plt.show()

		
		return pchi0_space


	#Lognormal dispersion in max. stellar luminosity in region
	def sigmaL_func(self, phi):
		logphi_lim = np.log(phi)
		logphi_lim = np.amax(np.array([-2.5*np.ones(len(phi)), logphi_lim]), axis=0)
		return 8./((3. +logphi_lim)**2. )


	#Lognormal dispersion in local flux 
	def sigmaF_func(self,psi0, phi, psi0f):
		
		sigmaL = self.sigmaL_func(phi)

		#Weighting function to avoid overshoot below 'field flux'
		W1 = special.erf((np.log(psi0)-np.log(psi0f))/(np.sqrt(2.)*self.sigmaFf))

		W1[np.where(W1<0.0)] = 0.
		
		#Limit value of sigmaL for numerical reasons
		sigmaL[np.where(sigmaL>10.)] = 10.0

		sigmaF = self.sigmaFf + sigmaL*W1 #*W2
		
		return sigmaF

	#Marginalise over the dispersion
	def convolve_psi(self,dpdpsi0, psi0,psi0f, sfe, iflg=False,extinct=False):
		
		psi_space = psi0 
		dpdpsi = np.zeros(len(psi0))

		if not extinct:
			phi = self.phi_psi_func(psi0, psi0f, sfe)
			sigmaF = self.sigmaF_func(psi0, phi, psi0f)

			sigmaF[np.where(psi0<psi0f)] = self.sigmaFf

		else:
			sigmaF= self.sigmaFf*np.ones(len(psi0))			

		for ipsi in range(len(psi_space)):
			dpsi_space = psi_space[ipsi]/psi0
			
			dpddpsi = self.dpdx_func(dpsi_space, mulnx=0.0, sig=sigmaF[ipsi])
			integrand  = dpdpsi0*dpddpsi/psi0
			dpdpsi[ipsi] = np.trapz(integrand, psi0)


		return dpdpsi

	#2D prob density function for SBE
	def d2Fdydpsi_func(self,rho_st,f_fuv, ext=False, convolve=True, phi_res=500, rfact=16.0, PFL=1e-20):

		rho0=self.rho0
		rho_g = self.rhog_rhost_func(rho_st)
		x = rho_g/rho0
		sfe = self.sfe_func(x)

		xd2pdxdpsi = np.zeros((len(x),len(f_fuv)))

		#Get value of x (gas over density) from rho_st
		x = rho_st/(sfe*rho0)

		dlnsfedx = np.gradient(np.log(sfe), x)

		#xdpdx is shorthand (also an approximation) - exact solution used.
		xdpdx = (1./((1./x)+dlnsfedx))*self.dpdx_func(x)

		for irho in range(len(x)):
			F0 = self.F0HM_func(rho_st[irho])
			#F0 = self.F0HM_func(self.rho0*temp_x[0]*temp_sfe[0])
			psi0f = self.F0f/F0
			
			#if psi0f is v. large, then approx. delta fn. (could improve this)
			if psi0f<100.:
				psi0_space = np.logspace(np.log10(psi0f/rfact),2., phi_res)
				if ext:
					dpdpsi0 = self.dpdpsi0ext_func(psi0_space, psi0f,x[irho], sfe[irho])
					#dpdpsi0 = self.dpdpsi0ext_func(psi0_space, psi0f,temp_x[0], temp_sfe[0])
				else:
					dpdpsi0 = self.dpdpsi0_func(psi0_space, psi0f, sfe[irho])

				
				if convolve:
					if self.time:
						tstart=time.time()
					dpdpsi = self.convolve_psi(dpdpsi0, psi0_space,psi0f, sfe[irho], extinct=ext)
					if self.time:
						print('Time for convolve_psi:', time.time()-tstart)
				else:
					dpdpsi = dpdpsi0

				

				psi_space = (f_fuv/F0)
				dpsi = psi0_space-psi0f
				idpsi = np.argmin(np.absolute(dpsi))
				psi0_space = np.append(np.array([0.0, psi0f*0.9/rfact]), psi0_space)
				psi0_space = np.append(psi0_space, np.array([1.1*psi0_space[-1],2.*psi0_space[-1], 1./PFL]))
				dpdpsi = np.append(np.array([0.0, 0.0]), dpdpsi)
				dpdpsi = np.append(dpdpsi, np.array([0.0,0.0, 0.0]))

				dpdpsi0 = copy.copy(dpdpsi)
				

				dpdpsi_interp = interpolate.interp1d(psi0_space, dpdpsi)
				dpdpsi_ivals = dpdpsi
				dpdpsi = dpdpsi_interp(psi_space)

				if np.trapz(dpdpsi, psi_space)<1e-10:
					imin= np.argmin(np.absolute(psi_space-psi0f))
					if psi_space[imin]>psi0f:
						imin-=1
					dpdpsi[imin-1:imin+1]= 1.0
					dpdpsi/= np.trapz(dpdpsi, psi_space)
				
				
			else:
				dpdpsi0 = np.zeros(len(f_fuv))
				psi0_space = f_fuv/F0
				idelta = np.argmin(np.absolute(psi0_space-psi0f))
				dpdpsi0[idelta-1] =1.
				
				dpdpsi0/=np.trapz(dpdpsi0,psi0_space)

			

				if convolve:
					dpdpsi = self.convolve_psi(dpdpsi0, psi0_space,psi0f, sfe[irho], extinct=ext)
				else:
					dpdpsi = dpdpsi0
				
				psi_space = f_fuv/F0

			xd2pdxdpsi[irho] = dpdpsi*xdpdx[irho]/np.trapz(dpdpsi, psi_space)
			
			isnan = np.where(np.isnan(xd2pdxdpsi[irho]))[0]

			xd2pdxdpsi[irho][isnan]=0.0

		return xd2pdxdpsi
	
	#Tidal destruction time-scale of PPDs
	def tau_tidal_func(self, rho_st, m_st):
		orig_shape = rho_st.shape
		m_st = np.array(m_st)
		
		if m_st.shape==rho_st.shape:
				mp_space = np.logspace(0.08, 100.,300)
				rr, mm = np.meshgrid(rho_st, mp_space,sparse=True)
				ss, mm = np.meshgrid(m_st, mp_space,sparse=True)
				Zmin = 3.1e-1*ss**(2./3.)*mm**(1./3.)
				dEdmp = 0.15*rr*(ss+mm+ 0.23*Zmin*self.vdisp*self.vdisp)*self.IMF_func(mm)/1e4/self.vdisp
				E = simps(np.swapaxes(dEdmp,0,1), mp_space).reshape(orig_shape)
		else:
			mp_space = np.logspace(0.08, 100.,276)
			rr, mm = np.meshgrid(rho_st, mp_space, sparse=True)
			Zmin = 3.1e-1*m_st**(2./3.)*mm**(1./3.)
			dEdmp = 0.15*rr*(m_st+mm+ 0.23*Zmin*self.vdisp*self.vdisp)*self.IMF_func(mm)/1e4/self.vdisp
			E = simps(np.swapaxes(dEdmp,0,1), mp_space).reshape(orig_shape)
		return 1./E
	
	#External photoevaporation dispersal time-scale
	def tau_photevap_func(self,fuv, mstar, alpha):
		tau_phot = self.theta0*(5.4e-3/alpha)**self.theta1*mstar**(self.theta2-0.5*self.theta1)*(1.+(fuv/5e3)**-self.theta4)*(1.+np.exp(-(fuv/5e3)**self.theta3))
		return tau_phot
		
	#External dispersal time-scale
	def tau_func(self, rho_st, fuv_flux, mstar=0.5, alpha=ALPHA):
		tau_p = self.tau_photevap_func(fuv_flux, mstar, alpha)
		tau_t = self.tau_tidal_func(rho_st, mstar)
		tau_disp = 1./(1./tau_t+1./tau_p)
		return tau_disp


	#Flux and rho arrays should be evenly spaced in logspace if logspace=True (and have their log values)
	#and the dF/dFdrho should be wrt to log variables
	def dFdtdisp_func(self, F_o,rho_o, dFsdFdrho_o, logspace=True, maxres_r=100, maxres_F=200, pthresh=1.01e-5, plttype=4,c='k', mst_mins = [0.08,1.0],	labels = ['All', '$m_*>1\, M_\odot$'],linestyles= ['solid', 'dashed']):
		tdisp = np.logspace(-3, 5, 200)

		orig_shape = dFsdFdrho_o.shape
		#Step 1 - make grid sparse to make calculation fast
		isbig_r, isbig_F = np.where(dFsdFdrho_o>pthresh)
		imax = np.argmax(dFsdFdrho_o[np.amax(isbig_r)])
		imaxrho = np.amax(isbig_r)
		iminrho = np.amin(isbig_r)

		iminF = np.amin(isbig_F)
		imaxF = np.amax(isbig_F)


		step_r = max(int((imaxrho-iminrho)/maxres_r),1)
		step_F = max(int((imaxF-iminF)/maxres_F),1)


		inds_r = np.arange(iminrho, imaxrho, step_r)
		inds_F = np.arange(iminF, imaxF, step_F)

		inds_grid = np.meshgrid(inds_r, inds_F, indexing='ij')
		
		rho = rho_o[inds_r]
		F = F_o[inds_F]
		dFsdFdrho =  dFsdFdrho_o[inds_grid]

		dFsdFdrho /= simps(simps(dFsdFdrho, F),rho)

		new_shape = dFsdFdrho.shape
		print('Reduced array from {0} to {1}'.format(orig_shape, new_shape))

		if plttype==1 or plttype==0:
			plt.rc('text',usetex=True)
			plt.rc('font', family='serif')
			fig = plt.figure(figsize=(4.0,4.))
		
		
		imst_min = 0
		handles = []

		tmeds = []

		for mst_min in mst_mins:

			#Define mass function array
			if logspace:
				ms = np.logspace(np.log10(mst_min), np.log10(100.0), 30)
				dFsdms =ms*self.IMF_func(ms, mmin=mst_min, mmax=100.0)
				ms = np.log(ms)
			else:
				ms = np.linspace(0.08, 100., 300)
				dFdm =self.IMF_func(ms, mmin=mst_min, mmax=100.0)
			
			dF = F[1]-F[0]
			dm = ms[1]-ms[0]
			dr = rho[1]-rho[0]
			
			dFsdrhodFdms = np.zeros((dFsdms.shape[0], dFsdFdrho.shape[0],dFsdFdrho.shape[1]))
			for im in range(len(ms)):
				dFsdrhodFdms[im] = np.multiply(dFsdms[im], dFsdFdrho)
			
			msmg, rhomg , Fmg= np.meshgrid(ms, rho,F,  indexing='ij')


			if not os.path.isdir(self.oname+'_tdisp_mgrid.npy'):
				if not logspace:
					tdispmg = self.tau_func(rhomg, Fmg, mstar=msmg, alpha=ALPHA)
				else:
					tdispmg = self.tau_func(np.exp(rhomg), np.exp(Fmg), mstar=np.exp(msmg), alpha=ALPHA)
				np.save(self.oname+'_tdisp_mgrid', np.array([msmg, rhomg, Fmg, tdispmg]))
			else:
				msmg, rhomg, Fmg, tdispmg = np.load(self.oname+'_tdisp_mgrid.npy')

					
			
			dFdtdisp = np.zeros(len(tdisp))
			cumF = np.zeros(len(tdisp))
			for it in range(1, len(tdisp)):
				dtdisp = tdisp[it]-tdisp[it-1]
				i_disp = np.where((tdispmg>tdisp[it-1])&(tdispmg<tdisp[it]))
				dtdFdtdisp  = np.sum(dFsdrhodFdms[i_disp]*dm*dr*dF)
				dFdtdisp[it] = dtdFdtdisp/dtdisp
				cumF[it] = cumF[it-1]+dtdFdtdisp
			cumF /= cumF[-1]

			cumF_func = interpolate.interp1d(cumF, np.log10(tdisp))
			
			tmed = float(10.**cumF_func(0.5))

			tmeds.append(tmed)
			if plttype<3:
				plt.plot(tdisp, cumF, c=c , ls=linestyles[imst_min],linewidth=0.8)
				plt.axvline(tmed, c=c, ls=linestyles[imst_min],linewidth=0.8)
			print(tmed)
			
			hand, = plt.plot([],[], c='k',ls= linestyles[imst_min], label=labels[imst_min])
			handles.append(hand)

			imst_min+=1

		
		if plttype==2 or plttype==0:
			plt.xscale('log')
			plt.xlim([1e-1, 1e1])
			plt.ylim([0., 1.0])
			plt.xlabel('$T$ (Myr)')
			plt.ylabel('Cum. Frac. - $\mathcal{F}_*(\\tau_\mathrm{disp.}< T)$')
			
			if plttype==0:
				plt.legend(loc='best')
				plt.savefig('paper_figure_cumtdisp.pdf', bbox_inches='tight', format='pdf')		
				plt.show()

		if plttype<3:		
			return handles

		return tmeds


	#Median dispersal time-scales for given FUV and density
	def dFdtdisp_mgrid_func(self, F_o,rho_o, dFsdFdrho_o, logspace=True, maxres_r=100, maxres_F=200,  msts = [0.2,0.5,1.0], pthresh=1e-5):
		tdisp = np.logspace(-3, 5, 200)

		orig_shape = dFsdFdrho_o.shape
		#Step 1 - make grid sparse to make calculation fast
		isbig_r, isbig_F = np.where(dFsdFdrho_o>pthresh)
		imax = np.argmax(dFsdFdrho_o[np.amax(isbig_r)])
		imaxrho = np.amax(isbig_r)
		iminrho = np.amin(isbig_r)

		iminF = np.amin(isbig_F)
		imaxF = np.amax(isbig_F)

		step_r = max(int((imaxrho-iminrho)/maxres_r),1)
		step_F = max(int((imaxF-iminF)/maxres_F),1)


		inds_r = np.arange(iminrho, imaxrho, step_r)
		inds_F = np.arange(iminF, imaxF, step_F)

		inds_grid = np.meshgrid(inds_r, inds_F, indexing='ij')
		
		rho = rho_o[inds_r]
		F = F_o[inds_F]
		dFsdFdrho =  dFsdFdrho_o[inds_grid]

		dFsdFdrho /= simps(simps(dFsdFdrho, F),rho)

		new_shape = dFsdFdrho.shape
		print('Reduced array from {0} to {1}'.format(orig_shape, new_shape))
		
		imst_min = 0
		tmeds = []

		dF = F[1]-F[0]
		dr = rho[1]-rho[0]
		
		rhomg , Fmg= np.meshgrid(rho,F,  indexing='ij')

		for mst in msts:
			
			if not logspace:
				tdispmg = self.tau_func(rhomg, Fmg, mstar=mst, alpha=ALPHA)
			else:
				tdispmg = self.tau_func(np.exp(rhomg), np.exp(Fmg), mstar=mst, alpha=ALPHA)
			
			dFdtdisp = np.zeros(len(tdisp))
			cumF = np.zeros(len(tdisp))
			for it in range(1, len(tdisp)):
				dtdisp = tdisp[it]-tdisp[it-1]
				i_disp = np.where((tdispmg>tdisp[it-1])&(tdispmg<tdisp[it]))
				dtdFdtdisp  = np.sum(dFsdFdrho[i_disp]*dr*dF)
				dFdtdisp[it] = dtdFdtdisp/dtdisp
				cumF[it] = cumF[it-1]+dtdFdtdisp
			cumF /= cumF[-1]

			cumF_func = interpolate.interp1d(cumF, np.log10(tdisp))
			
			tmed = float(10.**cumF_func(0.5))

			tmeds.append(tmed)
			
			imst_min+=1

		return np.array(tmeds)
		
		
		

	def plot_d2Fdydpsi(self,convolve=True,xlims = [-6.,4.5], ylims = [-.2,5.0],phlab=None, enclab=None, pmin = 1e-3, pmax = 1.1e-1, extinct=False, resx=100, resy=1000, sl=False, plot=True,interactive=True):


		if type(xlims)==type(None) or type(ylims)==type(None):
			rho_ = self.rho0
			mach_ = self.mach

			x_tmp = np.logspace(-20,20, 500)
			
			dpdx_tmp = self.dpdx_func(x_tmp)
			dpdlnx_tmp =x_tmp*dpdx_tmp
			xdpdlnx_tmp = x_tmp*dpdlnx_tmp
			sfe_tmp = self.sfe_func(x_tmp)
			xdpdlnx_tmp /= np.trapz(xdpdlnx_tmp, np.log10(x_tmp))
			sfexdpdlnx_tmp =sfe_tmp*xdpdlnx_tmp/ np.trapz(sfe_tmp*xdpdlnx_tmp, np.log10(x_tmp))
			dpdlnx_tmp /= np.trapz(dpdlnx_tmp, np.log10(x_tmp))
			dpdx_tmp /= np.trapz(dpdx_tmp, x_tmp)
			inds=  np.where(sfexdpdlnx_tmp>pmin)



			rhomax = np.amax((x_tmp*sfe_tmp)[inds])
			rhomin = np.amin((x_tmp*sfe_tmp)[inds])
			rhomax =np.log10(rho_*rhomax)
			rhomin = np.log10(rho_*rhomin)
			
			FUVmin =np.log10(self.F0f)
			FUVmax = np.log10(self.F0HM_func(10.*10.**rhomax))
			xlims = [float(rhomin-0.5),float(rhomax+0.5)]
			ylims = [float(FUVmin-1.), float(FUVmax+0.5)]

			

		
		logxmin = xlims[0]
		logxmax = xlims[1]
		logymin = ylims[0]
		logymax = ylims[1]
		dlogx = float(logxmax-logxmin)/float(resx)
		dlogy = float(logymax-logymin)/float(resy)
		
		rhost_space = np.logspace(xlims[0], xlims[1], resx)
		fuv_space = np.logspace(ylims[0], ylims[1], resy)


		rho0=self.rho0
		rho_g = self.rhog_rhost_func(rhost_space)
		x = rho_g/rho0
		sfe = self.sfe_func(x)

		
		suff = ''
		suff += '_r1'+str(resx)
		suff += '_r2'+str(resy)
		if extinct:
			suff+='_ext'
		if convolve:
			suff+='_conv'
			
		fname = self.oname+'_d2pdxdpsi'+suff
		if os.path.isfile(fname+'.npy') and os.path.isfile(fname+'_pgrid.npy') and sl:
			xd2pdxdpsi = np.load(fname+'.npy')
			rhost_space,fuv_space = np.load(fname+'_pgrid.npy')
		else:
			xd2pdxdpsi = self.d2Fdydpsi_func(rhost_space,fuv_space, convolve=convolve, ext=extinct)
			if sl:
				np.save(fname, xd2pdxdpsi)
				np.save(fname+'_pgrid', np.array([rhost_space,fuv_space]))
		
		rho0=self.rho0
		rho_g = self.rhog_rhost_func(rhost_space)
		sfe = rhost_space/rho_g
		
		

		#Renormalisation for each density -- not strictly necessary
		x = rho_g/rho0

		xfact = 1./(1./x+ np.absolute(np.gradient(np.log(sfe),x)))
		xdpdx = xfact*self.dpdx_func(x)
		xdpdx /= np.trapz(xdpdx, x)

		xdpdlogx = rhost_space*xdpdx/np.trapz(rhost_space*xdpdx, np.log10(rhost_space))

		
		xd2pdlnxdlnpsi = np.zeros((len(x), len(fuv_space)))
		for ix in range(len(rhost_space)):
			norm  = np.trapz(xd2pdxdpsi[ix]*fuv_space, np.log10(fuv_space))
			if norm>0.:
				xd2pdlnxdlnpsi[ix] = xd2pdxdpsi[ix]*fuv_space*xdpdlogx[ix]/norm


		iram = np.argmin(np.absolute(rhost_space/sfe - 2e4))
			

		if plot:

			ext = [np.log10(np.amin(rhost_space))-dlogx/2., np.log10(np.amax(rhost_space))+dlogx/2, np.log10(np.amin(fuv_space))-dlogy/2., np.log10(np.amax(fuv_space))+dlogy/2.]
			plt.rc('text',usetex=True)
			plt.rc('font', family='serif')
			fig = plt.figure(figsize=(3.2,3.))
			ax = plt.gca()

			xd2pdlnxdlnpsi_nz =copy.copy(xd2pdlnxdlnpsi)
			xd2pdlnxdlnpsi_nz[np.where(xd2pdlnxdlnpsi_nz<pmin)]=pmin
			xd2pdlnxdlnpsi_nz[np.isnan(xd2pdlnxdlnpsi)]=pmin
			imax = ax.imshow(np.rot90(xd2pdlnxdlnpsi_nz), aspect='auto',interpolation='bilinear', cmap=cm.gray_r, norm=LogNorm(vmin=pmin, vmax=pmax),  extent=ext)
			
			
			xvs = np.arange(int(ext[0]), int(ext[1]+1.), 1)
			if len(xvs)>8:
				xvs = np.arange(int(ext[0]), int(ext[1]+1.), 2)
			xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
			
			yvs = np.arange(int(ext[2]), int(ext[3]+1.))
			yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]
			plt.xticks(xvs, xls)
			plt.yticks(yvs, yls)
			plt.ylabel("FUV flux ($G_0$)")
			plt.xlabel("Stellar density ($M_\odot$ pc$^{-3}$)")

			rh_tmp = rhost_space[np.where((rhost_space>10.**xlims[0])&(rhost_space<10.**xlims[1]))]
			f_tmp = fuv_space[np.where((fuv_space>10.**ylims[0])&(fuv_space<10.**ylims[1]))]
			rh_tmp = np.logspace(xlims[0], xlims[1], 5000)
			f_tmp = np.logspace(ylims[0], ylims[1], 500)
			rs_mg, f_mg = np.meshgrid(rh_tmp, f_tmp, indexing='ij')

			tau_dips = self.tau_func(rs_mg, f_mg, mstar=2.0, alpha=ALPHA)

			plt.xlim(xlims)
			plt.ylim(ylims)
			conts= plt.contour(np.log10(rh_tmp), np.log10(f_tmp), np.swapaxes(tau_dips,0,1), levels=[1.,2.,3.], colors='b')

			fmt = {}
			strs = ['$1$~Myr', '$2$~Myr', '$3$~Myr']
			for l, s in zip(conts.levels, strs):
				fmt[l] = s
			plt.clabel(conts, conts.levels, inline=True, fmt=fmt, fontsize=10, manual =interactive)

			if extinct:
				iram = np.argmin(np.absolute(rhost_space/sfe - 2e4))
				print('Ram limit:', rhost_space[iram])
				plt.axvline(np.log10(rhost_space[iram]), c='b', label='$\\tau_\mathrm{ram} = 1$~Myr')

			
			#plt.legend(loc=2, fontsize=8)
			
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)

			#cbar = plt.colorbar(imax, cax=cax, label='${\partial^2 \mathcal{F}_*}/{\partial \log \\rho_* \partial \log F}$')
			cbar = plt.colorbar(imax, cax=cax, label='2D PDF (logarithmic)')
			cblog = np.array(np.arange(int(np.log10(pmin)-0.51), int(np.log10(pmax)+1.)), dtype='float')
			cbtick = np.power(10,cblog)
			cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
			cbar.set_ticks(cbtick)
			cbar.set_ticklabels(cblabs)
			
			plt.savefig('paper_figure_rhoFUV_PDF.pdf', bbox_inches='tight', format='pdf')
			plt.show()
			plt.close(fig)
		
		xd2pdlnxdlnpsi /= simps(simps(xd2pdlnxdlnpsi, np.log(fuv_space)), np.log(rhost_space))
		return rhost_space, fuv_space,  xd2pdlnxdlnpsi


	
	def plot_d2pdxdchi(self, plot=False, c='k'):

		xspace = np.array([1e-3, 1.0, 1e3])
		sfe = self.sfe_func(xspace)
		chispace = np.logspace(-3.5,3.5, 50)
		dpdchi = np.zeros((len(xspace), len(chispace)))
		
		if not plot:
			plt.rc('text',usetex=True)
			plt.rc('font', family='serif')
			fig = plt.figure(figsize=(4.0,4.))

		linestyles = ['dashed', 'solid', 'dotted']


		handles=  []
		for ix in range(len(xspace)):
			dpdchi[ix] = self.dpdchi1_func(chispace,xspace[ix], sfe[ix])
			dpdchi[ix] /= np.trapz(chispace*dpdchi[ix], np.log(chispace))
			plt.plot(chispace, chispace*dpdchi[ix], c=c, ls=linestyles[ix])
			hand, = plt.plot([],[], c='k',ls= linestyles[ix], label='$x=10^{%d}$'%(int(np.log10(xspace[ix]))))
			handles.append(hand)

		if plot:
			plt.xscale('log')
			plt.yscale('log')
			plt.xlim([1e-2, 1e3])
			plt.ylim([1e-2, 1e0])
			plt.xlabel('$\chi_1$')
			plt.ylabel('$\partial \mathcal{F}_*/\partial \ln \chi_1$')
			

		return handles

def grid_mmax(pmin=1e-3, pmax=1.0):

	oms = np.logspace(-3, 0, 8)
	sigs = np.logspace(0, 3,8)
	dlogx = np.log10(sigs[1])-np.log10(sigs[0])
	dlogy = np.log10(oms[1])-np.log10(oms[0])

	tff_2ds = np.zeros((len(sigs), len(oms)))
	tfbs = np.zeros((len(sigs), len(oms)))
	fcolls = np.zeros((len(sigs), len(oms)))

	for isig in range(len(sigs)):
		for iom in range(len(oms)):
			sbe = SBEClass(sigs[isig], 1.5, omega0=oms[iom])
			fc, tff, tfb = sbe.fcoll_func(rvals='all')

			fcolls[isig][iom] = fc
			tff_2ds[isig][iom] = tff
			tfbs[isig][iom] =tfb


	ext = [np.log10(np.amin(sigs))-dlogx/2., np.log10(np.amax(sigs))+dlogx/2, np.log10(np.amin(oms))-dlogy/2., np.log10(np.amax(oms))+dlogy/2.]
	plt.rc('text',usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(3.2,3.))
	ax = plt.gca()
	imax = ax.imshow(np.rot90(fcolls), aspect='auto',interpolation='bilinear', cmap=cm.gray_r, norm=LogNorm(vmin=pmin, vmax=pmax),  extent=ext)
	
	
	xvs = np.arange(int(ext[0]), int(ext[1]+1.), 1)
	if len(xvs)>8:
		xvs = np.arange(int(ext[0]), int(ext[1]+1.), 2)
	xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
	
	yvs = np.arange(int(ext[2]), int(ext[3]+1.))
	yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]
	plt.xticks(xvs, xls)
	plt.yticks(yvs, yls)
	plt.ylabel("$\Omega$ (Myr$^{-1}$)")
	plt.xlabel("$\Sigma_0$ ($M_\odot$ pc$^{-2}$)")

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	cbar = plt.colorbar(imax, cax=cax, label='${\partial^2 \mathcal{F}_*}/{\partial \log \\rho_* \partial \log F}$')
	cblog = np.array(np.arange(int(np.log10(pmin)-0.51), int(np.log10(pmax)+0.51)), dtype='float')
	cbtick = np.power(10,cblog)
	cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
	cbar.set_ticks(cbtick)
	cbar.set_ticklabels(cblabs)
	
	plt.savefig('tfb_grid.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	
		


if __name__=='__main__':

	ext=False  
	conv=True
	disc=True
	XLIMS = [-5.5,8.]
	YLIMS = [-0.5, 8.0]
	PMIN= 1e-3
	PMAX= 1.01e-0
	resrho = 300
	respsi = 1000
	if disc:
		Q= 1.5
		Om0 =  0.026
		sigma0= 12.
		rho0 =None
		mach=None
		PHLAB = [-1.5, 4.]
		ENLAB = None
		sbe_name = 'sn'
	else:
		Q= 1.5
		Om0 = 1.7
		sigma0= 1000.
		rho0 =None
		mach=None
		PHLAB = [1.0,5.0]
		ENLAB = [4.5, 2.5]
		sbe_name = 'cmz'


	if ext:
		sbe_name+='_ext'

	sbeclass = SBEClass(sigma0, Q,Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)
	
	"""
	#Plot ICMF
	
	phirange = np.logspace(0, 11, 100)/1e3
	dpdphi_sn = phirange*sbeclass.dpdphi_func(phirange)
	dpdphi_sn /= np.trapz(dpdphi_sn, np.log10(phirange))

	
	Q= 1.5
	Om0 = 1.7 #1.0 #None #0.025
	sigma0= 1000.
	rho0 =None
	mach=None
	XLIMS = None #[0.5,10.0]
	YLIMS = None #[1.5, 7.0]
	PHLAB = [1.0,5.0]
	ENLAB = [4.5, 2.5]
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'cmz'
	sbeclass = SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=False, oname=sbe_name)


	dpdphi_cmz =phirange*sbeclass.dpdphi_func(phirange)
	dpdphi_cmz /= np.trapz(dpdphi_cmz, np.log10(phirange))

	def numfmt(x, pos): # your custom formatter function: divide by 100.0
		s = '$10^{}$'.format(int(np.log10(x*1000.0)))
		return s

	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)    # create your custom formatter function
	import pylab
	# your existing code can be inserted here

	

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(4.,4.))
	ax1 = fig.add_subplot(111)
	ax2 = ax1.twiny()
	ax1.plot(phirange, dpdphi_sn, c='b', label='Solar Nbhd.')
	ax1.plot(phirange, dpdphi_cmz, c='r', label='CMZ')
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax2.set_xscale('log')
	pylab.gca().xaxis.set_major_formatter(yfmt)
	ax1.set_ylim([1e-2, 1e0])
	ax1.set_xlim([1e-3, 1e4])
	ax1.set_ylabel('Logarithmic PDF -- $\phi \\times {\partial \mathcal{F}_\mathrm{c}}/{\partial \log \phi}$')
	ax1.set_xlabel('$\phi \equiv M_\mathrm{c}/M_\mathrm{crit}$')
	
	tick_locations = np.logspace(-3., 4., 8)

	ax2.set_xlim(ax1.get_xlim())
	#ax2.set_xticks(tick_locations)
	#ax2.set_xticklabels(tick_function(tick_locations))
	ax2.set_xlabel("Mass of star forming region -- $M_\mathrm{c}$ ($M_\odot$)")
	ax1.legend(loc='best')
	plt.savefig('paper_figure_dFdphi.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	exit()"""
	
	"""
	#Plot distribution of chi1 values
	
	#rho, fuv, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, sl=True,plot=False)
	#sbeclass.dFdtdisp_func(np.log(fuv),np.log(rho), xd2dpdlnxdlnpsi, c='b', plttype=1)
	sbeclass.plot_d2pdxdchi(plot=False, c='b')
	
	Q= 1.5
	Om0 = 1.7 #None #0.025
	sigma0= 1000.
	rho0 =None
	mach=None
	XLIMS = None #[0.5,10.0]
	YLIMS = None #[1.5, 7.0]
	PHLAB = [1.0,5.0]
	ENLAB = [4.5, 2.5]
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'cmz'

	sbeclass = SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=False, oname=sbe_name)
	#rho, fuv, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, sl=True,plot=False)
	#handles = sbeclass.dFdtdisp_func(np.log(fuv),np.log(rho), xd2dpdlnxdlnpsi, c='r', plttype=2)
	
	handles = sbeclass.plot_d2pdxdchi(plot=True, c='r')
	
	l1, =plt.plot([],[], c='b', label='Solar Nbhd.')
	l2, = plt.plot([],[], c='r',label='CMZ')
	leg1 = plt.legend(handles=[l1,l2], loc=2, fontsize=8)
	plt.legend(handles=handles,loc=4, fontsize=8)
	plt.gca().add_artist(leg1)
	plt.savefig('paper_figure_dpdchi.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	exit()
	"""
	
	#Plot distribution of dispersal timescales
	
	rho, fuv, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, sl=True,plot=False)
	sbeclass.dFdtdisp_func(np.log(fuv),np.log(rho), xd2dpdlnxdlnpsi, c='b', plttype=1)
	#sbeclass.plot_d2pdxdchi(plot=False, c='b')
	
	Q= 1.5
	Om0 = 1.7 #None #0.025
	sigma0= 1000.
	rho0 =None
	mach=None
	XLIMS = None #[0.5,10.0]
	YLIMS = None #[1.5, 7.0]
	PHLAB = [1.0,5.0]
	ENLAB = [4.5, 2.5]
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'cmz'

	sbeclass = SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=False, oname=sbe_name)
	rho, fuv, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, sl=True,plot=False)
	handles = sbeclass.dFdtdisp_func(np.log(fuv),np.log(rho), xd2dpdlnxdlnpsi, c='r', plttype=2)
	
	#handles = sbeclass.plot_d2pdxdchi(plot=True, c='r')
	
	l1, =plt.plot([],[], c='b', label='Solar Nbhd.')
	l2, = plt.plot([],[], c='r',label='CMZ')
	leg1 = plt.legend(handles=[l1,l2], loc=2, fontsize=8)
	plt.legend(handles=handles,loc=4, fontsize=8)
	plt.gca().add_artist(leg1)
	plt.savefig('paper_figure_cdf_tdisp.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	exit()
	

	#sbeclass.plot_d2pdxdchi()
	#sbeclass.get_pchi0_func(plot=True, pmin=1e-3, pmax=1e-0)
	#exit()
	
	

	rho, fuv, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, resx=resrho, resy=respsi, sl=True, plot=True)


	tm = sbeclass.dFdtdisp_func(np.log(fuv),np.log(rho), xd2dpdlnxdlnpsi)

	print('Median dispersal timescales:', tm)
