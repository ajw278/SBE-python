from __future__ import print_function

import os
import numpy as np
import SBEclass as SBE
import multiprocessing as mp
from pylab import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import copy
import matplotlib.gridspec as gridspec


def fname_func(sig,rho, mach, Q, Om):

	fname = 'S%.2lf_R%.2lf_M%.2lf_Q%.2lf_O%.2lf'
	return fname%(np.log10(sig), np.log10(rho), np.log10(mach),np.log10(Q), np.log10(Om))



def get_tmeds(sig,rho, mach, Q, Om, mstars, Fname):
	sbeclass = SBE.SBEClass(sig, Q,omega0=Om, mach=mach, rho0=rho, timeit=False, oname=Fname)
	rho_s, fuv_s, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=True, xlims =None,  ylims = None, extinct=False, sl=True,resx=300, resy=500, plot=False, interactive=False)		

	tmeds = sbeclass.dFdtdisp_mgrid_func(np.log(fuv_s),np.log(rho_s), xd2dpdlnxdlnpsi, msts=mstars)
	
	return tmeds


def get_F0f(sig,rho, mach, Q, Om, mstars, Fname):
	sbeclass = SBE.SBEClass(sig, Q,omega0=Om, mach=mach, rho0=rho, timeit=False, oname=Fname)
	return sbeclass.F0f



def fname_func_red(sig,Om, Q):

	fname = 'S%.2lf_O%.2lf_Q%.2lf'
	return fname%(np.log10(sig), np.log10(Om), Q)



"""sig=1e3
Om = 10.**(-0.22)
Q = 1.5
Fname = fname_func_red(sig, Om,Q)
sbeclass = SBE.SBEClass(sig, Q,omega0=Om, timeit=False, oname=Fname)
rho_s, fuv_s, xd2dpdlnxdlnpsi= sbeclass.plot_d2Fdydpsi(convolve=True, xlims =None,  ylims = None, extinct=False, sl=True,resx=300, resy=500, plot=True, interactive=False)"""

def grid_run(sigs, rhos, machs, Qs, Oms):

	tmed_arr = np.zeros((len(sigs), 2))
	for irun in range(len(sigs)):
		Fname =  fname_func(sigs[irun], rhos[irun], machs[irun], Qs[irun], Oms[irun])
		print('Calculating tmed for %s (%d/%d)'%(Fname, irun,len(sigs)))
		tmeds = get_tmeds(sigs[irun],rhos[irun],machs[irun], Qs[irun],Oms[irun], Fname)
		print('Median dispersal timescales:', tmeds)
		tmed_arr[irun] = np.array(tmeds)

	return tmed_arr


def grid_run_reduced(sigs, Oms, Qs, mstars, rtype='tmeds'):
	if rtype=='tmeds':
		tmed_arr = np.zeros((len(Qs), len(sigs), len(Oms), len(mstars)))
	elif rtype=='F0f':
		tmed_arr = np.zeros((len(Qs), len(sigs), len(Oms)))
	irun=0
	Nruns = len(sigs)*len(Qs)*len(Oms)
	for iQ in range(len(Qs)):
		for isig in range(len(sigs)):
			for iOm in range(len(Oms)):
				irun+=1
				Fname=  fname_func_red(sigs[isig], Oms[iOm], Qs[iQ])
				print('Calculating tmed for %s (%d/%d)'%(Fname, irun,Nruns))
				if rtype=='tmeds':
					tmeds = get_tmeds(sigs[isig],None,None, Qs[iQ],Oms[iOm],mstars, Fname)
					print('Median dispersal timescales:', tmeds)
					tmed_arr[iQ][isig][iOm] = tmeds
				elif rtype=='F0f':
					F0f = get_F0f(sigs[isig],None,None, Qs[iQ],Oms[iOm],mstars, Fname)
					print('F0f:', F0f)
					tmed_arr[iQ][isig][iOm] = F0f

	return tmed_arr

def plot_paramspace(tmed_grid, sig_space, Om_space, Q_space,m_space, pmin=1e-1, pmax=1e1, rtype='tmeds'):

	if rtype=='tmed':
		logcol =False
		vmin =0.0
		vmax=10.0
	elif rtype=='F0f':
		logcol = True
		vmin=1e-3
		vmax= 1e4
	else:
		print("Results type '{0}' not recognised.".format(rtype))
		exit()

	dlogx = np.log10(sig_space[1])-np.log10(sig_space[0])
	dlogy = np.log10(Om_space[1])-np.log10(Om_space[0])


	ext = [np.log10(np.amin(sig_space))-dlogx/2., np.log10(np.amax(sig_space))+dlogx/2, np.log10(np.amin(Om_space))-dlogy/2., np.log10(np.amax(Om_space))+dlogy/2.]
	xlims  = [ext[0],ext[1]]
	ylims = [ext[2],ext[3]]

	plt.rc('text',usetex=True)
	plt.rc('font', family='serif')

	if len(tmed_grid.shape)==4:	
		rows = tmed_grid.shape[0]
		cols = tmed_grid.shape[1]
	else:
		cols = tmed_grid.shape[0]
		rows = 1 
		tmed_grid_cpy = np.zeros((1,tmed_grid.shape[0],tmed_grid.shape[1],tmed_grid.shape[2]))
		tmed_grid_cpy[0] = tmed_grid
		tmed_grid = tmed_grid_cpy
		

	if rows==5 and cols==3:
		fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8.0,10.0))
	elif rows==3 and cols==3:
		fig = plt.figure(figsize=(6.0,5.0))
	elif rows==1 and cols==3:
		fig = plt.figure(figsize=(7.0, 2.0))
	else:
		print('Figure size not determined for parameter space with shape:', tmed_grid.shape)

	gs1 = gridspec.GridSpec(rows, cols, width_ratios=[1]*cols,
         wspace=0.08, hspace=0.08, top=0.95, bottom=0.05, left=0.17, right=0.845) 

	xvs = np.arange(int(ext[0]), int(ext[1]+1.), 1)
	if len(xvs)>8:
		xvs = np.arange(int(ext[0]), int(ext[1]+0.9), 2)
	xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
	
	yvs = np.arange(int(ext[2]), int(ext[3]+1.0))
	yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]

	
	yls_h = yls
	#[:len(yls)-1]
	#yls_h.append('')


	xls_h = xls[:len(xls)-1]
	xls_h.append('')

	yls_void = ['' for i in range(len(yvs))]
	xls_void = ['' for i in range(len(xvs))]

	for irow in range(rows):
		for icol in range(cols):
			
			ax = plt.subplot(gs1[irow,icol]) #axes[irow][icol]
			if logcol:
				imax = ax.imshow(np.rot90(tmed_grid[irow][icol]), aspect='auto',interpolation='bicubic', cmap=cm.hot, norm=LogNorm(vmin=vmin, vmax=vmax),  extent=ext)
			else:
				imax = ax.imshow(np.rot90(tmed_grid[irow][icol]), aspect='auto',interpolation='bicubic', cmap=cm.hot, vmin=vmin, vmax=vmax,  extent=ext)

			if irow ==0:
				ax.text(2.0, -0.5, '$Q=%.1lf$'%(Q[icol]), color='lightgreen', weight='bold', size=15)
			if icol ==0 and rows>1:
				ax.text(1.3, -2.7, '$m_*=%.1lf \, M_\odot$'%(m_space[irow]), color='lightgreen', weight='bold', size=15)
				

			if Q[icol]==1.5:
				ax.scatter(np.log10(12.0), np.log10(0.025))
			
			if (irow==rows-1 and icol==0):
				ax.set_xticks(xvs)
				ax.set_yticks(yvs)
				
				ax.set_xticklabels(xls_h)
				ax.set_yticklabels(yls_h)


			elif irow==rows-1:
				ax.set_xticks(xvs)
				ax.set_yticks(yvs)

				if icol==cols-1:
					ax.set_xticklabels(xls)
				else:
					ax.set_xticklabels(xls_h)
				ax.set_yticklabels(yls_void)


			elif icol==0:
				ax.set_xticks(xvs)
				ax.set_yticks(yvs)

				ax.set_xticklabels(xls_void)
				if irow==0:
					ax.set_yticklabels(yls)
				else:
					ax.set_yticklabels(yls_h)
					


			else:
				ax.set_xticks(xvs)
				ax.set_yticks(yvs)

				ax.set_xticklabels(xls_void)
				ax.set_yticklabels(yls_void)

			if icol==0 and irow ==int(rows/2):
				ax.set_ylabel("$\Omega$ (Myr$^{-1}$)")
			if irow==rows-1 and icol ==int(cols/2):
				ax.set_xlabel("$\Sigma_0$ ($M_\odot$ pc$^{-2}$)")

			
			gal_line_Sig = np.logspace(0, 4)
			gal_line_Om = 0.058*(gal_line_Sig/100.)**0.49
			
			if rtype=='tmed':
				levels = [2., 4., 6.]
				labs = ['$2$ Myr', '$4$ Myr', '$6$ Myr'] 

				conts = ax.contour(np.rot90(tmed_grid[irow][icol]), levels, colors='w', origin='upper', extent=ext, linewidths=0.8)


				if irow==0:
					conts_orig = conts
			#ax.clabel(conts, inline=1, fontsize=8, fmt='%d Myr', manual=False)
			#ax.plot(np.log10(gal_line_Sig),np.log10(gal_line_Om), c='g')


			
			
			
			
		
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.9])
	fig.colorbar(imax, cax=cbar_ax)	

	if rtype=='tmed':		
		cbar = fig.colorbar(imax, cax=cbar_ax, label='$\\tau_\mathrm{disp.,1/2}$ (Myr)')
		cbtick = np.arange(0, 11,2, dtype=int)#np.power(10,cblog)
		#cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
		cbar.set_ticks(cbtick)
		cbar.set_ticklabels(cbtick)
		cbar.add_lines(conts_orig)
	elif rtype=='F0f':
		cbar = fig.colorbar(imax, cax=cbar_ax, label='$F_0^{\mathrm{f}}$ ($G_0$)')
		
	#cblog = np.array(np.arange(int(np.log10(pmin)-0.51), int(np.log10(pmax)+1.)), dtype='float')
	
	
	"""cbar.ax.plot([ext[0], ext[1]], [1.]*2, 'w')
	cbar.ax.plot([ext[0], ext[1]], [2.]*2, 'w')
	cbar.ax.plot([ext[0], ext[1]], [3.]*2, 'w')"""
	if rtype=='tmed':	
		plt.savefig('paper_figure_tdisp_exp.pdf', bbox_inches='tight', format='pdf')
	elif rtype=='F0f':
		plt.savefig('paper_figure_F0f.pdf', bbox_inches='tight', format='pdf')
	else:
		plt.savefig('paper_figure_other.pdf', bbox_inches='tight', format='pdf')
		
	plt.show()
	
	



if __name__ =='__main__':

	RTYPE = 'tmed'
	FNAME = RTYPE+'_grid'

	sig = np.logspace(0, 4, 25)
	Q = np.array([0.5,1.5, 3.0])
	Omega = np.logspace(-3,0.5, 30)

	mstars = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
	

	#Sigma0_mg,rho0_mg,mach_mg, Q_mg, Omega0_mg = np.meshgrid(sig, rho, mach, Q, Omega)

	
	Q_mg, Sigma0_mg,Omega0_mg= np.meshgrid(Q, sig,Omega, indexing='ij')
	
	if os.path.isfile(FNAME+'.npy'):
		tmed_arr = np.load(FNAME+'.npy')
		sig= np.load('sig.npy')
		Q= np.load('Q.npy')
		Omega= np.load('Omega.npy')
		mstars= np.load('mstars.npy')
	else:
		tmed_arr = grid_run_reduced(sig, Omega, Q, mstars, rtype=RTYPE)
		np.save(FNAME, tmed_arr)
		np.save('sig', sig)
		np.save('Q', Q)
		np.save('Omega', Omega)
		np.save('mstars', mstars)

	print(tmed_arr.shape)
	if len(tmed_arr.shape)==4:
		tmed_arr = np.rollaxis(tmed_arr, len(tmed_arr.shape)-1,0)
	print(tmed_arr.shape)
	#tmed_arr = tmed_arr[1:len(mstars)-1]
	#mstars = mstars[1:len(mstars)-1]

	plot_paramspace(tmed_arr, sig, Omega, Q,mstars, pmin=5e-1, pmax=1e1, rtype=RTYPE)
	
	

	
