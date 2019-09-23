from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import matplotlib.gridspec as gridspec
from pylab import cm
from matplotlib.colors import LogNorm

import SBEclass as SBE

XLIMS = [-5.5,8.]
YLIMS = [-0.5, 8.0]


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                 '#e41a1c', '#dede00']


def panel_2DPDF(ext=False):
	
	

	conv=True

	Q= 1.5
	Om0 = 0.026
	sigma0= 12.
	rho0 =None
	mach=None
	PHLAB = [-1.5, 4.]
	ENLAB = None
	
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'sn'

	if ext:
		sbe_name+='_ext'

	snclass = SBE.SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)

	rho, fuv, xd2dpdlnxdlnpsi_sn = snclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, sl=True, plot=False)

	xd2dpdlnxdlnpsi_sn /= simps(simps(xd2dpdlnxdlnpsi_sn, np.log10(fuv)), np.log10(rho))
	xd2dpdlnxdlnpsi_sn[np.where(xd2dpdlnxdlnpsi_sn<PMIN)] = PMIN

	logxmin = XLIMS[0]
	logxmax = XLIMS[1]
	logymin = YLIMS[0]
	logymax = YLIMS[1]
	dlogx = float(logxmax-logxmin)/float(len(rho))
	dlogy = float(logymax-logymin)/float(len(fuv))

	EXT = [np.log10(np.amin(rho))-dlogx/2., np.log10(np.amax(rho))+dlogx/2, np.log10(np.amin(fuv))-dlogy/2., np.log10(np.amax(fuv))+dlogy/2.]


	Q= 1.5
	Om0 = 1.7 
	sigma0= 1000.
	rho0 =None
	mach=None
	PHLAB = [1.0,5.0]
	ENLAB = [4.5, 2.5]
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'cmz'

	cmzclass = SBE.SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)

	rho, fuv, xd2dpdlnxdlnpsi_cmz = cmzclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=ext, sl=True, plot=False)

	xd2dpdlnxdlnpsi_cmz /= simps(simps(xd2dpdlnxdlnpsi_cmz, np.log10(fuv)), np.log10(rho))
	xd2dpdlnxdlnpsi_cmz[np.where(xd2dpdlnxdlnpsi_cmz<PMIN)] = PMIN


	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(9.0,4.0))

	
	gs1 = gridspec.GridSpec(1, 2, wspace=0.04, hspace=0.00, top=0.95, bottom=0.05, left=0.17, right=0.845) 

	xvs = np.arange(int(EXT[0]), int(EXT[1]+1.), 1)
	if len(xvs)>8:
		xvs = np.arange(int(EXT[0]), int(EXT[1]+1.0), 2)
	xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
	
	yvs = np.arange(int(EXT[2]), int(EXT[3]+1.0))
	if len(xvs)>8:
		yvs = np.arange(int(EXT[2]), int(EXT[3]+1.0), 2)
	yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]

	
	yls_h = yls[:len(yls)-1]
	yls_h.append('')


	xls_h = xls[:len(xls)-1]
	xls_h.append('')

	yls_void = ['' for i in range(len(yvs))]
	xls_void = ['' for i in range(len(xvs))]

	ax1 = plt.subplot(gs1[0,0]) #axes[irow][icol]
	imax = ax1.imshow(np.rot90(xd2dpdlnxdlnpsi_sn), aspect='auto',interpolation='bilinear', cmap=cm.gray_r, norm=LogNorm(vmin=PMIN, vmax=PMAX),  extent=EXT)

	



	ax1.set_ylabel("$F$ ($G_0$)")
	ax1.set_xlabel("$\\rho_*$ ($M_\odot$ pc$^{-3}$)")	


	ax = plt.subplot(gs1[0,1]) #axes[irow][icol]
	imax = ax.imshow(np.rot90(xd2dpdlnxdlnpsi_cmz), aspect='auto',interpolation='bilinear', cmap=cm.gray_r, norm=LogNorm(vmin=PMIN, vmax=PMAX),  extent=EXT)

	ax.set_xticks(xvs)
	ax.set_yticks(yvs)
	
	ax.set_xticklabels(xls)
	ax.set_yticklabels(yls_void)
	#ax1.tick_params(axis='x', direction='inout')

	ax1.set_xticks(xvs)
	ax1.set_yticks(yvs)
	
	ax1.set_xticklabels(xls)
	ax1.set_yticklabels(yls)

	ax.set_xlabel("$\\rho_*$ ($M_\odot$ pc$^{-3}$)")


	contours1 = ['ONC', 'Cygnus OB2', 'lambda Ori', 'sigma Ori']
	

	for icont in range(len(contours1)):
		nsp, fsp, fsp_h, fsp_l = np.load(contours1[icont]+'_cont_trunc.npy')
		nsp = np.log10(nsp*0.5)
		fsp = np.log10(fsp+1.)
		fsp_h = np.log10(fsp_h+1.)
		fsp_l= np.log10(fsp_l+1.)
		print(nsp, fsp)
		ax1.fill_between(nsp,fsp_h,fsp_l , where=fsp_l<= fsp_h, facecolor=CB_color_cycle[icont], alpha=0.4, interpolate=True)

		LABEL = contours1[icont]

		if LABEL=='lambda Ori':
			LABEL = '$\lambda$ Ori'
		if LABEL=='sigma Ori':
			LABEL = '$\sigma$ Ori'
		
		if LABEL=='Cygnus OB2':
			ax1.text(np.amax(nsp)-6., np.amax(fsp)-1.0, LABEL, color=CB_color_cycle[icont])
		elif LABEL=='$\sigma$ Ori':
			ax1.text(np.amax(nsp)-2., np.amax(fsp)+0.5, LABEL, color=CB_color_cycle[icont])
		elif LABEL=='ONC':
			ax1.text(np.amax(nsp)+0.5, np.amax(fsp)+0.5, LABEL, color=CB_color_cycle[icont])
		else:
			ax1.text(np.amax(nsp)-2., np.amax(fsp), LABEL, color=CB_color_cycle[icont])


	"""XERR = (np.log10(672./2.)-np.log10(2.5/2.))/4.
	
	ax1.errorbar(np.log10((2.5/2.+672./2.)/2.), 0.3, yerr=None, xerr=XERR, marker=None, color='orange')
	ax1.text(2.5, 0.5,'Serpens', color='orange')"""

	lup_n = 500.
	lup_f_up = 4.5
	lup_f_lo = 2.9
	lup_f = (lup_f_up+lup_f_lo)/2.
	llim_f = (np.absolute(np.log10(lup_f)-np.log10(lup_f_lo)))
	ulim_f = np.absolute(np.log10(lup_f)-np.log10(lup_f_up))
	
	ax1.errorbar(np.log10(lup_n/2.), np.log10(lup_f), xerr=0.2,xuplims=[1], yerr=[[llim_f],[ulim_f]], c='k')
	antext = 'Lupus'
	ax1.annotate(antext, (np.log10(lup_n*1.5), np.log10(lup_f)), color='k')
	
	contours2 = ['Arches', 'Wd 1', 'Quintuplet']
	

	for icont in range(len(contours2)):
		nsp, fsp, fsp_h, fsp_l = np.load(contours2[icont]+'_cont_trunc.npy')
		nsp = np.log10(nsp*0.5)
		fsp = np.log10(fsp+1.)
		fsp_h = np.log10(fsp_h+1.)
		fsp_l= np.log10(fsp_l+1.)
		print(nsp, fsp)
		ax.fill_between(nsp,fsp_h,fsp_l , where=fsp_l<= fsp_h, facecolor=CB_color_cycle[icont+len(contours1)], alpha=0.4, interpolate=True)

		LABEL = contours2[icont]

		if LABEL=='Wd 1':
			ax.text(np.amax(nsp)-3., np.amax(fsp)-0.5, LABEL, color=CB_color_cycle[icont+len(contours1)])
		elif LABEL=='Quintuplet':
			ax.text(np.amax(nsp)-4., np.amax(fsp)+0.5, LABEL, color=CB_color_cycle[icont+len(contours1)])
		else:
			ax.text(np.amax(nsp)-2., np.amax(fsp)+0.5, LABEL, color=CB_color_cycle[icont+len(contours1)])




	rh_tmp = rho[np.where((rho>10.**XLIMS[0])&(rho<10.**XLIMS[1]))]
	f_tmp = fuv[np.where((fuv>10.**YLIMS[0])&(fuv<10.**YLIMS[1]))]
	rs_mg, f_mg = np.meshgrid(rh_tmp, f_tmp, indexing='ij')

	tau_dips = snclass.tau_func(rs_mg, f_mg, mstar=0.5, alpha=5.4e-3)
			

	conts1= ax1.contour(np.log10(rh_tmp), np.log10(f_tmp), np.swapaxes(tau_dips,0,1), levels=[1.,2.,3.], colors='darkblue')
	
	fmt = {}
	strs = ['$1$~Myr', '$2$~Myr', '$3$~Myr']
	for l, s in zip(conts1.levels, strs):
		fmt[l] = s
	ax1.clabel(conts1, conts1.levels, inline=True, fmt=fmt, fontsize=10, manual =True)

	
	conts= ax.contour(np.log10(rh_tmp), np.log10(f_tmp), np.swapaxes(tau_dips,0,1), levels=[1.,2.,3.], colors='darkblue')
	
	fmt = {}
	strs = ['$1$~Myr', '$2$~Myr', '$3$~Myr']
	for l, s in zip(conts.levels, strs):
		fmt[l] = s
	ax.clabel(conts, conts.levels, inline=True, fmt=fmt, fontsize=10, manual =True)

	ax1.text(XLIMS[0]+0.5, YLIMS[1]-0.5, 'Solar Nbhd.')
	ax.text(XLIMS[0]+0.5, YLIMS[1]-0.5, 'CMZ')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.9])

	cblog = np.array(np.arange(int(np.log10(PMIN)-0.51), int(np.log10(PMAX)+1.)), dtype='float')
	cbtick = np.power(10,cblog)
	cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
	cbar = fig.colorbar(imax, cax=cbar_ax, label='Stellar PDF - $\partial^2 \mathcal{F}_*/\partial \log \\rho_* \partial \log F$')
	#cblog = np.array(np.arange(int(np.log10(pmin)-0.51), int(np.log10(pmax)+1.)), dtype='float')
	
	#cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
	cbar.set_ticks(cbtick)
	cbar.set_ticklabels(cblabs)


			
	plt.savefig('paper_figure_2panelPDF.pdf', bbox_inches='tight', format='pdf')
	plt.show()



def panel_2DPDF_ext():
	

	conv=True

	Q= 1.5
	Om0 =0.026
	sigma0= 12.
	rho0 =None
	mach=None
	PHLAB = [-1.5, 4.]
	ENLAB = None
	
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'sn'

	sbe_name+='_ext'

	snclass = SBE.SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)

	rho, fuv, xd2dpdlnxdlnpsi_sn = snclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=True, sl=True, plot=False)

	xd2dpdlnxdlnpsi_sn /= simps(simps(xd2dpdlnxdlnpsi_sn, np.log10(fuv)), np.log10(rho))


	xd2dpdlnxdlnpsi_sn[np.where(xd2dpdlnxdlnpsi_sn<PMIN)] = PMIN

	logxmin = XLIMS[0]
	logxmax = XLIMS[1]
	logymin = YLIMS[0]
	logymax = YLIMS[1]
	dlogx = float(logxmax-logxmin)/float(len(rho))
	dlogy = float(logymax-logymin)/float(len(fuv))

	EXT = [np.log10(np.amin(rho))-dlogx/2., np.log10(np.amax(rho))+dlogx/2, np.log10(np.amin(fuv))-dlogy/2., np.log10(np.amax(fuv))+dlogy/2.]


	Q= 1.5
	Om0 = 1.7 #None #0.025
	sigma0= 1000.
	rho0 =None
	mach=None
	PHLAB = [1.0,5.0]
	ENLAB = [4.5, 2.5]
	PMIN= 1e-3
	PMAX= 1.01e-0
	sbe_name = 'cmz'

	cmzclass = SBE.SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)

	rho, fuv, xd2dpdlnxdlnpsi_cmz = cmzclass.plot_d2Fdydpsi(convolve=conv, xlims = XLIMS, ylims = YLIMS, phlab=PHLAB, enclab=ENLAB,pmin = PMIN, pmax = PMAX, extinct=True, sl=True, plot=False)


	xd2dpdlnxdlnpsi_cmz /= simps(simps(xd2dpdlnxdlnpsi_cmz, np.log10(fuv)), np.log10(rho))

	
	xd2dpdlnxdlnpsi_cmz[np.where(xd2dpdlnxdlnpsi_cmz<PMIN)] = PMIN


	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(9.0,4.0))

	gs1 = gridspec.GridSpec(1, 2, wspace=0.04, hspace=0.00, top=0.95, bottom=0.05, left=0.17, right=0.845) 

	xvs = np.arange(int(EXT[0]), int(EXT[1]+1.), 1)
	if len(xvs)>8:
		xvs = np.arange(int(EXT[0]), int(EXT[1]+1.0), 2)
	xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
	
	yvs = np.arange(int(EXT[2]), int(EXT[3]+1.0))
	if len(xvs)>8:
		yvs = np.arange(int(EXT[2]), int(EXT[3]+1.0), 2)
	yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]

	
	yls_h = yls[:len(yls)-1]
	yls_h.append('')


	xls_h = xls[:len(xls)-1]
	xls_h.append('')

	yls_void = ['' for i in range(len(yvs))]
	xls_void = ['' for i in range(len(xvs))]

	ax1 = plt.subplot(gs1[0,0]) #axes[irow][icol]
	imax = ax1.imshow(np.rot90(xd2dpdlnxdlnpsi_sn), aspect='auto',interpolation='bilinear', cmap=cm.hot, norm=LogNorm(vmin=PMIN, vmax=PMAX),  extent=EXT)

	
	ax1.set_ylabel("$F$ ($G_0$)")
	ax1.set_xlabel("$\\rho_*$ ($M_\odot$ pc$^{-3}$)")	


	ax = plt.subplot(gs1[0,1]) #axes[irow][icol]
	imax = ax.imshow(np.rot90(xd2dpdlnxdlnpsi_cmz), aspect='auto',interpolation='bilinear', cmap=cm.hot, norm=LogNorm(vmin=PMIN, vmax=PMAX),  extent=EXT)

	ax.set_xticks(xvs)
	ax.set_yticks(yvs)
	
	ax.set_xticklabels(xls)
	ax.set_yticklabels(yls_void)
	#ax1.tick_params(axis='x', direction='inout')

	ax1.set_xticks(xvs)
	ax1.set_yticks(yvs)
	
	ax1.set_xticklabels(xls)
	ax1.set_yticklabels(yls)

	ax.set_xlabel("$\\rho_*$ ($M_\odot$ pc$^{-3}$)")


	rh_tmp = rho[np.where((rho>10.**XLIMS[0])&(rho<10.**XLIMS[1]))]
	f_tmp = fuv[np.where((fuv>10.**YLIMS[0])&(fuv<10.**YLIMS[1]))]
	rs_mg, f_mg = np.meshgrid(rh_tmp, f_tmp, indexing='ij')

	tau_dips = snclass.tau_func(rs_mg, f_mg, mstar=0.5, alpha=5.4e-3)
			


	
	RAMLIM = 9863.
	ax1.axvline(np.log10(RAMLIM), c='c')
	ax1.text(np.log10(RAMLIM)+0.3,6.5,'$\\tau_\mathrm{ram} = 1$ Myr',rotation=90, color='c')
	ax.axvline(np.log10(RAMLIM), c='c')
	ax.text(np.log10(RAMLIM)+0.3,6.5,'$\\tau_\mathrm{ram}=1$ Myr',rotation=90, color='c')

	conts1= ax1.contour(np.log10(rh_tmp), np.log10(f_tmp), np.swapaxes(tau_dips,0,1), levels=[1.,2.,3.], colors='w')
	
	fmt = {}
	strs = ['$1$~Myr', '$2$~Myr', '$3$~Myr']
	for l, s in zip(conts1.levels, strs):
		fmt[l] = s
	ax1.clabel(conts1, conts1.levels, inline=True, fmt=fmt, fontsize=10, manual =True)

	
	conts= ax.contour(np.log10(rh_tmp), np.log10(f_tmp), np.swapaxes(tau_dips,0,1), levels=[1.,2.,3.], colors='w')
	
	fmt = {}
	strs = ['$1$~Myr', '$2$~Myr', '$3$~Myr']
	for l, s in zip(conts.levels, strs):
		fmt[l] = s
	ax.clabel(conts, conts.levels, inline=True, fmt=fmt, fontsize=10, manual =True)

	ax1.text(XLIMS[0]+0.5, YLIMS[1]-0.5, 'Solar Nbhd.', color='w')
	ax.text(XLIMS[0]+0.5, YLIMS[1]-0.5, 'CMZ', color='w')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.9])

	cblog = np.array(np.arange(int(np.log10(PMIN)-0.51), int(np.log10(PMAX)+1.)), dtype='float')
	cbtick = np.power(10,cblog)
	cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
	cbar = fig.colorbar(imax, cax=cbar_ax, label='Stellar PDF - $\partial^2 \mathcal{F}_*/\partial \log \\rho_* \partial \log F$')
	cbar.set_ticks(cbtick)
	cbar.set_ticklabels(cblabs)
 
			
	plt.savefig('paper_figure_2panelPDF_ext.pdf', bbox_inches='tight', format='pdf')
	plt.show()

def panel_pchi0():
	
	XLIMS = [-3.,7.]
	YLIMS = [-2., 5.]


	conv=True

	Q= 1.5
	Om0 = 0.026
	sigma0= 12.
	rho0 =None
	mach=None
	PHLAB = [-1.5, 4.]
	ENLAB = None
	
	PMIN= 1e-3
	PMAX= 1.01e-1
	sbe_name = 'sn'

	xspace = np.logspace(XLIMS[0], XLIMS[1], 150)
	phispace = np.logspace(YLIMS[0],YLIMS[1], 200)


	snclass = SBE.SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)


	pchi0sn = snclass.get_pchi0_func(xspace=xspace, phispace=phispace, plot=False)


	logxmin = XLIMS[0]
	logxmax = XLIMS[1]
	logymin = YLIMS[0]
	logymax = YLIMS[1]
	dlogx = float(logxmax-logxmin)/float(len(xspace))
	dlogy = float(logymax-logymin)/float(len(phispace))

	EXT = [np.log10(np.amin(xspace))-dlogx/2., np.log10(np.amax(xspace))+dlogx/2, np.log10(np.amin(phispace))-dlogy/2., np.log10(np.amax(phispace))+dlogy/2.]


	Q= 1.5
	Om0 = 1.7 #None #0.025
	sigma0= 1000.
	rho0 =None
	mach=None
	PHLAB = [1.0,5.0]
	ENLAB = [4.5, 2.5]
	
	sbe_name = 'cmz'
	

	cmzclass = SBE.SBEClass(sigma0, Q,omega0=Om0, mach=mach, rho0=rho0, timeit=True, oname=sbe_name)


	pchi0cmz = cmzclass.get_pchi0_func(xspace=xspace, phispace=phispace, plot=False)


	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(4.0,7.0))

	gs1 = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.04, top=0.95, bottom=0.05, left=0.17, right=0.845) 

	xvs = np.arange(int(EXT[0]), int(EXT[1]+1.), 1)
	if len(xvs)>8:
		xvs = np.arange(int(EXT[0]), int(EXT[1]+1.0), 2)
	xls = ['$10^{%d}$'%(xvs[i]) for i in range(0,len(xvs),1)]
	
	yvs = np.arange(int(EXT[2]), int(EXT[3]+1.0))
	if len(xvs)>8:
		yvs = np.arange(int(EXT[2]), int(EXT[3]+1.0), 2)
	yls = ['$10^{%d}$'%(yvs[i]) for i in range(len(yvs))]

	
	yls_h = yls[:len(yls)-1]
	yls_h.append('')


	xls_h = xls[:len(xls)-1]
	xls_h.append('')

	yls_void = ['' for i in range(len(yvs))]
	xls_void = ['' for i in range(len(xvs))]

	ax1 = plt.subplot(gs1[0,0]) #axes[irow][icol]
	imax = ax1.imshow(np.rot90(pchi0sn), aspect='auto',interpolation='None', cmap=cm.hot, norm=LogNorm(vmin=PMIN, vmax=PMAX),  extent=EXT)

	
	ax1.set_ylabel("$\phi$")


	ax = plt.subplot(gs1[1,0]) #axes[irow][icol]
	imax = ax.imshow(np.rot90(pchi0cmz), aspect='auto',interpolation='None', cmap=cm.hot, norm=LogNorm(vmin=PMIN, vmax=PMAX),  extent=EXT)

	ax.set_xticks(xvs)
	ax.set_yticks(yvs)
	
	ax.set_xticklabels(xls)
	ax.set_yticklabels(yls)
	#ax1.tick_params(axis='x', direction='inout')

	ax1.set_xticks(xvs)
	ax1.set_yticks(yvs)
	
	ax1.set_xticklabels(xls_void)
	ax1.set_yticklabels(yls)

	ax.set_xlabel("$x$")
	ax.set_ylabel("$\phi$")


	
	ax1.text(XLIMS[0]+0.5, YLIMS[1]-0.5, 'Solar Nbhd.', color='w')
	ax.text(XLIMS[0]+0.5, YLIMS[1]-0.5, 'CMZ', color='w')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.9])

	cblog = np.array(np.arange(int(np.log10(PMIN)-0.51), int(np.log10(PMAX)+1.)), dtype='float')
	cbtick = np.power(10,cblog)
	cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
	cbar = fig.colorbar(imax, cax=cbar_ax, label='$p_\mathrm{S} (x,\phi)$')
	#cblog = np.array(np.arange(int(np.log10(pmin)-0.51), int(np.log10(pmax)+1.)), dtype='float')
	
	#cblabs = ['$10^{%d}$'%(cblog[i]) for i in range(len(cblog))]
	cbar.set_ticks(cbtick)
	cbar.set_ticklabels(cblabs)
 
			
	plt.savefig('paper_figure_pchi0.pdf', bbox_inches='tight', format='pdf')
	plt.show()

if __name__ =='__main__':

	panel_2DPDF()
	panel_2DPDF_ext()
	#exit()
	panel_pchi0()
	
