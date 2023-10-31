import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import os

parser = OptionParser()
(options,args) = parser.parse_args()

"""
dtype1 = np.dtype([('filter', '|U5'),('clrcoeff','f8'),('zpdiff', 'f8'),('JD', 'f8'),('flux','f8'),('flux_error','f8'),('nearestrefmag','f8'),('nearestrefmagunc','f8')])

data = np.loadtxt(args[0], usecols=[4,13,20,22,24,25,33,34], dtype=dtype1, skiprows=1)

#data1=np.loadtxt(args[1], delimiter = ',',skiprows = 1)   this is the PTF data import but skipped to focus on plotting ZTF in mag 


filter = data['filter']
ind = np.where(filter=='ZTF_r')
zpdiff = data['zpdiff'][ind]
jd = data['JD'][ind]
flux = data['flux'][ind]
flux_error = data['flux_error'][ind]
nearestrefmag = data['nearestrefmag'][ind]
nearestrefmagunc = data['nearestrefmagunc'][ind]
clrcoeff = data['clrcoeff'][ind]

nearestrefflux = 10**(0.4 * (zpdiff-nearestrefmag))
nearestreffluxunc = nearestrefmagunc * nearestrefflux / 1.0857
Fluxtot = flux + nearestrefflux
Fluxunctot = np.sqrt( flux_error**2-nearestreffluxunc**2)
SNRtot = Fluxtot/Fluxunctot

ind1 = np.where(SNRtot > 3 )
    # we have a “confident” detection, compute and plot mag with error bar:
#mag = zpdiff[ind1]-2.5*(np.log10(Fluxtot[ind1]))+clrcoeff[ind1]*(float(args[3])-float(args[4]))    I do not remember what these args were suppose to be so im using a different one for now to see if that will work just fine.
mag = zpdiff - 2.5 * log10( Flux_tot )
print(mag)
omag = 1.0857 / SNRtot[ind1]
jd = jd[ind1] - 2400000.5
#jd_all = np.append(data1[:,0], jd)
#mag_all = np.append(data1[:,1], mag)
#omag_all = np.append(data1[:,2], omag)

np.savetxt(args[2],np.c_[jd,mag,omag], fmt='%.5f %g %g')

plt.plot(jd, mag, 'ko')
plt.title(args[1][:-4])
plt.savefig(args[1])
plt.show()

#print(jd)
#print(flux)
#print(flux_error)
"""
#Combining the ZTF and PTF data files and creating a combined lightcurve png and data set:
dtype1 = np.dtype([('filter', '|U5'), ('zpdiff', 'f8'),('JD', 'f8'),('flux','f8'),('flux_error','f8'),('nearestrefmag','f8'),('nearestrefmagunc','f8')])

ztf_data = "/Volumes/MeganSSD/KupferResearch/Data/ZTF_data/Up_to_Date_ZTF_10.27.23"
#ptf_data = "/Users/megan/Research/KupferResearch/Data/old_data"

#ztf_list = [k[:-4] for k in os.listdir(ztf_data) if "J"==k[0]]
ztf_list = [k[:-4] for k in os.listdir(ztf_data) if "b"==k[0]]
#ptf_list = [k[:-4] for k in os.listdir(ptf_data) if "J"==k[0]]

#combined_list = np.unique(ztf_list + ptf_list)
combined_list = np.unique(ztf_list)

for i in combined_list:
    x = ztf_data + "/" + i + ".txt"

    if i + ".txt" in os.listdir(ztf_data) and False:
        zdata = np.loadtxt(x, usecols=[4,20,22,24,25,33,34], dtype=dtype1)

        filter = zdata['filter']
        ind = np.where(filter=='ZTF_r')
        zpdiff = zdata['zpdiff'][ind]
        jd = zdata['JD'][ind]
        flux = zdata['flux'][ind]
        flux_error = zdata['flux_error'][ind]
        nearestrefmag = zdata['nearestrefmag'][ind]
        nearestrefmagunc = zdata['nearestrefmagunc'][ind]

        nearestrefflux = 10**(0.4 * (zpdiff-nearestrefmag))
        nearestreffluxunc = nearestrefmagunc * nearestrefflux / 1.0857
        Fluxtot = flux + nearestrefflux
        Fluxunctot = np.sqrt( flux_error**2-nearestreffluxunc**2)
        SNRtot = Fluxtot/Fluxunctot
        ind1 = np.where(SNRtot > 3 )
        mag = zpdiff[ind1]-2.5*(np.log10(Fluxtot[ind1]))
        omag = 1.0857 / SNRtot[ind1]
        jd = jd[ind1] - 2400000.5 

    if i + ".txt" in os.listdir(ztf_data):
        jd = []
        zpdiff = []
        flux = []
        flux_error = []
        nearestrefmag = []
        nearestrefmagunc = []

        with open(x, "r") as readfile:
            for line in readfile.readlines():
                if line[0]=="#" or "null" in line:
                    continue
                if "ZTF_r" in line:
                    jd.append(line.split()[22])
                    zpdiff.append(line.split()[20])
                    flux.append(line.split()[24])
                    flux_error.append(line.split()[25])
                    nearestrefmag.append(line.split()[33])
                    nearestrefmagunc.append(line.split()[34])


        jd = np.array(jd, dtype = np.float64)
        zpdiff = np.array(zpdiff, dtype = np.float64)
        flux = np.array(flux, dtype = np.float64)
        flux_error = np.array(flux_error, dtype = np.float64) 
        nearestrefmag = np.array(nearestrefmag, dtype = np.float64) 
        nearestrefmagunc = np.array(nearestrefmagunc, dtype = np.float64)

        nearestrefflux = 10**(0.4 * (zpdiff-nearestrefmag))
        nearestreffluxunc = nearestrefmagunc * nearestrefflux / 1.0857
        Fluxtot = flux + nearestrefflux
        Fluxunctot = np.sqrt( flux_error**2-nearestreffluxunc**2)
        SNRtot = Fluxtot/Fluxunctot
        ind1 = np.where(SNRtot > 3 )
        mag = zpdiff[ind1]-2.5*(np.log10(Fluxtot[ind1]))
        omag = 1.0857 / SNRtot[ind1]
        jd = jd[ind1] - 2400000.5   

        
    #else:
        #continue
    #pdata = np.loadtxt(ptf_data + "/" + i + ".csv", delimiter = ',', skiprows = 1)


    #jd_all = np.append(pdata[:,0], jd)
    #mag_all = np.append(pdata[:,1], mag)
    #omag_all = np.append(pdata[:,2], omag)
    jd_all = jd
    mag_all = mag
    omag_all = omag

    np.savetxt("/Volumes/MeganSSD/KupferResearch/Data/ZTF_data/Up_to_Date_ZTF_10.27.23/LightcurveResults" + i + ".txt",np.c_[jd_all,mag_all,omag_all], fmt='%.5f %g %g')


    plt.plot(jd_all, mag_all, 'ko')
    plt.title(i)
    plt.xlabel('Days (MJD)')
    plt.ylabel('Magnitude')
    plt.savefig("/Volumes/MeganSSD/KupferResearch/Data/ZTF_data/Up_to_Date_ZTF_10.27.23/LightcurveResults" + i + ".png")
    plt.clf()
    plt.cla()
    
    #try:
        #PeriodFinder(jd_all, mag_all, omag_all, i)

    #except:
        #with open("results", "a") as results1:
            #results1.write(f"{i} failed\n")

