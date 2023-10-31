# Red-Supergiants-In-M31
A census of red supergiants in M31 

PeriodFitting.py uses gaussian processes to find the best period fit in days for red supergiants that exhibit periodic and non-periodic
variability. Sigma clipping is also included in this code. 

lightcurve.py combines PTF and ZTF lightcurve data and uses gaussian processes to period fit the combined data. 

ZTFlightcurvecolor.py generates all ZTF lightcurves at once (automated), and saves the PNG's as well as the .txt file containing the resulting lightsurve data (header: jd, mag, mag uncertainty).
