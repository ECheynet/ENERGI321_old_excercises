#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:10:07 2025

@author: ech022
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind Data Analysis Script
-------------------------
This script reads wind and temperature data, interpolates wind profiles at a given hub height,
analyzes wind direction and speed distributions using Weibull statistics, and investigates power law
and wind veer relationships. It also checks air-sea temperature differences.

Author: Etienne Cheynet
Created on: Fri Feb  3 14:39:24 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import scipy.stats as stats
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import calendar
from windrose import WindroseAxes

# =============================================================================
# Question 0: Read the Data and Prepare Time Vector
# =============================================================================

# Read the wind data file. The file is assumed to have 3 header rows.
data = pd.read_csv('NORA10_1957_2018_wind_and_sea.txt', header=3, sep='\s+')
data = np.array(data)  # Convert pandas DataFrame to a numpy array

# Number of records based on the second column (could be any column)
N = np.size(data[:, 1])

# Construct starting and ending datetime objects from the first four columns:
t1 = datetime(int(data[0, 0]), int(data[0, 1]), int(data[0, 2]), int(data[0, 3]))
t2 = datetime(int(data[-1, 0]), int(data[-1, 1]), int(data[-1, 2]), int(data[-1, 3]))

# Create a time vector with a time step of 3 hours. The extra timedelta ensures that t2 is included.
time = np.arange(t1, t2 + timedelta(hours=3), np.timedelta64(3, 'h'))

# Create a simple time axis (in hours) for interpolation and plotting.
dt = 3  # time step in hours
t = np.linspace(0, (N - 1) * dt, N)

# Display the data array for debugging
print(data)

# Split the array into mean wind speeds and wind direction components.
# Columns 8 to 12: mean wind speeds at different heights
oldU = data[:, 8:13]
# Columns 13 to 15: wind direction values at different heights
oldDir = data[:, 13:16]


# =============================================================================
# Question 1: Define Turbine and Height Vectors
# =============================================================================

# We use the DTU 10 MW wind turbine with a hub height of 119 m.
zHub = 119  # hub height in meters

# Original measurement heights for wind speed and direction.
oldZ = np.array([10, 50, 80, 100, 150])
oldZ_dir = np.array([10, 100, 150])

# New set of heights including the hub height. Sorted for interpolation.
newZ = np.sort(np.array([10, 50, 80, 100, zHub, 150]))

# Find the index in the newZ array corresponding to the hub height.
indZ = np.argmin(np.abs(newZ - zHub))


# =============================================================================
# Question 2: Interpolate Wind Speed and Wind Direction at Hub Height
# =============================================================================

# --- Interpolate Mean Wind Speed ---
# Use RectBivariateSpline for 2D spline interpolation over time (t) and height (oldZ).
speed_interpolator = RectBivariateSpline(t, oldZ, oldU)
# Interpolate on the new height vector newZ.
newU = speed_interpolator(t, newZ)
# Ensure no negative wind speeds.
newU[newU < 0] = 0

# --- Interpolate Wind Direction ---
# Since wind direction is circular, we first convert directions to vector components.
oldVx = np.cos(oldDir * np.pi / 180)  # Eastward component
oldVy = np.sin(oldDir * np.pi / 180)  # Northward component

# Create spline interpolators for the two components using linear interpolation (ky=1).
vx_interpolator = RectBivariateSpline(t, oldZ_dir, oldVx, ky=1)
newVx = vx_interpolator(t, newZ)

vy_interpolator = RectBivariateSpline(t, oldZ_dir, oldVy, ky=1)
newVy = vy_interpolator(t, newZ)

# Reconstruct wind direction from the interpolated vector components.
newDir = np.arctan2(newVy, newVx) * 180 / np.pi
# Adjust negative angles to lie between 0 and 360 degrees.
newDir[newDir < 0] += 360

# Quick plot to compare interpolated and original wind directions (for diagnostic purposes)
plt.figure()
plt.plot(newDir[:, indZ], oldDir[:, 1], 'r.', label='Interpolated vs Original')
plt.legend()

# Plot time series at hub height to check interpolation quality.
fig = plt.figure()
plt.plot(time, newDir[:, indZ], label='119 m')
plt.plot(time, oldDir[:, 1], label='100 m')
plt.xlim(100, 200)
plt.legend()
plt.title('Wind Direction Time Series')

fig = plt.figure()
plt.plot(time, newU[:, [indZ - 1, indZ, indZ + 1]])
plt.xlim(0, 100)
plt.legend(['100 m', '119 m', '150 m'])
plt.title('Wind Speed Time Series')


# =============================================================================
# Question 3: Wind Speed Distribution at Hub Height (Weibull Fit)
# =============================================================================

# Fit a 2-parameter Weibull distribution to the wind speed data at hub height.
params = stats.exponweib.fit(newU[:, indZ], loc=0, f0=1)
a = params[1]  # shape parameter
b = params[3]  # scale parameter

# Create a wind speed grid for plotting the PDF.
u = np.linspace(np.min(newU[:, indZ]), np.max(newU[:, indZ]), 100)
myPDF = weibull_min.pdf(u, a, scale=b)
print("Fitted Weibull parameters: a = {:.1f} and b = {:.1f}".format(a, b))

# Plot histogram of wind speeds and overlay the fitted Weibull PDF.
fig = plt.figure()
plt.hist(newU[:, indZ], bins=50, density=True, alpha=0.5, edgecolor='black',
         label='Histogram at hub height')
plt.plot(u, myPDF, linewidth=2, color='r', 
         label=f'a = {a:.1f} and b = {b:.1f}')
plt.xlim([0, 30])
plt.xlabel("u$_{hub}$ (m/s)", usetex=True)
plt.ylabel("PDF")
plt.legend()

# --- Wind Rose ---
# Filter out low wind speeds for the wind rose (only speeds > 5 m/s).
dummyU = newU[:, indZ]
D = newDir[:, indZ]
mask = dummyU > 5
D_filtered = D[mask]
dummyU_filtered = dummyU[mask]

ax = WindroseAxes.from_ax()
ax.bar(D_filtered, dummyU_filtered, nsector=30, opening=0.99,
       bins=np.arange(5, 25, 2), edgecolor='gray', lw=0.1, normed=True)
ax.set_theta_zero_location('W', offset=-180)
plt.legend()
plt.show()


# =============================================================================
# Question 4: Weibull Fit for Specific Wind Direction Sectors
# =============================================================================

# Define indices for four wind direction sectors.
d = newDir[:, indZ]
indDir = (
    tuple(np.where((d > 315) | (d < 45))[0]),
    tuple(np.where((d > 45) & (d < 135))[0]),
    tuple(np.where((d > 135) & (d < 225))[0]),
    tuple(np.where((d > 225) & (d < 315))[0])
)
sector_labels = ('315-45 deg', '45-135 deg', '135-225 deg', '225-315 deg')

# Plot histograms for each sector.
fig = plt.figure()
for ii in range(len(indDir)):
    u_sector = newU[indDir[ii], indZ].transpose()
    plt.hist(u_sector, bins=50, density=True, alpha=0.5, edgecolor='black',
             label=sector_labels[ii])
plt.xlim([0, 30])
plt.xlabel("u$_{hub}$ (m/s)", usetex=True)
plt.ylabel("PDF")
plt.legend()

# Fit and plot Weibull PDFs for each sector.
fig = plt.figure()
for ii in range(len(indDir)):
    u_sector = newU[indDir[ii], indZ].transpose()
    params = stats.exponweib.fit(u_sector, loc=0, f0=1)
    a_sector = params[1]
    b_sector = params[3]
    u_pdf = np.linspace(np.min(u_sector), np.max(u_sector), 50)
    pdf_vals = weibull_min.pdf(u_pdf, a_sector, scale=b_sector)
    print("Sector {}: a = {:.1f} and b = {:.1f}".format(sector_labels[ii], a_sector, b_sector))
    label_str = f'a = {a_sector:.1f} and b = {b_sector:.1f} ({sector_labels[ii]})'
    plt.plot(u_pdf, pdf_vals, linewidth=2, label=label_str)
plt.xlim([0, 30])
plt.xlabel("u$_{hub}$ (m/s)", usetex=True)
plt.ylabel("PDF")
plt.legend()


# =============================================================================
# Question 5: Rolling Statistics (Monthly Mean, Median, Max, and Std)
# =============================================================================

# Compute rolling statistics over a window corresponding to 720 hours.
window_size = int(np.round(720 / dt))
movMean = pd.Series(newU[:, indZ]).rolling(window_size).mean()
movMedian = pd.Series(newU[:, indZ]).rolling(window_size).median()
movMax = pd.Series(newU[:, indZ]).rolling(window_size).max()
movStd = pd.Series(newU[:, indZ]).rolling(window_size).std()

# Plot rolling statistics with linear trend fits.
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Rolling Mean
axs[0].plot(t, movMean, label='Rolling Mean')
# Fit linear trend on non-NaN values.
m, b = np.polyfit(t[~np.isnan(movMean)], movMean[~np.isnan(movMean)], deg=1)
trend_line = np.poly1d([m, b])
axs[0].plot(t[[0, -1]], trend_line(t[[0, -1]]), color='k', label=f'$y = {m:.2f}x {b:+.1f}$')
axs[0].legend()
axs[0].set_title('Rolling Monthly Mean')

# Rolling Median
axs[1].plot(t, movMedian, 'tab:orange', label='Rolling Median')
m, b = np.polyfit(t[~np.isnan(movMedian)], movMedian[~np.isnan(movMedian)], deg=1)
trend_line = np.poly1d([m, b])
axs[1].plot(t[[0, -1]], trend_line(t[[0, -1]]), color='k', label=f'$y = {m:.2f}x {b:+.1f}$')
axs[1].legend()
axs[1].set_title('Rolling Monthly Median')

# Rolling Max
axs[2].plot(t, movMax, 'tab:green', label='Rolling Max')
m, b = np.polyfit(t[~np.isnan(movMax)], movMax[~np.isnan(movMax)], deg=1)
trend_line = np.poly1d([m, b])
axs[2].plot(t[[0, -1]], trend_line(t[[0, -1]]), color='k', label=f'$y = {m:.2f}x {b:+.1f}$')
axs[2].legend()
axs[2].set_title('Rolling Monthly Max')

# Rolling Standard Deviation
axs[3].plot(t, movStd, 'tab:red', label='Rolling Std')
m, b = np.polyfit(t[~np.isnan(movStd)], movStd[~np.isnan(movStd)], deg=1)
trend_line = np.poly1d([m, b])
axs[3].plot(t[[0, -1]], trend_line(t[[0, -1]]), color='k', label=f'$y = {m:.2f}x {b:+.1f}$')
axs[3].legend()
axs[3].set_xlabel('Time (hours)')
axs[3].set_ylabel('Std (m/s)')
axs[3].set_title('Rolling Monthly Std')

plt.tight_layout()


# =============================================================================
# Question 6: Monthly Wind Speed Histograms
# =============================================================================

# Create a pandas DatetimeIndex from the time vector.
pd_time = pd.DatetimeIndex(time)

fig = plt.figure()
# Loop over all 12 months.
for month in range(1, 13):
    # Find indices corresponding to the current month.
    ind = np.where(pd_time.month == month)
    # Extract wind speeds at hub height for those indices.
    u_month = newU[ind, indZ].transpose()
    month_label = calendar.month_name[month]
    plt.hist(u_month, bins=50, density=True, alpha=0.5, edgecolor='black', label=month_label)
plt.xlim([0, 30])
plt.xlabel("u$_{hub}$ (m/s)", usetex=True)
plt.ylabel("PDF")
plt.legend()


# =============================================================================
# Question 7: Monthly Weibull PDF Fits
# =============================================================================

cmap = plt.cm.hsv
fig = plt.figure()
for month in range(1, 13):
    color = cmap(month / 12)
    ind = np.where(pd_time.month == month)
    u_month = newU[ind, indZ].transpose()
    params = stats.exponweib.fit(u_month, loc=0, f0=1)
    a_month = params[1]
    b_month = params[3]
    u_pdf = np.linspace(np.min(u_month), np.max(u_month), 50)
    pdf_vals = weibull_min.pdf(u_pdf, a_month, scale=b_month)
    print("Month {}: a = {:.1f} and b = {:.1f}".format(calendar.month_name[month], a_month, b_month))
    label_str = f'a = {a_month:.1f} and b = {b_month:.1f} ({calendar.month_name[month]})'
    plt.plot(u_pdf, pdf_vals, color=color, linewidth=2, label=label_str)
plt.xlim([0, 30])
plt.xlabel("u$_{hub}$ (m/s)", usetex=True)
plt.ylabel("PDF")
plt.legend()


# =============================================================================
# Question 8: Power Law and Wind Veer Analysis
# =============================================================================

# Define a list of datetimes for which to check the power law and wind veer.
dates_to_check = pd.to_datetime(['2016-01-23 00:00', '2016-04-11 21:00',
                                   '2016-07-01 12:00', '2015-11-25 11:00'])

def func(zr, a):
    """
    Power law function for curve fitting.
    
    Parameters:
        zr : array-like
            Non-dimensional height (z/zHub).
        a : float
            Exponent in the power law.
    
    Returns:
        array-like: zr raised to the power a.
    """
    return zr ** a

def powerLaw(zHub, uHub, z, a):
    """
    Computes the wind speed at height z using a power law profile.
    
    Parameters:
        zHub : float
            Hub height.
        uHub : float
            Wind speed at the hub height.
        z : array-like
            Heights at which to compute the wind speed.
        a : float
            Power law exponent.
    
    Returns:
        array-like: Wind speed at height z.
    """
    return uHub * (z / zHub) ** a

# Plot power law fits for each selected datetime.
fig = plt.figure()
for ii, d_val in enumerate(dates_to_check):
    color = cmap(ii / len(dates_to_check))
    # Find the index in pd_time closest to the current date.
    indTime = np.argmin(np.abs(pd_time - d_val))
    print(f"Data time: {pd_time[indTime]}")
    # Plot measured wind speeds vs. height.
    plt.plot(newU[indTime, :], newZ, 'o', color=color)
    
    # Normalize wind speeds and heights.
    ur = newU[indTime, :] / newU[indTime, indZ]
    zr = newZ / zHub
    # Fit the power law exponent using curve_fit with bounds between 0 and 1.
    alpha, _ = curve_fit(func, zr, ur, bounds=(0, 1))
    label_str = f'a = {float(alpha):.2f} ({str(d_val)})'
    
    # Generate a range of heights and plot the power law curve.
    z1 = np.logspace(np.log10(newZ[0]), np.log10(newZ[-1]), 20)
    plt.plot(powerLaw(zHub, newU[indTime, indZ], z1, alpha), z1, color=color, label=label_str)
plt.grid()
plt.xlabel("Mean wind speed (m/s)")
plt.ylabel("Height, z (m)")
plt.legend()

# Plot wind veer (difference in wind direction) with height.
fig = plt.figure()
for ii, d_val in enumerate(dates_to_check):
    color = cmap(ii / len(dates_to_check))
    indTime = np.argmin(np.abs(pd_time - d_val))
    print(f"Data time: {pd_time[indTime]}")
    label_str = str(d_val)
    # Plot the difference between wind direction at each height and the lowest level.
    plt.plot(newDir[indTime, :] - newDir[indTime, 0], newZ, 'o-', color=color, label=label_str)
plt.grid()
plt.xlabel("Wind veering (deg)")
plt.ylabel("Height, z (m)")
plt.legend()

# Note: The power coefficient in this dataset appears lower than typical offshore values (≈ 0.11).


# =============================================================================
# Question 9: Air-Sea Temperature Difference Analysis
# =============================================================================

# Read the temperature data file. Assumes 2 header rows.
temp_data = pd.read_csv('NORA10_1957_2018_SST_RH_T.txt', header=2, sep='\s+')

# Extract temperature columns.
T100 = temp_data["T100"]   # Air temperature at 100 m (assumed)
SST = temp_data["SST"]     # Sea Surface Temperature

# Convert the temperature data to a numpy array.
temp_array = np.array(temp_data)
N_temp = np.size(temp_array[:, 1])
# Create the time vector for temperature data.
t1_temp = datetime(int(temp_array[0, 0]), int(temp_array[0, 1]), int(temp_array[0, 2]), int(temp_array[0, 3]))
t2_temp = datetime(int(temp_array[-1, 0]), int(temp_array[-1, 1]), int(temp_array[-1, 2]), int(temp_array[-1, 3]))
time_temp = np.arange(t1_temp, t2_temp + timedelta(hours=3), np.timedelta64(3, 'h'))
pd_time_temp = pd.DatetimeIndex(time_temp)

# Analyze air-sea temperature difference at selected dates.
for d_val in dates_to_check:
    indTime = np.argmin(np.abs(pd_time_temp - d_val))
    dTemp = T100[indTime] - SST[indTime]
    label_str = f"On {pd_time_temp[indTime]}, ΔT = {float(dTemp):.2f} °C"
    if dTemp > 0:
        print("The atmosphere is statically stable")
    else:
        print("The atmosphere is statically unstable")
    print(label_str)

# Note:
#   - A positive air-sea temperature difference indicates warm air above cold water (stable atmosphere).
#   - A negative difference indicates cold air above warm water (unstable atmosphere).
