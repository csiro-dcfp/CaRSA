import xarray as xr
import numpy as np

def FFDI(precip, rh, tmax, wmax, time_dim='time'):
    """
        Returns the McArthur Forest Fire Danger Index following the formula provided in Dowdy (2018):
        FFDI = D ** 0.987 * exp (0.0338 * T - 0.0345 * H + 0.0234 * W + 0.243147)
        
        Parameters
        ----------
        precip : xarray DataArray
            Daily total precipitation [mm]. This is used to estimate the drought factor, D, as the 20-day accumulated rainfall
            scaled to lie between 0 and 10, with larger values indicating less precipitation.
        rh : xarray DataArray
            Daily max relative humidity at 2m [%] (or similar, depending on data availability). Richardson et al. (2021) uses mid-afternoon
            relative humidity at 2 m, Squire et al. (2021) uses daily mean relative humidity at 1000 hPa. This is used as H in the
            above equation.
        tmax : xarray DataArray
            Daily max 2 m temperature [deg C]. This is used as T in the above equation.
        wmax : xarray DataArray
            Daily max 10 m wind speed [km/h] (or similar, depending on data availability). Squire et al. (2021) uses daily mean wind 
            speed. This is used as W in the above equation.
        time_dim : str, optional
            Name of the time dimension
            
        Returns
        -------
        FFDI : xarray DataArray
            Array containing the FFDI.
            
        References
        ----------
        Dowdy, A. J. (2018). “Climatological Variability of Fire Weather in Australia”. Journal of Applied Meteorology and Climatology 57.2, pp. 221–234. issn: 1558-8424. doi: 10.1175/JAMC-D-17-0167.1.
    """
    
    p20 = precip.rolling({time_dim: 20}).sum()
    
    D = -10 * ((p20 - p20.min(time_dim)) / (p20.max(time_dim) - p20.min(time_dim))) + 10
    
    return D ** 0.987 * np.exp(0.0338 * tmax - 0.0345 * rh + 0.0234 * wmax + 0.243147)


def excess_heat_factor(temp, climatology_slice=None, time_dim='time'):
    """
        Calculates the Excess Heat Factor (EHF) following Nairn and Fawcett (2015):
        EHF = EHI_sig * max(1, EHI_accl),
        where the Excess Heat significance and acclimatisation indices, EHI_sig and EHI_accl, respectively, are defined as
        EHI_sig = [T_i + T_(i+1) + T_(i+2)] / 3 - T_95,
        EHI_accl = [T_i + T_(i+1) + T_(i+2)] / 3 - [T_(i-1) + ... + T_(i-30)] / 30.
        
        Parameters
        ----------
        temp : xarray DataArray
            Daily mean temperature [K or deg C]. This is T in the above equation.
        climatology_slice : slice, optional
            Climatological period used to calcluate the 95th percentile of temp. Should be in the format slice('first_year', 'last_year'). Default is None, which uses the full period of temp.
        time_name : string, optional
            Name of the time dimension.

        Returns
        -------
        EHF : xarray DataArray
            Array containing the EHF.
            
        References
        ----------
        Nairn, J.R.; Fawcett, R.J.B. The Excess Heat Factor: A Metric for Heatwave Intensity and Its Use in Classifying Heatwave Severity. Int. J. Environ. Res. Public Health 2015, 12, 227-253. https://doi.org/10.3390/ijerph120100227
    """      
    
    if climatology_slice is None: # Get heatwave threshold (T_95 in above equation)
        t_95 = temp.quantile(q=0.95, dim=time_dim)
    else:
        t_95 = temp.sel({time_dim: climatology_slice}).quantile(q=0.95, dim=time_dim)
    
    three_day_mean = temp.rolling({time_dim: 3}).sum() / 3
    thirty_day_mean = temp.rolling({time_dim: 30}).sum() / 30
    
    ehi_sig = three_day_mean - t_95
    
    ehi_accl = three_day_mean - thirty_day_mean.shift({time_dim: 3}) #  Shift thirty_day_mean forward 3 days so that last day of each averaging period has the same time
    
    ehi_accl_min_1 = xr.where(ehi_accl > 1, ehi_accl, 1) # Ensure minimum value is 1
    
    return ehi_sig * ehi_accl_min_1