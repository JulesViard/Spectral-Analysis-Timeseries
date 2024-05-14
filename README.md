# Spectral Analysis for Time Series

This Python module provides a suite of methods for performing spectral analysis (time consideration) on time series.

- Singular Spectrum Analysis (SSA with automatization thanks to agglomerative clustering) on time series data. The primary goal of these methods is to make a spectral decomposition of the time series and reconstruct the series from the grouped components, allowing for an analysis of its structure and underlying dynamics.

- Stationarity analysis aims to identify drift in time series distribution, based on weak stationarity. This allows to understand when important are taking place.



## Getting started

The A_SSA class can be used for a variety of time series analyses, including pattern recognition, anomaly detection, time series decomposition and forecasting. The user must specify the number of clusters, the method ('corr' or 'cov'), and the type of linkage ('single', 'complete', or 'average'), for the agglomerative clustering.

-----------------------------------------------------------
The AD_DD class can be used as part of a primary study to understand the behavior of a time series, or as pipeline processing (approximation of time series based on stationary behavior). The parameters allow to adjust the sensibilty of drifts detection, both on mean and variance aspects.




### Installing

    git clone ssh://git@git.xfel.eu:10022/viardj/spectral_analysis.git
    cd spectral_analysis
    python -mvenv my_environment
    source my_environment/bin/activate
    pip install -U pip
    pip install .



### Basic usage

On the following example, main methods (object, fit_transform, plot) are illustrated. Many examples are displayed in docs.

*Singular Spectral Analysis*

    ssa_obj = SSA_class(your_time_series_data)
    ssa_obj.fit_transform(50)
    ssa_obj_result = ssa_obj.time_reconstruction(4)
    ssa_obj.plot_reconstructed_TS(ssa_obj_result)
---
*Adaptative Window Drift Detection*

    addd_obj = AD_DD(your_time_series_data)
    addd_obj.fit_stationarity_detection()

## Authors

**Jules Viard**
