import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from river import drift
import pwlf


class AD_DD(object):

    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries):
        """
        Initialize an AD_DD (Adaptative Window Drift Detection) object with a given time series.

        :param tseries: The initial time series that this object will handle. This could be a Pandas Series, NumPy array or list.
        :type tseries: pandas.Series, numpy.array, or list
        :raises TypeError: If the provided time series is not a supported type (Pandas Series, NumPy array, or list).
        :raises ValueError: If the provided time series has length 1. The time series must have length > 1.
        :raises ValueError: If the provided time series has any NaN values. Missing values are not supported.
        :raises ValueError: If the provided time series has non-numerical values. The time series must only contain numerical values.
        :raises ValueError: If the provided time series has infinite values. Infinite values are not supported.
        """
        
        # Type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        self.orig_TS = pd.Series(tseries)   # Original series

        # Raise an error if the time series has length 1
        if len(self.orig_TS) == 1:
            raise ValueError("The time series must have length > 1.")
        
        # Raise an error if the time series has any NaN values
        if self.orig_TS.isnull().values.any():
            raise ValueError("Missing values are not supported.")
        
        # Raise an error if the time series has non-numerical values
        if not np.issubdtype(self.orig_TS.dtype, np.number):
            raise ValueError("The time series must only contain numerical values.")
        
        # Raise an error if the time series has infinite values
        if not np.isfinite(self.orig_TS).all():
            raise ValueError("Infinite values are not supported.")



    def ADWIN_app(self, time_series, param_delta = 0.002):
        """
        Conducts a stationary analysis of a given time series using the ADWIN (Adaptive Windowing) method, from River library, for change 
        point detection. This function computes the mean of the time series data to detect drifts and tracks the variance
        following the ADWIN process. 

        :param time_series: The input time series data.
        :type time_series: list, array
        :param param_delta: The desired false positive rate for drift detection. Default value is 0.002.
        :type param_delta: float
        :returns: A tuple containing two lists: 
                1. A list of indices where drifts were detected in the time series.
                2. A list tracking the variance following the ADWIN process.
        :rtype: tuple (list, list)
        """
        # np array conversion :
        ts_array = np.array(time_series)

        #Drift detecttion by mean :
        drift_detector = drift.ADWIN(delta=param_delta)
        drifts = [0]     #list of index drift

        #Variance following ADWIM process :
        var_detector = []

        for i, val in enumerate(ts_array):
            drift_detector.update(val)          # Data is processed one sample at a time
            var_detector.append(drift_detector.variance)
            if drift_detector.drift_detected:
                # The drift detector indicates after each sample if there is a drift in the data
                drifts.append(i)

        return drifts, var_detector

    

    def plot_ADWIN_app(self, drifts):
        """
        Generates a plot of the original time series data along with vertical lines at the points where drifts 
        were detected by the ADWIN algorithm. This visualization helps in understanding the results of the ADWIN
        change point detection method.

        :param drifts: A list of indices where drifts were detected in the time series.
        :type drifts: list
        :returns: None. Displays a plot.
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.grid()
        ax.plot(self.orig_TS, label="timeseries")
        if drifts is not None:
            for drift_detected in drifts:
                ax.axvline(drift_detected, color='red')
                
        ax.set_title("ADWIN Algorithm Results for Mean shift")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()



    def plot_variance_stream(self, var_data, x1=0, x2=0):
        """
        Generates a plot for the variance stream computed over the time series data. 
        The x-axis range can be specified.

        :param var_data: A list or array-like object containing variance data to be plotted.
        :type var_data: list, array
        :param x1: The starting point of the x-axis. Default is 0.
        :type x1: int
        :param x2: The ending point of the x-axis. Default is the length of `var_data`.
        :type x2: int
        :returns: None. Displays a plot.
        """
        if x2==0:
            x2=len(var_data)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.grid()
        ax.plot(var_data, label="Variance")
        ax.set_title("Variance stream")
        ax.set_xlabel("Time")
        ax.set_xlim(x1,x2)
        plt.show()



    def __variance_shift_detection(self ,mean_shift_list ,var_detection_list):     #var_detection_list is the data stream from adwin Algorithm
        """
        Performs variance shift detection in the time series data. This method uses a list of points 
        where mean shifts were detected and a variance detection list from the ADWIN algorithm. The function 
        segments the variance detection list based on the indices where mean shifts occurred. Overall, this
        method segments the variance detection list into sub-lists, each representing a window of variance data.

        :param mean_shift_list: A list of indices where mean shifts were detected in the time series.
        :type mean_shift_list: list
        :param var_detection_list: A list or array-like object containing variance data from the ADWIN algorithm.
        :type var_detection_list: list, array
        :returns: A list of lists, each inner list represents a segmented window of variance data.
        :rtype: list of lists
        """
        #Variance is set to zero one index after a shift mean is detected   (with .variance attribute of ADWIN)
        mean_L = [0] #index modification for futher variance segmentation according to precedent statement
        for i in range(1, len(mean_shift_list)):
            mean_L.append(mean_shift_list[i] +1)

        segmentation_result= []
        for i in range(1,len(mean_shift_list)):
            current_window = var_detection_list[mean_L[i-1]:mean_L[i]-1]
            segmentation_result.append(current_window)
        final_window = var_detection_list[mean_L[-1]:]
        segmentation_result.append(final_window)      #Final window
        
        return segmentation_result      #contains each subwindow associate with variance analysis



    #Iterative optmisation for number of breaks :
    def __linear_decompIT_var(self, var_segmentation_list, nb_max_breaks, threshold_piecewise):   #NNumber of break can be increased depending on data
        """
        Applies a piecewise linear fit on each segment of variance data using an iterative approach. 
        It stops adding breakpoints when the percentage change between the sum of squared residuals (ssr) 
        for current and previous fits falls below the defined threshold or when the maximum allowed 
        number of breakpoints is reached.

        :param var_segmentation_list: A list of lists, each containing a segment of variance data (variance data with transformation based on entropy).
        :type var_segmentation_list: list of lists
        :param nb_max_breaks: The maximum number of breakpoints to consider during the piecewise linear fit.
        :type nb_max_breaks: int
        :param threshold_piecewise: The threshold percentage change in ssr between two fits, to decide whether to stop adding breakpoints.
        :type threshold_piecewise: float
        :returns: A list of arrays, each array contains the breakpoints for a specific variance window.
        :rtype: list of arrays
        """
        
        breakpoints_list = []   #list of arrays, each array contains break point for a variance window

        for current_list in var_segmentation_list:
            x = np.linspace(0,len(current_list)-1,len(current_list))
            y = np.array(current_list)

            my_pwlf = pwlf.PiecewiseLinFit(x, y)
            ssr_old = float('inf')
            fit_breaks_history = [0]

            for i in range(1,nb_max_breaks):

                my_pwlf.fitfast(i)
                ssr_new = my_pwlf.ssr

                s = pd.Series([ssr_old, ssr_new])
                relative_change = abs(s.pct_change().iloc[-1])

                ssr_old = ssr_new
                fit_breaks_history.append(my_pwlf.fit_breaks[1:-1])

                if relative_change <= threshold_piecewise:       #Threshold of 65% by default : if the percentage of change between two ssr is inferior of 65% then stop adding
                    i = i-1
                    if i==1:
                        breakpoints_list.append(np.array([-1]))     #negative index equivalent to no breakpoint in a window variance
                    else:
                        breakpoints_list.append(fit_breaks_history[i])
                    break

                if i == nb_max_breaks-1:        #in case more shift variance than the maximum set, then modelisation with the number_max, which represents an approximation
                        breakpoints_list.append(my_pwlf.fit_breaks[1:-1])
        return breakpoints_list



    def __reconstruction_shift_var(self, break_list, mean_shift_list):
        """
        Reconstructs the global indices of variance shifts in the time series data. It uses the local 
        breakpoints within each variance window and adjusts them based on the global mean shift indices.
        Overall, this method returns a list of indices representing the global position of variance shifts

        :param break_list: A list of arrays, each array contains the breakpoints for a specific variance window.
        :type break_list: list of arrays
        :param mean_shift_list: A list of indices where mean shifts were detected in the time series.
        :type mean_shift_list: list
        :returns: A list of indices representing the global position of variance shifts in the time series.
        :rtype: list
        """

        result = []

        if len(break_list)==0:
            print("No breakpoints from variance")
            return result

        for i in range(0,len(mean_shift_list)):
            if break_list[i][0] != (-1):
                for j in break_list[i]:
                    current = j+mean_shift_list[i]        #index take in account precedent window
                    result.append(current)   #list of index of variance shift in a global index time series
        return result



    def plot_linear_decompIT_var(var_window, nb_seg):      #var_window is the variance stream in a specific window
        """
        Plots a piecewise linear fit for the given variance stream in a specific window. The function fits 
        a specified number of segments to the variance data and plots the actual data along with the fitted line.

        :param var_window: An array or list containing variance data for a specific window.
        :type var_window: list, array
        :param nb_seg: The number of segments to fit to the variance data.
        :type nb_seg: int
        :returns: The sum of squared residuals (ssr) for the fitted model.
        :rtype: float
        """
        # plot the results
        x_ref = np.linspace(0,len(var_window)-1,len(var_window))
        y_ref = np.array(var_window)
        my_pwlf = pwlf.PiecewiseLinFit(x_ref, y_ref)
        my_pwlf.fitfast(nb_seg)
        yHat = my_pwlf.predict(x_ref)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.grid()    
        ax.plot(x_ref, y_ref, 'o')
        ax.plot(x_ref, yHat, '-')
        ax.set_title('Piecewise Linear')
        # plot the results
        plt.show()
        return my_pwlf.ssr
    


    def __op_difference(self, time_series, lag=1):
        """
        Performs a difference operation on the time series data with a specified lag. 
        It replaces any resulting NaN values with the mean of the difference series.

        :param time_series: A list or Series containing the time series data.
        :type time_series: list, pd.Series
        :param lag: The lag to apply while differencing. Default is 1.
        :type lag: int
        :returns: The differenced time series.
        :rtype: pd.Series
        """

        time_series_diff = time_series.copy()
        time_series_diff = time_series_diff.diff(lag)
        time_series_diff.fillna(time_series_diff.mean(), inplace=True)
        return time_series_diff
    

    #Normalization of panda series between 0 and 1
    def __scaler_series(self, TS_to_scale):
        """
        Scales a time series using the MinMaxScaler, transforming values to the range [0, 1]. 
        The function reshapes the data as necessary for the scaler, then converts the scaled data 
        back to a pandas Series before returning it.

        :param TS_to_scale: A pandas Series containing the time series data to be scaled.
        :type TS_to_scale: pd.Series
        :returns: The scaled time series.
        :rtype: pd.Series
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(TS_to_scale.values.reshape(-1, 1))

        # Conversion np tab in pandas series
        scaled_series = pd.Series(scaled_data.flatten())
        return scaled_series


    #Standart function :
    def __standardize(self, x):
        """
        Standardizes the input data (x) by subtracting the mean and dividing by the standard deviation.
        The function converts the input into a numpy array before performing the operations.

        :param x: The data to be standardized. Could be a list or a pandas Series.
        :type x: list, pd.Series
        :returns: The standardized version of the input data.
        :rtype: np.array
        """
        x_array = np.array(x)
        mean_x = np.mean(x_array)
        std_x = np.std(x_array)
        standardized_x = (x_array - mean_x) / std_x
        return standardized_x


    # Define the G(u) function
    def __G(self, u):
        """
        A private method that implements a specific mathematical function used to compute the negentropy of a given input.
        It takes a number (u) as input and returns the negative exponential of half of the square of that number.

        :param u: The input to the function.
        :type u: float
        :returns: The result of the function applied to the input.
        :rtype: float
        """
        return -np.exp(-(u ** 2) / 2)


    def __negentropy_series(self, time_series):
        """
        Computes an approximation of the negentropy of a time series. This function first standardizes 
        the input time series, then computes the expectation of the G function (as implemented in __G) 
        applied to both the time series and a standard Gaussian random variable. It finally computes 
        the square of the difference between these two expected values, which serves as the negentropy approximation.

        :param time_series: The time series data for which to compute the negentropy approximation.
        :type time_series: pd.Series
        :returns: The negentropy approximation.
        :rtype: float
        """
        # Gaussian variable with zero mean and unit variance
        num_samples = len(time_series)
        v = np.random.normal(0, 1, num_samples)  # Gaussian variable with zero mean and unit variance

        TS_to_test = self.__standardize(time_series)

        # Calculate the expected values of G(y) and G(v)
        E_G_y = np.mean(self.__G(TS_to_test))
        E_G_v = np.mean(self.__G(v))

        # Compute the negentropy approximation
        J_y = (E_G_y - E_G_v) ** 2

        #print("Negentropy approximation:", J_y)
        return J_y



    def __determine_optimal_differencing_negentropy(self, time_series, threshold_negentropy):
        """
        Determines the optimal order of differencing (d) for a given time series based on a negentropy threshold.
        The function computes the negentropy for each order of differencing up to a maximum of 5, or until 
        the negentropy drops below the provided threshold.

        :param time_series: The time series data for which to determine the optimal differencing order.
        :type time_series: pd.Series
        :param threshold_negentropy: The negentropy threshold below which differencing order is considered as optimal.
        :type threshold_negentropy: float
        :returns: The optimal differencing order.
        :rtype: int
        """
        d = 0
        max_negentropy = -np.inf
        current_negentropy = self.__negentropy_series(time_series)
        while current_negentropy > threshold_negentropy:   # Can be a parameter to change, depending on the data and behvaior 
            max_negentropy = current_negentropy
            time_series = time_series.diff().dropna()
            d += 1
            if d==5:    #Maximum differentiation set to 5
                return d-1

            current_negentropy = self.__negentropy_series(time_series)

        # Subtract 1 since the loop will increment d one extra time
        d = d-1
        return d



    def __var_stream_by_entropy(self, time_series, mean_shift_list, threshold_negentropy):        #Differentiation is per window
        """
        Processes a time series to generate a new variance list using entropy-based methods. The time series
        is first divided into segments based on a list of mean shift indices. For each segment, the function
        determines the optimal order of differencing based on a negentropy threshold, applies the differencing,
        scales the result, and then applies the ADWIN algorithm. The variance stream outputs from ADWIN for each
        segment are combined to form the final output.

        :param time_series: The time series data to process.
        :type time_series: pd.Series
        :param mean_shift_list: List of indices where mean shifts occur, dividing the time series into segments.
        :type mean_shift_list: list of int
        :param threshold_negentropy: The negentropy threshold for determining the optimal order of differencing.
        :type threshold_negentropy: float
        :returns: The variance list generated by processing each segment of the time series, using differentiation based on entropy.
        :rtype: list of float
        """
        new_var_list_entropy = []

        if len(mean_shift_list)==1:             #Case if there isn't any mean shift
            d = self.__determine_optimal_differencing_negentropy(time_series, threshold_negentropy)
            time_series = self.__op_difference(time_series,d)
            time_series = self.__scaler_series(time_series)
            _, current_var_stream = self.ADWIN_app(time_series)
            return current_var_stream

        for i in range(0, len(mean_shift_list)-1):  #Case where there is at least one mean shift, so len(mean_shift_list)=2
            current_window = time_series[mean_shift_list[i]:mean_shift_list[i+1]]
            d = self.__determine_optimal_differencing_negentropy(current_window, threshold_negentropy)
            current_window = self.__op_difference(current_window,d)
            current_window = self.__scaler_series(current_window)
            no_use, current_var_stream = self.ADWIN_app(current_window)
            new_var_list_entropy = new_var_list_entropy+current_var_stream

        window_last = time_series[mean_shift_list[-1]:]
        d_last = self.__determine_optimal_differencing_negentropy(window_last, threshold_negentropy)
        window_last = self.__op_difference(window_last,d_last)
        window_last = self.__scaler_series(window_last)
        _, last_var_stream = self.ADWIN_app(window_last)
        new_var_list_entropy = new_var_list_entropy+last_var_stream

        return new_var_list_entropy




    def fit_stationarity_detection(self, delta_adwin=0.002, threshold_negentropy=10e-4, threshold_piecewise=0.65, max_breaks_piecewise=10, plot_bool=False, x1=0, x2=0):
        """
        Conducts a stationarity analysis on the time series by detecting both mean and variance shifts. The function
        applies the ADWIN algorithm to detect mean shifts and estimates a variance stream. This variance stream is
        further processed with an entropy-based method and segmented based on the detected mean shifts. Each variance 
        segment is then fitted using a piecewise linear method to detect variance shifts. 

        :param delta_adwin: The delta parameter for the ADWIN algorithm, default is 0.002.
        :type delta_adwin: float
        :param threshold_negentropy: The negentropy threshold for determining the optimal order of differencing, default is 10e-4.
        :type threshold_negentropy: float
        :param threshold_piecewise: The threshold for the relative change in the sum of squared residuals in the piecewise linear fitting, default is 0.65.
        :type threshold_piecewise: float
        :param max_breaks_piecewise: The maximum number of breaks in the piecewise linear fitting, default is 10.
        :type max_breaks_piecewise: int
        :param plot_bool: If True, plots the results of the stationarity detection.
        :type plot_bool: bool
        :param x1: Start index for the plot, default is 0.
        :type x1: int
        :param x2: End index for the plot, default is the length of the time series.
        :type x2: int
        :returns: A tuple containing a list of indices for mean shifts, a list of indices for variance shifts, the original variance list, and the entropy-based variance list.
        :rtype: tuple
        """
        if x2==0:
            x2=len(self.orig_TS)

        TS_to_analyse = self.orig_TS.copy()

        drift_mean_index, variance_list = self.ADWIN_app(TS_to_analyse, param_delta=delta_adwin)
        print("ADWIN done")
        variance_list_entropy = self.__var_stream_by_entropy(TS_to_analyse, drift_mean_index, threshold_negentropy)     #variance_list_entropy is the variance stream after optimal differentiation per window
        print("Variance by entropy done")
        variance_segmentation_list = self.__variance_shift_detection(drift_mean_index, variance_list_entropy)
        print("Variance Shift Detection done")
        variance_breakpoints_list = self.__linear_decompIT_var(variance_segmentation_list, max_breaks_piecewise, threshold_piecewise)
        print("Linear Piecewise Done")
        drift_variance_index = self.__reconstruction_shift_var(variance_breakpoints_list, drift_mean_index)
        print("Reconstruction done")

        self.comb_list = drift_mean_index + drift_variance_index
        self.comb_list = sorted(self.comb_list)        
        self.comb_list = [int(x) for x in self.comb_list]

        if plot_bool==True:
            self.plot_stationarity_detection(drift_mean_index, drift_variance_index, x1, x2)

        return drift_mean_index, drift_variance_index, variance_list, variance_list_entropy
    


    def plot_stationarity_detection(self, mean_index, var_index, x1=0, x2=0, bool_mean_shift=True, bool_var_shift=True):
        """
        Plots the time series with the detected mean and variance shifts. The shifts are indicated by vertical lines. 
        The function allows for customization of the range of the x-axis and whether to plot mean or variance shifts.

        :param mean_index: List of indices where mean shifts were detected.
        :type mean_index: list
        :param var_index: List of indices where variance shifts were detected.
        :type var_index: list
        :param x1: Start index for the x-axis, default is 0.
        :type x1: int
        :param x2: End index for the x-axis, default is the length of the time series.
        :type x2: int
        :param bool_mean_shift: If True, plot the mean shifts, default is True.
        :type bool_mean_shift: bool
        :param bool_var_shift: If True, plot the variance shifts, default is True.
        :type bool_var_shift: bool
        """
        if x2==0:
            x2=len(self.orig_TS)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.grid()
        ax.plot(self.orig_TS , label="Time Series")
        if bool_mean_shift==True:
            for i in mean_index:
                ax.axvline(x=i, color='r', label="mean shift" if i == mean_index[0] else None)
        if bool_var_shift==True:
            for i in var_index:
                ax.axvline(x=i, color='g', label="variance shift" if i == var_index[0] else None)
                
        ax.set_title("Stationarity Algorithm")
        ax.set_xlabel("Time")
        ax.set_xlim(x1,x2)
        ax.legend(loc='upper left')
        plt.show()



    def get_average_window(self, plot_bool=False):
        """
        Calculates the average of each detected stationary window in the time series and replaces the values in each 
        window with their respective mean. It can optionally plot the resulting series where the values in each stationary
        window have been replaced by their mean.

        :param plot_bool: If True, the averaged time series is plotted, default is False.
        :type plot_bool: bool
        """
        self.TS_average = self.orig_TS.copy()

        for i in range(0,len(self.comb_list)-1):
            mean_window = self.TS_average[self.comb_list[i]:self.comb_list[i+1]].mean()
            self.TS_average[self.comb_list[i]:self.comb_list[i+1]] = mean_window
        lastmean = self.TS_average[self.comb_list[-1]:].mean()
        self.TS_average[self.comb_list[-1]:] = lastmean

        if plot_bool==True:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.grid()
            ax.plot(self.TS_average, label="Time Series")
            ax.set_title("Average per stationary window")
            ax.set_xlabel("Time")
            plt.show()