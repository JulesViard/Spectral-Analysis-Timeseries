import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import moment
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform


class A_SSA(object):

    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries):
        """
        Initialize an A_SSA object with a given time series.

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
        

        
        
    def fit_transform(self, L, save_mem=True):
        """
        This function performs Singular Spectrum Analysis (SSA) on the original time series. Assumes the values of the time series are
    	recorded at equal intervals. It then calculates the w-correlation matrix.

        :param L: The window length for SSA. Must be in the interval [2, N/2], where N is the length of the original time series.
        :type L: int
        :param save_mem: If True, the method does not store the elementary matrices and the V matrix in order to save memory. 
                     To keep these matrices, set this parameter to False.
        :type save_mem: bool, optional
        :raises ValueError: If L is not in the interval [2, N/2].
        """
        
        # Value-checking for the window length
        self.N = len(self.orig_TS)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        

        # Definition of window length and number of windows
        self.L = L
        self.K = self.N - self.L + 1

        # First Stage : 
        # - Embed the time series in a trajectory matrix (Hankel matrix)
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T

        # - Decompose the trajectory matrix with SVD
        self.U, self.Sigma, VT = np.linalg.svd(self.X)  # SVD routine returns V^T, not V, so I'll tranpose it back

        self.d = np.linalg.matrix_rank(self.X)  # The intrinsic dimensionality of the trajectory space.
        
        self.TS_comps = np.zeros((self.N, self.d))


        # Second Stage : Reconstruction
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]

            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
            
        # Third Stage : w-correlation matrix
        #self.get_wcorr()
        self.Wcorr = None
        self.Wcov = None
        



    def components_to_df(self, n=0):
        """
        Returns the time series components as a pandas DataFrame.

        :param n: The number of components to include in the DataFrame. If n is 0 or greater than the number of components, all components are included.
        :type n: int, optional
        :returns: A DataFrame where each column represents a time series component. The columns are named 'F0', 'F1', 'F2', etc. The index of the DataFrame is the same as the index of the original time series.
        :rtype: pandas.DataFrame
        """

        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
    



    def reconstruct(self, indices):
        """
        Reconstructs the time series based on selected elementary components, using given indices.

        :param indices: The indices of the components to use for the reconstruction. This can be an int or a list of ints.
        :type indices: int or list of ints
        :returns: A reconstructed time series as a pandas Series. The index of the series is the same as the index of the original time series.
        :rtype: pandas.Series
        """

        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    



    def rank_component_var(self, nb_component):
        """
        Ranks the components based on the variance of the reconstructed time series for each component.

        :param nb_component: The number of components to include in the ranking.
        :type nb_component: int
        :returns: A DataFrame where the index is the component number and the 'Variance' column contains the variance of the reconstructed time series for each component. The DataFrame is sorted in descending order by variance.
        :rtype: pandas.DataFrame
        """

        if nb_component > self.d:   # If the number of components is greater than the intrinsic dimensionality of the trajectory space
            nb_component = self.d

        list_var = []
        for i in range(nb_component):
            list_var.append([i, self.reconstruct(i).var()])

        list_var = sorted(list_var, key=lambda x: x[1], reverse=True)
        # Convert the list to a Pandas DataFrame
        df_var = pd.DataFrame(list_var, columns=['Component', 'Variance'])
        df_var = df_var.set_index('Component')

        return df_var




    def get_wcorr(self):
        """
        Calculates and returns the w-correlation matrix for the time series. The function computes the weights, calculates weighted norms, computes weighted inner products, and then calculates the w-correlation matrix.

        :returns: A w-correlation matrix for the time series. If the time series has only one component, the returned w-correlation matrix will be of size (d, d) filled with zeros, where d is the intrinsic dimensionality of the trajectory space.
        :rtype: numpy.ndarray
        """
        if self.d > 1:
            # Calculate the weights
            w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

            # Reshape w for broadcasting
            w = w[:, np.newaxis]
            
            # Calculate weighted norms, ||F_i||_w, then invert.
            F_wnorms = np.sqrt((w * self.TS_comps ** 2).sum(axis=0))
            F_wnorms_inv = 1 / F_wnorms
            # Compute weighted inner products
            weighted_inner_products = (w*self.TS_comps).T.dot(self.TS_comps)

            # Calculate Wcorr
            self.Wcorr = np.abs(weighted_inner_products * np.outer(F_wnorms_inv, F_wnorms_inv))

            # Ensure the diagonal elements are exactly 1
            np.fill_diagonal(self.Wcorr, 1)

        else :
            self.Wcorr = np.zeros((self.d, self.d)) # The time series has only one component

        return self.Wcorr
        



    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.

        :param min: The minimum component index to include in the plot. If not specified, the default is 0.
        :type min: int, optional
        :param max: The maximum component index to include in the plot. If not specified, the default is the intrinsic dimensionality of the trajectory space.
        :type max: int, optional
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.get_wcorr()
        
        ax = plt.imshow(self.Wcorr, cmap='plasma')
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        plt.title("W-Correlation Matrix")
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)




    def get_wcov(self):
        """
        Calculates and returns the w-covariance matrix for the time series.

        The function computes the weights and then calculates the w-covariance matrix.

        :returns: A w-covariance matrix for the time series. If the time series has only one component, the returned w-covariance matrix will be of size (d, d) filled with zeros, where d is the intrinsic dimensionality of the trajectory space.
        :rtype: numpy.ndarray
        """
        if self.d > 1:
            # Calculate the weights
            w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

            # Reshape w for broadcasting
            w = w[:, np.newaxis]

            # Compute weighted inner products
            weighted_inner_products = (w*self.TS_comps).T.dot(self.TS_comps)

            # Calculate Wcov
            self.Wcov = np.abs(weighted_inner_products)

        else :
            self.Wcov = np.zeros((self.d, self.d))  # The time series has only one component

        return self.Wcov
    



    def plot_wcov(self, min=None, max=None):
        """
        Plots the w-covariance matrix for the decomposed time series.

        :param min: The minimum component index to include in the plot. If not specified, the default is 0.
        :type min: int, optional
        :param max: The maximum component index to include in the plot. If not specified, the default is the intrinsic dimensionality of the trajectory space.
        :type max: int, optional
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcov is None:
            self.get_wcov()
        

        ax = plt.imshow(self.Wcov)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.title("W-covariance matrix")
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)




    def grouping_elements_corr(self ,nb_cluster, link = 'single'):
        """
        Performs agglomerative clustering (single linkage) of the elementary matrices based on the W-correlation matrix.

        :param nb_cluster: The number of clusters to form as well as the number of centroids to generate.
        :type nb_cluster: int
        :param link: The linkage to use for agglomerative clustering. Must be either 'single' or 'complete' or 'average'.
        :type link: str, default 'single'
        :returns: An array of cluster labels for each elementary matrix in the time series. If the intrinsic dimensionality of the trajectory space is less than the number of clusters, an array of integers from 0 to nb_cluster-1 is returned.
        :rtype: numpy.ndarray
        """
        if self.Wcorr is None:
            self.get_wcorr()
            
        self.clusters = None
        # Apply the clustering algorithm to the W-correlation matrix

        if self.d >= nb_cluster :
            wmatrix_distance = 1-self.Wcorr # distance matrix
            agg_clustering = AgglomerativeClustering(n_clusters=nb_cluster, metric='precomputed', linkage=link)
            self.clusters = agg_clustering.fit_predict(wmatrix_distance)

        else : 
            self.clusters = np.arange(nb_cluster) # Create a list of nb_cluster elements assign to each elementary matrix

        return self.clusters




    def grouping_elements_cov(self ,nb_cluster, link = 'single'):
        """
        Performs agglomerative clustering (single linkage) of the elementary matrices based on the W-covariance matrix.

        :param nb_cluster: The number of clusters to form as well as the number of centroids to generate.
        :type nb_cluster: int
        :param link: The linkage to use for agglomerative clustering. Must be either 'single' or 'complete' or 'average'.
        :type link: str, default 'single'
        :returns: An array of cluster labels for each elementary matrix in the time series. If the intrinsic dimensionality of the trajectory space is less than the number of clusters, an array of integers from 0 to nb_cluster-1 is returned.
        :rtype: numpy.ndarray
        """
        if self.Wcov is None:
            self.get_wcov()

        self.clusters = None

        if self.d >1 :
            wmatrix_distance = squareform(pdist(self.Wcov)) # Conversion to distance matrix
            agg_clustering = AgglomerativeClustering(n_clusters=nb_cluster, metric='precomputed', linkage=link)
            self.clusters = agg_clustering.fit_predict(wmatrix_distance)

        else : 
            self.clusters = np.arange(nb_cluster) # Create a list of nb_cluster elements assign to each elementary matrix

        return self.clusters




    def plot_groups_components(self, F_groups):
        """
        Plots the original time series and its separated components, grouped by clusters, on a single plot.

        :param F_groups: A list of lists (givent by agglomerative clustering) where each sublist contains the indices of the components in a given cluster.
        :type F_groups: list of lists of int
        :returns: None
        """
        # Plot the toy time series and its separated components on a single plot, with a legend.
        plt.plot(self.orig_TS, lw=1)
        legend = ["$F$"]
        j=0
        for i in (F_groups):
            current_reconstruct_series = self.reconstruct(i)
            plt.plot(current_reconstruct_series)
            legend += [r"$\tilde{F}^{cluster(%s)}$"%j]
            j=j+1

            plt.legend(legend, loc = 'upper left')
        plt.xlabel("$t$")
        plt.ylabel(r"$\tilde{F}^{(j)}$")
        plt.title("Grouped Time Series Components")
        plt.show()




    def grouping_stage(self, nb_cluster, method='corr', link = 'single', bool_plot=False):
        """
        Performs the grouping stage of the algorithm by clustering the components of the time series.
    
        :param nb_cluster: The number of clusters to form.
        :type nb_cluster: int
        :param method: The method to use for grouping elements. Must be either 'corr' for correlation or 'cov' for covariance.
        :type method: str, default 'corr'
        :param link: The linkage to use for agglomerative clustering. Must be either 'single' or 'complete' or 'average'.
        :type link: str, default 'single'
        :param bool_plot: A flag that indicates whether to plot the grouped components or not.
        :type bool_plot: bool, default False
        :return: A list of lists where each sublist contains the indices of the components in a group.
        :rtype: list of lists of int
        :raises ValueError: If the method is not 'corr' or 'cov'.
        :raises ValueError: If the linkage for agglomerative clustering is not 'single' or 'complete' or 'average'.
        """
        # If link is not 'single' or ' completed' or 'average' raise an error
        if link not in ['single', 'complete', 'average']:
            raise ValueError("Link must be 'single' or 'complete' or 'average'.")

        if method == 'cov':
            self.grouping_elements_cov(nb_cluster, link)  # Agglomerative clustering with covariance matrix
        elif method == 'corr':
            self.grouping_elements_corr(nb_cluster, link) # Agglomerative clustering with correlation matrix
        else:
            raise ValueError("Method must be 'cov' or 'corr'.")
        

        # Assemble the grouped components of the time series based on the clustering, with a loop.
        F_groups = [] # list of list of grouped components. E.g. [[F0, F1], [F2, F3, F4]]
        
        for i in range(self.clusters.max()+1):
            F_groups.append([])

        for i in range(len(self.clusters)):
            value = self.clusters[i]
            F_groups[value].append(i)

        if bool_plot:
            self.plot_groups_components(F_groups)

        return F_groups
    
    


    def time_reconstruction(self, nb_cluster, method='corr', link = 'single'):
        """
        Reconstructs the time series from the grouped components.

        :param nb_cluster: The number of clusters to form.
        :type nb_cluster: int
        :param method: The method to use for grouping elements. Must be either 'corr' for correlation or 'cov' for covariance.
        :type method: str, default 'corr'
        :param link: The linkage to use for agglomerative clustering. Must be either 'single' or 'complete' or 'average'.
        :type link: str, default 'single'
        :return: A DataFrame containing the reconstructed time series for each cluster of components, sorted by variance from greatest to least.
        :rtype: pd.DataFrame
        """
        columns = ['Cluster {}'.format(i) for i in range(nb_cluster)]
        reconstructed_time_series = pd.DataFrame(columns=columns)
        F_groups = self.grouping_stage(nb_cluster, method, link)

        for i in range(len(F_groups)):
            if self.d > i:
                current_reconstruct_series = self.reconstruct(F_groups[i])
            else :
                #reconstruction with a null time series, if the number of clusters is greater than the intrinsic dimensionality of the trajectory space
                current_reconstruct_series = pd.Series(np.zeros(self.N), index=self.orig_TS.index)

            reconstructed_time_series['Cluster {}'.format(i)] = current_reconstruct_series

        # Sort the columns by variance, from greatest to least.
        var_TS = reconstructed_time_series.iloc[:,:] .var()
        sorted_columns = var_TS.sort_values(ascending=False).index
        reconstructed_time_series = reconstructed_time_series[sorted_columns.tolist()]

        return reconstructed_time_series
    



    def fit_rolling_with_reconstruction(self, window_size, window_size_ssa, step_size, nb_cluster, nb_feature=6, method='corr', link = 'single'):
        """
        Applies the SSA method over a rolling window of the original time series, then reconstructs the time series and computes 
        specific statistical moments (up to 6th order) for each cluster.

        :param window_size: The size of the window over which to apply the SSA method.
        :type window_size: int
        :param window_size_ssa: The size of the window for SSA method.
        :type window_size_ssa: int
        :param step_size: The step size for the rolling window.
        :type step_size: int
        :param nb_cluster: The number of clusters to form.
        :type nb_cluster: int
        :param nb_feature: The number of statistical moments to compute, defaults to 6.
        :type nb_feature: int, optional
        :param method: The method to use for grouping elements. Must be either 'corr' for correlation or 'cov' for covariance.
        :type method: str, default 'corr'
        :param link: The linkage to use for agglomerative clustering. Must be either 'single' or 'complete' or 'average'.
        :type link: str, default 'single'
        :raises ValueError: If the provided window size for SSA is greater than the window size for the rolling window.
        :raises ValueError: If the provided step size is greater than the window size for the rolling window.
        :raises ValueError: If the provided number of features is greater than 6.
        :raises ValueError: If the provided number of cluster is greater than the window size for SSA.
        :return: A tuple where the first element is a DataFrame containing the reconstructed time series for each cluster and 
                the second element is a DataFrame containing the computed moments of reconstructed time series for each cluster.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        if window_size_ssa > window_size:
            raise ValueError("The window size for SSA must be smaller than the window size for the rolling window.")
        
        if step_size > window_size:
            raise ValueError("The step size must be smaller than the window size.")
        
        if nb_feature > 6:
            raise ValueError("The number of features must be smaller than 6.")
        
        if nb_cluster > window_size_ssa:
            raise ValueError("The number of clusters must be smaller than the window size for SSA.")



        def h_order_moment(x, h): # return standardized h-order moment
                if moment(x, 2)**(h/2) == 0:
                    return 0
                
                return moment(x, h) / moment(x, 2)**(h/2)

        columns_reconstruct_series = ['Cluster {}'.format(i) for i in range(nb_cluster)]
        rolling_reconstructed_time_series = pd.DataFrame(columns=columns_reconstruct_series)

        # Column labels
        clusters = ['Cluster {}'.format(i) for i in range(nb_cluster)]
        moments = ['Moment {}'.format(i+1) for i in range(nb_feature)]
        columns = pd.MultiIndex.from_product([clusters, moments], names=['Cluster', 'Moment'])
        reconstructed_feature_series = pd.DataFrame(columns=columns)

        # Loop over the time series with the defined step size
        for i in tqdm(range(0, len(self.orig_TS) - window_size + 1, step_size)):
            # Extract the current window of the time series
            window = self.orig_TS.iloc[i:i+window_size]

            # Apply the SSA method on the current window
            a_ssa = A_SSA(window)
            a_ssa.fit_transform(window_size_ssa)
            output = a_ssa.time_reconstruction(nb_cluster, method, link)

            feature_series = pd.DataFrame([np.zeros(nb_cluster * nb_feature)], columns=columns)

            for j in range(nb_cluster):
                current_cluster = output['Cluster {}'.format(j)]

                # Calculate the 6 moments of the current cluster
                moments_values = [current_cluster.mean(), current_cluster.var(), current_cluster.skew(), current_cluster.kurtosis(), 
                                h_order_moment(current_cluster,5), h_order_moment(current_cluster,6)]

                for k in range(nb_feature):
                    feature_series.loc[0, ('Cluster {}'.format(j), 'Moment {}'.format(k+1))] = moments_values[k]


            # Concatenate the output with the previous SSA outputs
            rolling_reconstructed_time_series = pd.concat([rolling_reconstructed_time_series, output], axis=0, ignore_index=True)
            reconstructed_feature_series = pd.concat([reconstructed_feature_series, feature_series], axis=0, ignore_index=True)

        return rolling_reconstructed_time_series, reconstructed_feature_series
    
    


    def plot_reconstructed_TS(self, reconstruct_df, x_interval=None):
        """
        Plots the original time series and the reconstructed time series with legends and grid.

        :param reconstruct_df: DataFrame containing the reconstructed time series.
        :type reconstruct_df: pd.DataFrame
        :param x_interval: Interval for the x-axis of the plot. It should be a tuple (xmin, xmax), defaults to None.
        :type x_interval: tuple, optional
        """

        # Create a subplot with two plots vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot on the first subplot
        ax1.plot(self.orig_TS)
        ax1.set_title('Origin Time Series')
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(r"$\tilde{F}^{(j)}$")
        ax1.legend(["$F$"], loc = 'upper left')
        ax1.grid()

        # Plot on the second subplot
        ax2.plot(reconstruct_df)
        ax2.set_title('Rolling reconstructed Time Series')
        ax2.set_xlabel("$t$")
        ax2.set_ylabel(r"$\tilde{F}^{(j)}$")
        ax2.legend(list(reconstruct_df.columns), loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid()

        if x_interval:
            ax1.set_xlim(x_interval)
            ax2.set_xlim(x_interval)

        # Adjust the layout and spacing between subplots
        fig.tight_layout()

        # Display the plot
        plt.show()




    def plot_feature_extract(self, df_feature):
        """
        Plots features of a time series with and without normalization.

        :param df_feature: DataFrame containing the features to be plotted.
        :type df_feature: pd.DataFrame
        """

        # Create another df with MinMaxScaler
        scaler = MinMaxScaler()
        df_feature_norm = pd.DataFrame(scaler.fit_transform(df_feature), columns=df_feature.columns)

        # Create a subplot with two plots vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # First subplot
        ax1.plot(df_feature)
        ax1.set_title('Feature mean, var, skew')
        ax1.set_xlabel("$timestamp$")
        ax1.legend(list(df_feature.columns), loc='center left', bbox_to_anchor=(1, 0.5))

        ax1.grid()

        # Second subplot
        ax2.plot(df_feature_norm)

        ax2.set_title('Feature mean, var, skew with normalization')
        ax2.set_xlabel("$timestamp$")
        # Put the legend outside of the plot, on the right
        ax2.legend(list(df_feature.columns), loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid()

        # Adjust the layout and spacing between subplots
        fig.tight_layout()

        # Display the plot
        plt.show()
