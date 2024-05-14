import numpy as np
import pandas as pd

from a_ssa import A_SSA


def test_project_ssa():
    # Create a toy time series
    N = 200 # The number of time 'moments' in our toy series
    t = np.arange(0,N)
    trend = 0.001 * (t - 100)**2
    p1, p2 = 20, 30
    periodic1 = 2 * np.sin(2*np.pi*t/p1)
    periodic2 = 0.75 * np.sin(2*np.pi*t/p2)

    np.random.seed(123) # So we generate the same noisy time series every time.
    noise = 2 * (np.random.rand(N) - 0.5)
    F = trend + periodic1 + periodic2 + noise

    # Import True Dataframe
    df_result = pd.read_csv('tests/toy_result_1.csv')
    df_result = np.round(df_result, 3)

    # Create an instance of A_SSA
    ssa_obj = A_SSA(F)

    # Parameter : ssa window size=50 ; ssa number of components=4 ; clustering on correlation matrix ; linkage of agglomerative clustering='single'
    ssa_obj.fit_transform(50)
    ssa_obj_result = ssa_obj.time_reconstruction(4)
    ssa_obj_result = np.round(ssa_obj_result, 3)

    # Compare if the two pandas dataframes are equal
    assert df_result.equals(ssa_obj_result)

