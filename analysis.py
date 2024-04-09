import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, t
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm


def get_top_correlations(corr_matrix, n=1):
    # Excludes self-correlations (diagonal)

    num_features = len(corr_matrix)
    n = min(n, num_features)
    
    corr_matrix_no_diag = corr_matrix.mask(np.eye(num_features, dtype=bool))
    
    top_positive_corr = corr_matrix_no_diag.unstack().sort_values(ascending=False)[:n]
    top_negative_corr = corr_matrix_no_diag.unstack().sort_values()[:n]

    return top_positive_corr, top_negative_corr


def run_ttest_1samp(sample_data, popmean, alternative):
    """
    Perform a T-test for the mean of ONE group of scores.

    Parameters:
    sample_data (array-like): The sample data.
    popmean (float or array-like): The population mean under the null hypothesis.
    alternative (str): The alternative hypothesis. It must be one of the following:
                       'two-sided', 'greater', 'less'.

    Returns:
    tuple: A tuple containing the t-statistic and p-value.
    """
    try:
        t_statistic, p_value = ttest_1samp(a=sample_data, popmean=popmean, alternative=alternative)

        return t_statistic, p_value
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def run_proportions_ztest(p1, p2, prop_ztest_param, class_1_thresh, class_2_thresh):
    """
    Perform a z-test for proportions between two groups.

    Parameters:
    p1 (DataFrame): DataFrame containing data for the first group.
    p2 (DataFrame): DataFrame containing data for the second group.
    prop_ztest_param (str): Name of the parameter to use for the z-test.
    class_1_thresh (float): Threshold for determining success in the first group.
    class_2_thresh (float): Threshold for determining success in the second group.

    Returns:
    tuple: A tuple containing the z-statistic, p-value, proportion of successes in the first group,
           and proportion of successes in the second group.
    """
    try:
        # Count the number of successes (samples meeting the condition) in each class
        successes_1 = sum(p1[prop_ztest_param] > class_1_thresh)
        successes_2 = sum(p2[prop_ztest_param] > class_2_thresh)

        nobs_1 = len(p1)
        nobs_2 = len(p2)

        z_statistic, p_value = proportions_ztest([successes_1, successes_2], [nobs_1, nobs_2])
        
        # Calculate proportions
        prop_1 = successes_1 / nobs_1
        prop_2 = successes_2 / nobs_2

        return z_statistic, p_value, prop_1, prop_2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


def run_ci(sample_data_1, sample_data_2, confidence_level):
    """
    Calculate the confidence interval for the difference between means of two samples.

    Parameters:
    sample_1 (array-like): First sample data.
    sample_2 (array-like): Second sample data.
    confidence_level (float): Desired confidence level for the interval (e.g., 95%).

    Returns:
    tuple: A tuple containing the lower and upper bounds of the confidence interval,
           standard error, margin of error, t-value, and degrees of freedom.
    """
    mean_1 = np.mean(sample_data_1)
    mean_2 = np.mean(sample_data_2)
    std_1 = np.std(sample_data_1)
    std_2 = np.std(sample_data_2)
    
    # Calculate pooled standard error
    SE = np.sqrt((std_1**2 / len(sample_data_1)) + (std_2**2 / len(sample_data_2)))
    
    # Calculate degrees of freedom
    df = len(sample_data_1) + len(sample_data_2) - 2
    
    # Calculate t-value for the given confidence level
    alpha = (100 - confidence_level) / 100
    t_value = t.ppf(1 - alpha / 2, df)
    
    # Calculate margin of error
    ME = t_value * SE
    
    # Compute confidence interval
    CI_lower = (mean_1 - mean_2) - ME
    CI_upper = (mean_1 - mean_2) + ME

    return CI_lower, CI_upper, SE, ME, t_value, df


def run_linear_regression(model_type, data, predictor_variable, target_variable):

    X, y  = data[predictor_variable], data[target_variable]
    X = sm.add_constant(X)

    if model_type == "OLS":
        model = sm.OLS(y, X).fit()

    elif model_type == "WLS":

        st.info("Weighted Least Squares (WLS) requires specifying weights. If you're unsure, you can try equal weights or use variance-stabilizing transformations.")
        weights_option = st.selectbox("Select Weights Option", ["Equal Weights", "Variance-Stabilizing Transformations"])
        if weights_option == "Equal Weights":
            weights = np.ones_like(y)
        else:
            weights = np.log(y)  # TODO
        model = sm.WLS(y, X, weights=weights).fit()

    elif model_type == "GLS":

        st.info("Generalized Least Squares (GLS) requires specifying the covariance structure. You can try different covariance estimators based on your assumptions about the data.")
        cov_structure_option = st.selectbox("Select Covariance Structure", ["Autoregressive (AR)", "Moving Average (MA)", "Heteroscedasticity-Consistent (HC)"])
        if cov_structure_option == "Autoregressive (AR)":
            cov_structure = "ar"
        elif cov_structure_option == "Moving Average (MA)":
                cov_structure = "ma"
        else:
            cov_structure = "hc0"
            model = sm.GLS(y, X, cov_type=cov_structure).fit()

    return model, X