import numpy as np
import streamlit as st

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def use_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# TODO Complete tooltips

tooltips = dotdict({
    "sample_size": "Number of samples to include in calculation. Unless a set seed is used, samples will always be picked at random.",
    "random_seed": "Leave empty or use 'None' for random seed. Enter any integer for reproducibility.",

    "null_hypothesis_threshold": "Specify the value against which the null hypothesis is tested. This value defines the point of comparison for assessing the significance of the results.",
    "significance_level": "Determines the threshold for rejecting the null hypothesis. A lower significance level indicates a higher standard of evidence required to reject the null hypothesis.",

    "ttest_1samp_tstat": "Indicates the magnitude of the difference between the sample mean and the population mean, relative to the variability in the data.",
    "ttest_1samp_pval": " Indicates the probability of observing the given sample mean (or more extreme) if the null hypothesis (that the sample mean equals the population mean) is true.",

    "ci_level": "Specify the the percentage of confidence you will have in the accuracy of the results. A higher confidence level indicates a greater certainty in the accuracy of the calculated interval.",
    "ci_moe": "MOE quantifies the precision of an estimate, influenced by sample size and confidence level. It represents the range within which the true population parameter is likely to fall. Higher confidence levels result in wider MOEs, reflecting greater certainty but sacrificing precision. Given by t-value * std",

    "violin_plot": "Violin plots are particularly useful when you want to compare the shape of distributions between groups in addition to central tendency and spread."
})


def format_dataset_size(size_in_bytes):
    size_in_mb = size_in_bytes / (1024 * 1024)
    
    if size_in_mb > 0.5:
        return f"{size_in_mb:.2f} MB"
    else:
        size_in_kb = size_in_bytes / 1024
        return f"{size_in_kb:.2f} KB"


def format_loading_time(start_time, end_time):
    time_sec = end_time - start_time

    if time_sec > 1:
        return f"{time_sec:.2f} sec"
    else:
        return f"{time_sec * 1000:.2f} ms"


def format_alternative_hypothesis(value):
   # Converts alternative hypothesis params from SciPy format to StatsModels format

    if value == 'two-sided':
        return 'two-sided'
    elif value == 'less':
        return 'smaller'
    elif value == 'greater':
        return 'larger'
    else:
        return None

def get_top_correlations(corr_matrix, n=1):
    # Excludes self-correlations (diagonal)

    num_features = len(corr_matrix)
    n = min(n, num_features)
    
    corr_matrix_no_diag = corr_matrix.mask(np.eye(num_features, dtype=bool))
    
    top_positive_corr = corr_matrix_no_diag.unstack().sort_values(ascending=False)[:n]
    top_negative_corr = corr_matrix_no_diag.unstack().sort_values()[:n]

    return top_positive_corr, top_negative_corr


def style_df(val):
    return f'color: #000000'


def get_written_summary(data):

    return data


Pearson = ["""
Each cell in the above matrix represents the correlation between two variables, with values ranging from -1.0 to 1.0.

**Pearson's correlation coefficient $(r)$** is the most common way way of measuring a linear correlation.
It measures the strength and direction of the linear relationship between two variables.

- +1 indicates a perfect positive linear relationship,
- -1 indicates a perfect negative linear relationship,
- 0 indicates no linear relationship between the variables.

The formula for Pearson's correlation coefficient (r) is given by:

""", 

"\frac{\sum[(x - \bar{x}) (y - \bar{y})]}{\sqrt{{\sum(x - \bar{x})^2} {\sum(y - \bar{y})^2}}}",

"""
where:
- x and y are the variables,
- x̄ and ȳ are the means of x and y, respectively.
""",
]