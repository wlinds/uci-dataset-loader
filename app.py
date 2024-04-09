import time
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go

import plots
from utils import *
from uci_api import *
from gui_components import *

DEFAULT_DATASET_IDX = 66 # Set default loaded dataset when launching app

def main():
    st.set_page_config(page_title="UC Irvine ML Repo Dataset Loader", layout="wide")
    use_local_css(".streamlit/stylesheet.css")

    try:
        datasets = list_available_datasets()
        dataset_options = {f"{d['name']} (ID: {d['id']})": d['id'] for d in datasets}
    except Exception as e:
        st.error(f"An error occurred while fetching datasets: {e}")

    st.title('UC Irvine ML Repo Dataset Loader')
    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()), index=DEFAULT_DATASET_IDX)
    
    with st.spinner("Fetching dataset..."):
        start_time = time.time()
        dataset = fetch_ucirepo(id=dataset_options[selected_dataset])
        time_to_fetch = format_loading_time(start_time, time.time())

    corr_matrix = render_infobox(dataset,time_to_fetch)

    with st.spinner("Creating table..."):
        st.write(dataset.data.original)

    col1, col2 = st.columns(2)
    with col1:
        plots.render_correlation_matrix(corr_matrix)
    with col2:
        plots.render_histogram(dataset)

    active_button, target_column, random_seed = render_analysis_header(dataset)

    if active_button == "One-sample t-Test":
        render_ttest(dataset, target_column, random_seed, DEFAULT_DATASET_IDX)

    elif active_button == "Proportions z-Test":
        render_proportion_z_test(dataset, target_column, random_seed, DEFAULT_DATASET_IDX)

    elif active_button == "Confidence Interval":
        render_confidence_interval(dataset, target_column, random_seed, DEFAULT_DATASET_IDX)

    elif active_button == "Linear Regression":
        render_linear_regression(dataset, target_column, random_seed)

    else:
        st.warning("No idea how you managed to get here. Reload page.")



if __name__ == "__main__":
    main()