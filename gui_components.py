import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

import plots
from utils import *
from analysis import get_top_correlations, run_ttest_1samp, run_proportions_ztest, run_ci, run_linear_regression

def render_infobox(dataset, time_to_fetch):
    """Renders the top info box, calculates correlation. Returns corr matrix."""
        
    col1, corr_col = st.columns(2)
    with col1:
        if dataset.metadata.intro_paper and dataset.metadata.intro_paper['title']:
                st.write(f"**Intro paper**: [{dataset.metadata.intro_paper['title']}]({dataset.metadata.intro_paper['url']}), {dataset.metadata.intro_paper['year']}.")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Columns:**", str(dataset.metadata.num_features))
            st.write("**Fetched in:**", time_to_fetch)
            st.write("**Category:**", dataset.metadata.area)
            st.write("**Creators:**", ", ".join(dataset.metadata.creators))
            
        with col2:
            st.write("**Rows:**", str(dataset.metadata.num_instances))
            st.write("**Size:**", format_dataset_size(dataset.data.original.memory_usage(deep=True).sum()))
            st.write("**Year:**", str(dataset.metadata.year_of_dataset_creation))
            st.write("**Suggested tasks:**", ", ".join(dataset.metadata.tasks))

    with corr_col:
        with st.container():
            corr_method = st.selectbox("Select correlation method", ["Pearson", "Kendall", "Spearman"])
            try:
                corr_matrix = dataset.data.original.select_dtypes(include='number').corr(corr_method.lower())
                correlations = get_top_correlations(corr_matrix)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Positive Correlation:**  \n", f"{correlations[0].index[0][0]} & {correlations[0].index[0][1]}, r = {correlations[0].iloc[0]:.3f}")
                with col2:
                    st.write("**Top Negative Correlation:**  \n", f"{correlations[1].index[0][0]} & {correlations[1].index[0][1]}, r = {correlations[1].iloc[0]:.3f}")

            except Exception as e:
                st.error(f"An error occurred while calculating the correlations: {e}")

    with st.expander(f"Dataset details"):
        st.markdown(dataset.metadata.additional_info.summary)
        st.write("**Additional variable info:**")
        st.code(dataset.metadata.additional_info.variable_info, language=None)
        st.write(get_written_summary(dataset))
    
    # This should work if the try-except block fails, #TODO force .corr() to fail test
    if not corr_matrix.empty:
        return corr_matrix
    return None


def render_analysis_header(dataset):
    """Allows selection for target column, random seed and active button."""

    col1, col2, col3 = st.columns([0.2,0.4,0.2])

    with col1:
        st.empty()
    with col2:
        st.markdown("## Statistical Analysis Toolkit")
        use_label_finder = st.checkbox("Use smart label finder", value=True, help="Filter to find suggested label columns. Turn off to allow any column.")

        if use_label_finder:
            col1, col2 = st.columns(2)

            with col1:
                label_threshold = st.number_input("Label filtering threshold:", value=0.05, min_value=0.01, max_value=1.0, help="Unique % threshold. Reduce value to allow more options. ")

            with col2:
                filtered_columns = [col for col in dataset.data.original.columns if 
                                    (dataset.data.original[col].value_counts(normalize=True).max()) > label_threshold]

                # Calculate the percentage of unique values for each column & sort descending by unique values
                total_counts = dataset.data.original.shape[0]
                unique_value_percentages = {col: (dataset.data.original[col].nunique() / total_counts) * 100 for col in filtered_columns}
                sorted_columns = sorted(unique_value_percentages, key=unique_value_percentages.get)
                target_column = st.selectbox("Select Column", sorted_columns, help="Sorted descending by least unique values.")
        else:
            target_column = st.selectbox("Select Column", [col for col in dataset.data.original.columns])

        random_seed_input = st.text_input("Set Seed", value="2023", help=tooltips.random_seed)
        
        if random_seed_input.strip() == "" or random_seed_input.lower() == "none":
            random_seed = None
        else:
            try:
                random_seed = int(random_seed_input)
            except ValueError:
                st.error("Seed must be an integer or 'None'.")

        button_labels = ["One-sample t-Test", "Proportions z-Test", "Confidence Interval", "Linear Regression"]
        active_button = st.radio(label=" ", options=button_labels, horizontal=True, index=3)

    with col3:
        st.empty()

    return active_button, target_column, random_seed

def render_ttest(dataset, target_column, random_seed, DEFAULT_DATASET_IDX):
    st.markdown("### One-sample t-Test")
    st.markdown("""Construct a hypothesis test that compares the mean of a sample to a specified value, known as the null hypothesis mean ($\mu_0$).""")

    col1, plot_col = st.columns(2)

    with col1:

        # For Wine Dataset (Default)
        pre_select_index = 8 if DEFAULT_DATASET_IDX == 66 else 0
        ttest1_target_param = st.selectbox("Select Parameter:", [col for col in dataset.data.original.columns if col != target_column], index=pre_select_index)
        
        col1, col2 = st.columns(2)

        with col1:
            select_class = st.selectbox(f"Select category from {target_column}", dataset.data.original[target_column].unique())
            selected_classes = [select_class]
        with col2:
            n_samples = st.number_input("Number of samples", value=15, help=tooltips.sample_size)
        

        col1, col2, col3 = st.columns(3)
        with col1:
            popmean = st.number_input("$\mu_0$ (Hypothesized mean)", value=1.75, help=tooltips.null_hypothesis_threshold)
        with col2:
            alternative_hypothesis = st.selectbox("$H_1$ (Alternative hypothesis)", options=['Less', 'Greater', 'Two-sided'], index=0)
        with col3:
            significance_alpha = st.number_input("$a$ (Significance)", value=0.05, min_value=0.0, max_value=1.0, help=tooltips.significance_level)

        try:
            sample_data = dataset.data.original[dataset.data.original[target_column].isin(selected_classes)].sample(n_samples, random_state=random_seed)[ttest1_target_param]
            alternative = alternative_hypothesis.lower()

            if alternative == "less":
                h0_operand = "\geq"
                h1_operand = "<"
                h0_phrase = "or greater"
                h1_phrase = "or lesser"

            elif alternative == "greater":
                h0_operand = "\leq"
                h1_operand = ">"
                h0_phrase = "or lesser"
                h1_phrase = "or greater"

            else:  # Two-sided
                h0_operand = "="
                h1_operand = "≠" 
                h0_phrase = ""
                h1_phrase = "either greater or lesser"
            
            with st.expander("Show your hypothesis:"):

                st.write(f"""
                        Null Hypothesis: $H_0: \mu_0 {h0_operand} {popmean} $

                        The mean amount of {ttest1_target_param} in {target_column} {str(selected_classes[0])} is equal to {h0_phrase} {popmean}.

                        Alternative Hypothesis: $ H_1: \mu_0 {h1_operand} {popmean} $

                        The mean amount of {ttest1_target_param} in {target_column} {str(selected_classes[0])} is {h1_phrase} than {popmean}.

                        """)

            t_statistic, p_value = run_ttest_1samp(sample_data, popmean, alternative=alternative)

            st.write("**Results:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Sample mean:** {sample_data.mean():.{6}f}")
            with col2:
                st.markdown(f"**Sample STD:** {sample_data.std():.{6}f}")
            with col3:
                st.markdown(f"**T-Statistic:** {t_statistic:.{6}f}", help=tooltips.ttest_1samp_tstat)
            with col4:
                st.markdown(f"**P-Value:** {p_value:.{6}f}", help=tooltips.ttest_1samp_pval)


            if p_value < significance_alpha:
                st.warning("Reject the null hypothesis.")
                st.write(f"The mean amount of {ttest1_target_param} in {target_column} {str(selected_classes[0])} is with {((1-significance_alpha) * 100):.1f}% confidence significantly {alternative} than {popmean:.2f}.")
            else:
                st.info("Accept the null hypothesis.")
                if alternative_hypothesis == "Two-sided":
                    alternative = 'different'
                st.write(f"The mean amount of {ttest1_target_param} in {target_column} {str(selected_classes[0])} is with {((1-significance_alpha) * 100):.1f}% confidence **not** significantly {alternative} than {popmean:.2f}.")


        except Exception as e:
            st.error(f"An error occurred while performing hypothesis testing: {e}")

        with st.expander("About one-sample t-tests"):
            st.markdown("The t-test for one sample is a test used to determine whether the mean of a sample differs significantly from a known or hypothesized population mean. It is applicable when we have a single group of observations and want to assess whether the mean of that group is consistent with a specified value. The test statistic ($t$) is calculated as:")
            st.latex(r"t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}")
            st.markdown("""
            Where:

            - $ x̄ $ represents the sample mean,
            - $ \mu_0 $ represents the hypothesized population mean (the mean specified in the null hypothesis),
            - $ s $ represents the sample standard deviation, and
            - $ n $ represents the sample size.

            The formula calculates the t-statistic, which measures how many standard errors the sample mean ($ x̄ $) is away from the hypothesized population mean ($ \mu_0 $). If the calculated t-statistic is large in absolute value, it indicates that the sample mean is significantly different from the hypothesized population mean. This t-statistic is then used to determine the p-value, which helps in deciding whether to reject or fail to reject the null hypothesis.
            """)
            st.markdown("""
            The p-value is the probability of observing a t-statistic as extreme as, or more extreme than, the one obtained from the sample, under the assumption that the null hypothesis is true. It's represented as:
            """)
            st.latex(r"P(|T| > |t|)")
            st.markdown("""Where $ T $ follows a t-distribution with $ n - 1 $ degrees of freedom.""")

        with st.expander("About hypothesized mean"):
            st.markdown("""

            The hypothesized mean, also known as the null hypothesis mean, is a value that is assumed to be true for the population parameter being tested in a hypothesis test.
            It is often denoted as $ \mu_0 $. In the context of this one-sample t-test, the hypothesized mean represents the population mean that you are comparing your sample mean against.
            
            """)

    with plot_col:
        plots.render_ttest_histogram(sample_data, ttest1_target_param, popmean, significance_alpha)


def render_proportion_z_test(dataset, target_column, random_seed, DEFAULT_DATASET_IDX):

    st.markdown("### Hypothesis Test for Proportions")
    col1, col_plot = st.columns(2)

    with col1:
    
        pre_select_index = 3 if DEFAULT_DATASET_IDX == 66 else 0

        prop_ztest_param = st.selectbox("Select Variable:", [col for col in dataset.data.original.columns if col != target_column], index=pre_select_index)


        col1, col2, col3, col4 = st.columns(4)
        with col1:
            select_class1 = st.selectbox(f"$\hat{{p}}_1$: Select category from {target_column}", dataset.data.original[target_column].unique(), index=1, key="select_class1")
            select_class2 = st.selectbox(f"$\hat{{p}}_2$: Select category from {target_column}", dataset.data.original[target_column].unique(), index=2, key="select_class2")
        with col2:
            class_1_operand = st.selectbox(f"Select operand", ["<", ">"], index=1, key="ztest_operand_1")
            class_2_operand = st.selectbox(f"Select operand", ["<", ">"], index=1, key="ztest_operand_2")
        with col3:
            class_1_thresh = st.number_input(f"Select threshold", value=20, key="class_1_thresh")
            class_2_thresh = st.number_input(f"Select threshold", value=20, key="class_2_thresh")
        with col4:
            n_samples_1 = st.number_input(f"Number of samples for {target_column} {select_class1}", value=25, key="n_samples_1")
            n_samples_2 = st.number_input(f"Number of samples for {target_column} {select_class2}", value=25, key="n_samples_2")

        col1, col2, col3 = st.columns(3)
        with col1:
            null_hypothesis_threshold = st.number_input("Threshold for $H_0$ (Null hypothesis)", value=20, help=tooltips.null_hypothesis_threshold)
        with col2:
            alternative_hypothesis = st.selectbox("$H_1$ (Alternative hypothesis)", options=['Less', 'Greater', 'Two-sided'], index=2)
        with col3:
            significance_alpha = st.number_input("Threshold for $a$ (Significance)", value=0.05, min_value=0.0, max_value=1.0, help=tooltips.significance_level)


        with st.expander("Show your hypothesis:"):

            st.write(f"""
                    Null Hypothesis, $ H_0: \hat{{p}}_1 = \hat{{p}}_2 $

                    The proportion of samples of {prop_ztest_param} is equal for both classes.
                    
                    Alternative Hypothesis, $ H_0: \hat{{p}}_1 ≠ \hat{{p}}_2 $

                    The proportion of samples of {prop_ztest_param} differs between the two classes.

                    """)

        try:
            p1 = dataset.data.original[dataset.data.original[target_column] == select_class1].sample(n_samples_1, random_state=random_seed)
            p2 = dataset.data.original[dataset.data.original[target_column] == select_class2].sample(n_samples_2, random_state=random_seed)
        
            z_statistic, p_value, prop_1, prop_2 = run_proportions_ztest(p1, p2, prop_ztest_param, class_1_thresh, class_2_thresh)


            # if class_1_operand == ">":
            #     p1 = dataset.data.original[(dataset.data.original[prop_ztest_param] > class_1_thresh)].sample(n_samples_1, random_state=random_seed)
            # if class_2_operand == ">":
            #     p2 = dataset.data.original[(dataset.data.original[prop_ztest_param] > class_2_thresh)].sample(n_samples_2, random_state=random_seed)

            # if class_1_operand == "<":
            #     p1 = dataset.data.original[(dataset.data.original[prop_ztest_param] < class_1_thresh)].sample(n_samples_1, random_state=random_seed)
            # if class_2_operand == "<":
            #     p2 = dataset.data.original[(dataset.data.original[prop_ztest_param] < class_2_thresh)].sample(n_samples_2, random_state=random_seed)


            st.write("**Results:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Proportions for {target_column} {select_class1}:** {prop_1}", help="Placeholde")
            with col2:
                st.markdown(f"**Proportions for {target_column} {select_class2}:** {prop_2}", help="Placeholde")
            with col3:
                st.markdown(f"**Z-Statistic:** {z_statistic:.{6}f}", help="Placeholde")
            with col4:
                st.markdown(f"**P-Value:** {p_value:.{6}f}", help=tooltips.ttest_1samp_pval)

            if p_value < significance_alpha:
                st.warning("Reject the null hypothesis.")
                st.write(f"The proportion of samples with {prop_ztest_param} greater than {null_hypothesis_threshold} differs significantly between {target_column} variety {select_class1} and {target_column} variety {select_class2} at the {100*(1-significance_alpha)}% confidence level.")
            else:
                st.info("Accept the null hypothesis.")
                st.write(f"The proportion of samples with {prop_ztest_param} greater than {null_hypothesis_threshold} does not differ significantly between {target_column} variety {select_class1} and {target_column} variety {select_class2} at the {100*(1-significance_alpha)}% confidence level.")


        except Exception as e:
            st.error(f"An error occurred while performing hypothesis testing: {e}")

    with col_plot:
        plots.render_stacked_barplot(select_class1, select_class2, prop_1, prop_2, null_hypothesis_threshold, prop_ztest_param)

        # plots.render_barplot_proportions(select_class1, select_class2, prop_1, prop_2, null_hypothesis_threshold, prop_ztest_param)    
        # st.markdown("Placeholder", help=tooltips.violin_plot)

def render_confidence_interval(dataset, target_column, random_seed, DEFAULT_DATASET_IDX):
    st.write("### Confidence Interval")
    st.markdown("""Construct a confidence interval (CI) to estimate the range of plausible values for a population parameter based on samples.""")
    col1, col_plot = st.columns(2)
    with col1:

        # For Wine Dataset
        pre_select_index = 5 if DEFAULT_DATASET_IDX == 66 else 0
        ci_target_param = st.selectbox("Select Parameter:", [col for col in dataset.data.original.columns if col != target_column], index=pre_select_index)

        col1, col2 = st.columns(2)
        with col1:
            pre_select_index = 1 if DEFAULT_DATASET_IDX == 66 else 0
            select_class1 = st.selectbox(f"Select category from {target_column} column", dataset.data.original[target_column].unique(), index=1, key="select_class1")
            select_class2 = st.selectbox(f"Select category from {target_column} column", dataset.data.original[target_column].unique(), index=pre_select_index+1, key="select_class2")

        with col2:
            n_samples_1 = st.number_input("Number of samples", value=20, min_value=1, help=tooltips.sample_size, key="n_samples_1")
            n_samples_2 = st.number_input("Number of samples", value=20, min_value=1, key="n_samples_2")
        
        confidence_level = st.number_input("Confidence Level (%):", min_value=0.0, max_value=100.0, step=0.1, value=95.0, help=tooltips.ci_level)

        sample_data_1 = dataset.data.original[dataset.data.original[target_column] == select_class1][ci_target_param].sample(n_samples_1, random_state=random_seed)
        sample_data_2 = dataset.data.original[dataset.data.original[target_column] == select_class2][ci_target_param].sample(n_samples_2, random_state=random_seed)


        CI_lower, CI_upper, SE, ME, t_value, df = run_ci(sample_data_1, sample_data_2, confidence_level)

        st.write(f"**Results:** {confidence_level:.1f}% Confidence Interval for the Difference in {ci_target_param}:")


        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Lower Bound:** {CI_lower.round(5)}")
        with col2:
            st.markdown(f"**Upper Bound:** {CI_upper.round(5)}")
        with col3:
            st.markdown(f"**SE:** {SE.round(5)}")
        with col4:
            st.markdown(f"**Margin of Error (MOE):** {ME.round(5)}", help=tooltips.ci_moe)


        st.write(f"The true difference in {ci_target_param} between the {target_column} {select_class1} and {target_column} {select_class2} varieties falls within the range of {CI_lower:.3f} to {CI_upper:.3f}, with {confidence_level:.1f}% confidence.")

        if CI_lower > 0:
            st.info(f"""
            Since the confidence interval does not include zero, it implies that the difference in {target_column} is statistically significant.  \n
            If there were truly no difference in {target_column} between the {target_column} {select_class1} and {select_class2}, we would expect the confidence interval to include zero.
            """)

    with col_plot:
        plots.render_violin_plot(target_column, ci_target_param, select_class1, select_class2, sample_data_1, sample_data_2)


def render_linear_regression(dataset, target_column, random_seed):
    
    col1, plot_col = st.columns(2)
    with col1:
        st.markdown("### Linear Regression")
        st.markdown("Perform a regression analysis that models the relationship between a predictor variables and a target variable.")

        col1, col2, col3 = st.columns(3)
        with col1:
            select_class = st.selectbox(f"Select category from {target_column}", dataset.data.original[target_column].unique(), index=1, key="select_class1")
        with col2:
            predictor_variable = st.selectbox("Select predictor variable", dataset.data.original.columns, index=0)
        with col3:
            target_variable = st.selectbox("Select target variable", dataset.data.original.columns, index=1)

        model_type = st.selectbox("Select Model Type", ["OLS", "WLS", "GLS"])

        # TODO option to deselect filtering
        filtered_data = dataset.data.original[dataset.data.original[target_column] == select_class]

        model, X = run_linear_regression(model_type, filtered_data, predictor_variable, target_variable)

        st.markdown("#### Model Summary:") # TODO format
        st.code(model.summary())

    with plot_col:
        # Inference
        st.markdown("### Make Predictions:")
        st.markdown("Input new values for the predictor variable to generate predictions.")
        
        # TODO Improve inference GUI - allow multiple values
        col1, col2 = st.columns(2)
        with col1:
            new_value = st.number_input(f"Enter value for {predictor_variable}", value=filtered_data[predictor_variable].mean())
            prediction = model.predict([1, new_value])[0]
        with col2:
            st.write("##")
            st.write(f"**Predicted {target_variable}:** {prediction:.4f}")
        
        plots.render_linreg_scatter(model, filtered_data, predictor_variable, target_variable, X, [new_value], ["#d19e0f"])
            