import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

from utils import Pearson

def get_correlation(corr_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        width=600,
        height=600,
        autosize=False,
        margin=dict(l=0, r=0, t=33, b=0),
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )

    return fig

def render_correlation_matrix(corr_matrix):
    with st.spinner("Rendering matrix..."):
        with st.expander("Correlation Matrix", expanded=True):
            try:
                st.plotly_chart(get_correlation(corr_matrix), use_container_width=True, height=600, config={'displayModeBar': False})
            except Exception as e:
                st.error(f"An error occurred while creating the correlation plot: {e}")

        with st.expander("About Correlation Matrices"):
            st.write(Pearson[0])
            st.latex(r"\frac{\sum[(x - \bar{x}) (y - \bar{y})]}{\sqrt{{\sum(x - \bar{x})^2} {\sum(y - \bar{y})^2}}}")
            st.write(Pearson[2])


def get_histogram(dataset, means, select_x, select_y):
    fig = go.Figure()
    for y_value, mean in means.items():
        fig.add_trace(go.Histogram(x=dataset.data.original[dataset.data.original[select_y] == y_value][select_x], name=f'{select_y}={y_value}'))
        fig.add_vline(x=mean, line_dash="dash", line_color="red", annotation_text=f'Mean: {mean:.2f}', annotation_position="top left")
    
    fig.update_layout(barmode='overlay', 
                      xaxis_title=select_x,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                      width=700,
                      height=516,
                      autosize=False,
                      margin=dict(l=0, r=0, t=30, b=0),
                      yaxis=dict(scaleanchor='y', scaleratio=0.2)
                     )

    return fig

def render_histogram(dataset):
    with st.expander("Histogram Plot", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            numeric_columns = [col for col in dataset.data.original.columns if pd.api.types.is_numeric_dtype(dataset.data.original[col])]
            select_x = st.selectbox("Select X (Numerical columns)", numeric_columns)
        with col2:
            # Find the column with the least unique values # TODO handle non-numerics
            unique_value_counts = {col: dataset.data.original[col].nunique() for col in numeric_columns}
            sorted_columns = sorted(unique_value_counts, key=unique_value_counts.get)
            least_unique_column = sorted_columns[0]

            # Sort the dropdown options by least to most unique values
            select_y = st.selectbox("Select Y (Numerical columns)", sorted_columns, index=sorted_columns.index(least_unique_column))

        try:
            with st.spinner("Creating plot..."):
                means = dataset.data.original.groupby(select_y)[select_x].mean()
                stds = dataset.data.original.groupby(select_y)[select_x].std()
                st.plotly_chart(get_histogram(dataset,means,select_x,select_y), config={'displayModeBar': False})

        except TypeError as e:
            st.error("Cannot render plot with selected variables. Try again.")
            st.stop()

    with st.expander(f"Standard Deviation & Means of all {select_y} in {select_x}"):
                    std_table = pd.DataFrame({
                        "Class": stds.index,
                        "Means": means.values.round(3),
                        "Standard Deviation": stds.values.round(3)
            
                    })
                    st.table(std_table)

def render_violin_plot(target_column, ci_target_param, select_class1, select_class2, sample_data_1, sample_data_2):
    fig = go.Figure()
    fig.add_trace(go.Violin(y=sample_data_1, name=f"{str(target_column)} {str(select_class1)}", box_visible=True, meanline_visible=True))
    fig.add_trace(go.Violin(y=sample_data_2, name=f"{str(target_column)} {str(select_class2)}", box_visible=True, meanline_visible=True))

    fig.update_layout(
        yaxis_title=str(ci_target_param),
        showlegend=True,
        
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(showline=True, linewidth=2, linecolor='#d4d4d6'),
        yaxis=dict(showline=True, linewidth=2, linecolor='#d4d4d6'),
        width=800,
        height=600,
    )

    st.plotly_chart(fig, config={'displayModeBar': False})


def render_barplot_proportions(select_class1, select_class2, prop_1, prop_2, null_hypothesis_threshold, target_variable):
    fig = go.Figure(data=[
        go.Bar(name=str(select_class1), x=[str(select_class1)], y=[prop_1], text=[f"{prop_1:.2%}"], textposition='auto'),
        go.Bar(name=str(select_class2), x=[str(select_class2)], y=[prop_2], text=[f"{prop_2:.2%}"], textposition='auto')
    ])
    fig.update_layout(
        title=f'Proportion of Samples Exceeding {null_hypothesis_threshold} in {target_variable}',
        xaxis_title='Class',
        yaxis_title='Proportion',
        barmode='group',

        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(showline=True, linewidth=2, linecolor='#d4d4d6'),
        yaxis=dict(showline=True, linewidth=2, linecolor='#d4d4d6'),
        width=800,
        height=600,

    )
    st.plotly_chart(fig, config={'displayModeBar': False})


def render_stacked_barplot(select_class1, select_class2, prop_1, prop_2, null_hypothesis_threshold, target_variable):
    fig = go.Figure(data=[
        go.Bar(name=str(select_class1), x=[target_variable], y=[prop_1], text=[f"{prop_1:.2%}"], textposition='auto'),
        go.Bar(name=str(select_class2), x=[target_variable], y=[prop_2], text=[f"{prop_2:.2%}"], textposition='auto')
    ])

    fig.update_layout(
        title=f'Proportion of Samples Exceeding {null_hypothesis_threshold} in {target_variable}',
        xaxis_title=target_variable,
        yaxis_title='Proportion',
        barmode='stack',
        showlegend=True,
        legend=dict(title="Class", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, config={'displayModeBar': False})


def render_histogram_2(sample_data, ttest1_target_param, null_hypothesis, significance_alpha):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sample_data, histnorm='probability density', name='Sample Data'))
    fig.add_vline(x=null_hypothesis, line_dash="dash", line_color="red", annotation_text="Hypothesized Mean",
                annotation_position="top right", annotation_font=dict(size=12))

    fig.update_layout(title=f'Distribution of {ttest1_target_param}',
                    xaxis_title=ttest1_target_param,
                    yaxis_title='Probability Density',
                    width=800,
                    height=488),

    st.plotly_chart(fig)


def render_linreg_scatter(model, filtered_data, predictor_variable, target_variable, X):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data[predictor_variable], y=filtered_data[target_variable], mode='markers'))
    fig.add_trace(go.Scatter(x=filtered_data[predictor_variable], y=model.predict(X), mode='lines', line=dict(color='red')))
    fig.update_layout(title="Scatter Plot with Regression Line", xaxis_title=predictor_variable, yaxis_title=target_variable)
    fig.update_layout(
        width=900,
        height=600,
        autosize=False,
        margin=dict(l=0, r=0, t=33, b=0),
        yaxis=dict(scaleanchor='x', scaleratio=1),
        showlegend=False
    )

    st.plotly_chart(fig, config={'displayModeBar': False})