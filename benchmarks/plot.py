"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_runtimes(times_df, problem="primal_random", save_path=None):
    """Plots runtimes for given problem"""
    df = times_df.copy()
    # Combine Sparse Format and Operator Columns
    operator_string = df["operator_mode"].apply(lambda x: "operator" if x else "")
    df["sparse_format operator"] = df["sparse_format"] + " " + operator_string
    # compute error bars
    df["error (1st quartile)"] = df["time (median)"] - df["time (1st quartile)"]
    df["error (3rd quartile)"] = df["time (3rd quartile)"] - df["time (median)"]
    fig = px.bar(
        df[df["problem"] == problem],
        x="solver",
        y="time (median)",
        color="sparse_format operator",
        barmode="group",
        error_y="error (3rd quartile)",
        error_y_minus="error (1st quartile)",
    )

    if save_path:
        full_path = os.path.join(save_path, f"{problem}_runtimes.png")
        fig.write_image(full_path)
    else:
        fig.show()


def plot_residual_v_time(
    run_name, formatted_run_name, times_df, residuals_df, dash_style, fig
):
    fields = run_name.split("|")
    fields = [f.strip() for f in fields]

    runtime = times_df[
        (times_df["problem"] == fields[0])
        & (times_df["sparse_format"] == fields[1])
        & (times_df["solver"] == fields[2])
        & (times_df["operator_mode"] == (True if fields[3] == "op" else False))
    ]["time (median)"]

    runtime = runtime.values[0]
    iterations = residuals_df.shape[0]
    y = residuals_df[run_name + " (median)"]

    fig.add_trace(
        go.Scatter(
            x=np.array(range(iterations)) * runtime / iterations,
            y=y.values,
            error_y=dict(
                type="data",
                symmetric=False,
                #             array=residuals_df[f"{run_name} (3rd quartile)"].values,
                #             arrayminus=residuals_df[f"{run_name} (1st quartile)"].values
            ),
            line=dict(width=4, dash=dash_style),
            # mode='lines+markers',
            name=formatted_run_name,
        )
    )
    return fig


def get_x_max_time(run_name, times_df):
    fields = run_name.split("|")
    fields = [f.strip() for f in fields]

    runtime = times_df[
        (times_df["problem"] == fields[0])
        & (times_df["sparse_format"] == fields[1])
        & (times_df["solver"] == fields[2])
        & (times_df["operator_mode"] == (True if fields[3] == "op" else False))
    ]["time (median)"]

    runtime = runtime.values
    return runtime


def add_vertical_line(fig, run_name, y_max, value):
    """Adds a vertical line to the plot."""
    fig.add_trace(
        go.Scatter(
            x=[value, value],
            y=[0, y_max],
            line=dict(width=2, dash="dot", color="gray"),
            name=f"{run_name.replace('residual_norms', '')}",
        )
    )
    return fig


def plot_runs_over_time(
    run_names, formatted_run_names, times_df, residuals_df, title=None
):
    # Different style we use for the line plots. Loop cycles over these styles.
    dash_styles = ["solid", "dash", "longdash", "dashdot", "longdashdot"]
    fig = go.Figure()
    count = 0
    for run_name, formatted_run_name in zip(run_names, formatted_run_names):
        count = count + 1
        if "direct" in run_name:
            y_max = 1.1
            runtime = get_x_max_time(run_name, times_df)
            fig = add_vertical_line(fig, formatted_run_name, y_max, runtime)
        else:
            fig = plot_residual_v_time(
                run_name,
                formatted_run_name,
                times_df,
                residuals_df,
                dash_styles[count % 5],
                fig,
            )

    fig.update_layout(
        title=title,
        xaxis_title="runtime (in seconds)",
        yaxis_title="relative residual norm",
        yaxis_type="log",
    )
    return fig


def plot_residual_v_iteration(run_name, formatted_run_name, residuals_df, fig):
    """Adds a single trace to the existing figure"""
    y = residuals_df[run_name + " (median)"]
    iterations = y.count()

    fig.add_trace(
        go.Scatter(
            x=list(range(iterations)),
            y=y.values,
            error_y=dict(
                type="data",
                symmetric=False,
                #             array=residuals_df[f"{run_name} (3rd quartile)"].values,
                #             arrayminus=residuals_df[f"{run_name} (1st quartile)"].values
            ),
            mode="lines",
            name=formatted_run_name,
        )
    )

    return fig


def plot_runs_over_iterations(
    run_names, formatted_run_names, residuals_df, title="Relative Residual Norms",
):
    """Plots runs over iterations"""
    fig = go.Figure()

    for run_name, formatted_run_name in zip(run_names, formatted_run_names):
        run_name.replace("residual_norms", "")
        if "direct" not in run_name:
            fig = plot_residual_v_iteration(
                run_name, formatted_run_name, residuals_df, fig
            )

    fig.update_layout(
        title=title,
        xaxis_title="iteration",
        yaxis_title="relative residual norm",
        yaxis_type="log",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig
