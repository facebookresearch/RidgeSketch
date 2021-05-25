"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from itertools import cycle
from random import shuffle
import plotly.graph_objects as go
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator
from PIL import ImageColor
import functools


COLORS = px.colors.qualitative.Plotly
# blue, red, green, purple, cyan, pink, ...

LINE_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

SYMBOLS = [
    "circle",
    "square",
    "star",
    "x",
    "triangle-up",
    "pentagon",
    "cross",
]

# Parameters for exp 9.2, 9.3 and 9.4
COLOR_DICT = {
    "count": COLORS[0],
    "gaussian": COLORS[1],
    "hadamard": COLORS[2],
    "subcount": COLORS[3],
    "subsample": COLORS[4],
    "CD (sketch size = 1)": COLORS[5],
    "cg": COLORS[6],
    "no_mom": COLORS[0],
    "cst_mom": COLORS[1],
    "inc_mom": COLORS[2],
    "no momentum": COLORS[0],
    "constant momentum": COLORS[1],
    "increasing momentum": COLORS[2],
}

LINE_DICT = {
    "count": LINE_STYLES[0],
    "gaussian": LINE_STYLES[1],
    "hadamard": LINE_STYLES[2],
    "subcount": LINE_STYLES[3],
    "subsample": LINE_STYLES[4],
    "CD (sketch size = 1)": LINE_STYLES[5],
    "cg": LINE_STYLES[1],
    "no_mom": LINE_STYLES[0],
    "cst_mom": LINE_STYLES[1],
    "inc_mom": LINE_STYLES[2],
    "no momentum": LINE_STYLES[0],
    "constant momentum": LINE_STYLES[1],
    "increasing momentum": LINE_STYLES[2],
}

SYMBOL_DICT = {
    "count": SYMBOLS[0],
    "gaussian": SYMBOLS[1],
    "hadamard": SYMBOLS[2],
    "subcount": SYMBOLS[3],
    "subsample": SYMBOLS[4],
    "CD (sketch size = 1)": SYMBOLS[5],
    "cg": SYMBOLS[6],
    "no_mom": SYMBOLS[0],
    "cst_mom": SYMBOLS[1],
    "inc_mom": SYMBOLS[2],
    "no momentum": SYMBOLS[0],
    "constant momentum": SYMBOLS[1],
    "increasing momentum": SYMBOLS[2],
}


def apply_style(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig = func(*args, **kwargs)
        fig.update_layout(
            template="plotly_white", font=dict(size=20,),
        )
        return fig

    return wrapper


@apply_style
def plot_iterations_v_epsilon(
    y_values, x_values, labels, line_type="line+markers", fig=None
):
    """Plots the number of iterations (y-axis) against epsilon (tolerance).

    Args:
        y_values (np.array): containing iterations needed for each tolerance.
            shape (number of lables, numb of x_values)
        x_values (np.array): 1-D array containing tolerances
        labels (list): list of strings containing the run names. len = len(x_values)
        line_type (str): style to use for plot. Options: "lines+markers" or "lines"
        fig (plotly figure): appends to figure if given, else creates a new figure
    """
    if fig is None:
        fig = go.Figure()
    for y_val, label in zip(y_values, labels):
        if "Momentum" in label:
            line_style = "dot"
            color = COLORS[0]
        else:
            line_style = "solid"
            color = COLORS[1]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_val,
                name=label,
                mode=line_type,
                line=dict(color=color,dash=line_style),
            )
        )
    fig.update_layout(
        # title="Iterations Required vs. Tolerance",
        xaxis_title="epsilon",
        yaxis_title="iterations required",
        xaxis_type="log",
        yaxis_type="log",
        legend=dict(yanchor="middle", xanchor="right", y=0.9, x=1),
    )
    return fig


def plot_residual_v_iteration(run_name, formatted_run_name, residuals_df, fig):
    """
    Adds a single trace to the existing figure for experiment 9.2, 9.3 and 9.4
    """
    y = residuals_df[f"{run_name} (median)"]
    y = y[~np.isnan(y)]  # removing pading nans

    n_iterations = len(y)
    iterations = list(range(n_iterations))

    x = iterations.copy()

    y_upper = residuals_df[f"{run_name} (3rd quartile)"].values
    y_lower = residuals_df[f"{run_name} (1st quartile)"].values

    n_max_points = 25
    if n_iterations > n_max_points:
        step = int(n_iterations / n_max_points)
        iter_to_plot = iterations[::step]
        iter_to_plot.append(iterations[-1])
        iter_to_plot = list(dict.fromkeys(iter_to_plot))

        x_to_plot = [x[i] for i in iter_to_plot]
        y_to_plot = y[iter_to_plot]

        y_upper_to_plot = y_upper[iter_to_plot]
        y_lower_to_plot = y_lower[iter_to_plot]
    else:
        x_to_plot = x
        y_to_plot = y

        y_upper_to_plot = y_upper
        y_lower_to_plot = y_lower

    style_key = formatted_run_name.split(" | ")[0]

    current_color = COLOR_DICT[style_key]
    alpha = 0.2  # opacity of the error colored area
    current_color_rgba = ImageColor.getcolor(current_color, "RGB") + (alpha,)

    fig.add_trace(
        go.Scatter(
            x=x_to_plot,
            y=y_to_plot,
            name=formatted_run_name,
            mode="lines+markers",
            line=dict(color=current_color, dash=LINE_DICT[style_key]),
            marker=dict(symbol=SYMBOL_DICT[style_key], size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_to_plot,
            y=y_upper_to_plot,
            mode="lines",
            line=dict(width=0),
            marker=dict(color=f"rgba{current_color_rgba}"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_to_plot,
            y=y_lower_to_plot,
            mode="lines",
            line=dict(width=0),
            marker=dict(color=f"rgba{current_color_rgba}"),
            fillcolor=f"rgba{current_color_rgba}",
            fill="tonexty",
            showlegend=False,
        )
    )

    return fig


@apply_style
def plot_runs_over_iterations(
    run_names, formatted_run_names, residuals_df,
):
    """Plots runs over iterations for experiment 9.2, 9.3 and 9.4"""
    fig = go.Figure()

    for run_name, formatted_run_name in zip(run_names, formatted_run_names):
        run_name.replace("residual_norms", "")
        if "direct" not in run_name:
            fig = plot_residual_v_iteration(
                run_name, formatted_run_name, residuals_df, fig
            )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="iteration",
        yaxis_title="relative residual norm",
        yaxis_type="log",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_time_v_residuals(
    run_name, formatted_run_name, times_df, residuals_df, fig
):
    """Adds a single trace to the existing figure for experiment 9.3 and 9.4"""
    y = residuals_df[f"{run_name} (median)"]
    y = y[~np.isnan(y)]  # removing pading nans

    n_iterations = len(y)
    iterations = list(range(n_iterations))

    solver = run_name.split(" | ")[1]
    total_time = times_df[times_df["solver"] == solver]["time (median)"].values[0]
    x = [i * total_time / n_iterations for i in range(n_iterations)]

    y_upper = residuals_df[f"{run_name} (3rd quartile)"].values
    y_lower = residuals_df[f"{run_name} (1st quartile)"].values

    n_max_points = 25
    if n_iterations > n_max_points:
        step = int(n_iterations / n_max_points)
        iter_to_plot = iterations[::step]
        iter_to_plot.append(iterations[-1])
        iter_to_plot = list(dict.fromkeys(iter_to_plot))

        x_to_plot = [x[i] for i in iter_to_plot]
        y_to_plot = y[iter_to_plot]

        y_upper_to_plot = y_upper[iter_to_plot]
        y_lower_to_plot = y_lower[iter_to_plot]
    else:
        x_to_plot = x
        y_to_plot = y

        y_upper_to_plot = y_upper
        y_lower_to_plot = y_lower

    current_color = COLOR_DICT[formatted_run_name]
    alpha = 0.2  # opacity of the error colored area
    current_color_rgba = ImageColor.getcolor(current_color, "RGB") + (alpha,)

    fig.add_trace(
        go.Scatter(
            x=x_to_plot,
            y=y_to_plot,
            name=formatted_run_name,
            mode="lines+markers",
            line=dict(color=current_color, dash=LINE_DICT[formatted_run_name]),
            marker=dict(symbol=SYMBOL_DICT[formatted_run_name], size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_to_plot,
            y=y_upper_to_plot,
            mode="lines",
            line=dict(width=0),
            marker=dict(color=f"rgba{current_color_rgba}"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_to_plot,
            y=y_lower_to_plot,
            mode="lines",
            line=dict(width=0),
            marker=dict(color=f"rgba{current_color_rgba}"),
            fillcolor=f"rgba{current_color_rgba}",
            fill="tonexty",
            showlegend=False,
        )
    )

    return fig


@apply_style
def plot_runs_over_time(
    run_names, formatted_run_names, times_df, residuals_df,
):
    """Plots runs over time for experiment 9.3 and 9.4"""
    fig = go.Figure()

    for run_name, formatted_run_name in zip(run_names, formatted_run_names):
        run_name.replace("residual_norms", "")
        if "direct" in run_name:
            solver = run_name.split(" | ")[1]
            total_time = times_df[times_df["solver"] == solver]["time (median)"].values[0]
            fig.add_vline(x=total_time, annotation_text="direct")
        else:
            fig = plot_time_v_residuals(
                run_name, formatted_run_name, times_df, residuals_df, fig
            )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="time (seconds)",
        yaxis_title="relative residual norm",
        yaxis_type="log",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


@apply_style
def plot_runs_over_time_sketch_size(
    run_names, formatted_run_names, times_df, residuals_df,
):
    """Plots runs over time for experiment 9.3 and 9.4"""
    fig = go.Figure()

    for run_name, formatted_run_name in zip(run_names, formatted_run_names):
        run_name.replace("residual_norms", "")
        if "direct" in run_name:
            solver = run_name.split(" | ")[1]
            total_time = times_df[times_df["solver"] == solver]["time (median)"].values[0]
            fig.add_vline(x=total_time, annotation_text="direct")
        else:
            # print(f"run_name, formatted_run_name: {run_name}, {formatted_run_name}")
            fig = plot_time_v_residuals(
                run_name, formatted_run_name, times_df, residuals_df, fig
            )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="time (seconds)",
        yaxis_title="relative residual norm",
        yaxis_type="log",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def bar_plot_times_momentum(times_df):
    """Plots runtimes for given problem"""
    df = times_df.copy()
    df["error (1st quartile)"] = df["time (median)"] - df["time (1st quartile)"]
    df["error (3rd quartile)"] = df["time (3rd quartile)"] - df["time (median)"]

    fig_no_mom = px.bar(
        df[df["run_name"] == "no_mom"],
        x="sketch_size",
        y="time (median)",
        # color=COLOR_DICT["no_mom"],
        barmode="group",
        error_y="error (3rd quartile)",
        error_y_minus="error (1st quartile)",
    )
    fig_no_mom.update_xaxes(type='category')
    fig_no_mom.update_traces(marker_color=COLOR_DICT["no_mom"])
    fig_no_mom.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        template="plotly_white",
        font=dict(size=20,),
        xaxis_title="sketch size",
        yaxis_title="time (seconds)",
    )

    fig_cst_mom = px.bar(
        df[df["run_name"] == "cst_mom"],
        x="sketch_size",
        y="time (median)",
        # color=COLOR_DICT["cst_mom"],
        barmode="group",
        error_y="error (3rd quartile)",
        error_y_minus="error (1st quartile)",
    )
    fig_cst_mom.update_xaxes(type='category')
    fig_cst_mom.update_traces(marker_color=COLOR_DICT["cst_mom"])
    fig_cst_mom.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        template="plotly_white",
        font=dict(size=20,),
        xaxis_title="sketch size",
        yaxis_title="time (seconds)",
    )

    fig_inc_mom = px.bar(
        df[df["run_name"] == "inc_mom"],
        x="sketch_size",
        y="time (median)",
        # color=COLOR_DICT["inc_mom"],
        barmode="group",
        error_y="error (3rd quartile)",
        error_y_minus="error (1st quartile)",
    )
    fig_inc_mom.update_xaxes(type="category")
    fig_inc_mom.update_traces(marker_color=COLOR_DICT["inc_mom"])
    fig_inc_mom.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        template="plotly_white",
        font=dict(size=20,),
        xaxis_title="sketch size",
        yaxis_title="time (seconds)",
    )

    # Grouped bar plot summarizing all results
    fig_grouped = go.Figure(data=[
        go.Bar(
            name="no momentum",
            x=df["sketch_size"],
            y=df[df["run_name"] == "no_mom"]["time (median)"],
            error_y=dict(
                array=df[df["run_name"] == "no_mom"]["error (3rd quartile)"],
                arrayminus=df[df["run_name"] == "no_mom"]["error (1st quartile)"]
            ),
            marker_color=COLOR_DICT["no_mom"],
        ),
        go.Bar(
            name="increasing momentum",
            x=df["sketch_size"],
            y=df[df["run_name"] == "inc_mom"]["time (median)"],
            error_y=dict(
                array=df[df["run_name"] == "inc_mom"]["error (3rd quartile)"],
                arrayminus=df[df["run_name"] == "inc_mom"]["error (1st quartile)"]
            ),
            marker_color=COLOR_DICT["inc_mom"],
        ),
        go.Bar(
            name="constant momentum",
            x=df["sketch_size"],
            y=df[df["run_name"] == "cst_mom"]["time (median)"],
            error_y=dict(
                array=df[df["run_name"] == "cst_mom"]["error (3rd quartile)"],
                arrayminus=df[df["run_name"] == "cst_mom"]["error (1st quartile)"]
            ),
            marker_color=COLOR_DICT["cst_mom"],
        ),
    ])

    # Change the bar mode
    fig_grouped.update_layout(barmode='group')

    # Legend position
    fig_grouped.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        template="plotly_white",
        font=dict(size=20,),
        xaxis_title="sketch size",
        yaxis_title="time (seconds)",
        legend=dict(
            yanchor="top",
            y=0.99,
            x=0.01,
            xanchor="left",
        )
    )

    return fig_no_mom, fig_cst_mom, fig_inc_mom, fig_grouped
