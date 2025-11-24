# -*- coding: utf-8 -*-
# Copyright (c) 2019-2025 Prasanna
# Created on Tue Aug 27 12:21:33 2019

# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
This module is the core of the H1B analysis package, providing a comprehensive
suite of functions for data analysis, visualization, and reporting.

It includes a wide range of functionalities, such as:
- Plotting functions for generating various charts (box, histogram, distribution, CDF, scatter, linear regression, wage levels, and trends).
- Descriptive statistics functions for summarizing H1B data.
- Employer-specific analysis tools for examining data by employer.
- Data transformation utilities for creating and manipulating pandas DataFrames.
- Helper functions for plot styling and formatting.
- Functions for conducting analysis over a range of years.
- Document generation capabilities for creating reports (e.g., Excel files with job titles and wage levels).

This module serves as the backbone for the other analysis modules in the package.
"""

import collections
import operator
import random
import warnings
from math import fmod

import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from pandas.plotting import table
from scipy import stats
from tqdm import tqdm_notebook as tqdm

# added for company name in brackets
warnings.filterwarnings("ignore", "This pattern has match groups")

# from numpy import histogram
header_1 = {
    "x": 0.5,
    "y": 0.5,
    "fontsize": 25,
    "color": "gray",
    "alpha": 0.5,
    "rotation": 0,
    "ha": "center",
    "va": "center",
}

footer_1 = {
    "x": -0.1,
    "y": -0.1,
    "s": "".join(["_" for _ in range(138)]),
    "alpha": 0.75,
    "fontsize": 12,
    "verticalalignment": "center",
}
# print(footer_1)

footer_2 = {
    "x": -0.1,
    "y": -0.15,
    "s": "                              Source: https://www.foreignlaborcert.doleta.gov/performancedata.cfm           ",
    "alpha": 0.75,
    "fontsize": 15,
    "verticalalignment": "center",
}


# print(footer_2)


def set_plot_style():
    """
    sets the plot style
    """
    plt.style.use("fivethirtyeight")
    # change the figure size of all the plots in this notebook
    rcParams["figure.figsize"] = (9.0, 6.0)
    rcParams.update({"font.size": 14})


#  Takes range of years, returns the plot height factor
def get_plot_height_factor(a, b):
    """

    @param a:
    @param b:
    @return:
    """
    j = 0
    for i in range(a, b + 1):
        if i == 2009:
            continue
        j = j + 1
    if fmod(j, 2):
        return j + 1

    return j


# noinspection PyUnusedLocal
# pylint: disable=W0613
def thousands(x, pos):
    """The two args are the value and tick position"""
    return f"${x * 1e-3:1.0f}K"


# previous name of this function was employer_name, it was changed to avoid outer sope warning
# many functions parameter name is emploer_name
def employer_name_check(df, return_flag=False, unique_flag=True):
    """

    @param df:
    @param return_flag:
    @param unique_flag:
    @return:
    """
    dfe = pd.DataFrame()
    if isinstance(unique_flag, bool) is True:
        if unique_flag:
            dfe = pd.DataFrame(
                {"EMPLOYER_NAME": df.filter(regex="NAME$").iloc[:, 0].unique().tolist()}
            )
        else:
            dfe = pd.DataFrame(
                {"EMPLOYER_NAME": df.filter(regex="NAME$").iloc[:, 0].tolist()}
            )
    else:
        print("unique_flag is a boolean, set it to False or True, by default it is set to True")
        return None

    if return_flag:
        return dfe["EMPLOYER_NAME"]

    display(dfe.T)
    return None


def employer_descriptive_stats(df):
    """

    @param df:
    @return:
    """
    df_c = employer_name_check(df, return_flag=True, unique_flag=False)
    num = dict(collections.Counter(df_c))
    sorted_num = sorted(num.items(), key=operator.itemgetter(1), reverse=True)
    return collections.OrderedDict(sorted_num)


def top_n_employers(
    df: pd.DataFrame, n: int, return_flag: bool = False
) -> pd.DataFrame:
    """

    @param df:
    @param n:
    @param return_flag:
    @return:
    """
    dc = employer_descriptive_stats(df)
    k = 0
    t = pd.DataFrame()
    for i, j in dc.items():
        if k < n:
            t1 = pd.DataFrame(
                [[i, j]], index=[k + 1], columns=["Employer_Name", "Count"]
            )
            t = pd.concat([t, t1])
        k = k + 1
    t = pd.concat([t.T], keys=["TOP" + " " + str(n)]).T

    if return_flag:
        return t
    display(t)
    return pd.DataFrame()  # return empty


def descriptive_stats(pw, lw, **kwargs):
    """
    Descriptive Statistics
    ____
    Parameters
    ____
        Prevailing wages

        LCA wages

    Other Parameters
    ____
        **kwargs
            filter_type = 'Filtered'

            default: 'Raw' - in case no parameter is explicitly passed
    Returns
    ____
        count, mean, standard deviation, min, 25%, 50%(median), 75%, max
    """
    filter_type = kwargs.get("filter_type", "Raw")
    return_dfs = kwargs.get("return_df", False)
    year = kwargs.get(
        "year",
    )
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )

    plw_stat_df = pd.DataFrame(
        {
            "PREVAILING_WAGE": pw.describe().round().apply(lambda x: f"{x:,}"),
            "LCA_WAGE": lw.describe().round().apply(lambda x: f"{x:,}"),
        }
    )
    plw_stat_df_t = pd.concat(
        [plw_stat_df.T],
        keys=["DESCRIPTIVE STATISTICS " + str(year) + " (" + str(filter_type) + ") "],
    ).T

    skew_kurt_df = pd.DataFrame(
        [lw.skew(), lw.kurt()],
        index=["Skewness", "Kurtosis"],
        columns=["LCA_WAGE" + " " + str(year) + " (" + str(filter_type) + ")"],
    ).round(2)

    if return_dfs:
        return plw_stat_df_t, skew_kurt_df

    if ax is None:
        display(plw_stat_df_t)
        display(skew_kurt_df)

    else:
        ax.text(
            0.5,
            0.5,
            str(emp_name) + "_" + str(year),
            fontsize=30,
            color="gray",
            alpha=0.5,
            rotation=0,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            1,
            str(emp_name) + "_" + str(year),
            fontsize=30,
            bbox={"facecolor": "skyblue", "alpha": 0.5},
            fontfamily="sans-serif",
        )
        ax.text(
            0.0,
            0.4,
            str(plw_stat_df_t),
            fontsize="large",
            bbox={"facecolor": "skyblue", "alpha": 0.5},
            fontfamily="monospace",
        )
        ax.text(
            0.2,
            0.2,
            str(skew_kurt_df),
            fontsize="large",
            bbox={"facecolor": "skyblue", "alpha": 0.5},
            fontfamily="monospace",
        )
        ax.text(
            **footer_1,
            transform=ax.transAxes,
        )
        ax.text(
            **footer_2,
            transform=ax.transAxes,
        )
        ax.grid(False)
        ax.axis(False)
    return None


def plot_box(pw, lw, **kwargs):
    """
    Box plot
    ____
    Parameters
    ____
        DataFrame of Prevailing and LCA wages
    Returns
    ____
        Box plot of Prevailing and LCA wages
    """
    wage = pd.DataFrame({"PREVAILING_WAGE": pw, "LCA_WAGE": lw})

    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    formatter = FuncFormatter(thousands)

    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True

    ax.yaxis.set_major_formatter(formatter)
    sns.boxplot(data=wage, orient="v", palette=["khaki", "skyblue"])
    plt.title(
        "Box Plot - Prevailing and LCA Wages "
        + "("
        + filter_type
        + "_"
        + str(year)
        + ")"
    )
    plt.xlabel("Wage")
    plt.ylabel("Wage Range")
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    if enable_plt_show:
        plt.show()


def plot_hist(wage, bins, **kwargs):
    """
    Histogram Plot
    ____
    Parameters
    ____
        Wages

        Bins
    Returns
    ____
        Histogram Plot
    """
    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    formatter = FuncFormatter(thousands)
    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True
    ax.xaxis.set_major_formatter(formatter)
    plt.title(
        "Histogram - " + str(wage.name) + "(" + filter_type + "_" + str(year) + ")"
    )
    plt.xlim(left=0, right=1.1 * wage.max())
    if str(wage.name) == "LCA_WAGE":
        plt.hist(
            wage,
            bins=bins,
            edgecolor="white",
            facecolor="slateblue",
            rwidth=0.95,
            cumulative=False,
            label=str(wage.name),
            density=False,
        )
    elif str(wage.name) == "PREVAILING_WAGE":
        plt.hist(
            wage,
            bins=bins,
            edgecolor="white",
            facecolor="steelblue",
            rwidth=0.95,
            cumulative=False,
            label=str(wage.name),
            density=False,
        )
    else:
        plt.hist(
            wage,
            bins=bins,
            edgecolor="white",
            facecolor="slateblue",
            rwidth=0.95,
            cumulative=False,
            label=str(wage.name),
            density=False,
        )
    # the below line code is giving deprecated warning in Matplotlib 3.2 so commented out and added above new code for histogram.
    # wage.plot(kind = 'hist', bins = bins, edgecolor = 'white', rwidth = 0.95)
    plt.xlabel(wage.name)
    plt.legend(loc="best")
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    if enable_plt_show:
        plt.show()


# noinspection PyUnusedLocal
def plot_dist(wage, bins, **kwargs):
    """
    Distribution Plot
    ____
    Parameters
    ____
        Wages
    Returns
    ____
        Distribution Plot
    """
    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    # fit_type = kwargs.get('fit', stats.norm)

    formatter = FuncFormatter(thousands)

    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True

    ax.xaxis.set_major_formatter(formatter)
    # Normal Distribution of Wages
    plt.title(
        "Distribution Plot - "
        + str(wage.name)
        + "("
        + filter_type
        + "_"
        + str(year)
        + ")"
    )

    # generated 1000 norm random variables with mean and std from wage, this is for norm
    nrvs = stats.norm.rvs(size=1000, loc=wage.mean(), scale=wage.std())

    if str(wage.name) == "LCA_WAGE":
        sns.lineplot(
            x=sorted(nrvs),
            y=stats.norm.pdf(sorted(nrvs), loc=nrvs.mean(), scale=nrvs.std()),
            ax=ax,
            label="Normal Distribution",
            color="blue",
        )
        sns.kdeplot(
            x=wage,
            cumulative=False,
            bw_method="scott",
            ax=ax,
            label="Kernel Density Estimate",
            color="orange",
        )
        # Below sns displot is deprecated and will be removed in future version by Seaborn > 0.11.0, so changed the code to plt.hist()
        # sns.distplot(wage, bins, kde_kws = {'color': 'orange', 'label': 'Kernel Density Estimate'}, \
        #     fit = fit_type, hist = False,
        #     fit_kws ={'color' : 'blue', 'label' : str(fit_type.name).capitalize()  +  ' Distribution'},
        #     hist_kws ={'color': 'skyblue'})
    elif str(wage.name) == "PREVAILING_WAGE":
        sns.lineplot(
            x=sorted(nrvs),
            y=stats.norm.pdf(sorted(nrvs), loc=nrvs.mean(), scale=nrvs.std()),
            ax=ax,
            label="Normal Distribution",
            color="black",
        )
        sns.kdeplot(
            x=wage,
            cumulative=False,
            bw_method="scott",
            ax=ax,
            label="Kernel Density Estimate",
            color="olive",
        )
        # sns.distplot(wage, bins, kde_kws = {'color': 'olive', 'label': 'Kernel Density Estimate'},
        # fit = fit_type, hist = False, fit_kws ={'color' : 'black', 'label' : str(fit_type.name).capitalize() +  ' Distribution'},
        # hist_kws ={'color': 'red'})
    else:
        sns.lineplot(
            x=sorted(nrvs),
            y=stats.norm.pdf(sorted(nrvs), loc=nrvs.mean(), scale=nrvs.std()),
            ax=ax,
            label="Normal Distribution",
            color="blue",
        )
        sns.kdeplot(
            x=wage,
            cumulative=False,
            bw_method="scott",
            ax=ax,
            label="Kernel Density Estimate",
            color="orange",
        )
        # sns.distplot(wage, bins, kde_kws = {'color': 'orange', 'label': 'Kernel Density Estimate'},
        # fit = fit_type, hist = False,
        # fit_kws ={'color' : 'blue', 'label' : str(fit_type.name).capitalize() +  ' Distribution'},
        # hist_kws ={'color': 'skyblue'})

    plt.xlabel(wage.name)
    plt.ylabel("PDF")
    plt.xlim(left=0, right=1.1 * wage.max())
    plt.ylim(bottom=0)
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.legend(loc="best")
    plt.tight_layout()
    if enable_plt_show:
        plt.show()


def plot_cdf(wage, bins, **kwargs):
    """
    Cumulative Distribution Plot
    ____
    Parameters
    ____
        Wages

        Bins
    Returns
    ____
        Cumulative Distribution Plot
    """
    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    # fit_type = kwargs.get('fit', stats.norm)

    formatter = FuncFormatter(thousands)

    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True

    ax.xaxis.set_major_formatter(formatter)
    # Normal Distribution of Wages
    plt.title(
        "Cumulative Distribution Function - "
        + str("Wages")
        + "("
        + filter_type
        + "_"
        + str(year)
        + ")"
    )

    if str(wage.name) == "LCA_WAGE":
        plt.hist(
            wage,
            bins=bins,
            edgecolor="white",
            facecolor="slateblue",
            rwidth=0.95,
            cumulative=True,
            label=str(wage.name),
            density=True,
        )
        # Below sns displot is deprecated and will be removed in future version by Seaborn > 0.11.0, so changed the code to plt.hist()
        # sns.distplot(wage, bins, kde = False, \
        #              fit = fit_type,hist_kws = {'cumulative' : True, 'label' : str(wage.name), 'edgecolor':'white', 'facecolor' : 'blue', 'rwidth' : 0.95}, \
        #              kde_kws = {'cumulative': True, 'color' :'blue'})
    elif str(wage.name) == "PREVAILING_WAGE":
        plt.hist(
            wage,
            bins=bins,
            edgecolor="white",
            facecolor="steelblue",
            rwidth=0.95,
            cumulative=True,
            label=str(wage.name),
            density=True,
        )
        # sns.distplot(wage, bins, kde = False, \
        #              fit = fit_type, hist_kws = {'cumulative' : True, 'label' : str(wage.name), 'edgecolor':'white', 'facecolor' : 'navy', 'rwidth' : 0.95}, \
        #              kde_kws = {'cumulative': True, 'color': 'blue'})
    else:
        plt.hist(
            wage,
            bins=bins,
            edgecolor="white",
            facecolor="slateblue",
            rwidth=0.95,
            cumulative=True,
            label=str(wage.name),
            density=True,
        )
        # sns.distplot(wage, bins, kde = False,\
        #              fit = fit_type, hist_kws = {'cumulative' : True, 'label' : str(wage.name), 'edgecolor':'white', 'facecolor' : 'blue', 'rwidth': 0.95}, \
        #              de_kws = {'cumulative': True})

    plt.xlabel("Wage")
    plt.ylabel("CDF")
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1.0)
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.legend(loc="best")
    plt.margins(0.1)
    plt.tight_layout()
    if enable_plt_show:
        plt.margins(0.1)
        plt.show()


def plot_scatter(pw, lw, **kwargs):
    """
    Scatter Plot
    ____
    Parameters
    ____
        Prevailing wages

        LCA wages
    Returns
    ____
        Scatter Plot
    """
    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    formatter = FuncFormatter(thousands)
    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    scatter = ax.scatter(pw, lw, c=lw, cmap="viridis", alpha=0.2)
    cbar = plt.colorbar(scatter)
    cbar.set_label("LCA Wages")
    # plt.scatter(pw, lw)
    plt.title(
        "Scatter Plot - LCA Wages vs Prevailing "
        + "("
        + filter_type
        + "_"
        + str(year)
        + ")"
    )
    plt.xlabel(pw.name)
    plt.ylabel(lw.name)
    plt.xlim(left=0, right=1.1 * pw.max())
    plt.ylim(bottom=0, top=1.1 * lw.max())
    #    plt.tight_layout()
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    if enable_plt_show:
        plt.show()


def plot_linear_r(pw, lw, **kwargs):
    """
    Linear Regression
    ____
    Parameters
    ____
        Prevailing wages

        LCA wages
    Returns
    ____
        Plot with metrics slope, intercept, rvalue, pvalue, stderr
    """
    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    formatter = FuncFormatter(thousands)
    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True

    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    # Linear Regression
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pw, lw)
    if enable_plt_show:
        print(
            f"slope: {slope:f}, intercept: {intercept:f}, rvalue: {rvalue:f}, pvalue: {pvalue:f}, stderr: {stderr:f}"
        )
    plt.title(
        "Linear Regression - LCA Wages vs Prevailing"
        + "("
        + filter_type
        + "_"
        + str(year)
        + ")"
    )
    plt.scatter(pw, lw)
    plt.plot(
        pw,
        intercept + slope * pw,
        color="orange",
        label=f"slope: {slope:.3f} intercept: {intercept:.2f} correlation coefficient: {rvalue:.4f} stder: {stderr:.4f}",
    )
    plt.xlabel(pw.name)
    plt.xlim(left=0, right=1.1 * pw.max())
    plt.ylabel(lw.name)
    plt.ylim(bottom=0, top=1.1 * lw.max())
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    if enable_plt_show:
        plt.show()


def hcdf(x, pw, lw):
    """

    @param x:
    @param pw:
    @param lw:
    """
    v = stats.norm.cdf(x, lw.mean(), lw.std())
    print(f"Cummulative probability for {str(x)} is {str(v)}")
    if v <= 0.875:
        print("Potential issue")
    else:
        print("May not be an issue")
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(lw, pw)
    print(
        f"slope: {slope:f}, intercept: {intercept:f}, rvalue: {rvalue:f}, pvalue: {pvalue:f}, stderr: {stderr:f}"
    )
    print(f"Estimated Prevailing Wage: {intercept + slope * x:f}")

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pw, lw)
    print(
        f"slope: {slope:f}, intercept: {intercept:f}, rvalue: {rvalue:f}, pvalue: {pvalue:f}, stderr: {stderr:f}"
    )
    print(f"Estimated LCA Wage: {intercept + slope * x:f}")


def plot_wage_levels(wlevel, **kwargs):
    """
    Wage Plot
    ____
    Parameters
    ____
        Wage levels
    Returns
    ____
        Bar plot of wage levels
        DataFrame of sorted prevailing wage level counts
    """
    enable_plt_show = False
    filter_type = kwargs.get("filter_type", "Raw")
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    year = kwargs.get(
        "year",
    )
    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True

    wlevel_t = wlevel.copy()
    df_wl = pd.DataFrame(wlevel_t.value_counts())
    df_wl.sort_index(inplace=True)

    plt.title(f"Number of H1Bs vs Wage Levels({filter_type}_{str(year)})")
    plt.bar(df_wl.index, df_wl["count"].values, label="counts: " + str(wlevel.count()))
    plt.xlabel("Wage Level")
    plt.ylabel("Number of H1Bs")
    plt.text(
        **header_1,
        s=str(emp_name) + "_" + str(year),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    if enable_plt_show:
        plt.show()
    df_wl_1 = df_wl.reset_index().rename(
        columns={"index": "WAGE_LEVEL", "PW_WAGE_LEVEL": "COUNT"}
    )
    df_wl_t = pd.concat([df_wl_1.T], keys=["(" + str(filter_type) + ") "]).T
    return df_wl_t


def plot_wage_trends(t_mean, t_median, **kwargs):
    """
    Plot of wage trend
    ____
    Parameters
    ____
        mean and median wages with respect to years
    Returns
    ____
        wage trend
    """
    #    mean_label = kwargs.get('mean_label', '')
    #    median_label = kwargs.get('median_label', '')

    enable_plt_show = False

    wage_t = kwargs.get(
        "wage_type",
    )
    formatter = FuncFormatter(thousands)
    ax = kwargs.get(
        "ax",
    )
    emp_name = kwargs.get(
        "employer_name",
    )
    if ax is None:
        set_plot_style()
        _, ax = plt.subplots()
        enable_plt_show = True

    ax.yaxis.set_major_formatter(formatter)
    # ymin = min(t_mean.min(), t_median.min()) - 1000
    # ymax = max(t_mean.max(), t_median.max()) + 5000
    # plt.ylim(ymin, ymax)
    plt.title("Trends - Mean and Median " + str(wage_t))
    plt.plot(t_mean, label="mean " + str(wage_t))
    plt.plot(t_median, label="median " + str(wage_t))
    plt.legend(loc="upper left", fontsize="medium")
    plt.xlabel("Year")
    plt.ylabel("Mean and Median " + str(wage_t))
    plt.text(
        **header_1,
        s=str(emp_name),
        transform=ax.transAxes,
    )
    plt.text(
        **footer_1,
        transform=ax.transAxes,
    )
    plt.text(
        **footer_2,
        transform=ax.transAxes,
    )
    plt.tight_layout()

    if enable_plt_show:
        plt.show()


def analysis_range(df_ar, m, n, **kwargs):
    """

    @param df_ar:
    @param m:
    @param n:
    @param kwargs:
    @return:
    """
    #    df = dfa.copy()
    set_plot_style()
    bins = kwargs.get("bins", 50)
    dtype = kwargs.get("filter_type", "")
    #    year = kwargs.get('year', str(nan))
    #    pw_range = kwargs.get('pw_range', str(nan))
    #    lw_range = kwargs.get('lw_range', str(nan))
    emp_name = kwargs.get("emp_name", "ALL")

    plw_s_df = pd.DataFrame()
    sk_df = pd.DataFrame()
    print_once = False
    subplot_count = 1

    df_a = df_ar[
        (df_ar.YEAR.between(m, n))
        & (df_ar.EMPLOYER_NAME.str.contains(emp_name, case=False))
    ]

    df_f = df_a.copy()
    pw = df_a["PREVAILING_WAGE"]
    lw = df_a["LCA_WAGE"]
    en = employer_name_check(df_a, return_flag=True)

    task_dict = {
        descriptive_stats: [[pw, lw]],
        plot_box: [[pw, lw]],
        plot_hist: [[lw, bins]],
        plot_dist: [[lw, bins]],
        plot_cdf: [[lw, bins]],
        plot_scatter: [[pw, lw]],
        plot_linear_r: [[pw, lw]],
    }

    t_len = len(task_dict.values())
    t_count = 1

    with tqdm(
        total=7,
        desc="Overall Status",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:
        for task in task_dict:
            plt.figure(figsize=(26, 48))
            subplot_count = 1
            #        clear_output(wait = True)
            for j in range(m, n + 1):

                if j == 2009:
                    continue
                if task in (stats, plot_box, plot_scatter, plot_linear_r):
                    task_dict[task][0][0] = df_f[df_f.YEAR == j].PREVAILING_WAGE
                    task_dict[task][0][1] = df_f[df_f.YEAR == j].LCA_WAGE
                elif task in (plot_hist, plot_dist, plot_cdf):
                    task_dict[task][0][0] = df_f[df_f.YEAR == j].LCA_WAGE
                else:
                    display("Task schedular has an issue with arguments")

                for i in range(0, len(task_dict[task])):
                    if task == descriptive_stats:
                        (plw_stat_df, skew_kurt_df) = task(
                            *task_dict[task][i],
                            filter_type=dtype,
                            return_df=True,
                            employer_name=en,
                            year=j,
                        )
                        plw_s_df = pd.concat(
                            [plw_s_df, plw_stat_df], sort=False, axis=1
                        )
                        sk_df = pd.concat([sk_df, skew_kurt_df], sort=False, axis=1)
                    else:
                        if not print_once:
                            print_once = True
                            tcol_name = plw_s_df.columns.tolist()
                            if len(tcol_name) % 2:
                                x = len(tcol_name) + 1
                            else:
                                x = len(tcol_name)
                            if x > 10:
                                if x % 10:
                                    y = int((x / 10) + 1)
                                else:
                                    y = int(x / 10)
                                c = 10
                            else:
                                y = 1
                                c = x
                            k = 0
                            for _ in range(0, y):
                                display(plw_s_df.iloc[0:, range(k, c)])
                                k = c
                                if (2 * c) < len(tcol_name):
                                    c = 2 * c
                                else:
                                    c = len(tcol_name)
                            display(sk_df)

                        axs = plt.subplot(6, 2, subplot_count)
                        task(
                            *task_dict[task][i],
                            filter_type=dtype + "-" + str(j),
                            ax=axs,
                            employer_name=en,
                        )
                        subplot_count += 1
            print(f"Status: {t_count * 100 / t_len:.2f}" + str("%") + " completed")
            t_count += 1
            progress_bar.update(1)
            plt.show()

    if df_f.columns.str.contains("^PW_WAGE_LEVEL$").any():
        if not df_f.PW_WAGE_LEVEL.isna().all():
            axs = plt.subplot(6, 2, subplot_count)
            plot_wage_levels(df_f.PW_WAGE_LEVEL, filter_type=dtype, ax=axs)
    plt.show()
    return df_f


def create_wage_range_d_f(df, year):
    """

    @param df:
    @param year:
    @return:
    """
    if (df.columns.str.contains("^PREVAILING_WAGE_1$").any()) and (
        df.columns.str.contains("^WAGE_RATE_1$").any()
    ):
        pw = df.PREVAILING_WAGE_1
        lw = df.WAGE_RATE_1
    elif (df.columns.str.contains("^PW_1$").any()) and (
        df.columns.str.contains("^LCA_CASE_WAGE_RATE_FROM$").any()
    ):
        pw = df.PW_1
        lw = df.LCA_CASE_WAGE_RATE_FROM
    elif (df.columns.str.contains("PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^WAGE_RATE_OF_PAY$").any()
    ):
        pw = df.PREVAILING_WAGE
        lw = df.WAGE_RATE_OF_PAY
    elif (df.columns.str.contains("^PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^WAGE_RATE_OF_PAY_FROM$").any()
    ):
        pw = df.PREVAILING_WAGE
        lw = df.WAGE_RATE_OF_PAY_FROM
    else:
        return print("Error")

    return pd.DataFrame(
        [
            [
                str(year),
                lw.name,
                (lw.min(), lw.max()),
                pw.name,
                (pw.min(), pw.max()),
                "",
                "",
            ]
        ],
        columns=[
            "YEAR",
            "LCA_WAGE",
            "LCA_WAGE_RANGE",
            "PREVAILING_WAGE",
            "PREVAILING_WAGE_RANGE",
            "LCA_WAGE_RANGE_A",
            "PREVAILING_WAGE_RANGE_A",
        ],
    )


def create_wage_trends_d_f(df, year):
    """

    @param df:
    @param year:
    @return:
    """
    if (df.columns.str.contains("^PREVAILING_WAGE_1$").any()) and (
        df.columns.str.contains("^WAGE_RATE_1$").any()
    ):
        pw = df.PREVAILING_WAGE_1
        lw = df.WAGE_RATE_1
    elif (df.columns.str.contains("^PW_1$").any()) and (
        df.columns.str.contains("^LCA_CASE_WAGE_RATE_FROM$").any()
    ):
        pw = df.PW_1
        lw = df.LCA_CASE_WAGE_RATE_FROM
    elif (df.columns.str.contains("PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^WAGE_RATE_OF_PAY$").any()
    ):
        pw = df.PREVAILING_WAGE
        lw = df.WAGE_RATE_OF_PAY
    elif (df.columns.str.contains("^PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^WAGE_RATE_OF_PAY_FROM$").any()
    ):
        pw = df.PREVAILING_WAGE
        lw = df.WAGE_RATE_OF_PAY_FROM
    elif (df.columns.str.contains("^PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^LCA_WAGE$").any()
    ):
        pw = df.PREVAILING_WAGE
        lw = df.LCA_WAGE
    else:
        return print("Error")

    pds = pd.DataFrame({str(year): pw.describe()})
    lws = pd.DataFrame({str(year): lw.describe()})
    return pds, lws


def wages_d_f(df, i):
    """

    @param df:
    @param i:
    @return:
    """
    if (df.columns.str.contains("^PREVAILING_WAGE_1$").any()) and (
        df.columns.str.contains("^WAGE_RATE_1$").any()
    ):
        pw = "PREVAILING_WAGE_1"
        lw = "WAGE_RATE_1"
    elif (df.columns.str.contains("^PW_1$").any()) and (
        df.columns.str.contains("^LCA_CASE_WAGE_RATE_FROM$").any()
    ):
        pw = "PW_1"
        lw = "LCA_CASE_WAGE_RATE_FROM"
    elif (df.columns.str.contains("PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^WAGE_RATE_OF_PAY$").any()
    ):
        pw = "PREVAILING_WAGE"
        lw = "WAGE_RATE_OF_PAY"
    elif (df.columns.str.contains("^PREVAILING_WAGE$").any()) and (
        df.columns.str.contains("^WAGE_RATE_OF_PAY_FROM$").any()
    ):
        pw = "PREVAILING_WAGE"
        lw = "WAGE_RATE_OF_PAY_FROM"
    else:
        return print("Error")

    en = df.columns[df.columns.str.contains("NAME$")][0]

    if df.columns.str.contains("^PW_WAGE_LEVEL$").any():
        pw_level = df.columns[df.columns.str.contains("^PW_WAGE_LEVEL$")][0]
        pw_level_present = True
    else:
        pw_level_present = False
        pw_level = None

    if not pw_level_present:
        dfr = df.filter(items=[en, pw, lw])
    else:
        dfr = df.filter(items=[en, pw, lw, pw_level])

    dfr.index.rename("Year", inplace=True)
    dfr = dfr.reset_index()
    dfr.Year = i
    dfr = dfr.rename(
        columns={pw: "PREVAILING_WAGE", lw: "LCA_WAGE", en: "EMPLOYER_NAME"}
    )

    return dfr


def generate_doc_job_title_wage_level(df, **kwargs):
    """
    Generates xlsx document of wage levels and 'unique' job titles corresponding to wage levels
    Important: if filename suffix 'fns' is not passed as a parameter, a random suffix is generated
    ____
    Parameters
    ____
        dataframe
        **kwargs
            fns : 'string' used as suffix for filename
    Returns
    ____
        Generates xlsx document in the directory of the source file from which this function was called
    """
    suffix = random.randint(1, 1000000)
    fns = kwargs.get("fns", str(suffix))
    set_plot_style()
    if (
        df.columns.str.contains("^JOB_TITLE").any()
        and df.columns.str.contains("^PW_WAGE_LEVEL$").any()
    ):
        dfjp = df.filter(items=["JOB_TITLE", "PW_WAGE_LEVEL"]).copy()
        dfjp.JOB_TITLE.fillna("Job Title NA", inplace=True)
        dfjp.PW_WAGE_LEVEL.fillna("Level NA", inplace=True)
        pwl = dict(collections.Counter(dfjp.PW_WAGE_LEVEL))
        pwo = collections.OrderedDict(sorted(pwl.items()))
        xm = {}
        ym = {}
        for key in pwo:
            ym.update({key: pwo[key]})
            xm.update(
                {key: dfjp[dfjp.PW_WAGE_LEVEL == key].JOB_TITLE.unique().tolist()}
            )
        ya = pd.DataFrame.from_dict(ym, orient="index").T
        xa = pd.DataFrame.from_dict(xm, orient="index").T
        ya.append(xa).to_excel("JBTPWL-" + str(fns) + ".xlsx")
        print("The file " + "JBTPWL-" + str(fns) + ".xlsx" + " is generated")
    else:
        print("Either Job Title or Prevailing Wage Level not present\n")


# noinspection PyShadowingNames
def emp_stats_wage_range_display(df, employer_name, pw_range, lw_range, **kwargs):
    """

    @param df:
    @param employer_name:
    @param pw_range:
    @param lw_range:
    @param kwargs:
    @return:
    """
    gen_pdf = kwargs.get("generate_pdf", False)

    df_range1 = pd.DataFrame.from_dict(
        pw_range, orient="index", columns=["PW_LOWER_LIMIT", "PW_UPPER_LIMIT"]
    )
    df_range1["PW_LOWER_LIMIT"] = df_range1.PW_LOWER_LIMIT.apply(lambda x: f"${x:,}")
    df_range1["PW_UPPER_LIMIT"] = df_range1.PW_UPPER_LIMIT.apply(lambda x: f"${x:,}")
    df_range2 = pd.DataFrame.from_dict(
        lw_range, orient="index", columns=["LCA_LOWER_LIMIT", "LCA_UPPER_LIMIT"]
    )
    df_range2["LCA_LOWER_LIMIT"] = df_range2.LCA_LOWER_LIMIT.apply(lambda x: f"${x:,}")
    df_range2["LCA_UPPER_LIMIT"] = df_range2.LCA_UPPER_LIMIT.apply(lambda x: f"${x:,}")
    df_range = (
        pd.concat([df_range1, df_range2], sort=False, axis=1)
        .reset_index()
        .rename(columns={"index": "YEAR"})
    )

    es = dict(employer_descriptive_stats(df))
    df_es = (
        pd.DataFrame.from_dict(es, orient="index", columns=["COUNT"])
        .reset_index()
        .rename(columns={"index": "EMPLOYER_NAME"})
    )
    df_es["COUNT"] = df_es.COUNT.apply(lambda x: f"{x:,}")

    fig = plt.figure(figsize=(24, 12))
    fig.text(
        0.5,
        0.95,
        employer_name,
        fontsize=30,
        color="gray",
        alpha=0.5,
        rotation=0,
        ha="center",
    )

    ax = fig.add_subplot(1, 1, 1)

    ax.set_title("Total H1B employees count and wage ranges selected for analysis")
    table(ax=ax, data=df_es, bbox=[0.2, 0.85, 0.5, 0.1], cellLoc="center")
    table(ax=ax, data=df_range, bbox=[0.2, 0.2, 0.5, 0.5], cellLoc="center")
    # ax.text(0.5, 0.5, employer_name, fontsize=20, color='gray', alpha=0.5, rotation=0,
    #         ha='center', va='center', transform=ax.transAxes)
    # ax.text(0.0, 0.5, str(df_range), fontsize='medium', fontfamily='monospace', transform=ax.transAxes)
    ax.grid(False)
    ax.axis(False)
    if not gen_pdf:
        plt.savefig(employer_name + "_" + "emp_stats_wage_range_display.png")
    return fig


def wage_range_based_on_quartile(df, **kwargs):
    """
    Calculates wage range based on first and third quartile values

        Wage Low Limit = 0.5 * first quartile of wage values
        Wage Hight Limit = 1.5 * third quartile of wage values

    Input: DataFrame and optional keyword argument 'csv'
        if 'csv' is set to True, returns wages formatted with $ and 1000 separator. Default: False

    Returns: DataFrame containing following columns:
        YEAR, EMPLOYER_NAME, COUNT_BEFORE_RANGE,
        WR_LOW_PREV, WR_HIGH_PREV, WR_LOW_LCA, WR_HIGH_LCA,
        COUNT_AFTER_RANGE, COUNT_DIFF
    """
    csv_flag = kwargs.get("csv", False)
    df_return = pd.DataFrame()
    for year in df.YEAR.unique():
        for emp in df.EMPLOYER_NAME.unique():
            mask1 = df.EMPLOYER_NAME == emp
            mask2 = df.YEAR == year

            dfm = df[mask1 & mask2]

            r_low_p = int(0.5 * dfm.quantile(q=0.25)["PREVAILING_WAGE"])
            r_high_p = int(1.5 * dfm.quantile(q=0.75)["PREVAILING_WAGE"])
            r_low_l = int(0.5 * dfm.quantile(q=0.25)["LCA_WAGE"])
            r_high_l = int(1.5 * dfm.quantile(q=0.75)["LCA_WAGE"])

            dff = dfm[
                dfm.PREVAILING_WAGE.between(r_low_p, r_high_p)
                & dfm.LCA_WAGE.between(r_low_l, r_high_l)
            ]
            count_diff = dfm.index.size - dff.index.size

            # if df_return is used for storing to .csv file else for further analysis
            if csv_flag:
                dict_df = {
                    "YEAR": [year],
                    "EMPLOYER_NAME": [emp],
                    "COUNT_BEFORE_RANGE": [dfm.index.size],
                    "WR_LOW_PREV": [f"${r_low_p:,}"],
                    "WR_HIGH_PREV": [f"${r_high_p:,}"],
                    "WR_LOW_LCA": [f"${r_low_l:}"],
                    "WR_HIGH_LCA": [f"${r_high_l:,}"],
                    "COUNT_AFTER_RANGE": [dff.index.size],
                    "COUNT_DIFF": [count_diff],
                }
            else:
                dict_df = {
                    "YEAR": [year],
                    "EMPLOYER_NAME": [emp],
                    "COUNT_BEFORE_RANGE": [dfm.index.size],
                    "WR_LOW_PREV": [r_low_p],
                    "WR_HIGH_PREV": [r_high_p],
                    "WR_LOW_LCA": [r_low_l],
                    "WR_HIGH_LCA": [r_high_l],
                    "COUNT_AFTER_RANGE": [dff.index.size],
                    "COUNT_DIFF": [count_diff],
                }

            df_return = pd.concat([df_return, pd.DataFrame(dict_df)])

    return df_return.reset_index(drop=True)
