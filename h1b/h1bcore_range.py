# Copyright (c) 2019-2025 Prasanna
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
This module extends the core H1B analysis functionalities to support analysis over a range of years.

It provides a suite of functions for performing in-depth, longitudinal analysis of H1B data. 
The functions in this module, such as 'descriptive_stats2', 'box2', 'scatter2', 'hist2', 'dist2', 
'cdf2', 'linear_r2', 'wage_level2', and 'wage_trends2', are designed to work with data 
spanning multiple years. They generate plots for each year in a specified range and combine them 
for comparative analysis. The module also supports data filtering based on wage ranges.
"""

from math import ceil

import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from numpy import nan
from scipy import stats
from tqdm.notebook import tqdm

from h1b.h1bcore import (
    create_wage_trends_d_f,
    descriptive_stats,
    footer_1,
    footer_2,
    get_plot_height_factor,
    plot_box,
    plot_cdf,
    plot_dist,
    plot_hist,
    plot_linear_r,
    plot_scatter,
    plot_wage_levels,
    plot_wage_trends,
    set_plot_style,
)

# from h1b_analysis import *


# noinspection PyShadowingNames
def descriptive_stats2(df, employer_name, m, n, **kwargs):
    """
    Descriptive Statistics of wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary in the format: {year:(lower_limit, higher_limit)}
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary in the format {year:(lower_limit, higher_limit)}
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        displays the descriptive statistics of wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing"
            "and LCA wages needs to be passed as arguments when"
            "'wage_range_dict_flag is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(ceil(width_factor / 3))
    sp_columns = 3
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Box_Plot_'+str(m)+'_'+str(n), fontsize=20)
    if m == 2008:
        if m + 2 != n:
            fig.text(
                0.5,
                0.95,
                employer_name
                + " H1B Data analysis "
                + str(m)
                + ", "
                + str(m + 2)
                + "-"
                + str(n),
                fontsize=30,
                color="gray",
                alpha=0.5,
                rotation=0,
                ha="center",
            )
        else:
            fig.text(
                0.5,
                0.95,
                employer_name + " H1B Data analysis " + str(m) + ", " + str(n),
                fontsize=25,
                color="gray",
                alpha=0.5,
                rotation=0,
                ha="center",
            )
    else:
        fig.text(
            0.5,
            0.95,
            employer_name + " H1B Data analysis " + str(m) + "-" + str(n),
            fontsize=30,
            color="gray",
            alpha=0.5,
            rotation=0,
            ha="center",
        )

    subplot_value = 1

    plw_s_df = pd.DataFrame()
    sk_df = pd.DataFrame()

    with tqdm(
        total=n + 1 - m,
        desc="Descriptive Statistics",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:

        for i in range(m, n + 1):

            if i == 2009:
                progress_bar.update(1)
                continue

            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #             clear_output(wait = True)
            if not wage_range_dict_flag:
                pw = wages.PREVAILING_WAGE
                lw = wages.LCA_WAGE
                (plw_stat_df, skew_kurt_df) = descriptive_stats(
                    pw, lw, return_df=True, filter_type="Raw", year=i
                )
                plw_s_df = pd.concat([plw_s_df, plw_stat_df], sort=False, axis=1)
                sk_df = pd.concat([sk_df, skew_kurt_df], sort=False, axis=1)

                ax = plt.subplot(sp_rows, sp_columns, subplot_value)

                ax.text(
                    0.5,
                    0.5,
                    employer_name + "_" + str(i),
                    fontsize=0.5,
                    color="gray",
                    alpha=0.5,
                    rotation=0,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    0,
                    0.9,
                    employer_name + "_" + str(i),
                    fontsize=15,
                    fontfamily="sans-serif",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.0,
                    0.3,
                    str(plw_stat_df),
                    fontsize="medium",
                    fontfamily="monospace",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.1,
                    0.1,
                    str(skew_kurt_df),
                    fontsize="medium",
                    fontfamily="monospace",
                    transform=ax.transAxes,
                )
                ax.text(
                    -0.1,
                    -0.1,
                    "    __________________________________________________________________    ",
                    alpha=0.75,
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    -0.1,
                    -0.15,
                    "          Source: https://www.foreignlaborcert.doleta.gov/performancedata.cfm          ",
                    alpha=0.75,
                    fontsize=12,
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.grid(False)
                ax.axis(False)

                # table(ax, plw_stat_df, loc='center')
                subplot_value = subplot_value + 1

            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                pw = wages_filtered.PREVAILING_WAGE
                lw = wages_filtered.LCA_WAGE
                (plw_stat_df, skew_kurt_df) = descriptive_stats(
                    pw, lw, return_df=True, filter_type="Filtered", year=i
                )
                plw_s_df = pd.concat([plw_s_df, plw_stat_df], sort=False, axis=1)
                sk_df = pd.concat([sk_df, skew_kurt_df], sort=False, axis=1)

                ax = plt.subplot(sp_rows, sp_columns, subplot_value)
                ax.text(
                    0.5,
                    0.5,
                    employer_name + "_" + str(i),
                    fontsize=20,
                    color="gray",
                    alpha=0.5,
                    rotation=0,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    0,
                    0.9,
                    employer_name + "_" + str(i),
                    fontsize=15,
                    fontfamily="sans-serif",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.0,
                    0.3,
                    str(plw_stat_df),
                    fontsize="medium",
                    fontfamily="monospace",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.1,
                    0.1,
                    str(skew_kurt_df),
                    fontsize="medium",
                    fontfamily="monospace",
                    transform=ax.transAxes,
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
                # table(ax, plw_stat_df, loc='center')
                subplot_value = subplot_value + 1

            #                 display(pw_range[i][0], pw_range[i][1], lw_range[i][0], lw_range[i][1])
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)
    #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
    #             display('Descriptive Statistics status: ' + str(per_c) +'%' + ' completed')
    if not gen_pdf:
        plt.savefig(
            employer_name + "_" + str(m) + "_" + str(n) + "_" + "Descriptive.png"
        )
    return fig

    # Below code displays dataframes in JupyterLab notebook
    # tcol_name = plw_s_df.columns.tolist()
    # if len(tcol_name) % 2:
    #     x = (len(tcol_name) + 1)
    # else:
    #     x = len(tcol_name)
    # if x > 10:
    #     if x % 10:
    #         y = int((x / 10) + 1)
    #     else:
    #         y = int(x / 10)
    #     c = 10
    # else:
    #     y = 1
    #     c = x
    # k = 0
    # for i in range(0, y):
    #     display(plw_s_df.iloc[0:, range(k, c)])
    #     k = c
    #     if (2 * c) < len(tcol_name):
    #         c = 2 * c
    #     else:
    #         c = len(tcol_name)
    # display(sk_df)


# plw_s_df.to_excel('plw_s_df.xlsx')


# noinspection PyShadowingNames
def box2(df, employer_name, m, n, **kwargs):
    """
    Box plot of wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Box plot of wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(ceil(width_factor / 2))
    sp_columns = 2
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Box_Plot_'+str(m)+'_'+str(n), fontsize=20)
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
    subplot_value = 1
    with tqdm(
        total=n + 1 - m,
        desc="Box Plot Status",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:
        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue
            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #         clear_output(wait= True)
            if not wage_range_dict_flag:
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                # axs.text(0.5, 0.5, employer_name, fontsize=25, color='gray', alpha=0.5,
                #          rotation=0, ha='center', va = 'center', transform = axs.transAxes)
                plot_box(
                    wages.PREVAILING_WAGE,
                    wages.LCA_WAGE,
                    ax=axs,
                    filter_type="Raw",
                    year=i,
                    employer_name=employer_name,
                )
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                # axs.text(0.5, 0.5, employer_name, fontsize=25, color='gray', alpha=0.5,
                #          rotation=0, ha='center', va = 'center', transform = axs.transAxes)
                plot_box(
                    wages_filtered.PREVAILING_WAGE,
                    wages_filtered.LCA_WAGE,
                    ax=axs,
                    filter_type="Filtered",
                    year=i,
                    employer_name=employer_name,
                )
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)

        #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
        #             sys.stdout.write('Box plots status: ' + str(per_c) +'%' + ' completed')
        #             sys.stdout.flush()
        # plt.show()
        if not gen_pdf:
            plt.savefig(employer_name + "_" + str(m) + "_" + str(n) + "_" + "box.png")
        return fig


# noinspection PyShadowingNames
def scatter2(df, employer_name, m, n, **kwargs):
    """
    Scatter plot of wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Scatter plot of wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    yr = []
    cnt = []

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(width_factor / 2)
    sp_columns = 2
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Scatter_Plot_' + str(m) + '_' + str(n), fontsize=20)
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
    subplot_value = 1

    with tqdm(
        total=n + 1 - m,
        desc="Scatter analysis",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:

        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue
            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #         clear_output(wait = True)
            if not wage_range_dict_flag:
                pw = wages.PREVAILING_WAGE
                lw = wages.LCA_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_scatter(
                    pw,
                    lw,
                    ax=axs,
                    filter_type="Raw",
                    year=i,
                    employer_name=employer_name,
                )
                yr.append(i)
                cnt.append(pw.count())
                #                 display(pd.DataFrame({'Year': yr, 'Counts': cnt}))
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                pw = wages_filtered.PREVAILING_WAGE
                lw = wages_filtered.LCA_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_scatter(
                    pw,
                    lw,
                    ax=axs,
                    filter_type="Filtered",
                    year=i,
                    employer_name=employer_name,
                )
                yr.append(i)
                cnt.append(pw.count())
                #                 display(pd.DataFrame({'Year': yr, 'Counts': cnt}))
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)
        #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
        #             display('Scatter plots status: ' + str(per_c) +'%' + ' completed')
        # display(pd.DataFrame({'YEAR': yr, 'COUNT': cnt}))
    # plt.show()
    if not gen_pdf:
        plt.savefig(employer_name + "_" + str(m) + "_" + str(n) + "_" + "scatter.png")
    return fig


# noinspection PyShadowingNames
def hist2(df, employer_name, m, n, **kwargs):
    """
    Histogram (Frequency) of LCA wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Histogram (Frequency) of LCA wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(width_factor / 2)
    sp_columns = 2
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + '_Histogram_' + str(m) + '_' + str(n), fontsize=20)
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
    subplot_value = 1

    with tqdm(
        total=n + 1 - m,
        desc="Frequency analysis",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:
        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue
            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #             clear_output(wait= True)
            if not wage_range_dict_flag:
                lw = wages.LCA_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_hist(
                    lw,
                    bins=30,
                    ax=axs,
                    filter_type="Raw",
                    xlabel=lw.name,
                    year=i,
                    employer_name=employer_name,
                )
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                lw = wages_filtered.LCA_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_hist(
                    lw,
                    bins=30,
                    ax=axs,
                    filter_type="Filtered",
                    xlabel=lw.name,
                    year=i,
                    employer_name=employer_name,
                )
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
            #             display('Frequency distribution status: ' + str(per_c) +'%' + ' completed')
            progress_bar.update(1)
    # plt.show()
    if not gen_pdf:
        plt.savefig(employer_name + "_" + str(m) + "_" + str(n) + "_" + "histogram.png")
    return fig


# Distribution plot
# noinspection PyShadowingNames
def dist2(df, employer_name, m, n, **kwargs):
    """
    Distribution plot of LCA wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Distribution plot of LCA wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(width_factor / 2)
    sp_columns = 2
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Distribution_' + str(m) + '_' + str(n), fontsize=20)
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
    subplot_value = 1

    with tqdm(
        total=n + 1 - m,
        desc="Distribution",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:

        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue

            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #             clear_output(wait= True)
            if not wage_range_dict_flag:
                lw = wages.LCA_WAGE
                #             pw = wages.PREVAILING_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_dist(
                    lw,
                    bins=100,
                    ax=axs,
                    filter_type="Raw",
                    xlabel="LCA_Wage",
                    fit=stats.norm,
                    year=i,
                    employer_name=employer_name,
                )
                #             PlotNormDist(pw, bins = 100, ax = axs, filter_type = 'Raw-'+str(i), xlabel = 'Prevailing_Wage')
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                lw = wages_filtered.LCA_WAGE
                #             pw = wages.PREVAILING_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_dist(
                    lw,
                    bins=100,
                    ax=axs,
                    filter_type="Filtered",
                    xlabel="LCA_Wage",
                    fit=stats.norm,
                    year=i,
                    employer_name=employer_name,
                )
                #             PlotNormDist(pw, bins = 100, ax = axs, filter_type = 'Raw-'+str(i), xlabel = 'Prevailing_Wage')
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)
    #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
    #             display('Distribution plots status: ' + str(per_c) +'%' + ' completed')
    # plt.show()
    if not gen_pdf:
        plt.savefig(
            employer_name + "_" + str(m) + "_" + str(n) + "_" + "distribution.png"
        )
    return fig


# CDF plot
# noinspection PyShadowingNames
def cdf2(df, employer_name, m, n, **kwargs):
    """
    Cumulative Distribution plot of LCA wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Cumulative Distribution plot of LCA wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(width_factor / 2)
    sp_columns = 2
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Cumulative_Distribution_' + str(m) + '_' + str(n), fontsize=20)
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
    subplot_value = 1

    with tqdm(
        total=n + 1 - m,
        desc="Cummulative Dist:",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:

        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue

            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #             clear_output(wait= True)
            if not wage_range_dict_flag:
                lw = wages.LCA_WAGE
                #                 pw = wages.PREVAILING_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_cdf(
                    lw,
                    bins=15,
                    ax=axs,
                    filter_type="Raw",
                    fit=stats.norm,
                    year=i,
                    employer_name=employer_name,
                )
                #                 pw = wages.PREVAILING_WAGE
                #                 plot_c_d_f(pw, bins = 15, ax = axs, filter_type = 'Raw-'+str(i), fit = stats.norm)
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                lw = wages_filtered.LCA_WAGE
                #             pw = wages.PREVAILING_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_cdf(
                    lw,
                    bins=15,
                    ax=axs,
                    filter_type="Filtered",
                    fit=stats.norm,
                    year=i,
                    employer_name=employer_name,
                )
                #             plot_c_d_f(pw, bins = 15, ax = axs, filter_type = 'Raw-'+str(i), fit = stats.norm)
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)
    #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
    #             display('CDF plots status: ' + str(per_c) +'%' + ' completed')
    # plt.show()
    if not gen_pdf:
        plt.savefig(
            employer_name
            + "_"
            + str(m)
            + "_"
            + str(n)
            + "_"
            + "cumulative_distribution.png"
        )
    return fig


# noinspection PyShadowingNames
def linear_r2(df, employer_name, m, n, **kwargs):
    """
    Linear Regression plot of wages of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Linear Regression plot of  wages of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    set_plot_style()
    width_factor = get_plot_height_factor(m, n)
    sp_rows = int(width_factor / 2)
    sp_columns = 2
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Linear_Regression_' + str(m) + '_' + str(n), fontsize=20)
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
    subplot_value = 1

    with tqdm(
        total=n + 1 - m,
        desc="Linear Regression:",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:

        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue
            #             clear_output(wait= True)
            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )

            if not wage_range_dict_flag:
                pw = wages.PREVAILING_WAGE
                lw = wages.LCA_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_linear_r(
                    pw,
                    lw,
                    ax=axs,
                    filter_type="Raw",
                    year=i,
                    employer_name=employer_name,
                )
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                pw = wages_filtered.PREVAILING_WAGE
                lw = wages_filtered.LCA_WAGE
                axs = plt.subplot(sp_rows, sp_columns, subplot_value)
                plot_linear_r(
                    pw,
                    lw,
                    ax=axs,
                    filter_type="Filtered",
                    year=i,
                    employer_name=employer_name,
                )
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)
    #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
    #             display('Linear regression status: ' + str(per_c) +'%' + ' completed')
    # plt.show()
    if not gen_pdf:
        plt.savefig(
            employer_name + "_" + str(m) + "_" + str(n) + "_" + "linear_regression.png"
        )
    return fig


# plot wage level
# noinspection PyShadowingNames
def wage_level2(df, employer_name, m, n, **kwargs):
    """
    Wage Level plot of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Wage Level plot of an employer in the range of year
    """

    if m > n:
        return f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        display(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        return "Wage range dictionaries not present"

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                display("Employer selected not in the database or it is not matching")
                return "Employer not in database or not matching"
        else:
            return print("Database is empty")

    number_of_plots = 0
    wagelevel_na_list = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016]
    for i in range(m, n + 1):
        if i not in wagelevel_na_list:
            number_of_plots = number_of_plots + 1

    if not number_of_plots:
        print("Wage level plot is displayed for years that have wage level data")
        return "Wage_level plot is displayed for years that have wage level data"

    set_plot_style()
    sp_rows = int(ceil(number_of_plots / 2))
    fig = plt.figure(figsize=(24, sp_rows * 8))
    # fig.suptitle(employer_name + '_' + 'Wage_Levels_' + str(m) + '_' + str(n), fontsize=20)
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
    subplot_value = 1

    # set_plot_style()
    # plt.figure(figsize=(24, 48))
    # subplot_value = 1
    pwl_df = pd.DataFrame()

    with tqdm(
        total=n + 1 - m,
        desc="WageLevel Status:",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:

        for i in range(m, n + 1):
            if i in wagelevel_na_list:
                progress_bar.update(1)
                continue
            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE", "PW_WAGE_LEVEL"]
            )
            if not wage_range_dict_flag:
                if wages.PW_WAGE_LEVEL.isnull().all():
                    progress_bar.update(1)
                    continue
                wl = wages.PW_WAGE_LEVEL
                axs = plt.subplot(sp_rows, 2, subplot_value)
                #                 print(str(i) + '- Raw')
                r_pwl_df = plot_wage_levels(
                    wl,
                    ax=axs,
                    filter_type="Raw",
                    year=i,
                    employer_name=employer_name,
                )
                pwl_df = pd.concat([pwl_df, r_pwl_df], sort=False, axis=1)
                subplot_value = subplot_value + 1
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                if wages_filtered.PW_WAGE_LEVEL.isnull().all():
                    continue
                wl = wages_filtered.PW_WAGE_LEVEL
                axs = plt.subplot(sp_rows, 2, subplot_value)
                r_pwl_df = plot_wage_levels(
                    wl,
                    ax=axs,
                    filter_type="Filtered",
                    year=i,
                    employer_name=employer_name,
                )
                pwl_df = pd.concat([pwl_df, r_pwl_df], sort=False, axis=1)
                subplot_value = subplot_value + 1
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict is present"
                )
                break
            progress_bar.update(1)
        #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
        #             display('Wage level analysis status: ' + str(per_c) +'%' + ' completed')
        # plt.show()
        if not gen_pdf:
            plt.savefig(
                employer_name + "_" + str(m) + "_" + str(n) + "_" + "wagelevel.png"
            )
        return fig


# Plot wages trends
# noinspection PyShadowingNames
def wage_trends2(df, employer_name, m, n, **kwargs):
    """
    Wage trends plot of an employer in the range of year
    ____
    Parameters
    ____
        df: DataFrame containing columns ['YEAR', 'EMPLOYER_NAME', 'PREVAILING_WAGE', 'LCA_WAGE', 'PW_WAGE_LEVEL']

        employer_name:str

        m > n

        m: Lower Range Limit, Year in format YYYY, e.g. 2010

        n: Higher Range Limit, Year in format YYYY, e.g. 2011

    Other Parameters
    ____
        **kwargs
            wage_range_dict_flag: True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False

            pw_range: PREVAILING Wage Range Dictionary
                      Default: False. Raw

            lw_range: LCA WAGE Range Dictionary
                      Default: False. Raw

            dataframe_checked: True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
    Returns
    ____
        Wage trends plot of an employer in the range of year
    """

    if m > n:
        print(f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}")
        return None

    wage_range_dict_flag = kwargs.get("wage_range_dict_flag", False)
    dataframe_checked_status = kwargs.get("dataframe_checked", False)
    pw_range = kwargs.get(
        "pw_range",
    )
    lw_range = kwargs.get(
        "lw_range",
    )
    gen_pdf = kwargs.get("generate_pdf", False)
    filter_type = str(nan)

    if wage_range_dict_flag and pw_range is None and lw_range is None:
        print(
            "Wage range dictionaries 'pw_range' and 'lw_range' for Prevailing and LCA wages needs to be passed as "
            "arguments when 'wage_range_dict_flag' is set to True"
        )
        print("Wage range dictionaries not present")
        return None

    if not dataframe_checked_status:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if dataframe_checked_status:
        df_eyr = df[(df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)]
    else:
        if not df.empty:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                print("Employer selected not in the database or it is not matching")
                return None
        else:
            print("Database is empty")
            return None

    pwdf = pd.DataFrame()
    lwdf = pd.DataFrame()
    plot_flag = True

    with tqdm(
        total=n + 1 - m,
        desc="WageTrends Status:",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:
        for i in range(m, n + 1):
            if i == 2009:
                progress_bar.update(1)
                continue

            wages = df_eyr[df_eyr.YEAR.astype(str).str.contains(str(i))].filter(
                ["PREVAILING_WAGE", "LCA_WAGE"]
            )
            #         clear_output(wait= True)
            if not wage_range_dict_flag:
                (pwt, lwt) = create_wage_trends_d_f(wages, i)
                pwdf = pd.concat([pwdf, pwt], sort=False, axis=1)
                lwdf = pd.concat([lwdf, lwt], sort=False, axis=1)
                filter_type = "Raw"
            elif wage_range_dict_flag:
                wages_filtered = wages[
                    (wages.PREVAILING_WAGE.between(pw_range[i][0], pw_range[i][1]))
                    & (wages.LCA_WAGE.between(lw_range[i][0], lw_range[i][1]))
                ]
                (pwt, lwt) = create_wage_trends_d_f(wages_filtered, i)
                pwdf = pd.concat([pwdf, pwt], sort=False, axis=1)
                lwdf = pd.concat([lwdf, lwt], sort=False, axis=1)
                filter_type = "Filtered"
            else:
                print(
                    "Set the wage_range_dict_flag to boolean True or False. If set to True, make sure wage_range_dict "
                    "is present"
                )
                plot_flag = False
                break
            progress_bar.update(1)
        #             per_c = '{0:.{1}f}'.format(((i-m+1) * 100)/(n-m), 2)
        #             display('Wage trends analysis status: ' + str(per_c) +'%' + ' completed')
        if plot_flag:
            set_plot_style()
            #         fig = plt.figure(figsize= (23,15), dpi = 100)
            fig = plt.figure(figsize=(24, 10))
            # fig.suptitle(employer_name + '_' + 'Wage_Trends_' + str(m) + '_' + str(n), fontsize=20)
            # fig.text(0.5, 0.95, employer_name, fontsize=30, color='gray', alpha=0.5, rotation=0, ha='center')
            axs = plt.subplot(1, 1, 1)
            plot_wage_trends(
                lwdf.T["mean"],
                lwdf.T["50%"],
                wage_type="LCA_WAGE " + filter_type,
                ax=axs,
                employer_name=employer_name,
            )
            plot_wage_trends(
                pwdf.T["mean"],
                pwdf.T["50%"],
                wage_type="PREVAILING_WAGE " + filter_type,
                ax=axs,
                employer_name=employer_name,
            )
            # plt.show()
            if not gen_pdf:
                plt.savefig(
                    employer_name + "_" + str(m) + "_" + str(n) + "_" + "wagetrends.png"
                )
            return fig
    return None
