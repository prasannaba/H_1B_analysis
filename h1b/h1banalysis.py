# Copyright (c) 2019-2025 Prasanna
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
This module provides functions for analyzing H1B visa data.

It includes functions for performing descriptive statistical analysis, 
generating various plots (box, histogram, distribution, CDF, scatter), 
and conducting wage level analysis. The main function 'analysis' takes a 
pandas DataFrame of H1B data and produces a comprehensive analysis report. 
Helper functions are included to create and preprocess DataFrames from different 
data sources.
"""

from typing import Callable, Dict, List, Tuple

import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from h1b.h1bcore import (
    descriptive_stats,
    plot_box,
    plot_cdf,
    plot_dist,
    plot_hist,
    plot_linear_r,
    plot_scatter,
    plot_wage_levels,
    set_plot_style,
)


# noinspection PyShadowingNames
def analysis(
    df_a: pd.DataFrame,
    *,
    bins: int = 50,
    data_type: str = "Raw",
    all_flag: bool = False,
    employer_name: str = "MMXIX",
    year: int = None,
    pw_range: Dict[int, List[int]] = None,
    lw_range: Dict[int, List[int]] = None,
    generate_pdf: bool = False,
    dataframe_checked: bool = False,
):
    """
    analysis gives:
    Descriptive statistics, Box, Scatter Histogram,
    Distribution, CDF, Linear Regression,
    Wage Level and
    Trends of an employer along with alternate names of an employer in the database
    or if all_flag == True top 10 employers
    ____
    Parameters
    ____
        df: DataFrame of H1B database

    Other Parameters
    ____
        **kwargs
            bins: int, default: 50

            data_type: str, 'Raw' or 'Filtered'

            all_flag : True, if analysis is for complete database,
                       default: False,
                       select a valid employer by setting 'employer_name' flag to an employer name or starting with an employer name

            employer_name: str, employer name of interest.
                           Complete name or first word of an employer of an interest, in case of first word analysis is done on all names matching first word

            year: in format YYYY, e.g. 2019

            pw_range: PREVAILING Wage Range Dictionary in tuple format. e.g. (30000, 120000)

            lw_range: LCA WAGE Range Dictionary in tuple format. e.g (50000, 250000)

            dataframe_checked:  At present not implemented. True, if dataframe is already checked for consistencies
                               False, if dataframe consistency need to be checked in this module before proceeding further with analysis
                               Default: False
            wage_range_dict_flag: At present not implemented. True. Filtered, Prevailing and LCA Wage Range Dictionary is selected for analysis
                                  False. Raw
                                  Default : False
    Returns
    ____
        Descriptive statistics, Box, Scatter Histogram,
        Distribution, CDF, Linear Regression, Wage Level
        and Trends of an employer along with alternate names of an employer in the database
        or if all_flag == True top 10 employers
    """

    #    df = dfa.copy()

    set_plot_style()

    if (employer_name == "MMXIX") and (all_flag is False):
        return display(
            "Valid employer name has to be selected with 'all_flag' not set to True"
        )

    if (df_a.columns.str.contains("^PREVAILING_WAGE_1$").any()) and (
        df_a.columns.str.contains("^WAGE_RATE_1$").any()
    ):

        df = create_df_pw_wage_rate(df_a)

    elif (df_a.columns.str.contains("^PW_1$").any()) and (
        df_a.columns.str.contains("^LCA_CASE_WAGE_RATE_FROM$").any()
    ):

        df = create_df_pw_lca_case_wage_rate(df_a)

    elif (df_a.columns.str.contains("PREVAILING_WAGE$").any()) and (
        df_a.columns.str.contains("^WAGE_RATE_OF_PAY$").any()
    ):

        df = create_df_pw_wage_rate_pay(df_a)

    elif (df_a.columns.str.contains("^PREVAILING_WAGE$").any()) and (
        df_a.columns.str.contains("^WAGE_RATE_OF_PAY_FROM$").any()
    ):

        df = create_df_pw_wage_rate_pay_from(df_a)

    elif (df_a.columns.str.contains("^PREVAILING_WAGE$").any()) and (
        df_a.columns.str.contains("^LCA_WAGE$").any()
    ):
        df = create_df_pw_lca(df_a)
    else:
        return print("Error in column names")

    # Select the employer of interest
    if not dataframe_checked:
        if df.EMPLOYER_NAME.str.contains("^" + str(employer_name), case=False).any():
            df = df[
                df["EMPLOYER_NAME"].str.contains("^" + str(employer_name), case=False)
            ]
            en = employer_name

        else:
            return display(
                "Employer name not present in the database, please check the name"
            )
    else:
        df = df[df["EMPLOYER_NAME"] == employer_name]
        en = employer_name

    # Select the wage range
    if (pw_range is not None) and (lw_range is not None) and (year is not None):

        if (isinstance(pw_range, dict)) and (isinstance(lw_range, dict)):
            wages_filtered = df[
                (df.PREVAILING_WAGE.between(pw_range[year][0], pw_range[year][1]))
                & (df.LCA_WAGE.between(lw_range[year][0], lw_range[year][1]))
            ]

        elif (isinstance(pw_range, tuple)) and (isinstance(lw_range, tuple)):
            wages_filtered = df[
                (df.PREVAILING_WAGE.between(pw_range[0], pw_range[1]))
                & (df.LCA_WAGE.between(lw_range[0], lw_range[1]))
            ]
        else:
            return display(
                "Enter the range for Prevailing and LCA Wages in dictionary or tuple format"
            )

        pw = wages_filtered["PREVAILING_WAGE"]
        lw = wages_filtered["LCA_WAGE"]
        df_f = wages_filtered
    else:
        pw = df["PREVAILING_WAGE"]
        lw = df["LCA_WAGE"]
        df_f = df.copy()

    task_dict: Dict[Callable, List[Tuple]] = {
        descriptive_stats: [(pw, lw)],
        plot_box: [(pw, lw)],
        plot_hist: [(pw, bins), (lw, bins)],
        plot_dist: [(pw, bins), (lw, bins)],
        plot_cdf: [(pw, bins), (lw, bins)],
        plot_scatter: [(pw, lw)],
        plot_linear_r: [(pw, lw)],
    }

    if df_f.columns.str.contains("^PW_WAGE_LEVEL$").any():
        if not df_f.PW_WAGE_LEVEL.isna().all():
            pw_level = df_f["PW_WAGE_LEVEL"]
            task_dict.update({plot_wage_levels: [pw_level]})

    # if pw_level is present increase plot size
    if len(task_dict) == 8:
        fig = plt.figure(figsize=(26, 48))
        subplot_count = 1
    else:
        fig = plt.figure(figsize=(26, 40))
        subplot_count = 1

    t_len = len(task_dict)
    t_count = 1

    with tqdm(
        total=len(task_dict),
        desc="Overall Status",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:
        if plot_wage_levels in task_dict:
            for task in task_dict:
                #            clear_output(wait = True)
                for i in range(0, len(task_dict[task])):
                    axs = fig.add_subplot(6, 2, subplot_count)
                    if task == plot_wage_levels:
                        task(
                            task_dict[task][i],
                            filter_type=data_type,
                            ax=axs,
                            employer_name=en,
                            year=year,
                        )
                    else:
                        task(
                            *task_dict[task][i],
                            filter_type=data_type,
                            ax=axs,
                            employer_name=en,
                            year=year,
                        )
                    subplot_count += 1
                print(f"Status: {t_count * 100 / t_len:.2f}" + str("%") + " completed")
                t_count += 1
                progress_bar.update(1)
        else:
            for task in task_dict:
                #            clear_output(wait = True)
                for i in range(0, len(task_dict[task])):
                    axs = fig.add_subplot(5, 2, subplot_count)
                    task(
                        *task_dict[task][i],
                        filter_type=data_type,
                        ax=axs,
                        employer_name=en,
                        year=year,
                    )
                    subplot_count += 1
                print(f"Status: {t_count * 100 / t_len:.2f}" + str("%") + " completed")
                t_count += 1
                progress_bar.update(1)

    #    if (df_f.columns.str.contains('^PW_WAGE_LEVEL$').any()):
    #        if (df_f.PW_WAGE_LEVEL.isna().all() == False):
    #            axs = plt.subplot(6, 2, subplot_count)
    #            plot_wage_levels(df_f.PW_WAGE_LEVEL, filter_type = data_type, ax = axs)

    df_table = pd.DataFrame()
    # es_df = pd.DataFrame()
    if all_flag:
        df_table_t = pd.pivot_table(
            df_f,
            values=["PREVAILING_WAGE", "LCA_WAGE"],
            columns=["EMPLOYER_NAME"],
            aggfunc={
                "PREVAILING_WAGE": ["min", "max", "mean", "median"],
                "LCA_WAGE": ["count", "min", "max", "mean", "median"],
            },
        )
        df_table = df_table_t.T.reset_index()
        df_table = df_table.sort_values(("LCA_WAGE", "count"), ascending=False).iloc[
            0:10
        ]
        df_table = df_table.reset_index().drop(columns="index", level=0)
    # else:
    #     es = dict(employer_descriptive_stats(df_f))
    #     es_df = pd.DataFrame.from_dict(es, orient='index', columns=['COUNT']).reset_index().rename(
    #         columns={'index': 'EMPLOYER_NAME'})
    if not generate_pdf:
        plt.show()
    else:
        plt.savefig(
            fname=str(year) + "_" + str(en) + "_H1B_Analysis_" + str(data_type) + ".pdf"
        )
    plt.close()

    if all_flag:
        display(df_table)
    return 1


def create_df_pw_lca(df_a):
    """

    @param df_a:
    @return:
    """
    if df_a.columns.str.contains("^PW_WAGE_LEVEL$").any():
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE.values,
                "LCA_WAGE": df_a.LCA_WAGE.values,
                "PW_WAGE_LEVEL": df_a.PW_WAGE_LEVEL.values,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE.values,
                "LCA_WAGE": df_a.LCA_WAGE.values,
            }
        )
    return df


def create_df_pw_wage_rate_pay_from(df_a):
    """

    @param df_a:
    @return:
    """
    if df_a.columns.str.contains("^PW_WAGE_LEVEL$").any():
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE.values,
                "LCA_WAGE": df_a.WAGE_RATE_OF_PAY_FROM.values,
                "PW_WAGE_LEVEL": df_a.PW_WAGE_LEVEL.values,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE.values,
                "LCA_WAGE": df_a.WAGE_RATE_OF_PAY_FROM.values,
            }
        )
    return df


def create_df_pw_wage_rate_pay(df_a):
    """

    @param df_a:
    @return:
    """
    if df_a.columns.str.contains("^PW_WAGE_LEVEL$").any():
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE.values,
                "LCA_WAGE": df_a.WAGE_RATE_OF_PAY.values,
                "PW_WAGE_LEVEL": df_a.PW_WAGE_LEVEL.values,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE.values,
                "LCA_WAGE": df_a.WAGE_RATE_OF_PAY.values,
            }
        )
    return df


def create_df_pw_lca_case_wage_rate(df_a):
    """

    @param df_a:
    @return:
    """
    if df_a.columns.str.contains("^PW_WAGE_LEVEL$").any():
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PW_1.values,
                "LCA_WAGE": df_a.LCA_CASE_WAGE_RATE_FROM.values,
                "PW_WAGE_LEVEL": df_a.PW_WAGE_LEVEL.values,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PW_1.values,
                "LCA_WAGE": df_a.LCA_CASE_WAGE_RATE_FROM.values,
            }
        )
    return df


def create_df_pw_wage_rate(df_a):
    """

    @param df_a:
    @return:
    """
    if df_a.columns.str.contains("^PW_WAGE_LEVEL$").any():
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE_1.values,
                "LCA_WAGE": df_a.WAGE_RATE_1.values,
                "PW_WAGE_LEVEL": df_a.PW_WAGE_LEVEL.values,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "EMPLOYER_NAME": df_a.filter(regex="NAME$").iloc[:, 0],
                "PREVAILING_WAGE": df_a.PREVAILING_WAGE_1.values,
                "LCA_WAGE": df_a.WAGE_RATE_1.values,
            }
        )
    return df
