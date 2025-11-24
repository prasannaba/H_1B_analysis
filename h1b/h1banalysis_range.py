# Copyright (c) 2019-2025 Prasanna
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
This module provides advanced analysis functions for H1B visa data over a range of years.

It includes the main analysis function 'analysis2' which generates descriptive statistics, 
various plots (box, scatter, histogram, distribution, CDF, linear regression), wage level, 
and trend analysis for a specific employer over a given year range. The module also 
provides helper functions to display employer statistics and wage ranges, and to calculate 
wage ranges based on quartiles. This allows for a more in-depth, longitudinal analysis of H1B data.
"""

from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import table

# from IPython.core.display_functions import display
from PIL import Image
from tqdm.notebook import tqdm

from h1b.h1bcore import employer_descriptive_stats
from h1b.h1bcore_range import (
    box2,
    cdf2,
    descriptive_stats2,
    dist2,
    hist2,
    linear_r2,
    scatter2,
    wage_level2,
    wage_trends2,
)

__all__ = ["analysis2"]


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


# noinspection PyShadowingNames
def analysis2(
    df: pd.DataFrame,
    employer_name: str,
    m: int,
    n: int,
    *,
    wage_range_dict_flag: bool = False,
    dataframe_checked: bool = False,
    pw_range: Dict[int, List[int]] = None,
    lw_range: Dict[int, List[int]] = None,
    generate_pdf: bool = False,
):
    """
    analysis2 gives descriptive statistics, Box, Scatter Histogram, Distribution, CDF, Linear Regression, Wage Level and Trends
                of an employer in the range of year
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
        Descriptive statistics, Box, Scatter Histogram, Distribution, CDF, Linear Regression, Wage Level and Trends an employer in the range of year
    """
    if m > n:
        print(
            f"Lower Range Limit 'm': {m} should be less than Higher Range Limit 'n': {n}"
        )
        return False
    # wage_range_dict_flag = wage_range_dict_flag
    # dataframe_checked = dataframe_checked
    # pw_range = pw_range
    # lw_range = lw_range
    # generate_pdf = generate_pdf

    if not dataframe_checked:
        employer_name_str = str("^") + str(employer_name)
    else:
        employer_name_str = employer_name

    if not df.empty:

        if dataframe_checked:
            df_eyr = df[
                (df.YEAR.between(m, n)) & (df.EMPLOYER_NAME == employer_name_str)
            ]
            # progress_bar.update(1)
        else:
            if df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
                df_eyr = df[
                    (df.YEAR.between(m, n))
                    & (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False))
                ]
            else:
                print("Employer selected not in the database or it is not matching")
                return False

        # if not dataframe_checked:
        #     if not df.EMPLOYER_NAME.str.contains(employer_name_str, case=False).any():
        #         print('Employer selected not in the database, or it is not matching')
        #         return False
        #     else:
        #         df_eyr = df[(df.YEAR.between(m, n) == True) and (df.EMPLOYER_NAME.str.contains(employer_name_str, case=False) == True)]
        #     dataframe_checked = True
        #     progress_bar.update(1)
    else:
        print("Database is empty")
        return False
    # print(type(m), n)
    with tqdm(
        total=10,
        desc="Overall Status",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as progress_bar:
        plot_list = [
            descriptive_stats2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        ]
        #        out.clear_output()
        #        print(pw_range, lw_range)
        progress_bar.update(1)
        plot_list.append(
            box2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)
        plot_list.append(
            scatter2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)
        plot_list.append(
            hist2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)
        plot_list.append(
            dist2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)
        plot_list.append(
            cdf2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)
        plot_list.append(
            linear_r2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)
        plot_list.append(
            wage_level2(
                df_eyr.copy(),
                employer_name,
                m,
                n,
                wage_range_dict_flag=wage_range_dict_flag,
                pw_range=pw_range,
                lw_range=lw_range,
                dataframe_checked=dataframe_checked,
                generate_pdf=generate_pdf,
            )
        )
        progress_bar.update(1)

        if (n - m) >= 1:
            plot_list.append(
                wage_trends2(
                    df_eyr.copy(),
                    employer_name,
                    m,
                    n,
                    wage_range_dict_flag=wage_range_dict_flag,
                    pw_range=pw_range,
                    lw_range=lw_range,
                    dataframe_checked=dataframe_checked,
                    generate_pdf=generate_pdf,
                )
            )
        else:
            plot_list.append("No Plot")

        # if wage range dict flag is set(filter) then show wage range else no, because it will be raw.
        if wage_range_dict_flag:
            plot_list.append(
                emp_stats_wage_range_display(
                    df_eyr.copy(),
                    employer_name,
                    pw_range,
                    lw_range,
                    generate_pdf=generate_pdf,
                )
            )

        progress_bar.update(1)
        test_code = False
        # noinspection PyUnreachableCode
        if test_code is True:
            # noinspection PyUnreachableCode
            with tqdm(
                total=3,
                desc="Generating pdf",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ) as pdf_bar:
                if generate_pdf:
                    list_images = []
                    j = 1
                    for i in plot_list:
                        if isinstance(i, str) is False:
                            i.savefig("image" + str(j) + ".png")
                            j = j + 1
                    pdf_bar.update(1)
                    j = j - 1
                    for i in range(1, j + 1):
                        image1 = Image.open("image" + str(i) + ".png")
                        image2 = image1.convert("RGB")
                        list_images.append(image2)
                    pdf_bar.update(1)
                    list_images[0].save(
                        employer_name
                        + "_H1B_Analysis_"
                        + str(m)
                        + "_"
                        + str(n)
                        + ".pdf",
                        save_all=True,
                        append_images=list_images[1:],
                    )
                    pdf_bar.update(1)
            plt.close(fig="all")
        # pylint: disable=R1728
        progress_bar_count = sum(
            [1 for plot in plot_list if isinstance(plot, plt.Figure) is True]
        )
        with tqdm(
            total=progress_bar_count,
            desc="Generating pdf",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pdf_bar:
            if generate_pdf:

                file_name = (
                    employer_name + "_H1B_Analysis_" + str(m) + "_" + str(n) + ".pdf"
                )

                with PdfPages(filename=file_name) as pdf:
                    for plot in plot_list:
                        if isinstance(plot, plt.Figure) is True:
                            pdf.savefig(plot)
                            plt.close(plot)
                            pdf_bar.update(1)

        progress_bar.update(1)
    return True
