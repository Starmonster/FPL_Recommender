# This will be our main app file. We'll run this file to open the streamlit web app

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import math
from tqdm.auto import tqdm
import requests
import time
import datetime
import plotly_express as px
# from fpl_funcs import *
from fpl_funcs import *

plt.style.use("fivethirtyeight")


# pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page in wide mode
st.set_page_config(layout="wide")

# Set a function to make it easy to change styling through the page
# def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
#     html = f"""
#     <script>
#         var elems = window.parent.document.querySelectorAll('p');
#         var elem = Array.from(elems).find(x => x.innerText == '{label}');
#         elem.style.fontSize = '{font_size}';
#         elem.style.color = '{font_color}';
#         elem.style.fontFamily = '{font_family}';
#     </script>
#     """
#     st.components.v1.html(html)

st.title("FPL RECOMMENDER APP")

st.sidebar.subheader("Tool Selector")
tool = st.sidebar.radio("Select a tool to view", ["Recommender", "Analysis", "Forecast"])

if tool == "Analysis":

    st.subheader("Compare your squad")
    # Display the player comparison tables
    # First call in the player comparison df
    comparison_df = pd.read_csv("comparison_df.csv")
    squad_df = pd.read_csv("squad_history.csv")
    # Now get the current squad for the last / latest gw - 15 players in a squad
    my_squad = squad_df.head(15)

    # COMPARISON METRICS TABLE
    # Give user options for comparison tables
    st.sidebar.subheader("Squad Comparisons")
    comparison_metric = st.sidebar.radio("Select Metric to Compare", ["bps", "ict_index", "total_points"])
    position_to_compare = st.sidebar.slider("Select a Position Number", 1, 4, help="1:GK 2:DEF 3:MID 4:FWD")

    # Call in the comparis tabel with user inputs and display
    comparison_table, position_elements = player_comp(comp_metric=comparison_metric, position_sel=position_to_compare,
                                   df_for_comp=comparison_df, my_squad=my_squad
                                   )

    # Style the dataframe to highlight where our squad players rank
    highlighted_rows = np.where(comparison_table['element'].isin(position_elements),
                                'background-color: #e90052',
                                '')
    # Apply calculated styles to each column:
    styler = comparison_table.style.apply(lambda _: highlighted_rows)

    st.caption(f"Table to compare {comparison_metric} for your squad players (blue) against top 5 in each position")
    st.dataframe(styler)


    # Build plot
    st.subheader("Differential Plots")
    plot_type = st.radio("Static or Animated PLot", ["Static", "Animated"], horizontal=True)
    # Metric visuals - BETA - WE NEED TO CLEAN THIS SO WE SIMPLY PASS A fig from the plot function
    # STYLE IMPROVEMENTS REQUIRED
    st.sidebar.subheader("Differential Plot Variables")
    metric_sel = st.sidebar.radio("Select Metric to Compare", ["now_cost", "minutes", "selected"])
    metric_thresh = st.sidebar.number_input("Show players above value", value=48)
    points_thresh = st.sidebar.slider("Select the low points threhsold", min_value=0, value=20)

    if plot_type == "Static":
        st.pyplot(scoring_players=value_plots(completed_df_=comparison_df,
                                              value_by=metric_sel,
                                              non_scorers=True,
                                              position=position_to_compare,
                                              value_met=metric_thresh,
                                              points_threshold=points_thresh
                                              ))

    elif plot_type == "Animated":

        # Plot season animation with plotly
        gw_df_cumul = pd.read_csv("cumulative_gws.csv")


        comparison_position = comparison_df.query(f"element_type=={position_to_compare}")
        scout = st.selectbox("Select a player to scout", comparison_position.web_name.unique())

        plot_df = (gw_df_cumul.query(f"element_type=={position_to_compare}"))

        plot_df.loc[plot_df["web_name"] == scout, 'plot_color'] = "#e90052"

        # Make some local feature name changes so that plot labels make sense to users
        plot_df = plot_df.rename(columns={"name": "team", "tp_cumul": "total_points", "total_points": "gw_points",
                                          "minutes": "gw_minutes", "mins_cumul": "minutes", "value": "now_cost"
                                          })

        max_x = plot_df[metric_sel].max() * 1.1
        min_x = plot_df[metric_sel].min() / 1.1
        max_y = plot_df.total_points.max() * 1.1



        st.plotly_chart(animated_player_plot(plot_df=plot_df,
                                             max_x=max_x, min_x=min_x, max_y=max_y,
                                             metric_sel=metric_sel))










elif tool == "Forecast":

    st.subheader("Fixture Difficulty Ratings")
    # Call in latest fix diff.csv and display fdr dataframe
    fixture_diff_df = pd.read_csv("fixture_difficulty.csv")
    lookahead = st.radio("choose lookahead period", ["3 weeks", "5 weeks", "8 Weeks"], horizontal=True)
    sel_1, sel_2, sel_3 = st.columns(3)
    with sel_1:
        team_select = st.selectbox("Select a team to view fixture difficulty analysis", fixture_diff_df.name.unique())
    fixture_list = eval(fixture_diff_df[fixture_diff_df.name==team_select].iloc[0].next_eight)
    print(fixture_list)
    if lookahead == "3 weeks":
        fdr_df = fixture_diff_df.groupby("name").mean()[["fdr_three",
                                                         "fdr_three_std"]].sort_values("fdr_three", ascending=True)

    else: # only other current option is five weeks
        fdr_df = fixture_diff_df.groupby("name").mean()[["fdr_five",
                                                         "fdr_five_std"]].sort_values("fdr_five", ascending=True)


    # tab_1, tab_2 = st.tabs(["FDR RANKS", "FIXTURE LIST BY TEAM"])
    # with tab_1:
    #     st.dataframe(fdr_df.round(2))
    # # tab_1.write("Hello, I am tab 1")
    # with tab_2:
    #     st.write(fixture_list)


    fdr_ranks, fixtures = st.columns([1, 2])

    st.markdown(
        """<style>
            .col_heading {text-align: left !important;}
            .bigger-font {font-size: 50px !important;}
        </style>
        """, unsafe_allow_html=True)

    # st.markdown('<p class="bigger-font">Hello World !!</p>', unsafe_allow_html=True)

    with fdr_ranks:
        st.write(f"{lookahead[:-1].title()} Difficulty Lookahead Rankings")
        st.dataframe(fdr_df.round(2))
    with fixtures:
        st.write(f"{team_select} Current Strength:")
        fixture_df = pd.read_csv("fixtures.csv")

        # Read in the live strengths
        live_strengths_ = pd.read_csv("live_strengths.csv").set_index("name")
        # Read in the base / preseason strengths
        team_strengths_ = pd.read_csv("team_ratings.csv")

        # We'll need the team select short name to exclude from our strengths merge
        team_abbr = team_strengths_.query(f"team=='{team_select}'").short_name.values[0]

        fix_set = fixture_df[(fixture_df["name"] == team_select) |
                             (fixture_df.name_a == team_select)].sort_values("event").head(int(lookahead[0]))




        fix_set.loc[fix_set["name"] == team_select, "opponent"] = fix_set.short_name_a
        fix_set.loc[fix_set["name_a"] == team_select, "opponent"] = fix_set.short_name

        fix_set.loc[fix_set["name"] == team_select, "opponent_name"] = fix_set.name_a
        fix_set.loc[fix_set["name_a"] == team_select, "opponent_name"] = fix_set.name


        fix_set["team_review"] = team_select


        fix_set_ = fix_set[["event", "team_review", "opponent", "opponent_name"]]



        # Now get the strengths of all possible opponents - drop out our selected review team
        opponent_strengths = live_strengths_.loc[live_strengths_.index != team_abbr][["strength",
                                                                                     "strength_change",
                                                                                     "cumul_change"]]
                                                                                     # "opp_str_ave",
                                                                                     # "opp_str_std"]]

        # Get our review team strength for comparison
        team_strength = live_strengths_.loc[live_strengths_.index == team_abbr][["strength",
                                                                                     "strength_change",
                                                                                     "cumul_change"]]

        team_styled = team_strength.style.bar(
                                             subset=['strength', 'strength_change', 'cumul_change'],
                                             color=['#e90052', '#04f5ff'],
                                             align='zero', vmin=-0.5,
                                             )

        st.write(team_styled.to_html(), unsafe_allow_html=True)
        st.write(" ")

        fix_strengths = fix_set_.merge(opponent_strengths, left_on="opponent", right_index=True)
        # Drop unecessary columns
        fix_strengths = fix_strengths.drop(["team_review", "opponent"], axis=1)
        fix_strengths.set_index("event", inplace=True)
        # Style the dataframe
        fix_styled = fix_strengths.style.bar(
                                             subset=['strength', 'strength_change', 'cumul_change'],
                                             color=['#e90052', '#04f5ff'],
                                             align='zero', vmin=-0.5,
                                             )
        # st.dataframe(fix_styled.to_html(), unsafe_allow_html=True)



        st.write(f"{lookahead[0]} Game Fixture List: {team_select}")
        st.write(fix_styled.to_html(), unsafe_allow_html=True)

    # For our selected teams fixture list, plot out theirs and their opponents season history
    fdr_set = pd.read_csv("fdr_results.csv")

    # We now need the short name for each opponent
    opponents = team_strengths_[team_strengths_.team.isin(fix_strengths.opponent_name.tolist())]
    # Set a select box to check opponent results history and strength change progress
    opp_1, opp_2, opp_3 = st.columns(3)
    with opp_1:
        opp_select = st.selectbox("select an opponent to view results", opponents.short_name.values, )

    #
    tm_A = fdr_set.query(f"name=='{team_abbr}'").team.iloc[0]
    tm_B = fdr_set.query(f"name=='{opp_select}'").team.iloc[0]

    fdr_fig = fdr_plot(fdr_set=fdr_set, tm_A=tm_A, tm_B=tm_B)

    st.pyplot(fdr_fig)



    # PLayer detailed view
    list_of_players = fixture_diff_df.web_name.unique().tolist()
    player_to_view = st.selectbox("Select a player for detailed view", list_of_players,

                                  )

elif tool == "Recommender":

    st.subheader("Recommender Tool")


# END
