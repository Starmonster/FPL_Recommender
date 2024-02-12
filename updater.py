# In this file we'll store the code that updates the local data sets that store our latest
# information


# Import libraries
import streamlit as st
# Import libraries
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
from fpl_funcs import *

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

# Call all main file updates once a week!

fixture_diff = full_player_summary_build()
time.sleep(2)
updated_gw_data = updated_gw_data_build()
time.sleep(2)
squad_hist = squad_history_build(mgr_id=5497071)
time.sleep(2)
completed_df = comparison_df_build()
time.sleep(2)
gw_df_cumul = cumulative_gw_build()
time.sleep(2)
results, fdr_set, live_strengths, fixtures = fixtures_strengths_build()



# # Run code that calls in all latest data and updates and saves the local .csv files
# # Get the last completed gw
# # Call the main api endpoint
# url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
# r = requests.get(url)
# json = r.json()
#
# # Get and store the last endpoint
# events = pd.DataFrame(json["events"])
# completed_events = events.query("finished == True")
# # Store the last completed gw in the game
# last_completed_gw = completed_events.iloc[-1].id
#
# # Call in the last updated_gw_data file and see what was last gw to have full update
# current_gw_df = pd.read_csv("updated_gw_data.csv")
# # get the most recent gw update
# last_updated_gw = current_gw_df["round"].max()
#
# # Now check between our two values... if a new gw has completed: update, else: do nothing
# # We'll need the difference in gws as we may need more than one gw to call
# gw_diff = last_completed_gw - last_updated_gw
# print(gw_diff)
# if gw_diff == 0:
#     print("file up to date")
# else:
#     print(f"Updating file to last {gw_diff} gw/s...")
#     # Read in the latest player summaries -
#     players = pd.read_csv("full_player_summary.csv")
#     players = players.drop(["next_three", "next_five", "next_eight"], axis=1)
#
#     # Call the update data func from our fpl_funcs file with the number of missing gameweeks
#     new_gw_data = update_ply_data(no_gws=gw_diff)
#
#     # Now add the new_gw_data to the current gw df
#     updated_gw_data = current_gw_df.append(new_gw_data)
#     # Sort by element and round
#     updated_gw_data = updated_gw_data.sort_values(["element", "round"]).reset_index()
#     # Save updated data to file
#     updated_gw_data.to_csv("updated_gw_data.csv", index=False)
#
#     # Call in the latest mgr squad data
#     mgr_squad_df, transfer_hist = update_mgr_data(last_completed_gw=last_completed_gw, players=players)
#     # Save latest squad and transfer history to file
#     mgr_squad_df.to_csv("squad_history.csv", index=False)
#     transfer_hist.to_csv("transfer_history.csv", index=False)
#
#     plot_sets = build_plot_sets(squad_capt=mgr_squad_df, players=players,
#                                 complete_gw=last_completed_gw, gw_data=updated_gw_data,
#
#                                 )
#     plot_sets.to_csv("comparison_df.csv", index=False)
#
#     # BASIC PLAYER DATA and FIXTURE DIFFICULTY
#     print("Getting updated player data and fixture difficulty...")
#     main_df = pd.DataFrame(json["elements"])
#     teams_df = pd.DataFrame(json["teams"])
#
#     # ply_details = main_df[["code", "id", "element_type", "web_name", "first_name",
#     #                        "second_name", "team", "total_points", "now_cost",
#     #                        ]]
#     #
#     team_details = teams_df[["id", "name", "short_name"]]
#     #
#     # ply_details_ = ply_details.rename(columns={"id": "element"})
#     team_details_ = team_details.rename(columns={"id": "team_id"})
#     #
#     # player_basic = ply_details_.merge(team_details_, left_on="team", right_on="team_id")
#     #
#     # opposition_team = team_details_.rename(columns={"team_id": "opp_team_id", "name": "opp_name"})
#     #
#     # round_data_ = player_basic.merge(key_metrics, on="element")
#     # # Now merge the opposition team with teams
#     # round_data = round_data_.merge(opposition_team, left_on="opponent_team", right_on="opp_team_id")
#     # round_data["ict_index"] = round_data.ict_index.astype(float)
#
#     # BUild a replacement set for the round set that just has totals, basic info and team info
#     latest_data = main_df[["code", "id", "element_type", "web_name", "first_name", "second_name", "team",
#                            "total_points", "now_cost", "news", "minutes", "goals_scored", "assists",
#                            "clean_sheets", "goals_conceded", "penalties_saved", "penalties_missed", "yellow_cards",
#                            "red_cards", "saves", "bonus", "bps", "ict_index"
#                            ]]
#     # Add in our players team info
#     latest_data_ = latest_data.merge(team_details_, left_on="team", right_on="team_id")
#     # Rename the id feature to element to keep consistent
#     latest_data_ = latest_data_.rename(columns={"id": "element"})
#
#     fixture_diff = update_fixture_diff(latest_data_=latest_data_, team_details_=team_details_)
#
#     data = fixture_diff.round(2)[["code", "element", "web_name", "minutes", "now_cost", "name", "element_type",
#                                   "bps", "total_points", "bonus", "ict_index", "fdr_one", "fdr_three",
#                                   "fdr_five", "fdr_eight", "fdr_three_std", "fdr_five_std", "fdr_eight_std",
#                                   "next_eight", "news"
#                                   ]]
#
#     data.to_csv("fixture_difficulty.csv", index=False) # Note this is the same file as "key_player_summary.csv"
#
#
#     # UPDATE CUMULATIVE GWS DF
#     print("Updating cumulative_gws.csv")
#     # First get the gw data for each player
#     updated_gw_data = pd.read_csv("updated_gw_data.csv")
#     # Now get the key player summary to access some basic player data
#     key_player_summary = pd.read_csv("fixture_difficulty.csv")
#     # Get the team ratings to build the historic FDR set
#     team_ratings = pd.read_csv("team_ratings.csv")
#     # Get the manager squad history
#     squad_df_ = pd.read_csv("squad_history.csv")
#
#     # We'll want only the data that we'll be using to plot and visualise - just key stats
#     gw_df = updated_gw_data[["element", "was_home", "fixture", "opponent_team", "total_points", "round", "minutes",
#                              "bps", "ict_index", "value", "transfers_balance", "selected"
#                              ]]
#
#     # Let's add a bit more basic player data - code, web_name, element_type
#     gw_df = gw_df.merge(key_player_summary[["element", "code", "web_name", "element_type", "name"]], on="element")
#     # Merge the team ratings and clean features
#     gw_df = gw_df.merge(team_ratings[["team_h", "team_rating", "short_name"]],
#                         left_on="opponent_team", right_on="team_h"
#                         ).drop("team_h", axis=1).rename(
#         columns={"short_name": "opposition", "team_rating": "opp_rating"})
#
#     # Reorder features in a sensible manner
#     gw_df = gw_df[["code", "element", "web_name", "element_type", "name", "value", "round", "was_home", "fixture",
#                    "opponent_team", "opposition", "opp_rating", "total_points", "minutes", "bps", "ict_index",
#                    "transfers_balance", "selected",
#                    ]]
#
#     # Now build cumulative features for the key performance metrics ict, bps, total_points
#     gw_df_cumul = pd.DataFrame()
#     for code_ in tqdm(gw_df.code.unique()):
#         ply_df = gw_df.query(f"code=={code_}")
#         ply_df = ply_df.sort_values("round")
#
#         ply_df["tp_cumul"] = ply_df.total_points.cumsum()
#         ply_df["bps_cumul"] = ply_df.bps.cumsum()
#         ply_df["ict_cumul"] = ply_df.ict_index.cumsum()
#         ply_df["mins_cumul"] = ply_df.minutes.cumsum()
#         ply_df["sel_size"] = ply_df.selected / 500
#
#         #  Build the average historic difficulty
#         ave_rating = ply_df.opp_rating.rolling(window=ply_df.shape[0], min_periods=1).mean().round(2)
#         ply_df.insert(12, "mean_rating", ave_rating)
#
#         gw_df_cumul = gw_df_cumul.append(ply_df)
#
#     # Build feature to measure the difficulty weighted points - a point againts MANCITY > pt against Shef Utd
#     gw_df_cumul["weighted_pts"] = gw_df_cumul.tp_cumul / (1 / gw_df_cumul.mean_rating)
#     gw_df_cumul["weighted_gw_pts"] = gw_df_cumul.total_points / (1 / gw_df_cumul.opp_rating)
#
#     # Isolate the merge feature from my_squad
#     squad_df_merge = squad_df_[["element", "round", "position"]]
#
#     # Merge with the master cumulative st
#     gw_df_cumul = gw_df_cumul.merge(squad_df_merge, on=["element", "round"], how="left").fillna(0)
#     # Fill nans with 0, x==0 will be proxy for "not in gw squad", x>0 means "in gw squad"
#
#     # Assign a plot color
#     gw_df_cumul["plot_color"] = gw_df_cumul.position.apply(lambda x: '#ff7f0e' if x > 0 else "#1f77b4")
#
#     # Send to csv
#     gw_df_cumul.to_csv("cumulative_gws.csv", index=False)
#
#
#     # FIXTURES DF
#     # Call in the base team strengths
#     team_strengths_ = pd.read_csv("team_strengths.csv")
#     # GET THE FIXTURE LIST
#     # First Call in all fixtures from end point
#     fix_ep = "https://fantasy.premierleague.com/api/fixtures/"
#     teams = ""
#     f = requests.get(fix_ep)
#     # Let's transform the response body into a JSON object
#     fix_json = f.json()
#     fixtures = pd.DataFrame(fix_json)
#     fixtures.kickoff_time = pd.to_datetime(fixtures.kickoff_time)
#     fixtures = fixtures[["event", "id", "team_a", "team_a_score", "team_h", "team_h_score",
#                          "team_h_difficulty", "team_a_difficulty"]]
#     fixtures = fixtures.rename(columns={"id": "match_id"})
#
#     # Isolate the fixtures, ignoring all completed gws
#     # last_complete_gw = 11 # Last completed gw is set at top of this updater page
#     future_fix = fixtures.query(f"event > {last_completed_gw}")
#
#     # Merge home teams
#     future_fix_ = future_fix.merge(team_strengths_[["id", "name",
#                                                     "short_name", "strength"]],
#                                    left_on="team_h", right_on="id").drop("id", axis=1)
#     # Merge away teams
#     future_fix_ = future_fix_.merge(team_strengths_[["id", "name",
#                                                      "short_name", "strength"]],
#                                     left_on="team_a", right_on="id", suffixes=("", "_a"))
#
#     # Organise the features
#     fixture_df = future_fix_[["event", "match_id", "team_h", "team_h_difficulty", "strength", "short_name", "name",
#                               "team_h_score", "team_a_score", "name_a", "short_name_a", "strength_a",
#                               "team_a_difficulty",
#                               "team_a"
#                               ]]
#     # Send the fixtures set to file
#     fixture_df.to_csv("fixtures.csv", index=False)
#
#     # LIVE STRENGTHS DATASEST
#     print("Building live strengths dataset...")
#     # First we'll need to build a results set
#     results = fixtures.dropna()
#
#     # Let's add the home / away team names for clarity
#     # Merge home teams
#     results_ = results.merge(team_strengths_[["id", "short_name", "strength"]], left_on="team_h", right_on="id")
#     results_["home_team"] = results_.short_name
#     results_["home_strength"] = results_.strength
#     results_ = results_.drop(["short_name", "id", "strength"], axis=1)
#
#     results_ = results_.merge(team_strengths_[["id", "short_name", "strength"]], left_on="team_a", right_on="id")
#     results_["away_team"] = results_.short_name
#     results_["away_strength"] = results_.strength
#
#     results_ = results_.drop(["short_name", "id", "strength"], axis=1)
#
#     results_ = results_[["event", "match_id", "team_h", "team_h_difficulty", "home_strength", "home_team", "team_h_score",
#          "team_a_score", "away_team", "away_strength", "team_a_difficulty", "team_a"]]
#
#     # Send latest results to file
#     results_.to_csv("latest_results.csv", index=False)
#
#     # Build a result feature to store the final result of each match
#     result = []
#
#     for ind, row in results_.iterrows():
#
#         if row["team_h_score"] > row["team_a_score"]:
#             result.append("home_win")
#         elif row["team_a_score"] > row["team_h_score"]:
#             result.append("away_win")
#         else:
#             result.append("draw")
#
#     results_["result"] = result
#     results_ = results_.sort_values(["event", "match_id"])
#
#     # Now build a function to calculate the updated team strengths based on the experimental scoring strategy
#
#     # Set empty lists to store the updated home and away strengths
#     home_updated_strength = []
#     away_updated_strength = []
#
#     # Set some arrays to store the win / lose / draw metrics as per strategy link
#     winners_array = np.array([[0.1, 0.2, 0.3, 0.35], [0.05, 0.1, 0.2, 0.3],
#                               [0, 0.05, 0.1, 0.2], [0, 0, 0.05, 0.1]])
#
#     losers_array = np.array([[-0.1, -0.2, -0.3, -0.35], [0, -0.1, -0.2, -0.3],
#                              [0, 0, -0.1, -0.2], [0, 0, 0, -0.1]])
#
#     draw_array = np.array([[0.05, 0.1, 0.2, 0.3], [-0.1, 0.05, 0.1, 0.2],
#                            [-0.2, -0.1, 0.05, 0.1], [-0.3, -0.2, -0.1, 0.05]])
#
#     # CAll in fdr set from builder function
#     fdr_set = build_fdr_set(results_, team_strengths_)
#
#     # Build live strengths
#     ## Display the live strength table
#     live_strengths = fdr_set.groupby("name").last().sort_values("strength",
#                                                                 ascending=False)  # [["strength", "cumul_change"]]
#     # live_strengths = live_strengths[["event", "team", "strength", "strength_change", "cumul_change"]]
#     # live_strengths = live_strengths.merge(ave_fdr_df, left_index=True, right_index=True)
#     live_strengths_ = live_strengths.rename(columns={"event": "last_gw", "details": "last_match"})
#
#     # Send live strengths to file
#     live_strengths_.to_csv("live_strengths.csv", index=True)
#
#     # Send fdr set to file
#     fdr_set.to_csv("fdr_results.csv", index=False)
#
#
#
#
#
#
#
#
#
#
#
#
#
