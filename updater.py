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


# Run code that calls in all latest data and updates and saves the local .csv files
# Get the last completed gw
# Call the main api endpoint
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
json = r.json()

# Get and store the last endpoint
events = pd.DataFrame(json["events"])
completed_events = events.query("finished == True")
# Store the last completed gw in the game
last_completed_gw = completed_events.iloc[-1].id

# Call in the last updated_gw_data file and see what was last gw to have full update
current_gw_df = pd.read_csv("updated_gw_data.csv")
# get the most recent gw update
last_updated_gw = current_gw_df["round"].max()

# Now check between our two values... if a new gw has completed: update, else: do nothing
# We'll need the difference in gws as we may need more than one gw to call
gw_diff = last_completed_gw - last_updated_gw
print(gw_diff)
if gw_diff == 0:
    print("file up to date")
else:
    print(f"Updating file to last {gw_diff} gw/s...")
    # Read in the latest player summaries -
    players = pd.read_csv("full_player_summary.csv")
    players = players.drop(["next_three", "next_five", "next_eight"], axis=1)

    # Call the update data func from our fpl_funcs file with the number of missing gameweeks
    new_gw_data = update_ply_data(no_gws=gw_diff)

    # Now add the new_gw_data to the current gw df
    updated_gw_data = current_gw_df.append(new_gw_data)
    # Sort by element and round
    updated_gw_data = updated_gw_data.sort_values(["element", "round"]).reset_index()
    # Save updated data to file
    updated_gw_data.to_csv("updated_gw_data.csv", index=False)

    # Call in the latest mgr squad data
    mgr_squad_df, transfer_hist = update_mgr_data(last_completed_gw=last_completed_gw, players=players)
    # Save latest squad and transfer history to file
    mgr_squad_df.to_csv("squad_history.csv", index=False)
    transfer_hist.to_csv("transfer_history.csv", index=False)

    plot_sets = build_plot_sets(squad_capt=mgr_squad_df, players=players,
                                complete_gw=last_completed_gw, gw_data=updated_gw_data,

                                )
    plot_sets.to_csv("comparison_df.csv", index=False)

    # BASIC PLAYER DATA and FIXTURE DIFFICULTY
    print("Getting updated player data and fixture difficulty...")
    main_df = pd.DataFrame(json["elements"])
    teams_df = pd.DataFrame(json["teams"])

    # ply_details = main_df[["code", "id", "element_type", "web_name", "first_name",
    #                        "second_name", "team", "total_points", "now_cost",
    #                        ]]
    #
    team_details = teams_df[["id", "name", "short_name"]]
    #
    # ply_details_ = ply_details.rename(columns={"id": "element"})
    team_details_ = team_details.rename(columns={"id": "team_id"})
    #
    # player_basic = ply_details_.merge(team_details_, left_on="team", right_on="team_id")
    #
    # opposition_team = team_details_.rename(columns={"team_id": "opp_team_id", "name": "opp_name"})
    #
    # round_data_ = player_basic.merge(key_metrics, on="element")
    # # Now merge the opposition team with teams
    # round_data = round_data_.merge(opposition_team, left_on="opponent_team", right_on="opp_team_id")
    # round_data["ict_index"] = round_data.ict_index.astype(float)

    # BUild a replacement set for the round set that just has totals, basic info and team info
    latest_data = main_df[["code", "id", "element_type", "web_name", "first_name", "second_name", "team",
                           "total_points", "now_cost", "news", "minutes", "goals_scored", "assists",
                           "clean_sheets", "goals_conceded", "penalties_saved", "penalties_missed", "yellow_cards",
                           "red_cards", "saves", "bonus", "bps", "ict_index"
                           ]]
    # Add in our players team info
    latest_data_ = latest_data.merge(team_details_, left_on="team", right_on="team_id")
    # Rename the id feature to element to keep consistent
    latest_data_ = latest_data_.rename(columns={"id": "element"})

    fixture_diff = update_fixture_diff(latest_data_=latest_data_, team_details_=team_details_)

    data = fixture_diff.round(2)[["code", "element", "web_name", "minutes", "now_cost", "name", "element_type",
                                  "bps", "total_points", "bonus", "ict_index", "fdr_one", "fdr_three",
                                  "fdr_five", "fdr_eight", "fdr_three_std", "fdr_five_std", "fdr_eight_std",
                                  "next_eight", "news"
                                  ]]

    data.to_csv("fixture_difficulty.csv", index=False)

    # WORK TO BE COMPLETED
    # UPDATE CUMULATIVE GWS DF





