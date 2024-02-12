# This file will host any bespoke functions required to run and maintain the application

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

import plotly_express as px
import datetime

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")


def full_player_summary_build():
    """
    This function builds a dataset that stores all players updated stats. This includes:
    Primary Stats -
    * Total Points
    * BPS / ICT
    * Current Value
    * Goals Scored
    Secondary Stats -
    * Clean Sheets
    * Penalties Concede
    * Total Minutes
    Fixture Difficulty -
    * 1/3/5/8 FDR metric
    * Actual fixtures to be played


    """

    # Set our fantasy football api endpoint
    main_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

    # Create a response object
    main_end = requests.get(main_url)
    # Let's transform the response body into a JSON object
    main_data = main_end.json()
    main_df = pd.DataFrame(main_data["elements"])
    teams_df = pd.DataFrame(main_data["teams"])

    ply_details = main_df[["code", "id", "element_type", "web_name", "first_name",
                           "second_name", "team", "total_points", "now_cost",
                           ]]

    team_details = teams_df[["id", "name", "short_name"]]

    ply_details_ = ply_details.rename(columns={"id": "element"})
    team_details_ = team_details.rename(columns={"id": "team_id"})

    player_basic = ply_details_.merge(team_details_, left_on="team", right_on="team_id")

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

    # First Call in all fixtures from end point
    fix_ep = "https://fantasy.premierleague.com/api/fixtures/"
    teams = ""
    f = requests.get(fix_ep)
    # Let's transform the response body into a JSON object
    fix_json = f.json()
    fixtures = pd.DataFrame(fix_json)
    fixtures.kickoff_time = pd.to_datetime(fixtures.kickoff_time)
    # Get last completed week
    last_week_index = pd.DataFrame(main_data["events"]).query("finished == True").iloc[-1].id
    # Get rid of the wierd top two rows as these appear to be some kind of error
    fixtures = fixtures.loc[last_week_index:]

    # Create a blank master fdr dataframe
    fixture_diff = pd.DataFrame()

    # Iterate thro all teams
    for team_ref in tqdm(latest_data_.team_id.unique()):
        print(team_ref, type(team_ref))

        # Get the team's fixtures for full season and isolate key features
        all_fixs = fixtures.query(f"team_a == {team_ref} or team_h == {team_ref}")[["event", "kickoff_time",
                                                                                    "team_a", "team_h",
                                                                                    "team_h_difficulty",
                                                                                    "team_a_difficulty"]]

        # Set a blank dataframe to store teams fixture data
        all_fix_rev = pd.DataFrame()
        # We iterate through each fixture to get more details about the fixture
        for ind, row in all_fixs.iterrows():

            fix_dp = all_fixs.loc[ind].to_frame().T
            if fix_dp.team_h.values[0] == team_ref:
                # Merge fixtures with away opposition team info if our index team is at home
                fix_dp = fix_dp.merge(team_details_, left_on="team_a", right_on="team_id")
                fix_dp["is_home"] = True
                fix_dp["fdr"] = row["team_h_difficulty"]  # Our index team is at home
                fix_dp["opp_loc"] = f"{fix_dp['name'].values[0]} (h)"  # Who is opponent and where is game for our team

            else:
                # Merge fixtures with home opposition team info if our index team is away
                fix_dp = fix_dp.merge(team_details_, left_on="team_h", right_on="team_id")
                fix_dp["is_home"] = False
                fix_dp["fdr"] = row["team_a_difficulty"]  # our team is away from home
                fix_dp["opp_loc"] = f"{fix_dp['name'].values[0]} (a)"  # Who is opponent and where is game for our team

            all_fix_rev = all_fix_rev.append(fix_dp)

        # Reset the index
        all_fix_rev = all_fix_rev.reset_index(drop=True)
        #     print(all_fix_rev)
        # Ensure the event reference is stored as integers
        all_fix_rev.event = all_fix_rev.event.astype(int)

        # Now get only the future data - for this test run it's all games after the last completed gw
        future_opps = all_fix_rev[all_fix_rev.event > last_week_index]

        # Start to construct the fixture difficulty dataframe
        fdr_df = latest_data_.copy()
        #     future_opps = all_fix_rev[all_fix_rev.event>1]
        # Get the index round - the round that has just passed into history - we want all future fixture diffs
        #     index_round = fdr_df["round"].loc[0]

        # Practice with Arsenal
        team_fdr = fdr_df[fdr_df.team == team_ref]

        # Ok now build the data
        # Get the fixture list
        next_fixture = future_opps.head(1).opp_loc.values[0]
        next_three_fixtures = future_opps.head(3).opp_loc.to_list()
        next_five_fixtures = future_opps.head(5).opp_loc.to_list()
        next_eight_fixtures = future_opps.head(8).opp_loc.to_list()

        # Get the FDR for each lookahead
        next_fix_fdr = future_opps.head(1).fdr.values[0]
        next_three_fdr = future_opps.head(3).fdr.sum() / 3
        next_five_fdr = future_opps.head(5).fdr.sum() / 5
        next_eight_fdr = future_opps.head(8).fdr.sum() / 8

        # Get the stdded for the fdrs - we might have a low fdr mean but the variance could still be relatively high
        next_three_std = np.std(future_opps.head(3).fdr)
        next_five_std = np.std(future_opps.head(5).fdr)
        next_eight_std = np.std(future_opps.head(8).fdr)

        # Now build the fdr features
        team_fdr["fdr_one"] = next_fix_fdr
        team_fdr["fdr_three"] = next_three_fdr
        team_fdr["fdr_five"] = next_five_fdr
        team_fdr["fdr_eight"] = next_eight_fdr

        # Now build fdr stddev features
        team_fdr["fdr_three_std"] = next_three_std
        team_fdr["fdr_five_std"] = next_five_std
        team_fdr["fdr_eight_std"] = next_eight_std

        # Now build the fixture lists
        # df['col5'] = [v for _ in range(len(df))]
        team_fdr["next_game"] = [next_fixture for _ in range(len(team_fdr))]
        team_fdr["next_three"] = [next_three_fixtures for _ in range(len(team_fdr))]
        team_fdr["next_five"] = [next_five_fixtures for _ in range(len(team_fdr))]
        team_fdr["next_eight"] = [next_eight_fixtures for _ in range(len(team_fdr))]

        # Update the fixture_diff dataframe
        fixture_diff = fixture_diff.append(team_fdr)

        # Check data type , nans  etc
    # Change ict index to float type
    fixture_diff["ict_index"] = fixture_diff.ict_index.astype(float)

    data = fixture_diff.round(2)[["code", "element", "web_name", "minutes", "now_cost", "name", "element_type",
                                  "bps", "total_points", "bonus", "ict_index", "fdr_one", "fdr_three",
                                  "fdr_five", "fdr_eight", "fdr_three_std", "fdr_five_std", "fdr_eight_std",
                                  "next_eight", "news"
                                  ]]

    data.to_csv("fixture_difficulty.csv", index=False)
    fixture_diff.to_csv("full_player_summary.csv", index=False)

    return fixture_diff


def updated_gw_data_build():
    """
    This function stores all game data for every player on a gw basis. This includes a player's opposition and KO,
    Selection data, metric totals

    """

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()

    # Get and store the last endpoint
    events = pd.DataFrame(json["events"])
    completed_events = events.query("finished == True")
    # Store the last completed gw in the game
    last_completed_gw = completed_events.iloc[-1].id

    # Read in the old gw data
    old_gw_data = pd.read_csv("updated_gw_data.csv")
    last_updated_gw = old_gw_data["round"].max()

    gw_diff = last_completed_gw - last_updated_gw

    # If our data is up to date we don't need to run the update loop
    # In this case we can print that everything is up to date and exit the function
    if last_completed_gw == last_updated_gw:
        print("GW data file is up to date!")
        return None

    elif last_completed_gw > last_updated_gw:
        print("Getting latest player data")

        last_gw = pd.DataFrame()
        # Loop through every player and get their last completed gw data
        for idx in range(1, gw_diff + 1):
            print(f"Updating {idx} gws ago")
            for id_ in tqdm(range(1, 725)):

                try:
                    player_url = f"https://fantasy.premierleague.com/api/element-summary/{id_}/"
                    # Create a response object
                    ply = requests.get(player_url)
                    # Let's transform the response body into a JSON object
                    ply_data = ply.json()

                    ply_df = pd.DataFrame(ply_data["history"][-idx], index=[0])

                    last_gw = last_gw.append(ply_df)

                    time.sleep(0.5)
                except:
                    print(id_)
                    continue

        # Append the last gw data to the "old" gw data

        updated_gw_data = old_gw_data.append(last_gw)

        # Sort by element and round
        updated_gw_data = updated_gw_data.sort_values(["element", "round"]).reset_index(drop=True)

        # Save to file
        updated_gw_data.to_csv("updated_gw_data.csv", index=False)

        return updated_gw_data


def squad_history_build(mgr_id):
    """
    Builds and stores every squad for any manager based on user input: manager id.
    Also saves a standalone csv for the latest squad at date for selected manager.
    """

    # First we need to get the transfer history numbers to see if we've ever spent 4pts for extra transfers
    address = f"https://fantasy.premierleague.com/api/entry/{mgr_id}/history"
    gw_sum_request = requests.get(address)
    gw_dict = gw_sum_request.json()

    # Store transfer hist numbers in a dataframe
    transfer_hist = pd.DataFrame(gw_dict["current"])
    # Get the extra over points spent on transfers
    transfer_expense = transfer_hist.event_transfers_cost.sum()

    # Send this manager's history to file
    transfer_hist.to_csv("transfer_history.csv", index=False)

    # Now build your full squad - note this builds from gw each time - this shouldn't be a hige time debt
    # But can be improved by simply updating the squad from the last save point
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()

    # Get and store the last endpoint
    events = pd.DataFrame(json["events"])
    completed_events = events.query("finished == True")
    # Store the last completed gw in the game
    last_completed_gw = completed_events.iloc[-1].id
    # Set an empty df to store all dataframes
    squad_df = pd.DataFrame()

    for gw in range(1, last_completed_gw + 1):
        # Call in each gw squad finalised with completed transfers
        gw_sum = f"https://fantasy.premierleague.com/api/entry/{mgr_id}/event/{gw}/picks/"
        gw_sum_request = requests.get(gw_sum)
        gw_dict = gw_sum_request.json()

        #  save as a dataframe and stack on to previous gws
        gw_squad = pd.DataFrame(gw_dict["picks"])
        # Add the gw feature
        gw_squad["round"] = gw
        squad_df = squad_df.append(gw_squad)
        time.sleep(0.5)

    # CLEAN UP DATE AND SAVE
    # Read in the updated player gameweek data
    gw_data = pd.read_csv("updated_gw_data.csv")
    # Read in the latest player summaries -
    players = pd.read_csv("full_player_summary.csv")
    players = players.drop(["next_three", "next_five", "next_eight"], axis=1)

    # Merge latest squad with key data
    squad_df_ = squad_df.merge(players[["code", "element", "web_name", "element_type", "name"]], on="element")

    squad_df_ = squad_df_.merge(gw_data[["element", "round", "total_points", "value", "selected", "minutes"]],
                                on=["element", "round"])
    # Sort with latest squad at the top of the frame
    squad_df_ = squad_df_.sort_values(["round", "position"], ascending=[False, True])
    #  Add a feature for points including captained pts
    squad_df_["points_scored"] = squad_df_.total_points * squad_df_.multiplier
    # Save to file
    squad_df_.to_csv("squad_history.csv", index=False)

    # Store the latest squad as a standalone dataframe
    my_squad = squad_df_.head(15)
    my_squad.to_csv("my_latest_squad.csv", index=False)

    return squad_df_


def comparison_df_build():
    """
    This function builds an alternative player summary suitable for plotting and comparisons.
    It's likely that this can be merged with the full plyer summary set at some point.

    """

    # Get last completed gw
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()

    # Get and store the last endpoint
    events = pd.DataFrame(json["events"])
    completed_events = events.query("finished == True")
    # Store the last completed gw in the game
    complete_gw = completed_events.iloc[-1].id

    # Read in the updated player gameweek data
    gw_data = pd.read_csv("updated_gw_data.csv")
    # Read in the latest player summaries -
    players = pd.read_csv("full_player_summary.csv")
    players = players.drop(["next_three", "next_five", "next_eight"], axis=1)
    # Read in my squad
    my_squad = pd.read_csv("my_latest_squad.csv")

    if gw_data["round"].max() != complete_gw:
        print("Updated gw file not up to date!")
        return None

    elif gw_data["round"].max() == complete_gw:
        # Get the completed gw data
        completed_df = gw_data[gw_data["round"] <= complete_gw]

        # Get the non sum features i.e. value...
        non_summed_df = gw_data[gw_data["round"] == complete_gw][["element", "value", "selected"]]
        # rename value to now_cost for this situation to align with other sets
        non_summed_df = non_summed_df.rename(columns={"value": "now_cost"})

        # Get required basic player info features unique to the players set for merge
        players_ = players[["element", "element_type", "web_name", "short_name", "next_game",
                            "fdr_one", "fdr_three", "fdr_five", "fdr_eight"]]

        # Confirm our features
        completed_cols = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                          "own_goals",
                          "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
                          "ict_index", "starts"]

        #  Group player stats and Sum the various metrics
        completed_df_ = completed_df.groupby("element").sum()[completed_cols].reset_index()
        # Rename old index to element in line with other sets
        completed_df_ = completed_df_.rename(columns={"index": "element"})
        # Merge the summed feats and the value / latest_df
        completed_df_ = completed_df_.merge(non_summed_df, on="element")

        # Merge with the unique players data
        completed_df_ = completed_df_.merge(players_, on="element")

        # Drop any duplicated players - where players have a double gameweek two rows are currently constructed
        # Note at this stage the duplicated datapoints will be identical once a gw is completed
        completed_df_ = completed_df_.drop_duplicates("element")

        # BUild the pts per pound feat
        completed_df_["pts_per_pound"] = round(completed_df_["total_points"] / (completed_df_["now_cost"] / 10), 2)
        # Build the pts per minute feat - this is actually pts per 10mins to make the values slightly clearer
        completed_df_["pts_per_min"] = round(completed_df_["total_points"] / (completed_df_["minutes"] / 10), 2)
        # Build the pts per ownner feat
        completed_df_["pts_per_owner"] = round(completed_df_["total_points"] / (completed_df_["selected"] / 1000), 4)

        # Build a plot color and plot marker size so we can differentiate our squad players
        colors = completed_df_.element.apply(lambda x: '#ff7f0e' if x in my_squad.element.tolist() else "#1f77b4")
        completed_df_["plot_color"] = colors

        sizes = completed_df_.element.apply(lambda x: 45 if x in my_squad.element.tolist() else 25)
        completed_df_["plot_size"] = sizes

        in_squad = completed_df_.element.apply(lambda x: True if x in my_squad.element.tolist() else False)
        completed_df_["in_squad"] = in_squad

        completed_df_.to_csv("comparison_df.csv", index=False)

        return completed_df_


def cumulative_gw_build():
    """
    Generates cumulative totals for each player gw so we can plot cumulative progress through the season.
    Initially intended to be used for plotly animated plots.

    """

    # READ IN REQUIRED DATASETS

    # First get the gw data for each player
    updated_gw_data = pd.read_csv("updated_gw_data.csv")
    # Now get the key player summary to access some basic player data
    player_summary = pd.read_csv("full_player_summary.csv")
    # Get the team ratings to build the historic FDR set
    team_ratings = pd.read_csv("team_ratings.csv")

    # We'll want only the data that we'll be using to plot and visualise - just key stats
    gw_df = updated_gw_data[["element", "was_home", "fixture", "opponent_team", "total_points", "round", "minutes",
                             "bps", "ict_index", "value", "transfers_balance", "selected"
                             ]]

    # Let's add a bit more basic player data - code, web_name, element_type
    gw_df = gw_df.merge(player_summary[["element", "code", "web_name", "element_type", "name"]], on="element")
    # Merge the team ratings and clean features
    gw_df = gw_df.merge(team_ratings[["team_h", "team_rating", "short_name"]],
                        left_on="opponent_team", right_on="team_h"
                        ).drop("team_h", axis=1).rename(
        columns={"short_name": "opposition", "team_rating": "opp_rating"})

    # Reorder features in a sensible manner
    gw_df = gw_df[["code", "element", "web_name", "element_type", "name", "value", "round", "was_home", "fixture",
                   "opponent_team", "opposition", "opp_rating", "total_points", "minutes", "bps", "ict_index",
                   "transfers_balance", "selected",
                   ]]

    # Now build cumulative features for the key performance metrics ict, bps, total_points
    gw_df_cumul = pd.DataFrame()
    for code_ in tqdm(gw_df.code.unique()):
        ply_df = gw_df.query(f"code=={code_}")
        ply_df = ply_df.sort_values("round")

        ply_df["tp_cumul"] = ply_df.total_points.cumsum()
        ply_df["bps_cumul"] = ply_df.bps.cumsum()
        ply_df["ict_cumul"] = ply_df.ict_index.cumsum()
        ply_df["mins_cumul"] = ply_df.minutes.cumsum()
        ply_df["sel_size"] = ply_df.selected / 500

        #  Build the average historic difficulty
        ave_rating = ply_df.opp_rating.rolling(window=ply_df.shape[0], min_periods=1).mean().round(2)
        ply_df.insert(12, "mean_rating", ave_rating)

        gw_df_cumul = gw_df_cumul.append(ply_df)

    # Build feature to measure the difficulty weighted points - a point againts MANCITY > pt against Shef Utd
    gw_df_cumul["weighted_pts"] = gw_df_cumul.tp_cumul / (1 / gw_df_cumul.mean_rating)
    gw_df_cumul["weighted_gw_pts"] = gw_df_cumul.total_points / (1 / gw_df_cumul.opp_rating)

    # Isolate the merge feature from my_squad
    squad_df_ = pd.read_csv("squad_history.csv")
    squad_df_merge = squad_df_[["element", "round", "position"]]

    # Merge with the master cumulative st
    gw_df_cumul = gw_df_cumul.merge(squad_df_merge, on=["element", "round"], how="left").fillna(0)
    # Fill nans with 0, x==0 will be proxy for "not in gw squad", x>0 means "in gw squad"

    # Assign a plot color
    gw_df_cumul["plot_color"] = gw_df_cumul.position.apply(lambda x: '#ff7f0e' if x > 0 else "#1f77b4")

    # Send to csv
    gw_df_cumul.to_csv("cumulative_gws.csv", index=False)

    return gw_df_cumul


def fixtures_strengths_build():
    """
    Builds and stores latest results, future fixtures and live strengths of all teams in the league.

    ---------------
    LIVE STRENGTHS
    ---------------
    Live strengths are calculated using the experimental coeficients to increase and decrease strength from teams
    based on their past results and the strengths of teams against which the results are earned. See the project
    sheets file for more information.

    One way to improve the live strengths is to use the updated strength metric for each change in strength. At the
    moment the metric uses the base strength of every team allocated at the start of the season.

    We could practice on all past seasons, the strength metric should track a teams strength change throughout the
    season and should roughly reflect the league table at the end of a season!! I think... !

    """

    # GET THE FIXTURE LIST
    # First Call in all fixtures from end point
    fix_ep = "https://fantasy.premierleague.com/api/fixtures/"
    teams = ""
    f = requests.get(fix_ep)
    # Let's transform the response body into a JSON object
    fix_json = f.json()
    fixtures = pd.DataFrame(fix_json)
    fixtures.kickoff_time = pd.to_datetime(fixtures.kickoff_time)
    fixtures = fixtures[["event", "id", "team_a", "team_a_score", "team_h", "team_h_score",
                         "team_h_difficulty", "team_a_difficulty"]]
    fixtures = fixtures.rename(columns={"id": "match_id"})

    # Get the results
    results = fixtures.dropna()

    # Get the team strengths
    team_strengths_ = pd.read_csv("team_strengths.csv")

    # Let's add the home / away team names for clarity
    # Merge home teams
    results_ = results.merge(team_strengths_[["id", "short_name", "strength"]], left_on="team_h", right_on="id")
    results_["home_team"] = results_.short_name
    results_["home_strength"] = results_.strength
    results_ = results_.drop(["short_name", "id", "strength"], axis=1)

    results_ = results_.merge(team_strengths_[["id", "short_name", "strength"]], left_on="team_a", right_on="id")
    results_["away_team"] = results_.short_name
    results_["away_strength"] = results_.strength

    results_ = results_.drop(["short_name", "id", "strength"], axis=1)

    results_ = results_[
        ["event", "match_id", "team_h", "team_h_difficulty", "home_strength", "home_team", "team_h_score",
         "team_a_score", "away_team", "away_strength", "team_a_difficulty", "team_a"]]

    # Build a result feature to store the final result of each match
    result = []
    for ind, row in results_.iterrows():

        if row["team_h_score"] > row["team_a_score"]:
            result.append("home_win")
        elif row["team_a_score"] > row["team_h_score"]:
            result.append("away_win")
        else:
            result.append("draw")

    results_["result"] = result

    # Store results set
    results_.to_csv("latest_results.csv", index=False)

    # Now build a function to calculate the updated team strengths based on the experimental scoring strategy

    # Set empty lists to store the updated home and away strengths
    home_updated_strength = []
    away_updated_strength = []

    # Set some arrays to store the win / lose / draw metrics as per strategy link
    winners_array = np.array([[0.1, 0.2, 0.3, 0.35], [0.05, 0.1, 0.2, 0.3],
                              [0, 0.05, 0.1, 0.2], [0, 0, 0.05, 0.1]])

    losers_array = np.array([[-0.1, -0.2, -0.3, -0.35], [0, -0.1, -0.2, -0.3],
                             [0, 0, -0.1, -0.2], [0, 0, 0, -0.1]])

    draw_array = np.array([[0.05, 0.1, 0.2, 0.3], [-0.1, 0.05, 0.1, 0.2],
                           [-0.2, -0.1, 0.05, 0.1], [-0.3, -0.2, -0.1, 0.05]])

    # Set an empty master strength tracking set to append each teams rolling strengths to
    fdr_set = pd.DataFrame()

    # team_fdr = pd.DataFrame()
    for tm in tqdm(results_.team_h.unique()):
        print(tm)

        latest_strength = results_[results_.team_h == tm].home_strength.iloc[0]
        updated_strengths = [latest_strength]
        delta_strength = [0]
        event = [0]
        result_info = ["Pre Season"]
        opponent = ["Pre Season"]
        opp_strengths = ["Pre Season"]
        home_away = ["Pre Season"]

        # get the team set
        team_set = results_.query(f"team_h == {tm}or team_a == {tm}").sort_values("event")
        team_name = team_strengths_.query(f"id == {tm}").short_name.iloc[0]
        for ind, row in team_set.iterrows():

            if row["team_h"] == tm:
                team_strength = row["home_strength"]
                opp_strength = row["away_strength"]

                team_indx = team_strength - 2
                opp_indx = opp_strength - 2

                if row["result"] == "home_win":

                    # Get the fdr score from the array
                    fdr_score = winners_array[team_indx, opp_indx]
                    print("win", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"won at home vs {row.away_team}")
                    opponent.append(row.away_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "away_win":

                    # Get the fdr score from the array
                    fdr_score = losers_array[opp_indx, team_indx]
                    print("lose", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"lost at home vs {row.away_team}")
                    opponent.append(row.away_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "draw":

                    fdr_score = draw_array[team_indx, opp_indx]
                    print("draw", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"draw at home vs {row.away_team}")
                    opponent.append(row.away_team)
                    opp_strengths.append(opp_strength)

                home_away.append("H")




            elif row["team_a"] == tm:
                team_strength = row["away_strength"]
                opp_strength = row["home_strength"]

                team_indx = team_strength - 2
                opp_indx = opp_strength - 2

                if row["result"] == "away_win":

                    # Get the fdr score from the array
                    fdr_score = winners_array[team_indx, opp_indx]
                    print("win", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"won away vs {row.home_team}")
                    opponent.append(row.home_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "home_win":

                    # Get the fdr score from the array
                    fdr_score = losers_array[opp_indx, team_indx]
                    print("lose", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"lost away vs {row.home_team}")
                    opponent.append(row.home_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "draw":

                    fdr_score = draw_array[team_indx, opp_indx]
                    print("draw", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"draw away vs {row.home_team}")
                    opponent.append(row.home_team)
                    opp_strengths.append(opp_strength)

                home_away.append("A")

        team_fdr = pd.DataFrame([event, delta_strength, updated_strengths,
                                 result_info, home_away, opponent, opp_strengths]).T

        team_fdr.columns = ["event", "strength_change", "strength", "details", "home/away",
                            "opponent", "opp_strength"]
        team_fdr["cumul_change"] = team_fdr.strength_change.cumsum()
        team_fdr["team"] = tm
        team_fdr["name"] = team_name

        # Append our updated team srength to the master set
        fdr_set = fdr_set.append(team_fdr)

    fdr_set = fdr_set[["event", "name", "team", "strength", "strength_change",
                       "cumul_change", "details", "home/away", "opponent", "opp_strength"]]
    fdr_set["event"] = fdr_set.event.astype(int)
    fdr_set["cumul_change"] = fdr_set.cumul_change.astype("float").round(2)
    fdr_set["strength_change"] = fdr_set.strength_change.astype("float").round(2)
    fdr_set["strength"] = fdr_set.strength.astype("float").round(2)

    # Send fdr_set to file
    fdr_set.to_csv("fdr_results.csv", index=False)

    ## Build and store the live strength dataset
    live_strengths = fdr_set.groupby("name").last().sort_values("strength", ascending=False)
    live_strengths_ = live_strengths.rename(columns={"event": "last_gw", "details": "last_match"})
    # Send live strengths to file
    live_strengths_.to_csv("live_strengths.csv", index=True)

    ##### GET FIXTURE LIST
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()
    # Get and store the last endpoint
    events = pd.DataFrame(json["events"])
    completed_events = events.query("finished == True")
    # Store the last completed gw in the game
    last_completed_gw = completed_events.iloc[-1].id

    # Isolate the fixtures, ignoring all completed gws

    future_fix = fixtures.query(f"event > {last_completed_gw}")

    # Merge home teams
    future_fix_ = future_fix.merge(team_strengths_[["id", "name",
                                                    "short_name", "strength"]],
                                   left_on="team_h", right_on="id").drop("id", axis=1)
    # Merge away teams
    future_fix_ = future_fix_.merge(team_strengths_[["id", "name",
                                                     "short_name", "strength"]],
                                    left_on="team_a", right_on="id", suffixes=("", "_a"))

    # Organise the features
    fixture_df = future_fix_[["event", "match_id", "team_h", "team_h_difficulty", "strength", "short_name", "name",
                              "team_h_score", "team_a_score", "name_a", "short_name_a", "strength_a",
                              "team_a_difficulty",
                              "team_a"
                              ]]

    fixture_df.to_csv("fixtures.csv", index=False)

    return results_, fdr_set, live_strengths_, fixture_df


############### SUPERSEDED FUNCS #########

def update_ply_data(no_gws):

    """
    Updates the data to the last complete gw
    :return:
    """

    new_gw_data = pd.DataFrame()
    for idx in range(1, no_gws+1): # Ideally this will just be 1 gw behind, but this function can manage multiple gws
        for id_ in tqdm(range(1, 725)):

            try:
                player_url = f"https://fantasy.premierleague.com/api/element-summary/{id_}/"
                # Create a response object
                ply = requests.get(player_url)
                # Let's transform the response body into a JSON object
                ply_data = ply.json()

                ply_df = pd.DataFrame(ply_data["history"][-idx], index=[0])

                new_gw_data = new_gw_data.append(ply_df)

                time.sleep(0.5)
            except:
                print(id_)
                continue

    return new_gw_data


def update_mgr_data(last_completed_gw, players, mgr_id=5497071, ):

    # Call in my last finalised squad gw.. the last completed gw
    # mgr_id = 5497071
    # Set an empty df to store all dataframes
    squad_df = pd.DataFrame()
    # last_completed_gw = 10
    for gw in range(1, last_completed_gw + 1):
        # Call in each gw squad finalised with completed transfers
        gw_sum = f"https://fantasy.premierleague.com/api/entry/{mgr_id}/event/{gw}/picks/"
        gw_sum_request = requests.get(gw_sum)
        gw_dict = gw_sum_request.json()

        #  save as a dataframe and stack on to previous gws
        gw_squad = pd.DataFrame(gw_dict["picks"])
        # Add the gw feature
        gw_squad["round"] = gw
        squad_df = squad_df.append(gw_squad)
        time.sleep(0.5)

    # Read in the updated player gameweek data
    gw_data = pd.read_csv("updated_gw_data.csv")

    # Merge latest squad with key data
    squad_df_ = squad_df.merge(players[["code", "element", "web_name", "element_type", "name"]], on="element")

    squad_df_ = squad_df_.merge(gw_data[["element", "round", "total_points", "value", "selected", "minutes"]],
                                on=["element", "round"])
    # Sort with latest squad at the top of the frame
    squad_df_ = squad_df_.sort_values(["round", "position"], ascending=[False, True])
    # Add a column for the points scored adjusted for captained player etc
    squad_df_["points_scored"] = squad_df_.total_points * squad_df_.multiplier



    # Get transfer history
    address = f"https://fantasy.premierleague.com/api/entry/{mgr_id}/history"
    gw_sum_request = requests.get(address)
    gw_dict = gw_sum_request.json()

    # Store transfer hist numbers in a dataframe
    transfer_hist = pd.DataFrame(gw_dict["current"])

    # We can use the transfer history to monitor how many points we've spent on extra transfers
    # Among other things
    return squad_df_, transfer_hist

def update_fixture_diff(latest_data_, team_details_):
    # First Call in all fixtures from end point
    fix_ep = "https://fantasy.premierleague.com/api/fixtures/"
    teams = ""
    f = requests.get(fix_ep)
    # Let's transform the response body into a JSON object
    fix_json = f.json()
    fixtures = pd.DataFrame(fix_json)
    fixtures.kickoff_time = pd.to_datetime(fixtures.kickoff_time)
    last_week_index = 10
    # Get rid of the wierd top two rows as these appear to be some kind of error
    fixtures = fixtures.loc[last_week_index:]

    # Create a blank master fdr dataframe
    fixture_diff = pd.DataFrame()

    # Iterate thro all teams
    for team_ref in tqdm(latest_data_.team_id.unique()):
        print(team_ref, type(team_ref))

        # Get the team's fixtures for full season and isolate key features
        all_fixs = fixtures.query(f"team_a == {team_ref} or team_h == {team_ref}")[["event", "kickoff_time",
                                                                                    "team_a", "team_h",
                                                                                    "team_h_difficulty",
                                                                                    "team_a_difficulty"]]

        # Set a blank dataframe to store teams fixture data
        all_fix_rev = pd.DataFrame()
        # We iterate through each fixture to get more details about the fixture
        for ind, row in all_fixs.iterrows():

            fix_dp = all_fixs.loc[ind].to_frame().T
            if fix_dp.team_h.values[0] == team_ref:
                # Merge fixtures with away opposition team info if our index team is at home
                fix_dp = fix_dp.merge(team_details_, left_on="team_a", right_on="team_id")
                fix_dp["is_home"] = True
                fix_dp["fdr"] = row["team_h_difficulty"]  # Our index team is at home
                fix_dp["opp_loc"] = f"{fix_dp['name'].values[0]} (h)"  # Who is opponent and where is game for our team

            else:
                # Merge fixtures with home opposition team info if our index team is away
                fix_dp = fix_dp.merge(team_details_, left_on="team_h", right_on="team_id")
                fix_dp["is_home"] = False
                fix_dp["fdr"] = row["team_a_difficulty"]  # our team is away from home
                fix_dp["opp_loc"] = f"{fix_dp['name'].values[0]} (a)"  # Who is opponent and where is game for our team

            all_fix_rev = all_fix_rev.append(fix_dp)

        # Reset the index
        all_fix_rev = all_fix_rev.reset_index(drop=True)
        #     print(all_fix_rev)
        # Ensure the event reference is stored as integers
        all_fix_rev.event = all_fix_rev.event.astype(int)

        # Now get only the future data - for this test run it's all games after the last completed gw
        future_opps = all_fix_rev[all_fix_rev.event > last_week_index]

        # Start to construct the fixture difficulty dataframe
        fdr_df = latest_data_.copy()
        #     future_opps = all_fix_rev[all_fix_rev.event>1]
        # Get the index round - the round that has just passed into history - we want all future fixture diffs
        #     index_round = fdr_df["round"].loc[0]

        # Practice with Arsenal
        team_fdr = fdr_df[fdr_df.team == team_ref]

        # Ok now build the data
        # Get the fixture list
        next_fixture = future_opps.head(1).opp_loc.values[0]
        next_three_fixtures = future_opps.head(3).opp_loc.to_list()
        next_five_fixtures = future_opps.head(5).opp_loc.to_list()
        next_eight_fixtures = future_opps.head(8).opp_loc.to_list()

        # Get the FDR for each lookahead
        next_fix_fdr = future_opps.head(1).fdr.values[0]
        next_three_fdr = future_opps.head(3).fdr.sum() / 3
        next_five_fdr = future_opps.head(5).fdr.sum() / 5
        next_eight_fdr = future_opps.head(8).fdr.sum() / 8

        # Get the stdded for the fdrs - we might have a low fdr mean but the variance could still be relatively high
        next_three_std = np.std(future_opps.head(3).fdr)
        next_five_std = np.std(future_opps.head(5).fdr)
        next_eight_std = np.std(future_opps.head(8).fdr)

        # Now build the fdr features
        team_fdr["fdr_one"] = next_fix_fdr
        team_fdr["fdr_three"] = next_three_fdr
        team_fdr["fdr_five"] = next_five_fdr
        team_fdr["fdr_eight"] = next_eight_fdr

        # Now build fdr stddev features
        team_fdr["fdr_three_std"] = next_three_std
        team_fdr["fdr_five_std"] = next_five_std
        team_fdr["fdr_eight_std"] = next_eight_std

        # Now build the fixture lists
        # df['col5'] = [v for _ in range(len(df))]
        team_fdr["next_game"] = [next_fixture for _ in range(len(team_fdr))]
        team_fdr["next_three"] = [next_three_fixtures for _ in range(len(team_fdr))]
        team_fdr["next_five"] = [next_five_fixtures for _ in range(len(team_fdr))]
        team_fdr["next_eight"] = [next_eight_fixtures for _ in range(len(team_fdr))]

        # Update the fixture_diff dataframe
        fixture_diff = fixture_diff.append(team_fdr)

    fixture_diff["ict_index"] = fixture_diff.ict_index.astype(float)

    return fixture_diff


def build_plot_sets(squad_capt, players, complete_gw, gw_data):

    # First get my latest squad
    my_squad = squad_capt.head(15)

    # Get the most played element type at each positions
    # This will allow us to plot an average position, by element type at the plot function
    # Use the mode to get the element type that has taken the position in most gw
    pos_indices = []
    mode_type = []
    for pos_idx in range(1, 16):
        main_element = squad_capt.query(f"position == {pos_idx}").element_type.mode().iloc[0]
        pos_indices.append(pos_idx)
        mode_type.append(main_element)

    pos_element_df = pd.DataFrame([pos_indices, mode_type]).T
    pos_element_df = pos_element_df.rename(columns={0: "position", 1: "element_type"})
    pos_element_df = pos_element_df.set_index("position")

    # FOR FUTURE PLOTS #
    # Now get the total_points scored by each position, the average cost of each position, ave mins and ave popularity
    summed_points = squad_capt.groupby("position").sum()[["points_scored", "minutes"]]
    aved_feats = squad_capt.groupby("position").mean()[["selected", "value"]].round(1)

    aggregated_squad = summed_points.join(aved_feats)
    aggregated_squad = aggregated_squad.rename(columns={"value": "now_cost", "points_scored": "total_points"})
    aggregated_squad = aggregated_squad.join(pos_element_df)


    # BUILD THE MAIN PLOTTING SET
    # Get the completed gw data
    completed_df = gw_data[gw_data["round"] <= complete_gw]

    # Get the non sum features i.e. value...
    non_summed_df = gw_data[gw_data["round"] == complete_gw][["element", "value", "selected"]]
    # rename value to now_cost for this situation to align with other sets
    non_summed_df = non_summed_df.rename(columns={"value": "now_cost"})

    # Get required basic player info features unique to the players set for merge
    players_ = players[["element", "element_type", "web_name", "short_name", "next_game",
                        "fdr_one", "fdr_three", "fdr_five", "fdr_eight"]]

    # Confirm our features
    completed_cols = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                      "own_goals",
                      "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
                      "ict_index", "starts"]

    #  Group player stats and Sum the various metrics
    completed_df_ = completed_df.groupby("element").sum()[completed_cols].reset_index()
    # Rename old index to element in line with other sets
    completed_df_ = completed_df_.rename(columns={"index": "element"})
    # Merge the summed feats and the value / latest_df
    completed_df_ = completed_df_.merge(non_summed_df, on="element")

    # Merge with the unique players data
    completed_df_ = completed_df_.merge(players_, on="element")

    # Drop any duplicated players - where players have a double gameweek two rows are currently constructed
    # Note at this stage the duplicated datapoints will be identical once a gw is completed
    completed_df_ = completed_df_.drop_duplicates("element")

    # BUild the pts per pound feat
    completed_df_["pts_per_pound"] = round(completed_df_["total_points"] / (completed_df_["now_cost"] / 10), 2)
    # Build the pts per minute feat - this is actually pts per 10mins to make the values slightly clearer
    completed_df_["pts_per_min"] = round(completed_df_["total_points"] / (completed_df_["minutes"] / 10), 2)
    # Build the pts per ownner feat
    completed_df_["pts_per_owner"] = round(completed_df_["total_points"] / (completed_df_["selected"] / 1000), 4)

    # Build a plot color and plot marker size so we can differentiate our squad players
    colors = completed_df_.element.apply(lambda x: '#ff7f0e' if x in my_squad.element.tolist() else "#1f77b4")
    completed_df_["plot_color"] = colors

    sizes = completed_df_.element.apply(lambda x: 45 if x in my_squad.element.tolist() else 25)
    completed_df_["plot_size"] = sizes

    in_squad = completed_df_.element.apply(lambda x: True if x in my_squad.element.tolist() else False)
    completed_df_["in_squad"] = in_squad

    return completed_df_


def player_comp(comp_metric, position_sel, df_for_comp, my_squad):
    """
    A function to build a comparison set to compare your current squad players with the best available players
    """

    # We want total point to be in any comparison set regardless of the selected metric.
    # Only add comparison features if not total_points
    if comp_metric == "total_points":
        base_cols = ["web_name", "element", "element_type", "now_cost", "short_name", "total_points",
                     "pts_per_pound", "minutes", "pts_per_min", "pts_per_owner", "next_game"]
        my_squad_ = my_squad.drop(["total_points", "minutes", "element_type",
                                   "value", "selected", "web_name"], axis=1)

        my_summary = my_squad_.merge(df_for_comp, on="element", suffixes=("_gw", "_sum"))
        my_sum = my_summary[base_cols]
    else:
        base_cols = ["web_name", "element", "element_type", "now_cost", "short_name", "total_points",
                     comp_metric, "pts_per_pound", "minutes", "pts_per_min", "pts_per_owner", "next_game"]

        my_squad_ = my_squad.drop(["total_points", "minutes", "element_type",
                                   "value", "selected", "web_name"], axis=1)

        my_summary = my_squad_.merge(df_for_comp, on="element", suffixes=("_gw", "_sum"))
        my_sum = my_summary[base_cols]

    # Get our squad members in the requested position
    squad_position = my_sum[my_sum.element_type == position_sel]
    position_elements = squad_position.element

    # Get the top players from the league excluding any from our squad
    # Sort by the requested comparison metric

    top_in_position = (df_for_comp[(df_for_comp.element_type == position_sel) &
                                   (~df_for_comp.element.isin(position_elements))].sort_values(comp_metric,
                                                                                               ascending=False).head(
        5))[base_cols]

    # Add our squad players in and compare
    comparison = top_in_position.append(squad_position).sort_values(comp_metric, ascending=False)

    comparison = comparison.reset_index(drop=True).rename(columns={'index': "rank"})
    rank = list(range(1, comparison.shape[0] + 1))
    comparison.insert(0, "rank", rank)
    comparison.set_index("rank")



    return comparison, position_elements


def value_plots(completed_df_, value_by="now_cost", non_scorers=True,
                position=False, value_met=0, points_threshold=20
                ):
    """
    A function to display value plots based on a range of metrics
    1. Cost value ie points per million cost
    2. Time return ie points per 10 minuts played
    3. Under / Over Selected - ie points per 1000 managers owned.

    """
    # Build a dictionary to make bespoke text placement for each value metric
    text_dict = {"now_cost": []}  ### WIP

    # Remove non scoring players if function arguments reques this
    if non_scorers == True:
        scoring_players = completed_df_[completed_df_.total_points > 0]
    else:
        scoring_players = completed_df_[completed_df_.total_points >= 0]

    # If user has provided a position query - only plot that position value metrics
    if position != False:

        # Get the position queried, 1,2,3,4 / GK,DEF,MID,FWD
        scoring_players = scoring_players.query(f"element_type == {position}")

        #######

        base_cols = ["web_name", "element", "element_type", "now_cost", "short_name", "total_points",
                     "pts_per_pound", "minutes", "pts_per_min", "pts_per_owner", "selected", "next_game",
                     "plot_color", "plot_size"]
        query = scoring_players.query(f"{value_by} > {value_met} | total_points > {points_threshold}")[base_cols]

        #######

        positions = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]

        # Plot the position value
        plt.figure(figsize=(14, 8))

        plt.scatter(scoring_players[value_by], scoring_players.total_points,
                    c=scoring_players.plot_color, s=scoring_players.selected/10000)
        plt.xlabel(value_by.replace("_", " ").title(), fontsize=13)
        plt.ylabel("Total Points", fontsize=13)
        plt.title(f"Current Player Value - {positions[position - 1]}", fontsize=16)

        text_coords = []

        for ind, row in query.iterrows():
            text_plotted = False

            # If our text plot coordinate is too close to a previous text plot - move it
            for coord in text_coords:
                if abs(row["total_points"] - coord[0] < 3) or abs(row[value_by] - coord[1] < 1000000):  # 100
                    plt.text(x=row[value_by] - 1, y=row["total_points"] + 0.25, s=row["web_name"],
                             fontsize=9, rotation=25)
                    text_plotted = True
                    break

            text_coords.append([row["total_points"], row[value_by]])

            if text_plotted == False:
                #                 text_coords.append((row["total_points"], row[value_by]))
                plt.text(x=row[value_by] + 1, y=row["total_points"] - 0.15, s=row["web_name"],
                         fontsize=9, rotation=25)

        plt.tight_layout()
        plt.show()



    else:
        # We include all positions in the plot
        # Plot the position value
        plt.figure(figsize=(14, 8))
        plt.scatter(scoring_players[value_by], scoring_players.total_points)
        plt.xlabel(value_by.replace("_", " ").title(), fontsize=13)
        plt.ylabel("Total Points", fontsize=13)
        plt.title("Current Player Value", fontsize=16)

    #  plt.text(x=6247522, y=28, s="Estupinan")
    return scoring_players

def animated_player_plot(plot_df, max_x, min_x, max_y, metric_sel):
    # plot_df["plot_color"] = "#38003c"

    fig = px.scatter(plot_df, x=metric_sel, y="total_points",
                     animation_frame="round", animation_group="code",
                     range_x=[min_x, max_x], range_y=[0, max_y], size="sel_size",
                     hover_name="web_name", color="plot_color", opacity=0.9,
                     hover_data={'now_cost': True, 'round': False, "gw_points": True, "total_points": True,
                                              'minutes': True, 'plot_color': False, 'sel_size': False},
                     width=1000, height=550)

    # Change the legend to simply state whether you own payer or not
    newnames = {'#1f77b4': 'Not Selected', '#ff7f0e': 'Selected', "#e90052": "Scouting"}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                          )
                       )

    # Add text to the queried player set

    fig.update_layout(legend_title_text='Squad')
    fig.update_layout(title=f"Player Stats by Gameweek: {metric_sel}")

    # fig.show()

    return fig


def fdr_plot(fdr_set, tm_A, tm_B):
    fig, ax = plt.subplots(figsize=(12, 7))
    # plt.figure(figsize=(12, 7))
    # tm_A, tm_B = 20, 17
    team_A = fdr_set[fdr_set.team == tm_A].reset_index()
    team_B = fdr_set[fdr_set.team == tm_B].reset_index()
    ax.plot(team_A.event, team_A.strength, label=team_A.name.iloc[0], marker="o")
    ax.plot(team_B.event, team_B.strength, label=team_B.name.iloc[0], marker='o');

    st.write(team_A.event[1])
    for idx in range(1, team_A.shape[0]):
        plt.text(team_A.event[idx] - 0.3, team_A.strength[idx] + 0.05,
                 f"{team_A.opponent[idx]} ({team_A.details[idx][0].upper()}, "
                 f"{team_A['home/away'][idx]})", fontsize=8, color="darkblue")

    for idx in range(1, team_B.shape[0]):
        plt.text(team_B.event[idx] - 0.3, team_B.strength[idx] - 0.07,
                 f"{team_B.opponent[idx]} ({team_B.details[idx][0].upper()},"
                 f"{team_B['home/away'][idx]})", fontsize=8, color="darkred")

    plt.title(f"Rolling Strength Change - Fixture: {team_A.name.iloc[0]} vs {team_B.name.iloc[0]}", fontsize=16)
    plt.xlabel("GW", fontsize=14)
    plt.ylabel("Strength", fontsize=14)
    plt.legend(loc=1)

    return fig


def build_fdr_set(results_, team_strengths_):
    # Set an empty master strength tracking set to append each teams rolling strengths to
    fdr_set = pd.DataFrame()

    # team_fdr = pd.DataFrame()
    for tm in tqdm(results_.team_h.unique()):
        print(tm)

        latest_strength = results_[results_.team_h == tm].home_strength.iloc[0]
        updated_strengths = [latest_strength]
        delta_strength = [0]
        event = [0]
        result_info = ["Pre Season"]
        opponent = ["Pre Season"]
        opp_strengths = ["Pre Season"]
        home_away = ["Pre Season"]

        # get the team set
        team_set = results_.query(f"team_h == {tm}or team_a == {tm}").sort_values("event")
        team_name = team_strengths_.query(f"id == {tm}").short_name.iloc[0]
        for ind, row in team_set.iterrows():

            if row["team_h"] == tm:
                team_strength = row["home_strength"]
                opp_strength = row["away_strength"]

                team_indx = team_strength - 2
                opp_indx = opp_strength - 2

                if row["result"] == "home_win":

                    # Get the fdr score from the array
                    fdr_score = winners_array[team_indx, opp_indx]
                    print("win", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"won at home vs {row.away_team}")
                    opponent.append(row.away_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "away_win":

                    # Get the fdr score from the array
                    fdr_score = losers_array[opp_indx, team_indx]
                    print("lose", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"lost at home vs {row.away_team}")
                    opponent.append(row.away_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "draw":

                    fdr_score = draw_array[team_indx, opp_indx]
                    print("draw", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"draw at home vs {row.away_team}")
                    opponent.append(row.away_team)
                    opp_strengths.append(opp_strength)

                home_away.append("H")




            elif row["team_a"] == tm:
                team_strength = row["away_strength"]
                opp_strength = row["home_strength"]

                team_indx = team_strength - 2
                opp_indx = opp_strength - 2

                if row["result"] == "away_win":

                    # Get the fdr score from the array
                    fdr_score = winners_array[team_indx, opp_indx]
                    print("win", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"won away vs {row.home_team}")
                    opponent.append(row.home_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "home_win":

                    # Get the fdr score from the array
                    fdr_score = losers_array[opp_indx, team_indx]
                    print("lose", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"lost away vs {row.home_team}")
                    opponent.append(row.home_team)
                    opp_strengths.append(opp_strength)


                elif row["result"] == "draw":

                    fdr_score = draw_array[team_indx, opp_indx]
                    print("draw", fdr_score)

                    latest_strength = latest_strength + fdr_score

                    updated_strengths.append(latest_strength)
                    delta_strength.append(fdr_score)
                    event.append(row["event"])
                    result_info.append(f"draw away vs {row.home_team}")
                    opponent.append(row.home_team)
                    opp_strengths.append(opp_strength)

                home_away.append("A")

        team_fdr = pd.DataFrame([event, delta_strength, updated_strengths,
                                 result_info, home_away, opponent, opp_strengths]).T

        team_fdr.columns = ["event", "strength_change", "strength", "details", "home/away",
                            "opponent", "opp_strength"]
        team_fdr["cumul_change"] = team_fdr.strength_change.cumsum()
        team_fdr["team"] = tm
        team_fdr["name"] = team_name

        # Append our updated team srength to the master set
        fdr_set = fdr_set.append(team_fdr)
    fdr_set = fdr_set[["event", "name", "team", "strength", "strength_change",
                       "cumul_change", "details", "home/away", "opponent", "opp_strength"]]
    fdr_set["event"] = fdr_set.event.astype(int)

    return fdr_set



