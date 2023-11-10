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
                     width=1000, height=550

                     )

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
        plt.text(team_A.event[idx] - 0.3, team_A.strength[idx] + 0.025,
                 f"{team_A.opponent[idx]} ({team_A.details[idx][0].upper()}, {team_A['home/away'][idx]})", fontsize=8)

    for idx in range(1, team_B.shape[0]):
        plt.text(team_B.event[idx] - 0.3, team_B.strength[idx] - 0.025,
                 f"{team_B.opponent[idx]} ({team_B.details[idx][0].upper()}, {team_B['home/away'][idx]})", fontsize=8)

    plt.title("Plot of Future Opponent Results and Strength Updates", fontsize=16)
    plt.xlabel("Round", fontsize=14)
    plt.ylabel("Strength", fontsize=14)
    plt.legend(loc=2)

    return fig


