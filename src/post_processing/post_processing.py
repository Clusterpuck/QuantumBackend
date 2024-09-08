"""Driver to post-process param sweeper output"""

import sys
import os
import pandas as pd
import ast
import graph_storer as gs
import route_storer as rs

# cd to post_processing
# python post_processing.py "output" "test1" "Locations.json"

def post_process() -> None:
    input_name = sys.argv[1]
    output_file = sys.argv[2]
    orders_file = sys.argv[3]

    df = pd.read_csv(os.path.join("data", input_name + ".csv"))
    success_df = df[df["relative_cost"] != -1]

    average(success_df, output_file)
    best(success_df, output_file)
    failed_occurences(df, output_file)
    visualise_deliveries(df, orders_file)

def average(success_df, output_file) -> None:
    # Average performance
    average_df = (
        success_df.groupby(["cost_constraint_ratio", "chain_strength"])
        .agg({"relative_cost": "mean"})
        .reset_index()
    )
    gs.save_heatmap(
        average_df,
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        ("avg", output_file),
    )
    gs.save_contour_plot(
        average_df,
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        ("avg", output_file),
    )

def best(success_df, output_file) -> None:
    idx = success_df.groupby(["cost_constraint_ratio", "chain_strength"])[
        "relative_cost"
    ].idxmin()
    min_relative_cost_df = success_df.loc[idx].reset_index(drop=True)

    gs.save_heatmap(
        min_relative_cost_df,
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        ("best", output_file),
    )
    gs.save_contour_plot(
        min_relative_cost_df,
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        ("best", output_file),
    )

def failed_occurences(df, output_file) -> None:
    failed_routes_df = df[df["relative_cost"] == -1]
    if not failed_routes_df.empty:
        failed_routes_df = (
            failed_routes_df.groupby(
                ["cost_constraint_ratio", "chain_strength"]
            )
            .size()
            .reset_index(name="failed_routes_count")
        )
        gs.save_heatmap(
            failed_routes_df,
            "cost_constraint_ratio",
            "chain_strength",
            "failed_routes_count",
            ("fails", output_file),
        )

def visualise_deliveries(df, orders_file) -> None:
    for _, row in df.iterrows():
        cost_constraint_ratio = str(row["cost_constraint_ratio"])
        chain_strength = str(row["chain_strength"])
        trial = str(row["trial"])
        relative_cost = str(row["relative_cost"])
        filename = (
            f"{cost_constraint_ratio}_{chain_strength}_{trial}_{relative_cost}"
        )
        route_list = ast.literal_eval(row["route"])
        rs.create_graph(orders_file, route_list, "routes", filename)

if __name__ == '__main__':
    post_process()