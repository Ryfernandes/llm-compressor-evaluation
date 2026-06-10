# Simple, dummy pipeline with an artifact to demonstrate KFP functionality and verify pipeline execution

from typing import List, Optional

from kfp import dsl, compiler

PIPELINE_NAME = "dummy_artifact"

@dsl.component
def get_shooting_statistics(
    output_raw: dsl.Output[dsl.Artifact], seed: int, num_shots: int
) -> None:
    import random
    import json

    random.seed(seed)

    players = [
        {
            "name": "Wemby",
            "team": "Spurs",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Fox",
            "team": "Spurs",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Vassell",
            "team": "Spurs",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Champagnie",
            "team": "Spurs",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Castle",
            "team": "Spurs",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Brunson",
            "team": "Knicks",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Hart",
            "team": "Knicks",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Bridges",
            "team": "Knicks",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Anunoby",
            "team": "Knicks",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
        {
            "name": "Towns",
            "team": "Knicks",
            "points": 0,
            "shots": 0,
            "made": 0,
        },
    ]

    for i in range(num_shots):
        player = random.choice(players)
        points = random.choices([0, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        player["shots"] += 1
        player["points"] += points
        if points > 0:
            player["made"] += 1
    
    with open(output_raw.path, "w") as f:
        f.write(json.dumps(players))

@dsl.component
def get_shooting_percentage(
    input_raw: dsl.Input[dsl.Artifact], output_averages: dsl.Output[dsl.Artifact]
) -> None:
    import json

    with open(input_raw.path, "r") as f:
        players = json.loads(f.read())
    
    for player in players:
        player["fg%"] = player["made"] / player["shots"] if player["shots"] > 0 else 0
    
    with open(output_averages.path, "w") as f:
        f.write(json.dumps(players))
    
@dsl.component
def report_most_points(
    input_points: dsl.Input[dsl.Artifact]
) -> str:
    import json

    with open(input_points.path, "r") as f:
        players = json.loads(f.read())
    
    most_points = max(players, key=lambda p: p["points"])
    return f'{most_points["name"]}: {most_points["points"]} points'

@dsl.component
def report_best_percentage(
    input_percentages: dsl.Input[dsl.Artifact]
) -> str:
    import json

    with open(input_percentages.path, "r") as f:
        players = json.loads(f.read())
    
    best_percentage = max(players, key=lambda p: p["fg%"])
    return f'{best_percentage["name"]}: {best_percentage["fg%"] * 100:.2f}%'

@dsl.component
def report_game_result(
    input_points: dsl.Input[dsl.Artifact]
) -> str:
    import json

    with open(input_points.path, "r") as f:
        points_players = json.loads(f.read())
    
    team_scores = {}
    
    for player in points_players:
        if player["team"] not in team_scores:
            team_scores[player["team"]] = 0
        team_scores[player["team"]] += player["points"]
    
    return ", ".join([f"{team}: {score}" for team, score in team_scores.items()])

@dsl.component
def combine_results(
    most_points: str, best_percentage: str, game_result: str
) -> str:
    return f"{most_points}\n{best_percentage}\n{game_result}"

@dsl.pipeline
def basketball_pipeline(seed: int=1234, num_shots: int=100) -> str:
    stats_task = get_shooting_statistics(seed=seed, num_shots=num_shots)
    most_points_task = report_most_points(input_points=stats_task.outputs["output_raw"])
    shooting_percentage_task = get_shooting_percentage(input_raw=stats_task.outputs["output_raw"])
    game_result_task = report_game_result(input_points=stats_task.outputs["output_raw"])
    best_percentage_task = report_best_percentage(input_percentages=shooting_percentage_task.outputs["output_averages"])
    combine_results_task = combine_results(
        most_points=most_points_task.output,
        best_percentage=best_percentage_task.output,
        game_result=game_result_task.output
    )
    return combine_results_task.output

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=basketball_pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )