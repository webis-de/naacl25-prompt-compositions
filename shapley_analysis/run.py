"""This script loads and inspects a random prompt file from the dataset."""
import copy
import os
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support
from shapiq.games import Game
from shapiq.exact import ExactComputer
from shapiq import powerset, InteractionValues

from si_graph import si_graph_plot

TIMESTAMP = "2024-07-31"  # folder name of the data dump


def _print_files(files_list: list[str]) -> None:
    """Print all files in the list.

    Args:
        files_list (list[str]): List of files.
    """
    print(f"\nNumber of files: {len(files_list)}")
    print("-" * 80)
    # print all files per line
    for i, f_name in enumerate(files_list):
        i = str(i).rjust(3)
        print(f"{i}: {f_name}")
    print()


def _get_n_players(files_list: list[str]) -> int:
    """Get the number of players from the files.

    Args:
        files_list (list[str]): List of files.

    Returns:
        int: Number of players.

    Raises:
        ValueError: If the number of files is not a power of 2.
    """
    for i in range(20):
        if len(files_list) == 2 ** i:
            return i
    else:
        raise ValueError(f"Number of files is not a power of 2: {len(files_list)}")


def _get_coalition_from_file_name(file_name: str) -> tuple[str, ...]:
    """Get the component from the file name.

    For a 2-component file with chain-of-thought (cot), the file name is encoded as
    `<DATASET>-cot-greedy-<ml_setting>_<model_id>_<component1>_<component2>.parquet` and the
    same file without cot is encoded as
    `<DATASET>-greedy-<ml_setting>_<model_id>_<component1>_<component2>.parquet`. The empty
    component is encoded as
    `<DATASET>-greedy-<ml_setting>_<model_id>_task-description-only.parquet` (without cot).

    Args:
        file_name (str): File name.
    """
    file_name = file_name.replace(".parquet", "")
    _coalition = []
    if "-cot-" in file_name:
        _coalition.append("cot")
    if "task-description-only" not in file_name:
        component_parts = file_name.split("_")[2:]
        _coalition.extend(component_parts)
    return tuple(sorted(_coalition))


def _compute_macro_f1(targets: pd.Series, predictions: pd.Series) -> float:
    """Compute the macro F1 score.

    Args:
        targets (pd.Series): True labels.
        predictions (pd.Series): Predicted labels.

    Returns:
        float: Macro F1 score.
    """
    scores = precision_recall_fscore_support(
        y_true=targets, y_pred=predictions, pos_label=1
    )
    f1 = scores[2]
    f1_macro = (f1[0] + f1[1]) / 2
    return f1_macro


def get_games(
    dataset: str = "stereoset",  # stereoset, sbic
    model_id: str = "llama3-70b-instruct",  # mistral-7b-instruct-v2, llama3-70b-instruct, command-r-v01
    ml_setting: str = "test",  # test, train, dev
    few_shot_component_name: str = "similar",  # random, category, similar
) -> tuple[dict[str, Game], list[str], str, list[dict]]:
    """Get the game object depending on the dataset and model name."""

    data_dir = os.path.join("data", TIMESTAMP, dataset, model_id)  # Path to the data dump

    prefix = f"{ml_setting}_{model_id}"

    few_shot_component_name += "-few-shot"

    save_name = f"{prefix}_{few_shot_component_name}"

    all_files = sorted(os.listdir(data_dir))
    _print_files(all_files)

    # select the ml_setting, model_id, and few-shot component
    all_files = [file for file in all_files if prefix in file]
    all_files = [
        f for f in all_files if not ("few-shot" in f and few_shot_component_name not in f)
    ]
    _print_files(all_files)

    # test if all files are there (must be a power of 2) and get the number of players
    n_players = _get_n_players(all_files)

    # create all combinations
    player_names = set()
    grand_coalition = tuple()
    coalitions: list[dict] = []
    for file in all_files:
        coalition = _get_coalition_from_file_name(file)
        player_names.update(coalition)
        coal = {
            "size": len(coalition),
            "coalition": coalition,
            "file": file,
            "path": os.path.join(data_dir, file),
        }
        coalitions.append(coal)
        if len(coalition) == n_players:
            grand_coalition = coalition
    print(f"List of all coalitions: {coalitions}")
    print(f"Grand coalition: {grand_coalition}")
    print(f"Player names: {player_names}")

    # print all elements sorted in ascending length
    coalitions.sort(key=lambda x: x["size"])
    for i, coalition in enumerate(coalitions):
        print(coalition)

    assert len(grand_coalition) == n_players
    assert len(coalitions) == 2 ** n_players

    name_mapping: dict[str, int] = {
        few_shot_component_name: 0,
    }
    feature_names = [few_shot_component_name]
    player_id = 1
    for player in sorted(player_names):
        if player not in name_mapping:
            name_mapping[player] = player_id
            player_id += 1
            feature_names.append(player)
    # abbreviate the feature names
    feature_names_abbrev = []
    for name in feature_names:
        name_parts = name.split("-")
        name_parts = [part[0:3] + "." for part in name_parts]
        feature_names_abbrev.append(" ".join(name_parts))
    feature_names = feature_names_abbrev

    # read the files and compute worth of coalition
    coalition_worth: list[dict] = []
    empty_value_log = 0
    empty_value_f1 = 0
    max_performance_f1, max_performance_log = 0, 1_000_000
    max_performance_coalition_f1, max_performance_coalition_log = None, None
    for coalition_dict in coalitions:
        # read_file
        coalition = coalition_dict["coalition"]
        df = pd.read_parquet(coalition_dict["path"])
        targets = df["true_label"]
        all_cols = [col for col in df.columns if col.startswith("output_")]
        all_cols = [col for col in all_cols if "prob" not in col]
        performance_log, performance_f1 = 0, 0
        for col in all_cols:
            predictions = df[col]
            performance_log += log_loss(targets, predictions)
            performance_f1 += _compute_macro_f1(targets, predictions)
        performance_log /= len(all_cols)
        performance_f1 /= len(all_cols)
        coalition_ints = tuple(sorted([name_mapping[name] for name in coalition]))
        coalition_worth.append(
            {"LogLoss": performance_log, "coalition": coalition_ints,
             "F1": performance_f1}
        )
        if coalition_dict["size"] == 0:
            empty_value_log = performance_log
            empty_value_f1 = performance_f1
        if performance_f1 > max_performance_f1:
            max_performance_f1 = performance_f1
            max_performance_coalition_f1 = coalition_ints
        if performance_log < max_performance_log:
            max_performance_log = performance_log
            max_performance_coalition_log = coalition_ints

    print("\nCoalition Worth", coalition_worth)

    # print the empty value ------------------------------------------------------------------------
    print(f"\nEmpty Value F1: {empty_value_f1}")
    print(f"Empty Value Log: {empty_value_log}")

    # print the name mapping -----------------------------------------------------------------------
    print(f"Naming Mapping: {name_mapping}")

    # print the max performance --------------------------------------------------------------------
    f1_best_coalition = [feature_names[i] for i in max_performance_coalition_f1]
    print(
        f"Max Performance F1: {max_performance_f1} for coalition {max_performance_coalition_f1} "
        f"({f1_best_coalition})"
    )
    log_best_coalition = [feature_names[i] for i in max_performance_coalition_log]
    print(
        f"Max Performance Log: {max_performance_log} for coalition {max_performance_coalition_log}"
        f" ({log_best_coalition})"
    )

    games_dict = {}

    # f1 values ------------------------------------------------------------------------------------
    values_array_f1 = [c["F1"] for c in coalition_worth]
    coalition_lookup_f1 = {c["coalition"]: i for i, c in enumerate(coalition_worth)}
    llm_game_f1 = Game(n_players=n_players, normalize=NORMALIZE, normalization_value=empty_value_f1)
    llm_game_f1.value_storage = values_array_f1
    llm_game_f1.coalition_lookup = coalition_lookup_f1
    games_dict["F1"] = llm_game_f1

    # log loss values ------------------------------------------------------------------------------
    values_array_ll = [c["LogLoss"] for c in coalition_worth]
    coalition_lookup_ll = {c["coalition"]: i for i, c in enumerate(coalition_worth)}
    llm_game_ll = Game(n_players=n_players, normalize=NORMALIZE, normalization_value=empty_value_log)
    llm_game_ll.value_storage = values_array_ll
    llm_game_ll.coalition_lookup = coalition_lookup_ll
    games_dict["LogLoss"] = llm_game_ll

    feature_names_renamed = _rename_feature_names(feature_names)

    return games_dict, feature_names_renamed, save_name, coalition_worth


def _rename_feature_names(feature_names: list[str]) -> list[str]:
    """Rename the feature names.

    Renames the following components:
        - cot.: rea.
        - cat. few. sho.: cat. dem.
        - ran. few. sho.: ran. dem.
        - sim. few. sho.: sim. dem.
        - sys. pro.: per.


    All others remain unchanged.

    Args:
        feature_names (list[str]): Feature names.

    Returns:
        list[str]: Renamed feature names.
    """
    feature_names = [
        "rea." if name == "cot." else name for name in feature_names
    ]
    feature_names = [
        "cat. dem." if name == "cat. few. sho." else name for name in feature_names
    ]
    feature_names = [
        "ran. dem." if name == "ran. few. sho." else name for name in feature_names
    ]
    feature_names = [
        "sim. dem." if name == "sim. few. sho." else name for name in feature_names
    ]
    feature_names = [
        "per." if name == "sys. pro." else name for name in feature_names
    ]
    return feature_names


def compute_interaction_values(game: Game) -> dict[str, InteractionValues]:
    """Computes a set of interaction values.

    The following interaction values are computed:
        - Shapley Value (SV) of order 1
        - 2-SII values (2-SII) of order 2
        - 2-FSII values (2-FSII) of order 2
        - Moebius values (Moebius) of order n_players

    Args:
        game (Game): Game object.

    Returns:
        dict[str, InteractionValues]: Interaction values.
    """
    interactions: dict[str, InteractionValues] = {}

    # compute the values exactly
    n_players = game.n_players
    computer = ExactComputer(n_players=n_players, game_fun=game)

    # sv_values
    sv_values = computer(index="SV", order=1)
    interactions["SV"] = sv_values
    print("\n", sv_values)

    # 2-SII values
    two_sii_values = computer(index="k-SII", order=2)
    interactions["2-SII"] = two_sii_values
    print("\n", two_sii_values)

    # 2-FSII values
    two_fsii_values = computer(index="FSII", order=2)
    interactions["2-FSII"] = two_fsii_values
    print("\n", two_fsii_values)

    # moebius values
    moebius_values = computer(index="Moebius", order=n_players)
    moebius_values.values[moebius_values.interaction_lookup[tuple()]] = 0
    interactions["Moebius"] = moebius_values
    print("\n", moebius_values)

    return interactions


def plot_and_save(
    interactions: dict[str, InteractionValues],
    performance_metric: str,
    feature_names: Optional[list[str]] = None,
    save_name: str = "",
    show: bool = True,
    plot_dir: str = "plots",
) -> None:

    for index, interaction_values in interactions.items():
        print(f"\n{index} of order {interaction_values.max_order}")

        name = f"{save_name}_{performance_metric}_{index}"

        # plot force_plot if index is not Moebius
        if index != "Moebius":
            interaction_values.plot_force(feature_names=feature_names, show=False)
            if FORCE_LIMITS is not None:
                plt.xlim(FORCE_LIMITS)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{name}_force_plot.pdf"))
            if show:
                plt.show()

        # if index is max_order = 2, plot the network plot
        if PLOT_TWO_SI and interaction_values.max_order == 2:
            interaction_values.plot_network(feature_names=feature_names)
            plt.title(f"Performance Metric: {performance_metric} Score\n")
            plt.savefig(os.path.join(plot_dir, f"{name}_network_plot.pdf"))
            if show:
                plt.show()

        # plot the si graph for all indices
        if PLOT_SI_GRAPH:
            si_graph_nodes = list(
                powerset(range(interaction_values.n_players), min_size=2, max_size=2)
            )
            si_graph_interaction = copy.deepcopy(interaction_values)
            try:
                si_graph_interaction.values[si_graph_interaction.interaction_lookup[tuple()]] = 0
            except KeyError:
                pass
            si_graph_plot(
                si_graph_interaction,
                graph=si_graph_nodes,
                size_factor=3,
                node_size_scaling=3,
                compactness=100,
                label_mapping={i: f"{feature_names[i]}" for i in range(interaction_values.n_players)},
            )
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{name}_si_graph.pdf"))
            if show:
                plt.show()


def _get_coal_label_name(
    coalition: tuple[int],
    feature_names: list[str]
) -> str:
    """Get the coalition label name.

    Args:
        coalition (tuple[int]): Coalition.
        feature_names (list[str]): Feature names.

    Returns:
        str: Coalition label name.
    """
    return "-".join([feature_names[i] for i in coalition]).replace(" ", "")


def _get_best_coalition(
    worths: list[dict],
    feature_names: list[str],
    performance_metric: str = "F1"
) -> tuple[str, float]:
    """Get the best coalition.

    Args:
        worths (list[dict]): Worths of the coalitions.
        feature_names (list[str]): Component names.
        performance_metric (str): Performance metric.

    Returns:
        tuple[str, float]: Best coalition and its worth.
    """
    max_worth = max([worth[performance_metric] for worth in worths])
    max_coalition = [worth["coalition"] for worth in worths if worth[performance_metric] == max_worth][0]
    max_coalition = _get_coal_label_name(max_coalition, feature_names)
    return max_coalition, max_worth


def get_si_selection(
    interaction_values: InteractionValues,
    game: Game,
    feature_names: list[str]
) -> tuple[str, float]:
    """Makes a selection of the best coalition based on the interaction values.

    The selection is made by using the SIs as a faithful decomposition of the game. The powerset of
    all coalitions is used to predict the output of the game by using the interaction values. The
    highest value is selected as the best coalition. Note that this mechanism is not guaranteed to
    find the best coalition (only in case of Moebius values) and does not scale well with the
    number of players.

    Args:
        interaction_values (InteractionValues): Interaction values.
        game (Game): Game object.
        feature_names (list[str]): Feature names.

    Returns:
        A tuple of the best coalition and its worth.
    """
    best_prediction = -10000000
    best_coalition = tuple()
    for coalition in powerset(range(game.n_players), min_size=0):
        coalition = tuple(sorted(coalition))
        predicted_score = interaction_values.baseline_value
        for interaction in powerset(coalition, min_size=0):
            predicted_score += interaction_values[interaction]
        if predicted_score > best_prediction:
            best_prediction = predicted_score
            best_coalition = coalition
    coalition_array = np.zeros(game.n_players, dtype=bool)
    coalition_array[list(best_coalition)] = True
    coalition_worth = game(coalition_array)
    best_coalition = _get_coal_label_name(best_coalition, feature_names)
    return best_coalition, float(coalition_worth[0])


def add_values_to_csv(
    data: dict[str, Union[str, float]],
    file_name: str = "results.csv"
) -> None:
    """Write the data to a csv file.

    Args:
        data (dict[str, Union[str, float]]): Data to be written.
        file_name (str): File name.
    """
    data_keys = [
        "dataset",
        "model",
        "few-shot-variant",
        "empty-coal-worth",
        "grand-coal-worth",
        "is-grand-better-empty",
        "best-coal",
        "best-coal-worth",
        "best-coal-sv",
        "best-coal-sv-worth",
        "is-coal-sv-best",
        "is-coal-sv-better-grand-and-empty",
        "best-coal-2-sii",
        "best-coal-2-sii-worth",
        "is-coal-2-sii-best",
        "is-coal-2-sii-better-grand-and-empty",
    ]
    try:
        data_stored = pd.read_csv(file_name)
    except FileNotFoundError:
        data_stored = pd.DataFrame(columns=data_keys)

    # check if the data is already in the file
    if data_stored[
        (data_stored["dataset"] == data["dataset"])
        & (data_stored["model"] == data["model"])
        & (data_stored["few-shot-variant"] == data["few-shot-variant"])
    ].shape[0] > 0:
        return
    data_to_add = pd.DataFrame(data, index=[0])
    data_stored = pd.concat([data_stored, data_to_add], ignore_index=True)
    data_stored.to_csv(file_name, index=False)


def run_inspection(
        dataset: str = "cobra",
        model_name: str = "llama3-70b-instruct",
        few_shot: str = "category",
) -> None:
    plot_dir = os.path.join("plots", f"{model_name}_{dataset}")
    os.makedirs(plot_dir, exist_ok=True)

    # get validation values ------------------------------------------------------------------------
    games_val, _, _, worths_val = get_games(
        dataset, model_name, "dev", few_shot
    )
    game_val = games_val["F1"]
    interactions_val = compute_interaction_values(games_val["F1"])

    # get test values ------------------------------------------------------------------------------
    games_test, component_names, save_prefix, worths_test = get_games(
        dataset, model_name, "test", few_shot
    )
    game_test = games_test["F1"]
    interactions_test = compute_interaction_values(games_test["F1"])

    # get the best coalition for the test set ------------------------------------------------------
    max_coalition_test, max_worth_test = _get_best_coalition(worths_test, component_names, "F1")
    max_coalition_val, max_worth_val = _get_best_coalition(worths_val, component_names, "F1")

    # get grand and empty coalition worths ---------------------------------------------------------
    grand_coal_worth_test = float(game_test(game_test.grand_coalition)[0])
    empty_coal_worth_test = float(game_test(game_test.empty_coalition)[0])

    # make plots for test values -------------------------------------------------------------------
    plot_and_save(
        interactions_test,
        performance_metric="F1",
        feature_names=component_names,
        save_name=save_prefix,
        show=SHOW,
        plot_dir=plot_dir,
    )

    # get the best coalition and worth for test set ------------------------------------------------
    sv_values_val = interactions_val["SV"]
    best_coalition_sv, best_worth_sv = get_si_selection(sv_values_val, game_test, component_names)
    print(f"\nBest Coalition (SV): {best_coalition_sv} with worth {best_worth_sv}")

    two_sii_val = interactions_val["2-SII"]
    best_coalition_two_sii, best_worth_two_sii = get_si_selection(
        two_sii_val, game_test, component_names
    )
    print(f"\nBest Coalition (2-SII): {best_coalition_two_sii} with worth {best_worth_two_sii}")

    two_fsii_val = interactions_val["2-FSII"]
    best_coalition_two_fsii, best_worth_two_fsii = get_si_selection(
        two_fsii_val, game_test, component_names
    )
    print(f"\nBest Coalition (2-FSII): {best_coalition_two_fsii} with worth {best_worth_two_fsii}")

    moebius_val = interactions_val["Moebius"]
    best_coalition_moebius, best_worth_moebius = get_si_selection(
        moebius_val, game_test, component_names
    )
    _, best_worth_moebius_val = get_si_selection(
        moebius_val, game_val, component_names
    )
    print(f"\nBest Coalition (Moebius): {best_coalition_moebius} with worth {best_worth_moebius}")

    # get the best coalition and worth for test set ------------------------------------------------
    is_grand_better_empty = "\\cmark" if grand_coal_worth_test > empty_coal_worth_test else "\\xmark"

    is_coal_sv_best = "\\xmark"
    is_coal_sv_better_grand_empty = "\\xmark"
    is_coal_two_sii_best = "\\xmark"
    is_coal_two_sii_better_grand_empty = "\\xmark"
    is_coal_two_sii_better_sv = "\\xmark"
    is_coal_two_fsii_best = "\\xmark"
    is_coal_two_fsii_better_grand_empty = "\\xmark"
    is_coal_two_fsii_better_sv = "\\xmark"
    is_coal_moebius_best = "\\xmark"
    is_coal_moebius_better_grand_empty = "\\xmark"
    is_coal_moebius_better_sv = "\\xmark"

    if best_worth_sv >= max_worth_test:
        is_coal_sv_best = "\\cmark"
    if best_worth_sv >= grand_coal_worth_test and best_worth_sv >= empty_coal_worth_test:
        is_coal_sv_better_grand_empty = "\\cmark"
    if best_worth_two_sii >= max_worth_test:
        is_coal_two_sii_best = "\\cmark"
    if best_worth_two_sii >= grand_coal_worth_test and best_worth_two_sii >= empty_coal_worth_test:
        is_coal_two_sii_better_grand_empty = "\\cmark"
    if best_worth_two_sii >= best_worth_sv:
        if best_worth_two_sii > best_worth_sv:
            is_coal_two_sii_better_sv = "\\cmark"
        else:
            is_coal_two_sii_better_sv = "--"
    if best_worth_two_fsii >= max_worth_test:
        is_coal_two_fsii_best = "\\cmark"
    if best_worth_two_fsii >= grand_coal_worth_test and best_worth_two_fsii >= empty_coal_worth_test:
        is_coal_two_fsii_better_grand_empty = "\\cmark"
    if best_worth_two_fsii >= best_worth_sv:
        if best_worth_two_fsii > best_worth_sv:
            is_coal_two_fsii_better_sv = "\\cmark"
        else:
            is_coal_two_fsii_better_sv = "--"
    if best_worth_moebius >= max_worth_test:
        is_coal_moebius_best = "\\cmark"
    if best_worth_moebius >= grand_coal_worth_test and best_worth_moebius >= empty_coal_worth_test:
        is_coal_moebius_better_grand_empty = "\\cmark"
    if best_worth_moebius >= best_worth_sv:
        if best_worth_moebius > best_worth_sv:
            is_coal_moebius_better_sv = "\\cmark"
        else:
            is_coal_moebius_better_sv = "--"

    is_coal_moebius_best_validation = "\\xmark"
    if best_worth_moebius_val >= max_worth_val:
        is_coal_moebius_best_validation = "\\cmark"

    # add the values to the csv file ---------------------------------------------------------------
    data = {
        "dataset": dataset,
        "model": model_name,
        "few-shot-variant": few_shot,
        "empty-coal-worth": empty_coal_worth_test,
        "grand-coal-worth": grand_coal_worth_test,
        "is-grand-better-empty": is_grand_better_empty,
        "best-coal": max_coalition_test,
        "best-coal-worth": max_worth_test,
        "best-coal-sv": best_coalition_sv,
        "best-coal-sv-worth": best_worth_sv,
        "is-coal-sv-best": is_coal_sv_best,
        "is-coal-sv-better-grand-and-empty": is_coal_sv_better_grand_empty,
        "best-coal-2-sii": best_coalition_two_sii,
        "best-coal-2-sii-worth": best_worth_two_sii,
        "is-coal-2-sii-best": is_coal_two_sii_best,
        "is-coal-2-sii-better-grand-and-empty": is_coal_two_sii_better_grand_empty,
        "is-coal-2-sii-better-sv": is_coal_two_sii_better_sv,
        "best-coal-2-fsii": best_coalition_two_fsii,
        "best-coal-2-fsii-worth": best_worth_two_fsii,
        "is-coal-2-fsii-best": is_coal_two_fsii_best,
        "is-coal-2-fsii-better-grand-and-empty": is_coal_two_fsii_better_grand_empty,
        "is-coal-2-fsii-better-sv": is_coal_two_fsii_better_sv,
        "best-coal-moebius": best_coalition_moebius,
        "best-coal-moebius-worth": best_worth_moebius,
        "is-coal-moebius-best": is_coal_moebius_best,
        "is-coal-moebius-better-grand-and-empty": is_coal_moebius_better_grand_empty,
        "is-coal-moebius-better-sv": is_coal_moebius_better_sv,
        "is-coal-moebius-best-validation": is_coal_moebius_best_validation,
    }
    add_values_to_csv(data)


if __name__ == '__main__':

    NORMALIZE = False

    DATASETS = ["stereoset", "sbic", "cobra"]
    MODELS = ["mistral-7b-instruct-v2", "llama3-70b-instruct", "command-r-v01"]
    FEW_SHOT_VARIANTS = ["category", "similar", "random"]

    SHOW = False
    SAVE_FIG = True
    PLOT_TWO_SI = True
    PLOT_SI_GRAPH = False

    FORCE_LIMITS = None  # (0.682, 0.789)

    for dataset in DATASETS:
        for model in MODELS:
            for few_shot_variant in FEW_SHOT_VARIANTS:

                print(f"\n Dataset: {dataset}, Model: {model}, Few-shot: {few_shot_variant}")
                run_inspection(dataset, model, few_shot_variant)
