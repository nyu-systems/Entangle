# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import os
import os.path as osp
import re
from itertools import chain
from subprocess import Popen

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import seaborn as sns
from pytimeparse.timeparse import timeparse

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
pd.set_option("max_colwidth", 400)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
font = {"family": "Times New Roman", "size": 12}
matplotlib.rc("font", **font)


def get_run_info(dirname):
    assert osp.exists(dirname), f"{dirname} does not exist"
    # Get total time from "output.log"
    log_content = open(f"{dirname}/output.log", "r").read()
    try:
        total_time_str = re.search(r"Elapased for Inference: ([\d:.]+)", log_content).group(1).removesuffix(".")
    except Exception as e:
        print(f"Error parsing {dirname}/output.log")
        raise e
    total_time = timeparse(total_time_str)
    # Get node, edge count from "graph_stats.json"
    try:
        with open(f"{dirname}/graph_stats.json", "r") as f:
            graph_stats = json.load(f)
            num_nodes = sum([g["num_nodes"] for g in graph_stats])
            num_edges = sum([g["num_edges"] for g in graph_stats])
    except Exception as e:
        print(f"Error parsing {dirname}/graph_stats.json")
        raise e
    return total_time, num_nodes, num_edges


def get_group_info(dirname):
    assert osp.exists(dirname), dirname
    ag_log_path = osp.join(dirname, "ag.log")
    p = Popen(
        f""" find {dirname}/group* | grep output.log """,
        shell=True,
        stdout=open(ag_log_path, "w"),
    )
    p.wait()
    log_paths = open(ag_log_path, "r").read().strip("\t\r\n ").split("\n")

    data_list = []
    for log_path in log_paths:
        assert osp.exists(log_path), f"{log_path} does not exist"
        try:
            group_id = int(re.search(r"group(\d+)", log_path).group(1))
            reason = None
            saturation_time = None
            extraction_time = None
            num_iterations = None
            num_enodes = None
            num_eclasses = None
            num_edges = None
            lemma_applied_count = {}
            for line in open(log_path, "r").readlines():
                if matched := re.match(r"  Stopped: (.+)", line):
                    assert reason is None
                    reason = matched.group(1)
                if matched := re.search(r"Saturation done in ([\d:.]+)", line):
                    assert saturation_time is None
                    saturation_time = timeparse(matched.group(1))
                if matched := re.search(r"Extraction done in ([\d:.]+)", line):
                    assert extraction_time is None
                    extraction_time = timeparse(matched.group(1))
                if matched := re.match(r"  Number of iterations: (\d+)", line):
                    assert num_iterations is None
                    num_iterations = int(matched.group(1))
                if matched := re.match(r"  Number of enodes: (\d+)", line):
                    assert num_enodes is None
                    num_enodes = int(matched.group(1))
                if matched := re.match(r"  Number of classes: (\d+)", line):
                    assert num_eclasses is None
                    num_eclasses = int(matched.group(1))
                if matched := re.match(r"  Number of edges: (\d+)", line):
                    assert num_edges is None
                    num_edges = int(matched.group(1))
                if matched := re.match(r"Lemma (.+) was applied (\d+) times.", line):
                    lemma_name = matched.group(1)
                    lemma_count = int(matched.group(2))
                    lemma_applied_count[lemma_name] = lemma_count
            data_list.append(
                (
                    group_id,
                    saturation_time,
                    extraction_time,
                    num_iterations,
                    num_enodes,
                    num_eclasses,
                    num_edges,
                    reason,
                    lemma_applied_count,
                )
            )
        except Exception as e:
            print(f"Error parsing {log_path}")
            raise e

    df = (
        pd.DataFrame(
            data_list,
            columns=[
                "group_id",
                "saturation",
                "extraction",
                "num_iterations",
                "num_nodes",
                "num_classes",
                "num_edges",
                "reason",
                "lemma_applied_count",
            ],
        )
        .sort_values("group_id", ascending=True)
        .reset_index(drop=True)
    )
    df["total_applied_count"] = df["lemma_applied_count"].apply(lambda x: sum(x.values()))
    return df


def collect_lemma_data(dirnames: list[str]):
    collected_lemma_names = set()
    data_list = []
    for dirname in dirnames:
        assert osp.exists(dirname), dirname
        ag_log_path = osp.join(dirname, "ag.log")
        p = Popen(
            f""" find {dirname}/group* | grep output.log """,
            shell=True,
            stdout=open(ag_log_path, "w"),
        )
        p.wait()
        log_paths = open(ag_log_path, "r").read().strip("\t\r\n ").split("\n")

        for log_path in log_paths:
            assert osp.exists(log_path), f"{log_path} does not exist"
            try:
                for line in open(log_path, "r").readlines():
                    if matched := re.match(r"Rewrite for <(.+)>: .+ -> .+, following (.+) -> (.+)", line):
                        name = matched.group(1).strip(" ")
                        name = re.match(r"([^()]+)(\(.+\))?", name).group(1)
                        src = matched.group(2).strip(" ")
                        dst = matched.group(3).strip(" ")
                        if name not in collected_lemma_names:
                            data_list.append((name, src, dst))
                        collected_lemma_names.add(name)
            except Exception as e:
                print(f"Error parsing {log_path}")
                raise e

    df = (
        pd.DataFrame(
            data_list,
            columns=["name", "src", "dst"],
        )
        .sort_values("name", ascending=True)
        .reset_index(drop=True)
    )
    return df


def remove_cdf_tailing_vertical_line(ax):
    poly = ax.findobj(plt.Polygon)[0]
    vertices = poly.get_path().vertices

    # Keep everything above y == 0. You can define this mask however
    # you need, if you want to be more careful in your selection.
    keep = vertices[:, 1] > 0

    # Construct new polygon from these "good" vertices
    new_poly = plt.Polygon(
        vertices[keep],
        closed=False,
        fill=False,
        edgecolor=poly.get_edgecolor(),
        linewidth=poly.get_linewidth(),
    )
    poly.set_visible(False)
    ax.add_artist(new_poly)


COLORS = [
    "#8CB368",  # #8CB368
    "#F4E284",  # #F4E284
    "#F3A259",  # #F3A259
    "#5A8E7D",  # #5A8E7D
    "#BC4A51",  # #BC4A51
    "#70AD47",  # #70AD47
    "#0563C1",  # #0563C1
    "#954F72",  # #954F72
]

model_rename = {"aws_llama": "Llama-3", "gpt": "GPT", "qwen2": "QWen2"}


def process_target_paral(model, target_paral, num_layers_list):
    data_list = []
    for num_layers in num_layers_list:
        if model in ("gpt", "qwen2"):
            dirname = f"precondition.{model}.fw.g0.paral1_layer{num_layers}-paral{target_paral}_layer{num_layers}.greedy"
        else:
            dirname = f"precondition.{model}.fw.g0.tp1_layer{num_layers}-tp{target_paral}_layer{num_layers}.greedy"
        total_time, num_nodes, num_edges = get_run_info(dirname)
        df = get_group_info(dirname)
        saturation, extraction, total_applied_count = (
            df["saturation"].sum(),
            df["extraction"].sum(),
            df["total_applied_count"].sum(),
        )
        data_list.append(
            (
                model_rename[model],
                num_layers,
                target_paral,
                num_nodes,
                num_edges,
                total_time,
                saturation,
                extraction,
                total_applied_count,
            )
        )
    return data_list


def plot_performance():
    num_layers = 1
    target_paral = 2

    models = ["gpt", "aws_llama", "qwen2"]

    with mp.Pool(processes=len(models)) as pool:
        data_list = list(chain(*pool.starmap(process_target_paral, [(model, target_paral, [1]) for model in models])))
    all_df = pd.DataFrame(
        data_list,
        columns=[
            "Model",
            "Num Layers",
            "TP-size",
            "#of Nodes",
            "#of Edges",
            "Total Time",
            "Saturation Time",
            "Extraction Time",
            "Total Applied Count",
        ],
    )
    all_df["Others"] = all_df["Total Time"] - all_df["Saturation Time"] - all_df["Extraction Time"]
    all_df["ModelLabel"] = all_df.apply(lambda row: f"{row['Model']}\n({row['#of Nodes']})", axis=1)
    fig = plt.figure(figsize=(1.618 * 3, 1 * 3))
    ax = fig.gca()
    all_df.plot.bar(x="ModelLabel", y="Total Time", color=COLORS, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Total Time (s)")
    ax.xaxis.set_tick_params(rotation=0)
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(osp.join("figures", f"one_layer_time.pdf"), bbox_inches="tight", dpi=300)

    print(f" One-layer Performance Figure generated to figures/one_layer_time.pdf")


def plot_scalability():
    for model_idx, model in enumerate(["gpt", "aws_llama"]):
        fig = plt.figure(figsize=(16, 4))
        num_layers_list = [1, 2, 4, 8]
        target_paral_list = [2, 4, 6, 8] if model == "gpt" else [2, 4, 8]

        with mp.Pool(processes=len(target_paral_list)) as pool:
            data_list = list(
                chain(
                    *pool.starmap(
                        process_target_paral, [(model, target_paral, num_layers_list) for target_paral in target_paral_list]
                    )
                )
            )
        all_df = pd.DataFrame(
            data_list,
            columns=[
                "Model",
                "Num Layers",
                "TP-size",
                "#of Nodes",
                "#of Edges",
                "Total Time",
                "Saturation Time",
                "Extraction Time",
                "Total Applied Count",
            ],
        )
        overview_df = pd.DataFrame()
        for model, num_layers, target_paral, _, _, total_time, _, _, _ in all_df.itertuples(index=False):
            overview_df.loc[target_paral, num_layers] = total_time
        fig = plt.figure(figsize=(1.618 * 3, 1 * 3))
        ax = fig.gca()
        overview_df.plot(kind="line", marker="o", color=COLORS, ax=ax)
        ax.set_xlabel("Parallelism Size", fontsize=16)
        ax.set_ylabel("Total Time (s)", fontsize=16)
        ax.set_xticks(target_paral_list)
        ax.tick_params(axis="both", which="major", labelsize=16)
        plt.ylim(0)
        plt.legend(title="#of Layers", ncol=2, labelspacing=0.05)
        plt.tight_layout()
        plt.savefig(osp.join("figures", f"{model}_scalability.pdf"), bbox_inches="tight", dpi=300)

        print(f" Scalability Figure generated to figures/{model}_scalability.pdf")


def plot_heatmap():
    def normalize_lemma_name(name):
        if matched := re.search(r"\((.+)\)>$", name):
            num_items_str = matched.group(1)
            name = name.replace(f"({num_items_str})>", ">")
        return name.removeprefix("<").removesuffix(">")

    def get_lemma_applied_count(dirname):
        df = get_group_info(dirname)
        lemma_applied_count = {}
        for d in df["lemma_applied_count"]:
            for name, count in d.items():
                name = normalize_lemma_name(name)
                if name in lemma_applied_count:
                    lemma_applied_count[name] += count
                else:
                    lemma_applied_count[name] = count
        return lemma_applied_count

    lemma_applied_count_list = []
    for paral in [2, 4, 8]:
        lemma_applied_count_list.append(
            (f"GPT({paral})", get_lemma_applied_count(f"precondition.gpt.fw.g0.paral1_layer1-paral{paral}_layer1.greedy"))
        )
    for paral in [4]:
        lemma_applied_count_list.append(
            (f"Qwen2({paral})", get_lemma_applied_count(f"precondition.qwen2.fw.g0.paral1_layer1-paral{paral}_layer1.greedy"))
        )
    for paral in [4]:
        lemma_applied_count_list.append(
            (f"Llama-3({paral})", get_lemma_applied_count(f"precondition.aws_llama.fw.g0.tp1_layer1-tp{paral}_layer1.greedy"))
        )

    seen_lemma_names = set()
    lemma_names = []
    for run_name, lemma_applied_count in lemma_applied_count_list:
        names = [name for name in lemma_applied_count.keys() if name not in seen_lemma_names]
        common_names = []
        hlo_names = []
        vllm_names = []
        internal_names = []
        for name in names:
            if name.startswith("hlo"):
                hlo_names.append(name)
            elif name.startswith("vllm"):
                vllm_names.append(name)
            elif name.startswith("internal") or name.startswith("rms_"):
                internal_names.append(name)
            else:
                common_names.append(name)
        lemma_names.extend(common_names)
        lemma_names.extend(hlo_names)
        lemma_names.extend(vllm_names)
        lemma_names.extend(internal_names)
        seen_lemma_names.update(lemma_applied_count.keys())

    df = pd.DataFrame(columns=list(lemma_names))
    for name, lemma_applied_count in lemma_applied_count_list:
        df.loc[name, lemma_applied_count.keys()] = list(lemma_applied_count.values())
    df = df.fillna(float("nan")).sort_values("GPT(4)", axis=1, ascending=False).apply(lambda x: np.log2(x))

    # Save a un-named version
    # fmt: off
    clean_lemmas = ['reduce_add-to-matadd-rewrite', 'index_put-mask_select-to_sum-swap', 'transpose-reshape-swap', 'reshape-slice-swap', 'slice-concat-swap', 'transpose-slice-swap', 'reshape-to-same-shape', 'reshape-concat-swap', 'reshape-reshape-collapse', 'reshape-masked_select-swap', 'transpose-concat-swap', 'reshape-matadd-swap', 'matadd-slice-swap', 'reshape-reduce_add-swap', 'matadd-dual_concat-swap', 'matadd-0-back', 'index_put-mask_select-swap', 'layernorm-concat-swap', 'consecutive-slices-back', 'transpose-reduce_add-swap', 'tranpose-matadd-swap', 'matmul-both-concat-swap', 'matadd-concat-swap', 'full-slice-back', 'mask_fill_scalar-concat-swap',  'matadd-rhs-zeros-back', 'concat-slice-swap', 'slice-slice-swap', 'hlo_broadcast-maybe-concat', 'concat-concat-2-2-swap', 'hlo_broadcast-concat-swap', 'hlo_select-all-concat-swap', 'hlo_max-concat-swap']
    # fmt: on
    def unamed_mapper(i, name):
        if name.startswith("vllm"):
            return f"{i}\nv"
        elif name.startswith("hlo"):
            return f"{i}\nh"
        elif name.startswith("internal") or name.startswith("rms"):
            return f"{i}\ni"
        elif name in clean_lemmas:
            return f"{i}\nc"
        else:
            return f"{i}"

    unamed_df = df.copy()
    unamed_df.columns = [unamed_mapper(i, name) for i, name in enumerate(df.columns)]
    fig = plt.figure(figsize=(18, 1.8))
    ax = fig.gca()
    sns.heatmap(
        unamed_df, ax=ax, linewidths=1, cmap="Oranges", mask=unamed_df.isnull(), cbar_kws=dict(orientation="vertical", pad=0.01)
    )

    plt.xticks(ticks=[i + 0.5 for i in range(len(unamed_df.columns))], labels=unamed_df.columns.to_list(), rotation=0)
    ax.tick_params(axis="x", which="major", labelsize=10)
    plt.yticks(rotation=0)
    plt.tight_layout()
    ax.collections[0].cmap.set_bad("0.98")
    plt.savefig(osp.join("figures", "lemma_applied_count_heatmap.pdf"), dpi=600, bbox_inches="tight")

    print(" Lemma Applied Count Figure (Figure 4) generated to figures/lemma_applied_count_heatmap.pdf")


def plot_lemma_complexity():
    # NOTE: The numbers here are manually counted.
    # name, LOC, src ops, src depth, dst ops, dst depth [, #of helper_pattern]
    world_size = 2
    GPT = [
        ("layernorm-concat-swap", 8, 2, 2, 3, 2, 0),
        ("layernorm-slice-swap", 8, 2, 2, 2, 2, 0),
    ]
    QWEN2 = [
        ("vllm_rotary_embedding-concat-swap", 27, 2, 2, 3, 2, 0),
        ("vllm_attention-concat-swap", 22, 4, 2, 4, 2, world_size),
        ("vllm_silu_and_mul-rewrite", 11, 4, 3, 5, 3, 0),
        ("vllm_rotary_embedding_ignore_q", 5, 1, 1, 1, 1, 1),
    ]
    LLAMA = [
        ("hlo_broadcast-concat-swap", 20, 2, 2, 3, 2, 0),
        ("hlo_broadcast-maybe-concat", 29, 2, 1, 3, 2, 0),
        ("hlo_dot-concat-lhs-swap", 19, 2, 2, 3, 2, 0),
        ("hlo_dot-concat-rhs-swap", 19, 2, 2, 3, 2, 0),
        ("hlo_dot-dual-concat-swap", 21, 3, 2, 3, 2, 0),
        ("hlo_dot-slice-swap", 16, 2, 2, 2, 2, 0),
        ("hlo_max-concat-swap", 9, 2, 2, 3, 2, 0),
        ("hlo-reshape-concat-swap", 37, 2, 2, 3, 2, 2),
        ("hlo_select-all-concat-swap", 2, 4, 2, 3, 2, 0),
    ]

    columns = [
        "name",
        "LOC",
        "src_ops",
        "src_depth",
        "dst_ops",
        "dst_depth",
        "#of helper patterns",
    ]
    gpt_df = pd.DataFrame(GPT, columns=columns)
    qwen2_df = pd.DataFrame(QWEN2, columns=columns)
    llama_df = pd.DataFrame(LLAMA, columns=columns)

    ALL = GPT + QWEN2 + LLAMA
    all_df = pd.DataFrame(ALL, columns=columns)

    data = []
    for name, df, num_new_ops in [("GPT", gpt_df, 1), ("Qwen2", qwen2_df, 4), ("Llama", llama_df, 5)]:
        data.append(
            (name, num_new_ops, len(df), df["LOC"].mean().astype(int), np.round((df["src_ops"] + df["dst_ops"]).mean(), 1))
        )
    df = pd.DataFrame(
        data, columns=["Name", "#of New Operators", "#of New Lemmas", "Avg. LOC", "Avg. #of Operators in a Lemma"]
    )

    # 1. Plot CDF of LOC
    fig = plt.figure(figsize=(1.618 * 3, 1 * 3))
    ax = plt.gca()
    res = all_df.hist(column="LOC", bins=100, color=COLORS[5], grid=True, ax=ax, cumulative=True, histtype="step", density=True)
    # ax.hist(all_df["LOC"], bins=100, histtype = 'step', fill = None)
    ax.set_title("")
    ax.set_xlabel("LOC")
    ax.set_ylabel("Percent")
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    # ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlim(all_df["LOC"].min(), all_df["LOC"].max())
    ax.set_ylim(0, 1)
    remove_cdf_tailing_vertical_line(ax)
    plt.tight_layout()
    plt.savefig("figures/lemma_loc.pdf", bbox_inches="tight", dpi=600)

    # 2. Plot statistics numbers
    ys = ["#of New Operators", "#of New Lemmas", "Avg. #of Operators in a Lemma"]
    cmap_dict = {k / 2: v for k, v in enumerate(COLORS)}
    cmap = lambda x: cmap_dict[x]
    ax = df.plot(x="Name", y=ys, kind="bar", figsize=(3 * 1.618, 3), legend=True, cmap=cmap)
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(rotation=0)
    plt.ylim(0, 20)
    plt.xlabel("")
    plt.tight_layout()
    plt.legend(ncol=1, labelspacing=0.05, fontsize=10)
    plt.savefig("figures/number_of_ops_and_lemmas.pdf", bbox_inches="tight", dpi=600)



if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("Generating Visualization Figures...")
    plot_performance()
    plot_scalability()
    plot_heatmap()
    plot_lemma_complexity()
    print("Figures Generated Successfully!")
