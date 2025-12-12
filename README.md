# Entangle

<p align="left">
    <a href="https://arxiv.org/abs/2508.09505">
    <img src="https://img.shields.io/badge/Entangle-Paper-green"></a>
    <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-blue"></a>
</p>

This is the open-source codes repository for paper [It Takes Two to Entangle](https://arxiv.org/abs/2508.09505) (ASPLOS'26). This project provides a tool to inductively infer the relations for tensors between sequential and distributed computation graphs, allowing checking the their equivalence scalably and efficiently. It supports verification for graphs dumped with our wrapped version of TorchDynamo, thus covers a wide range of PyTorch programs.

**NOTE for Artifact Evaluation Reviewers**: 
This also serves as the artifact for ASPLOS's Artifact Evaluation, targetting Availability and Evaluted badges. Please follow the instructions in [Artifact Evaluation](#artifact-evaluation) to reproduce the results in the paper.


## Setup

## Recommended Environment

For best reproducing, we recommend using the environment we used for the experiments:

- Ubuntu-22.04
    - Should run on any other Linux distributions (not tested)
    - Runs on MacOS, but significantly slower (tested)
    - Cannot run on Windows since we used some Unix-like commands, but you can run on WSL.
- Python 3.12
- Rust supporting edition 2018

Other software recommendation are described in [pyproject.toml](./pyproject.toml) and [Cargo.toml](./egger/Cargo.toml).

## Install Dependencies

Assuming you have installed
- `Python` >= 3.12 with `pip` available.

```sh
# 1. Install Rust (for Linux, check the link for other systems)
# You can refer to https://rust-lang.org/learn/get-started/ to install Rust, or run the command below:
# WARNING: this command may update your shell profile(~/.bashrc or ~/.zshrc) to add Rust to PATH.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# NOTE: After installing Rust, you usually need to restart your terminal to activate it.

# 2. Install Entangle
# Assume you are starting under the root directory of this repository.
# This command installs `entangle` and `torchgraph`(deprecated) python packages and `tg` and `tginfer` commands.
pip install -e .

# 3. Build egger and then go back.
cd egger && cargo build --release && cd ..
```

## Examples

Please refer to the [examples/README.md](examples/README.md) for instructions on running examples.


## Artifact Evaluation

Thank you for your reviews.

This artifact is our implementation of the inductive relation refinement method described in the paper. We also provides input computation graphs for the open-source frameworks, but we choose not to publish the graphs of the company's proprietary models. 

With this repository, the performance results for verification graphs from open-source frameworks can be reproduced, including

- (Figure 3) End-to-end verification time across different models
- (Figure 4) Scalability on verifying parallelized models.
- (Figure 6) Heatmap showing the number of times each lemmas is used for different models.

This repository does NOT contain codes to draw Figure 5 since it relies human efforts to manually count the lines of codes (LOC).

We first introduce the [Code Structure](#code-structure), then provide [Evaluation Instructions](#evaluation-instructions) to reproduce open-source part results in the paper.

### Code Structure

(Note: only key components are introduced here.)

The roles of the code files/directories are as described below in the comments.

```sh
egger
├── extractor # helper to extract output relations
├── src
│   ├── load.rs     # load egraph dumped from Python side
│   ├── main.rs     # entry point
│   ├── metadata.rs # internal types
│   ├── model.rs    # operators
│   ├── rewrite     # lemmas
│   ├── special     # operators and lemmas for new frameworks
│   ├── symval      # symbolic value manager
entangle    # the key Python package
├── convert     # converts graphs between different formats
├── graph       # graph dumping tools wrapping TorchDynamo
├── ops         # defines ATen and customized operators
├── pgraph      # graph format dumped through our wrapping. 
├── sgraph      # core graph format used during verification
│   ├── egraph.py    # egrpah format
│   ├── infer        # contains different exploration strategies.
│   ├── sexpr.py     # for expressions
│   ├── sgraph.py    # for the whole computional graph
│   ├── transform.py # lower distributed operators
examples    # examples, see examples/README.md
```

### Evaluation Instructions

#### 1. Setup

Setup your environment following instructions in previous [Setup](#setup).

#### 2. Run all experiments

The example input compution graphs are in [examples/data](./examples/data/). Run the commands below to run the refinement inference and verification.

```sh
# Assuming you are in `examples`
# gpt starts 16 runs sequentially
python run_all.py gpt --all
# qwen2 starts 2 runs sequentially
python run_all.py qwen2 --all
# aws_llama starts 12 runs sequentially
python run_all.py aws_llama --all 
```

#### 3. Visualization

To easily compare with results in the paper, run the visualization script to generate figures after step 2.

```sh
python visualization.py
```

The figures will be saved to `examples/figures`, including
- `one_layer_time.pdf`: the end-to-end verification time results (Figure 3). The performance results can vary when you use different hardwares.
- `GPT_scalability.pdf`, `Llama-3_scalibility.pdf`: the scalability results (Figure 4), where you should see a bit super-linear time cost increasing with the parallel degrees and linear time cost increasing with the model sizes.
- `lemma_applied_count_heatmap.pdf`: the heatmap of lemma application counts (Figure 6), where you should see something similar to the one in the paper.

#### 4. (Optional) Try for Your Models

Refer to [Use Your Own Models](#use-your-own-models) for instructions on capturing graphs from your own models and running verification.

-----

## Use Your Own Models

### Capture Graphs

#### PyTorch Models

Our implementation supports the most commonly used operators defined in ATen. Thus, as long as your PyTorch model can be captured by TorchDynamo completely without break, our tool can verify your model directly.

We wrapped TorchDynamo to generate computation graphs in the format of `PickleableGraph` (or `pgraph`, see [definition](./entangle/pgraph/pickleable.py)) for convenience. We recommend you to use it to capture your model. 

An example can be found in [capture_example.py](./examples/capture_example.py). Simply run the command below to run this example:

```sh
# In `examples` directory
python capture_example.py
```

The output files will be printed on the terminal for you to check.

#### New Operators

Since our implementation doesn't include some rerely used operators, and it is possible your model used some customized operators, you might need to 
- Define the new operators. Examples can be found [here](./entangle/ops).
- Define converter from `PickleableGraph` (or `pgraph`) to `sexpr`. Examples can be found [here](./entangle/convert/mappings).

#### Other Frameworks

For frameworks other than PyTorch (like TensorFlow, JAX, etc.), you must implement a converter from your framework's computation graph IR to Entangle's `pgraph` representation.

In this repository, we provided a simple incomplete [example](./entangle/pgraph/hlo.py) for [HLO IR](https://github.com/openxla/stablehlo), where only a few operators required by our evaluation are implemented. You can refer to this example to implement your own converter.

Besides, new frameworks usually introduces new operators. Thus you should also follow [New Operators](#new-operators) to define the new operators used in your framework.

### Run Refinement Inference and Verification

#### Configuration

To run for your own model, you must provide clean input relations as mentioned in the paper. Examples can be found in [verify_gpt.py](./examples/verify_gpt.py), [verify_aws_llama.py](./examples/verify_aws_llama.py) and [verify_qwen2.py](./examples/verify_qwen2.py). The tool automatically found first Python `class` that inherits from `ExplorativeConfig`.

The commandline arguments are passed to the initializer of the module such that you can define your own arguments (like TP size).

#### Run

With the captured graphs, you can run the tool using the command below:

```sh
export OUTDIR=<output directory>
mkdir -p $OUTDIR
tg infer \
    --infer_manager greedy \
    --stats \
    -g <path to your graphs> \
    --origin <directory name of sequential graph> \
    --target <directory name of distributed graph> \
    --graph_prefix <name of the graph break> \
    --config_module <Python config file path> \
    -o $OUTDIR \
    [other options you defined inside your config module]
```

Note for performance:
- We used Python `rich` package for fancy terminal outputs. However, we found it significantly slows down the overal performance. You may disable it by `--disable_rich` for better performance.

#### New Operators and Lemmas

If your model used new operatros, you will need to also
- Define new operators. Examples can be found [here](./egger/src/special). 
- Define new lemmas regarding new operators if necessary. Examples can be found [here](./egger/src/special).


-----

## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
