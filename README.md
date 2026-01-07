<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white" style="height:20px"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" style="height:20px"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white" style="height:20px"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white" style="height:20px"></a>
</p>

<p align="center" style="margin-top: -20px; margin-bottom: -30px;">
    <img src=https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216 style="width: 40%">
</p>

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

### Option 1 (Recommended): CloudLab Setup

If you are using CloudLab, we provide a CloudLab profile and a single script to automatically set up the environment.

You can find the profile [here](https://www.cloudlab.us/p/rdma-prefetch/Entangle). After starting an experiment with this profile, login to the shell and run the following commands to setup (you don't need to clone the repository in advance):

```sh
wget https://raw.githubusercontent.com/nyu-systems/Entangle/refs/heads/main/setup.sh -O $HOME/setup.sh && source $HOME/setup.sh
```

The script will
1. install Rust (cargo), uv
2. clone the repository to `/opt/tiger/Entangle`
3. set up the Python environment with uv
4. build and installs Entangle

### Option 2: Manual Setup

#### Recommended Environment

For best reproducing, we recommend using the environment we used for the experiments:

- Ubuntu-22.04
    - ✅️ Linux: should run (not tested)
    - ✅️ MacOS: can run, but significantly slower (tested)
    - ❌ Windows: cannot run due to usage of bash commands
      - ✅️ WSL: can run the tool (tested)
- Python 3.12
- Rust supporting edition 2018

Other software recommendation are described in [pyproject.toml](./pyproject.toml) and [Cargo.toml](./egger/Cargo.toml), and can be installed following the [Install Dependencies](#install-dependencies) instructions below.

#### Install Dependencies

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

# Alternatively, if you are using `uv`, run the 3 commands below:
# uv venv
# uv sync
# source .venv/bin/activate

# 3. Build egger and then go back.
cd egger && cargo build --release && cd ..
```

## Examples

Please refer to the [examples/README.md](examples/README.md) for instructions on running examples.


-----


## Artifact Evaluation

Thank you for your reviews.

This artifact is our implementation of the inductive relation refinement method described in the paper. We also provides input computation graphs for all the open-source models/frameworks we used. But unfortunately, we cannot publish the graphs of the company's proprietary models. 

With this repository, the performance results for verification graphs from open-source frameworks can be reproduced, including

- (Figure 3) End-to-end verification time across different models
- (Figure 4) Scalability on verifying parallelized models.
- (Figure 5) Lemmas complexity statistics (which requires manual counting and the results are put in the [visualization script](./examples/visualization.py)).
- (Figure 6) Heatmap showing the number of times each lemmas is used for different models.

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

#### Step 1. Setup

Setup your environment following instructions in previous [Setup](#setup).

#### Step 2. Run all experiments

The example input compution graphs are in [examples/data](./examples/data/). Run the commands below to run the refinement inference and verification.

##### Step 2.1 Model Verification

These three commands run all verification task for each model sequentially. All the verified graphs are bug-free.

```sh
# Assume you are in directory `examples`
# gpt starts 16 runs sequentially (TP degrees are 2, 4, 6, 8, and numbers of layers are 1, 2, 4, 8)
python run_all.py gpt --all
# qwen2 starts 2 runs sequentially (TP degrees are 2, 4, and number of layers is 1)
python run_all.py qwen2 --all
# aws_llama starts 12 runs sequentially (TP degrees are 2, 4, 8, and numbers of layers are 1, 2, 4, 8)
python run_all.py aws_llama --all 
```

If you see `Refinement verification succeeded for ...` messages after each run, then it means the verification is successful.

**A Note for OUTPUT directory**: By default, these commands only output the real running commands, which set `$OUTPUT` environment variable as the directory for output logs. The output directory includes:
- `$OUTPUT/group*`: logs for each step of inference
- `$OUTPUT/check_impl`: logs for checking user expectations
- `$OUTPUT/index.html`: visualization for the graphs
- `$OUTPUT/output.log`: overall log for the whole verification

##### Step 2.2 Bug Detection

The 4 commands run the verification for all the 4 pairs of graphs **with bugs** introduced in the paper. 

The verification is **expected to raise errors**. If you get any of the errors below, then it means the bug is successfully detected:

- entangle.sgraph.egraph.CannotFindPostconditions: <span style="color: red">Failed, check the conditions above.</span>
- entangle.tools.egg.FailedImplyingEquivalence: <span style="color: red">User expectation violated.</span>

```sh
# Assume you are in directory `examples`
python ./run_all.py grad_accumulation  # (Bug 6 in paper)
python ./run_all.py missing_allreduce_under_wrong_config  # (Bug 7 in paper)
python ./run_all.py missing_switchmlp_allreduce  # (Bug 8 in paper)
python ./run_all.py missing_layernorm_allreduce  # (Bug 9 in paper)
```


#### Step 3. Visualization

To easily compare with results in the paper, run the visualization script to generate figures after step 2 ([Model Verification Part](#21-model-verification)).

```sh
python visualization.py
```

The figures will be saved to `examples/figures`, including
- `one_layer_time.pdf`: the end-to-end verification time results (Figure 3). The performance results can vary when you use different hardwares.
- `GPT_scalability.pdf`, `Llama-3_scalability.pdf`: the scalability results (Figure 4), where you should see a bit super-linear time cost increasing with the parallel degrees and linear time cost increasing with the model sizes.
- `number_of_ops_and_lemmas.pdf`: the number of operators and lemmas used (Figure 5.a), where you should see the same number as the one in the paper (except that data of company's proprietary models are removed).
- `lemma_loc.pdf`: the CDF of LOC of the lemmas (Figure 5.b), where you should see a similar trend as the one in the paper (you will see some differences since we only include open-sourced models/framworks here).
- `lemma_applied_count_heatmap.pdf`: the heatmap of lemma application counts (Figure 6), where you should see something similar to the one in the paper.

For your convenience, we also provide reference figures in [examples/reference_figures](./examples/reference_figures) to compare.


**End of Artifact Evaluation Instructions**

-----

## Use Your Own Models

### Capture Graphs

#### PyTorch Models

Our implementation supports the most commonly used operators defined in ATen. Thus, as long as your PyTorch model can be captured by TorchDynamo completely without break, our tool can verify your model directly.

We wrapped TorchDynamo to generate computation graphs in the format of `PickleableGraph` (or `pgraph`, see [definition](./entangle/pgraph/pickleable.py)) for convenience. We recommend you to use it to capture your model. 

An example can be found in [capture_example.py](./examples/capture_example.py). Simply run the command below to run this example:

```sh
# Assume you are in directory `examples`
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

## About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
