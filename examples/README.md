# Examples

This directory contains examples for verifying open-sourced models.

## Download Captured Graphs

As a first step, you need to download the graphs used for verification into `./data`
```sh
# Assume you are starting under the root directory of this repository.
cd examples
git clone git@github.com:RabbitWhite1/entangle-data.git data
```

## Running Examples

The general command to run an example is as follows:
```sh
python run_all.py <MODEL> --tp <TP_SIZE> --num_layers <NUM_LAYERS> [--stdout]
```
For more details, you can refer to `python run_all.py --help`.

Note that `run_all.py` is just a wrapper of `tg infer` command. You can also directly use `tg infer` command to run the verification. When you run `run_all.py`, the corresponding `tg infer` command will be printed on the terminal, you can use them as examples to run your own codes.

### Verifying GPT

```sh
python run_all.py gpt --tp 2 --num_layers 1 --stdout
# or equivalently
# export OUTDIR=./precondition.gpt.fw.g0.paral1_layer1-paral2_layer1.greedy && mkdir -p $OUTDIR; tg infer -g data/gpt --origin paral1_layer1 --target paral2_layer1 --graph_prefix fw.g0 --config_module verify_gpt.py --tp=2 --num_layers=1 --infer_manager greedy --stats -o ./precondition.gpt.fw.g0.paral1_layer1-paral2_layer1.greedy  | tee $OUTDIR/output.log 
```

All available `TP_SIZE` are 2, 4, 6, 8, and all available `NUM_LAYERS` are 1, 2, 4, 8.

### Verifying Llama-3

```sh
python run_all.py aws_llama --tp 2 --num_layers 1 --stdout
# or equivalently
# export OUTDIR=./precondition.aws_llama.fw.g0.tp1_layer1-tp2_layer1.greedy && mkdir -p $OUTDIR; tg infer -g data/aws_llama --origin tp1_layer1 --target tp2_layer1 --graph_prefix fw.g0 --config_module verify_aws_llama.py --tp=2 --num_layers=1 --infer_manager greedy --stats -o ./precondition.aws_llama.fw.g0.tp1_layer1-tp2_layer1.greedy  | tee $OUTDIR/output.log
```

All available `TP_SIZE` are 2, 4, 8, and all available `NUM_LAYERS` are 1, 2, 4, 8.

### Verifying Qwen2

```sh
python run_all.py qwen2 --tp 2 --num_layers 1 --stdout
# or equivalently
# export OUTDIR=./precondition.qwen2.fw.g0.paral1_layer1-paral2_layer1.greedy && mkdir -p $OUTDIR; tg infer -g data/qwen2 --origin paral1_layer1 --target paral2_layer1 --graph_prefix fw.g0 --config_module verify_qwen2.py --tp=2 --num_layers=1 --infer_manager greedy --stats -o ./precondition.qwen2.fw.g0.paral1_layer1-paral2_layer1.greedy  | tee $OUTDIR/output.log
```

All available `TP_SIZE` are 2, 4 and all available `NUM_LAYERS` are 1.
