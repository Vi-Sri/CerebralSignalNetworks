# Project Name

Applying Barlow twins for EEG-Image Cross modality Self supervised learning

## Table of Contents

- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Usage

To train the model, run the following command:

```shell
srun python -m torch.distributed.launch \
    --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
    --master_port=12345 \
    train.py \
    --data /path/to/dataset \
    --workers 8 \
    --epochs 1000 \
    --batch-size 16 \
    --learning-rate-weights 0.2 \
    --learning-rate-biases 0.0048 \
    --weight-decay 1e-6 \
    --lambd 0.0051 \
    --print-freq 100 \
    --checkpoint-dir /path/to/checkpoints

```


## Contributing

Guidelines on how to contribute to the project and any specific requirements.

## License

Information about the project's license and any additional terms or conditions.
