# ASR

This is a repository for ASR homework with implementation of Conformer model and some experiments with it. Also that repository can be used as a template for other projects.

## Getting Started

These instructions will help you to run a project on your machine.

### Prerequisites

Install [poetry](https://python-poetry.org/docs/#installation) following the instructions in the official repository.

### Installation

Clone the repository into your local folder:

```bash
cd <path/to/local/folder>
git clone https://github.com/toobrainless/dla_template -b asr
```

Setup requirements with poetry

```bash
cd <path/to/cloned/ASR/project>
poetry install
```
<!-- Надо написать на английском: теперь вы можете использовать poetry run чтобы использовать среду -->
Now you can use `poetry run` to use environment. For example, `poetry run python3 train.py`, for more usability read [poetry documentation](https://python-poetry.org/docs/).

### Setup heavy stuff

```bash
poetry run python3 setup_lm_model.py
poetry run python3 setup_conformer.py
```

## Training

To train model run with default config:

```bash
poetry run python3 train.py
```

To resume training from checkpoint

```bash
poetry run python3 train.py resume=path\to\saved\checkpoint.pth
```

To change config parameters use the following approach:

```bash
poetry run python3 train.py param1=value1 param2=value2
```

For example, to change batch size and number of epochs:

```bash
poetry run python3 train.py data.train.batch_size=32 trainer.epochs=10
```

The train config is stored in `configs/train.yaml`, it contains path to the datasets, model, optimizer, scheduler, trainer and other parameters. Also you can change it manually or create your own config file.

## Testing

To evaluate model run the following script:

```bash
poetry run python3 test.py
```

The test config is stored in `configs/test.yaml`, it contains path to the checkpoint, datasets and metrics for evaluation. To change config parameters use the same approach as for training.
