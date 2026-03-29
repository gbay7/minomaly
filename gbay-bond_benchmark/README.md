# GBay BOND : Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs

This repository focuses on benchmarking various algorithms for unsupervised outlier node detection on static attributed graphs. This work is an improvement on the implementation by the PyGOD team, with enhancements to performance, usability, and extensibility.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Outlier node detection in graphs is a crucial task in many applications such as fraud detection, network security, and social network analysis. This project benchmarks several state-of-the-art algorithms for unsupervised outlier node detection on static attributed graphs, providing a comprehensive evaluation of their performance.

This project is built upon the work by the [PyGOD team](https://github.com/pygod-team/pygod/tree/main/benchmark), with several improvements to enhance the benchmarking process.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/bay127prem/gbay-bond_benchmark.git
cd gbay-bond_benchmark
pip install -r requirements.txt
```

## Usage

To run the benchmarking suite, use the following command:

```bash
python gbay_bench.py --models models.json --datasets datasets_synth.json --epochs 10 100 300 400 --num_trial 20 --gpu 0 --out bench_synth_results 2>&1 | tee -a bench_synth_results.log
```

```bash
python gbay_bench_all.py --models models.json --datasets datasets.json --epochs 10 100 300 400 --num_trial 20 --gpu 0 --out bench_results 2>&1 | tee -a bench_results.log
```

### gbay_bench.py vs gbay_bench_all.py

- `gbay_bench.py`: This script separates between contextual and structural anomalies in the benchmark. Use this script if you need to distinguish between these types of anomalies.
- `gbay_bench_all.py`: This script does not distinguish between contextual and structural anomalies. Use this script for a general benchmark that treats all anomalies the same.

### Command Parameters

- `--models models.json`: Specifies the JSON file containing the model configurations. Each model listed in this file will be used in the benchmarking process.

- `--datasets datasets.json`: Specifies the JSON file containing the dataset configurations. Each dataset listed in this file will be used in the benchmarking process.

- `--epochs 10 100 300 400`: A list of epochs (iterations) for which the benchmarking will be performed. The script will run the models for the specified number of epochs and record the performance metrics at each stage.

- `--num_trial 20`: The number of trials to run for each model and dataset combination. Running multiple trials helps in obtaining statistically significant results with random hyperparameters set each time.

- `--gpu 0`: Specifies the GPU device ID to be used for the benchmarking. Set to `0` for the first GPU, `1` for the second, and so on. Use `-1` to run on the CPU.

- `--out bench_results`: The directory where the benchmarking results will be saved. Each run creates a timestamped subdirectory within this directory containing performance metrics, visualizations, and logs.

- `2>&1 | tee -a bench_results.log`: This part of the command redirects both standard output and standard error to the `bench_results.log` file, while also displaying the output in the terminal. This ensures that all logs are saved for later analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
