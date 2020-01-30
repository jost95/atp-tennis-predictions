# cs-ml-project

## Data generation

To generate data that can be used in training both pure
statistics generation via `stats.py` and pre processing via `pre_processing.py` is needed.

To generate data, configure your settings in the `config.json` file and then run `main.py`.

### Statistics generation
The approximate run-time is about 5 minutes. Some statistics that are generated include
mutual player statistics, total wins, surface wins etc.

### Pre processing
The approximate run-time is about 4 hours. This is the script that actually creates a useful datafile by taking the statistics
file and generating opponent different statistics.

## Notebooks

Three different notebooks are available:

1. For feature evaluation, model selection and testing: `eval_select_test.ipynb`
2. For merging main data set with historic odds: `results_odds_merge.ipynb`
3. For evaluating final model with an all-in betting strategy: `odds_evaluation.ipynb`
