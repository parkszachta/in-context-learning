The "eval.ipynb" file can be run via the following steps:

- Download the code.
- Download the trained models.
- Move the folder containing trained models directly into the outermost folder of the code.
- From directly inside the checkpoint folder, run "conda env create -f environment.yml" and "conda activate in-context-learning".
- Run the relevant commented commands that are at the bottom of "environment.yml".
- In the first code block of eval.ipynb, set the last line to "run_dir = 'trained_models/models\_\*'" depending on the models you would like to analyze.
- In the fourth code block of "eval.ipynb", enable the relevant "task", set "run_id" to the relevant run (which can be found after running the first two code blocks), and set "skip_baselines = False" if you want baselines to be computed. Note that setting "skip_baselines = False" may cause significantly increased runtime.
- Run the relevant code blocks in "eval.ipynb".
