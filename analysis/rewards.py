import math
import json
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import pandas as pd


plt.style.use('ggplot')

FIGURE_X = 7.0
FIGURE_Y = 5.0

# Data to plot.
data = [
    {
        'mdp': 'Min. speed delta',
        'paths': {
            'QL':  '/path/to/XXXX.YYY.tar.gz',
            'DQN': '/path/to/XXXX.YYY.tar.gz',
            'DDPG':  '/path/to/XXXX.YYY.tar.gz',
        }
    },
    {
        'mdp': 'Min. delay',
        'paths': {
            'QL':  '/path/to/XXXX.YYY.tar.gz',
            'DQN': '/path/to/XXXX.YYY.tar.gz',
            'DDPG':  '/path/to/XXXX.YYY.tar.gz',
        }
    },
]


def calculate_CI_bootstrap(x_hat, samples, num_resamples=20000):
    """
        Calculates 95 % interval using bootstrap.

        REF: https://ocw.mit.edu/courses/mathematics/
            18-05-introduction-to-probability-and-statistics-spring-2014/
            readings/MIT18_05S14_Reading24.pdf

    """
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    means = np.mean(resampled, axis=0)
    diffs = means - x_hat
    bounds = [x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95)]

    return bounds


def main():

    num_cols = 3
    num_rows = math.ceil(len(data) / num_cols)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    fig.tight_layout(pad=3.0)

    counter = 0
    for (idx, d) in enumerate(data):

        if num_rows > 1:
            r = counter // num_cols
            c = counter % num_cols
            curr_ax = ax[r][c]
        else:
            curr_ax = ax[counter]


        names = []
        values = []
        errors = []

        for (method, path) in d['paths'].items():
            # Get test eval json file from experiment root folder.

            tar = tarfile.open(path)

            all_names = tar.getnames()
            json_file = [x for x in all_names if Path(x).name == 'rollouts_test.json'][0]

            # Create temporary directory.
            dirpath = tempfile.mkdtemp()

            # Extract json file to temporary directory.
            tar.extract(json_file, dirpath)

            # Load JSON data.
            with open(dirpath + '/' + json_file) as f:
                json_data = json.load(f)

            # Clean temporary directory.
            shutil.rmtree(dirpath)


            # Calculate cumulative reward.
            id = str(json_data['id'][0])
            dfs_r = [pd.DataFrame(r) for r in json_data['rewards'][id]]
            df_concat = pd.concat(dfs_r, axis=1)
            df_rewards = df_concat.to_numpy()
            cum_rewards = np.sum(df_rewards, axis=0)

            bounds = calculate_CI_bootstrap(np.mean(cum_rewards), cum_rewards)

            names.append(method)
            values.append(np.mean(cum_rewards))
            errors.append(bounds)

            # print(np.mean(cum_rewards))
            # print(bounds)
            # print(df_rewards)
            # print(df_rewards.shape)

        errors = np.array(errors).T
        errors = np.flip(errors, axis=0)
        error_lengths = np.abs(np.subtract(errors, values))

        x = 0.5*np.arange(len(names))

        curr_ax.bar(x, values, width=0.25, yerr=error_lengths, capsize=4)

        curr_ax.set_xticks(x)
        curr_ax.set_xticklabels(names)

        low = min(values)
        high = max(values)
        curr_ax.set_ylim([low-1.25*(high-low), high+1.25*(high-low)])

        counter += 1

    plt.savefig('analysis/plots/rewards_comparison.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/rewards_comparison.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()


if __name__ == "__main__":
    main()
