"""Compare Q tables generated by different experiments"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-20'
import os
import numpy as np
import dill

# current project dependencies
from ilurl.envs.base import TrafficLightQLEnv

ROOT_DIR = os.environ['ILURL_HOME']
# EMISSION_DIR = f"{ROOT_DIR}/data/experiments/0x02/"
EMISSION_DIR = f"{ROOT_DIR}/data/experiments/0x04/"
# CONFIG_DIRS = ('4545', '5040', '5434', '6030')
CONFIG_DIRS = ('6030',)


filenames = ('intersection_20200227-2043431582836223.796797.Q.1-99',
             'intersection_20200227-2128041582838884.952684.Q.1-100',
             'intersection_20200227-2128511582838931.2348.Q.1-100')


def qdist(Q, Q1):
    """Computes the maximium value absolute value

    Params:
    -------
        * Q: dictionary of dictionaries
        Nested dictionary representing a table
        * Q1: dictionary of dictionaries
        Nested dictionary representing a table

    Returns:
    -------
        * distance: float
            sum(|a -b|) / sqrt(A * B)
    """
    states = set(Q.keys()).union(set(Q1.keys()))
    distance = 0
    total, total1 = 0, 0
    for state in states:
        actions = set(Q[state].keys()).union(set(Q1[state].keys()))
        for action in actions:
            total += Q[state][action]
            total1 += Q1[state][action]
            distance += np.abs(Q[state][action] - Q1[state][action])

    ret = np.round(distance / np.sqrt(total * total1), 2)
    return ret


def jacd(pola, polb):
    """Computes the jaccard distance

    Params:
    -------
    * pola: dictionary from states to actions
        deterministic policy computed by experiment a
    * polb: dictionary from states to actions
        deterministic policy computed by experiment b

    Returns:
    --------
    * jaccard distance
        1 - jaccard similarity
    """
    states = set(pola.keys()).union(set(polb.keys()))
    d = 0
    for s in states:
        set_a = set(pola[s])
        set_b = set(polb[s])
        d += 1 - (len(set_a & set_b) / len(set_a | set_b))

    return d


def policy(Q):
    """Hard max over prescriptions

    Params:
    -------
        * Q: dictionary of dictionaries
        Nested dictionary representing a table

    Returns:
    -------
        * policy: dictonary of states to policies
    """
    pol = {}
    for s in Q:
        pol[s] = max(Q[s].items(), key=lambda x: x[1])[0]

    return pol

def visited(Q):
    """Visited states

    Params:
    -------
        * Q: dictionary of dictionaries
        Nested dictionary representing a table

    Returns:
    -------
        * vis: a list of unique visited states
    """
    vis = sorted([s for s in Q if any([v for v in Q[s].values()])])

    return vis


def num_state_actions(Q):
    t = 0
    n = 0
    for s in Q:
        for v in Q[s].values():
            n += int(v != 0)
            t += 1
    return n, t


if __name__ == '__main__':
    QS = []
    PI = []
    for config_dir in CONFIG_DIRS:
        for filename in filenames:
            path = f'{EMISSION_DIR}{config_dir}/{filename}.pickle'
            with open(path, 'rb') as f:
                Q = dill.load(f)
            QS.append(Q)
            PI.append(policy(Q))

    N = len(QS)
    D = np.zeros((N, N), dtype=np.float)
    JD = np.zeros((N, N), dtype=np.float)
    for i in range(N - 1):
        for j in range(i + 1, N):
            D[i, j] = qdist(QS[i], QS[j])
            JD[i, j] = jacd(PI[i], PI[j])

    # number of state-action pairs explored
    states = {s for Q in QS for s in Q}
    vis = {s for Q in QS for s in visited(Q)}
    msa = np.round(np.mean([len(visited(Q)) for Q in QS]), 2)
    tsa = len(states)
    print(f'mean number of states visited:{msa} of {tsa}')
    print(f'visited states: {vis}')

    # percentual distance
    print('Manhattan distance (%)')
    print(D)
    # policy disagreement
    print('Jaccard distance')
    print(JD)

