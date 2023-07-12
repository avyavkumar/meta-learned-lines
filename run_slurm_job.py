import os
import random
from argparse import ArgumentParser


def main(params):
    totalTrials = params.trials
    outerLRs = [5e-3, 5e-4, 5e-5]
    innerLRs = [5e-2, 5e-3, 5e-4]
    outputLRs = [1e-1, 1e-2, 1e-3]
    stepSize = [3, 5, 7]

    indicesTried = []
    trialsLaunched = 0

    while trialsLaunched < totalTrials:
        # pick random indices for each hyperparameter
        indices = [random.randint(0, 2) for i in range(4)]
        key = str(indices[0]) + "_" + str(indices[1]) + "_" + str(indices[2]) + "_" + str(indices[3])
        if key not in indicesTried:
            indicesTried.append(key)
            trialsLaunched += 1
            command = "sbatch -p gpu --gres gpu run-job.sh -o {} -i {} -l {} -s {}".format(str(outerLRs[indices[0]]),
                                                                                           str(innerLRs[indices[1]]),
                                                                                           str(outputLRs[indices[2]]),
                                                                                           str(stepSize[indices[3]]))
            print(command)
            os.system(command)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-t', '--trials', type=int)
    params = parser.parse_args()
    # launch the required jobs
    main(params)
