# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# RUN FAST 9.2

for r in "0" "1e-3" "1" "100"
do
    # Boston
    taskset -c 0-14 python experiments/momentum_regimes.py boston -k False -r $r -t 1e-6 -m 700 -n 10
    # Boston + kernel
    taskset -c 0-14 python experiments/momentum_regimes.py boston -k True -r $r -t 1e-6 -m 8000 -n 10
    # California housing
    taskset -c 0-14 python experiments/momentum_regimes.py cali -k False -r $r -t 1e-6 -m 800 -n 10
    # YearPredictionMSD
    taskset -c 0-14 python experiments/momentum_regimes.py year -k False -r $r -t 1e-6 -m 1000 -n 10
done
