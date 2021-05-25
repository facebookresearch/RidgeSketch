# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# RUN SLOW 9.2

for r in "100"
do
    # Cali kernel
    taskset -c 97-106 python experiments/momentum_regimes.py cali -k True -r $r -t 1e-3 -m 7000 -n 1

    # Rcv1
    taskset -c 97-106 python experiments/momentum_regimes.py rcv1 -k False -r $r -t 1e-3 -m 4000 -n 1
done
