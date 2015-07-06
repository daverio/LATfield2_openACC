#!/bin/bash
set -e
#Note: -e makes the script fail if any subcommand fails (i.e. return code != 0)
# --> our tests need to return a failure code if something is wrong
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"

#parameters
n_proc=2
m_proc=2
problem_size=128

#setup
total_proc=`expr $n_proc \* $m_proc`

#run tests
echo "Running fft_test"
aprun -n ${total_proc} ${script_dir}/fft_test -n ${n_proc} -m ${m_proc} -b ${problem_size}
echo "Running poisson"
aprun -n ${total_proc} ${script_dir}/poisson -n ${n_proc} -m ${m_proc} -b ${problem_size}
echo "All tests passed"