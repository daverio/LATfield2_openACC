#!/bin/bash
set -e
#Note: -e makes the script fail if any subcommand fails (i.e. return code != 0)
# --> our tests need to return a failure code if something is wrong
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"

#parameters
n_proc=2
m_proc=2

function run_test {
	echo "Running $1 with problem size $2"
	aprun -n ${total_proc} ${script_dir}/$1 -n ${n_proc} -m ${m_proc} -b $2
}

#setup
total_proc=`expr $n_proc \* $m_proc`

#run tests
run_test fft_test 64
run_test poisson 64
run_test fft_test 128
run_test poisson 128
echo "All tests passed"