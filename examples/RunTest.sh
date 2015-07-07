#!/bin/bash
set -e
#Note: -e makes the script fail if any subcommand fails (i.e. return code != 0)
# --> our tests need to return a failure code if something is wrong
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"

#parameters
n_proc=4
m_proc=4

function run_test {
	echo "Running $1 with problem size $2"
	aprun -n ${total_proc} ${script_dir}/$1 -n ${n_proc} -m ${m_proc} -b $2
}

#setup
total_proc=`expr $n_proc \* $m_proc`

#run tests
# run_test fft_test_cpu 64
run_test poisson_cpu 64
# run_test fft_test_cpu 128
run_test poisson_cpu 128
# run_test fft_test_openacc 64
# run_test poisson_openacc 64
# run_test fft_test_openacc 128
# run_test poisson_openacc 128
echo "All tests passed"