#!/bin/bash
set -e
#Note: -e makes the script fail if any subcommand fails (i.e. return code != 0)
# --> our tests need to return a failure code if something is wrong
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"

#parameters
n_proc_default=4
m_proc_default=4

function run_test {
	exec_name=$1
	problem_size=$2
	n_proc=${n_proc_default}
	m_proc=${m_proc_default}
	if [ ! -z $3 ]; then
		n_proc=$3
	fi
	if [ ! -z $4 ]; then
		n_proc=m_proc
	fi
	total_proc=`expr $n_proc \* $m_proc`
	echo "Running ${exec_name} with problem size ${problem_size}, ${n_proc} x ${m_proc} processes"
	aprun -n ${total_proc} ${script_dir}/${exec_name} -n ${n_proc} -m ${m_proc} -b ${problem_size}
}

#run tests
run_test gettingStarted_openacc
# run_test fft_test_cpu 64
# run_test poisson_cpu 64
# run_test fft_test_cpu 128
# run_test poisson_cpu 128
# run_test fft_test_openacc 64
run_test poisson_openacc 64 4 2
# run_test fft_test_openacc 128
# run_test poisson_openacc 128
run_test wave_test_openacc 16
run_test wave_test_openacc 64 2 4
echo "All tests passed"
