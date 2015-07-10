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
	if [ -n "$3" ]; then
		n_proc=$3
	fi
	if [ -n "$4" ]; then
		m_proc=$4
	fi
	total_proc=`expr $n_proc \* $m_proc`
	echo "Running ${exec_name} with problem size ${problem_size}, ${n_proc} x ${m_proc} processes"
	aprun -n ${total_proc} ${script_dir}/${exec_name} -n ${n_proc} -m ${m_proc} -b ${problem_size}
}

#run tests
run_test wave_test_cpu 256 4 4
run_test wave_test_openacc 256 4 4

echo "All tests passed"
