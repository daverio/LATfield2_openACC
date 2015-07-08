#---------------- parameters -----------------------------------------------#
EXEC                      := poisson wave_test gettingStarted #fft_test
TEST_WITH_EVERY_BUILD     := true
COMPILER                  := CC
INC                       := -I./../
LIB_OPENACC               := -L/opt/nvidia/cudatoolkit6.5/default/lib64/
LIB_CPU                   := -lfftw3 -lm
DEF_LATFIEILD             := -DFFT3D_ACC
ifdef DEBUG
OPT_OPENACC               := -DGPU
OPT_CPU                   :=
CFLAG                     := -g -std=c++11 -Mbounds -traceback -w --gnu
else
OPT_OPENACC               := -O3 -acc -DGPU -Minfo=accel -ta=tesla:cc35 -VH15.7
OPT_CPU                   := -O3
CFLAG                     := -std=c++11 -DNO_OUTPUT -w
# CFLAG                     := -std=c++11 -w -Y0,/users/hck60/pgi -Yc,/users/hck60/pgi -Mqq,expand+beforeaccel,printblocks
# CFLAG                     := -E
endif

#---------------- setup -----------------------------------------------------#
HEADER                    := $(wildcard ../*.hpp) $(wildcard LATfield2d/*.h)
EXEC_CPU                  := $(addsuffix _cpu,$(EXEC))
EXEC_OPENACC              := $(addsuffix _openacc,$(EXEC))
DEF                       := $(DEF_LATFIEILD)
ifeq ($(TEST_WITH_EVERY_BUILD),true)
ADDITIONAL_BUILD_TARGETS=tests
else
ADDITIONAL_BUILD_TARGETS
endif
.PHONY: all tests clean openacc cpu

#---------------- target build rules ----------------------------------------#
#mpic++ main.cpp -I./LATfield2d -I../local/include/gsl/  -DFFT3D -DPHINONLINEAR -DCHECK_B -lfftw3 -lm -lhdf5 -lgsl -lgslcblas -std\
# =c++0x -O3  -o test

all: openacc cpu ${ADDITIONAL_BUILD_TARGETS}

openacc: ${EXEC_OPENACC}

cpu: ${EXEC_CPU}

tests: ${EXEC_OPENACC} ${EXEC_CPU}
	./RunTest.sh

%_cpu: %.cpp $(HEADER) makefile
	$(COMPILER) $< $(INC) $(DEF) $(LIB_CPU) $(OPT_CPU) $(CFLAG) -o $@

%_openacc: %.cpp $(HEADER) makefile
	$(COMPILER) $< $(INC) $(DEF) $(LIB_OPENACC) $(OPT_OPENACC) $(CFLAG) -o $@

clean:
	rm -f ${EXEC_CPU} ${EXEC_OPENACC} *.o *~
