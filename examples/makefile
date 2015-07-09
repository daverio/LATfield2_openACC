#---------------- parameters -----------------------------------------------#
EXEC                      := poisson wave_test #fft_test
TEST_WITH_EVERY_BUILD     := true
COMPILER                  := CC
INC                       := -I./../
#LIB                       := -lfftw3 -lm
LIB			  := -lcufftw -lcufft
DEF_LATFIELD_OPENACC      := -DFFT3D_ACC
DEF_LATFIELD_CPU          := -DFFT3D
ifdef DEBUG
OPT_OPENACC               :=
OPT_CPU                   :=
CFLAG                     := -g -std=c++11 -Mbounds -traceback -w
else
OPT_OPENACC               := -O3 -acc -Minfo=accel -ta=tesla:cc35
OPT_CPU                   := -O3
CFLAG                     := -std=c++11 -w -DNO_OUTPUT -DDO3D -VH15.7
endif

#---------------- setup -----------------------------------------------------#
HEADER                    := $(wildcard ../*.hpp) $(wildcard LATfield2d/*.h)
EXEC_CPU                  := $(addsuffix _cpu,$(EXEC))
EXEC_OPENACC              := $(addsuffix _openacc,$(EXEC))
DEF_OPENACC               := $(DEF_LATFIELD_OPENACC)
DEF_CPU                   := $(DEF_LATFIELD_CPU)
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
	$(COMPILER) $< $(INC) $(DEF_CPU) $(LIB) $(OPT_CPU) $(CFLAG) -o $@

%_openacc: %.cpp $(HEADER) makefile
	$(COMPILER) $< $(INC) $(DEF_OPENACC) $(LIB) $(OPT_OPENACC) $(CFLAG) -o $@

clean:
	rm -f $(EXEC_CPU) $(EXEC_OPENACC) *.o *~
