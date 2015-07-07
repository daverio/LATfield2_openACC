EXEC                      := poisson fft_test
HEADER                    := $(wildcard ../*.hpp) $(wildcard LATfield2d/*.h)
COMPILER                  := CC
ifdef DEBUG
OPT                       :=
CFLAG                     := -g -std=c++11 -w -Mbounds -traceback
else
OPT                       := -O3
#OPT                      := -O3 -acc -Minfo=accel -ta=tesla:cc35 -w
CFLAG                     := -std=c++11 -w
endif
INC                       := -I./../
LIB                       := -lfftw3 -lm
DEF_LATFIEILD             := -DFFT3D
DEF                       := $(DEF_LATFIEILD)
TEST_WITH_EVERY_BUILD     := true

ifeq ($(TEST_WITH_EVERY_BUILD),true)
ADDITIONAL_BUILD_TARGETS=tests
else
ADDITIONAL_BUILD_TARGETS
endif

.PHONY: all tests clean

#mpic++ main.cpp -I./LATfield2d -I../local/include/gsl/  -DFFT3D -DPHINONLINEAR -DCHECK_B -lfftw3 -lm -lhdf5 -lgsl -lgslcblas -std\
# =c++0x -O3  -o test

all: ${EXEC} ${ADDITIONAL_BUILD_TARGETS}

tests: ${EXEC}
	./RunTest.sh

fft_test: fft.cpp $(HEADER) makefile
	$(COMPILER) fft.cpp $(INC) $(DEF) $(LIB) $(OPT) $(CFLAG) -o fft_test

poisson: poissonSolver.cpp $(HEADER) makefile
	$(COMPILER) poissonSolver.cpp $(INC) $(DEF) $(LIB) $(OPT) $(CFLAG) -o poisson

clean:
	rm -f $(EXEC) *.o *~
