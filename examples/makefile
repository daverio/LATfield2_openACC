EXEC         := fft_test
SOURCE       := fft.cpp
HEADER       :=  $(wildcard ../*.hpp) $(wildcard LATfield2d/*.h)
COMPILER     := CC
OPT          := -std=c++11  -O3 -acc -Minfo=accel -ta=tesla:cc35 -w
INC          := -I./../
LIB          := -lfftw3 -lm
DEF_LATFIEILD:= -DFFT3D
DEF          := $(DEF_LATFIEILD) 

#mpic++ main.cpp -I./LATfield2d -I../local/include/gsl/  -DFFT3D -DPHINONLINEAR -DCHECK_B -lfftw3 -lm -lhdf5 -lgsl -lgslcblas -std\
=c++0x -O3  -o test                                                                                                                

$(EXEC): $(SOURCE) $(HEADER) makefile
	$(COMPILER) $(SOURCE) $(INC) $(DEF) $(LIB) $(OPT) -o $(EXEC)
