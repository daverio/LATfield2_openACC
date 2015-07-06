#ifndef LATFIELD2_HPP
#define LATFIELD2_HPP

/*! \file LATfield2.hpp
 \brief Library header
 
 
 */ 

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <typeinfo>



//to insure old code works without any changes, we keep the class PlanFFT, set at compilation to be the one on cpu or the one on gpu. 
// only on cpu: flag FFT3D: PlanFFT = PlanFFT_CPU. PlanFFT_ACC is not compiled.
// only on acc: flag FFT3D_ACC: PlanFFT = PlanFFT_ACC. PlanFFT_CPU is not compiled.
// both flag: PlanFFT = PlanFFT_CPU and PlanFFT_ACC is compiled.
// flag acc type: default GPU, ACC_MIC for mic but not to develop now.


#ifdef FFT3D 
#include "fftw3.h"
#define PlanFFT PlanFFT_CPU 
#define WITH_FFTS
#endif



#ifdef FFT3D_ACC
#include "fftw3.h"

#ifdef ACC_MIC
//mic fft library..


#else
//cufft lib...


#endif


#ifndef FFT3D 
#define PlanFFT PlanFFT_ACC
#define WITH_FFTS
#endif
#endif

using namespace std;


#ifdef EXTERNAL_IO
#include "LATfield2_IO_server.hpp"
IOserver IO_Server;
#endif

#include "LATfield2_parallel2d.hpp"
Parallel2d parallel;




//IOserver Io_Server;
#include "LATfield2_SettingsFile.hpp"

namespace LATfield2
{
	#include "Imag.hpp"

	#include "LATfield2_Lattice.hpp"
	#include "LATfield2_Site.hpp"
	#include "LATfield2_Field.hpp"
    
#ifdef WITH_FFTS
#include "LATfield2_FFT_tempMem.hpp"
    
    temporaryMemFFT tempMemory;
#endif
	#ifdef FFT3D
	#include "LATfield2_PlanFFT_CPU.hpp"
	#endif
    #ifdef FFT3D_ACC
    #include "LATfield2_PlanFFT_ACC_test.hpp"
    #endif
}

#endif


