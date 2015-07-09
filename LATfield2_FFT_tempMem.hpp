#ifndef LATFIELD2_FFT_TEMPMEM_HPP
#define LATFIELD2_FFT_TEMPMEM_HPP



//alocSize * 2 because of crapy cray computer



/*! \file LATfield2_FFT_tempMem.hpp
 \brief FFTs temporary memory
 
  
 */ 



#ifdef SINGLE
#define MPI_DATA_PREC MPI_FLOAT
#endif

#ifndef SINGLE
#define MPI_DATA_PREC MPI_DOUBLE
#endif


const int FFT_FORWARD = 1;
const int FFT_BACKWARD = -1;
const int FFT_IN_PLACE = 16;
const int FFT_OUT_OF_PLACE = -16;


/*! \class temporaryMemFFT
 \brief A class wich handle the additional memory needed by the class PlanFFT_CPU; No documentation!
 */
class temporaryMemFFT
	{
	public:
	    temporaryMemFFT();
	    ~temporaryMemFFT();
	    temporaryMemFFT(long size);
	    
	    int setTemp(long size);
	    int setTempReal(long size);
#ifdef SINGLE
	    fftwf_complex* temp1(){return temp1_;}
	    fftwf_complex* temp2(){return temp2_;}
	    float * tempReal(){return tempReal_;}
#endif
		
#ifndef SINGLE
	    fftw_complex * temp1(){return temp1_;}
	    fftw_complex * temp2(){return temp2_;}
	    double * tempReal(){return tempReal_;}
#endif
		
	private:
#ifdef SINGLE
	    fftwf_complex * temp1_;
	    fftwf_complex * temp2_;
	    float * tempReal_;
#endif
		
#ifndef SINGLE
	    fftw_complex * temp1_;
	    fftw_complex * temp2_;
	    double * tempReal_;
#endif
	    long allocated_; //number of variable stored (bit = allocated*sizeof(fftw(f)_complex))
	    long allocatedReal_;
};
temporaryMemFFT::temporaryMemFFT()
{
	allocated_=0;
#pragma acc enter data create(this)

}
temporaryMemFFT::~temporaryMemFFT()
{
    if(allocated_!=0){

#pragma acc exit data delete(temp1_)
#pragma acc exit data delete(temp2_)

#ifndef FFT3D_ACC
#ifdef SINGLE
	fftwf_free(temp1_);
	fftwf_free(temp2_);	
#endif
#ifndef SINGLE
	fftw_free(temp1_);
	fftw_free(temp2_);
#endif
#else
	free(temp1_);
	free(temp2_);
#endif
	}


    if(allocatedReal_!=0){
#pragma acc exit data delete(tempReal_)
	free(tempReal_);
    }


#pragma acc exit data delete(this)

}

temporaryMemFFT::temporaryMemFFT(long size)
{
    long alocSize;
    alocSize = 2*size;

#ifndef FFT3D_ACC
#ifdef SINGLE
	temp1_ = (fftwf_complex *)fftwf_malloc(alocSize*sizeof(fftwf_complex));
	temp2_ = (fftwf_complex *)fftwf_malloc(alocSize*sizeof(fftwf_complex));


#endif
	
#ifndef SINGLE
	temp1_ = (fftw_complex *)fftw_malloc(alocSize*sizeof(fftw_complex));
	temp2_ = (fftw_complex *)fftw_malloc(alocSize*sizeof(fftw_complex));
#endif
#else
#ifdef SINGLE
        temp1_ = (fftwf_complex *)malloc(alocSize*sizeof(fftwf_complex));
        temp2_ = (fftwf_complex *)malloc(alocSize*sizeof(fftwf_complex));
#endif

#ifndef SINGLE
        temp1_ = (fftw_complex *)malloc(alocSize*sizeof(fftw_complex));
        temp2_ = (fftw_complex *)malloc(alocSize*sizeof(fftw_complex));
#endif
#endif
	allocated_=size;
#pragma acc enter data copyin(this)
#pragma acc enter data create(temp1_[0:alocSize][0:2])
#pragma acc enter data create(temp2_[0:alocSize][0:2])
}


int temporaryMemFFT::setTempReal(long size)
{
    if(size>allocatedReal_)
    {
	if(allocatedReal_!=0)
	{
#pragma acc exit data delete(tempReal_)

	    free(tempReal_);
	}

	long alocSize;
	alocSize = 2*size;

#ifdef SINGLE
	tempReal_ = (float*)malloc(alocSize * sizeof(float));
#else
	tempReal_ = (double*)malloc(alocSize * sizeof(double));
#endif

#pragma acc enter data create(tempReal_[0:alocSize])

    }
}
int temporaryMemFFT::setTemp(long size)
{
	if(size>allocated_)
	{

	    if(allocated_!=0){

#pragma acc exit data delete(temp1_)
#pragma acc exit data delete(temp2_)

#ifndef FFT3D_ACC
#ifdef SINGLE
		fftwf_free(temp1_);
		fftwf_free(temp2_);
#endif
#ifndef SINGLE
		fftw_free(temp1_);
		fftw_free(temp2_);
#endif
#else
		free(temp1_);
		free(temp2_);
#endif
	    }

	    long alocSize;
	    alocSize = 2*size;

#ifndef FFT3D_ACC
#ifdef SINGLE
	    temp1_ = (fftwf_complex *)fftwf_malloc(alocSize*sizeof(fftwf_complex));
	    temp2_ = (fftwf_complex *)fftwf_malloc(alocSize*sizeof(fftwf_complex));
#endif  

#ifndef SINGLE
	    temp1_ = (fftw_complex *)fftw_malloc(alocSize*sizeof(fftw_complex));
	    temp2_ = (fftw_complex *)fftw_malloc(alocSize*sizeof(fftw_complex));
#endif
#else
#ifdef SINGLE
	    temp1_ = (fftwf_complex *)malloc(alocSize*sizeof(fftwf_complex));
	    temp2_ = (fftwf_complex *)malloc(alocSize*sizeof(fftwf_complex));
#endif

#ifndef SINGLE
	    temp1_ = (fftw_complex *)malloc(alocSize*sizeof(fftw_complex));
	    temp2_ = (fftw_complex *)malloc(alocSize*sizeof(fftw_complex));
#endif
#endif
/// maybe add temp2_ if segfault
#pragma acc enter data create(temp1_[0:alocSize][0:2])
#pragma acc enter data create(temp2_[0:alocSize][0:2])
	}
	return 1;
}


//////////////////////Temp memory///////////////////////////



#endif
