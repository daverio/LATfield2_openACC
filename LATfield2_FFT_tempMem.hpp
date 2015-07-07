#ifndef LATFIELD2_FFT_TEMPMEM_HPP
#define LATFIELD2_FFT_TEMPMEM_HPP

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
		
#ifdef SINGLE
	    fftwf_complex* temp1(){return temp1_;}
	    fftwf_complex* temp2(){return temp2_;}
#endif
		
#ifndef SINGLE
	    fftw_complex * temp1(){return temp1_;}
	    fftw_complex * temp2(){return temp2_;}
#endif
		
	private:
#ifdef SINGLE
	    fftwf_complex * temp1_;
	    fftwf_complex * temp2_;
#endif
		
#ifndef SINGLE
	    fftw_complex * temp1_;
	    fftw_complex * temp2_;
#endif
	    long allocated_; //number of variable stored (bit = allocated*sizeof(fftw(f)_complex))
	};
temporaryMemFFT::temporaryMemFFT()
{
	allocated_=0;
}
temporaryMemFFT::~temporaryMemFFT()
{
    if(allocated_!=0){
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
}

temporaryMemFFT::temporaryMemFFT(long size)
{

#ifndef FFT3D_ACC
#ifdef SINGLE
	temp1_ = (fftwf_complex *)fftwf_malloc(2*size*sizeof(fftwf_complex));
	temp2_ = (fftwf_complex *)fftwf_malloc(2*size*sizeof(fftwf_complex));
#endif
	
#ifndef SINGLE
	temp1_ = (fftw_complex *)fftw_malloc(2*size*sizeof(fftw_complex));
	temp2_ = (fftw_complex *)fftw_malloc(2*size*sizeof(fftw_complex));
#endif
#else
#ifdef SINGLE
        temp1_ = (fftwf_complex *)malloc(2*size*sizeof(fftwf_complex));
        temp2_ = (fftwf_complex *)malloc(2*size*sizeof(fftwf_complex));
#endif

#ifndef SINGLE
        temp1_ = (fftw_complex *)malloc(2*size*sizeof(fftw_complex));
        temp2_ = (fftw_complex *)malloc(2*size*sizeof(fftw_complex));
#endif
#endif


	allocated_=size;
}
int temporaryMemFFT::setTemp(long size)
{
	if(size>allocated_)
	{

	    if(allocated_!=0){
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
	

#ifndef FFT3D_ACC
#ifdef SINGLE
	    temp1_ = (fftwf_complex *)fftwf_malloc(2*size*sizeof(fftwf_complex));
	    temp2_ = (fftwf_complex *)fftwf_malloc(2*size*sizeof(fftwf_complex));
#endif  

#ifndef SINGLE
	    temp1_ = (fftw_complex *)fftw_malloc(2*size*sizeof(fftw_complex));
	    temp2_ = (fftw_complex *)fftw_malloc(2*size*sizeof(fftw_complex));
#endif
#else
#ifdef SINGLE
	    temp1_ = (fftwf_complex *)malloc(2*size*sizeof(fftwf_complex));
	    temp2_ = (fftwf_complex *)malloc(2*size*sizeof(fftwf_complex));
#endif

#ifndef SINGLE
	    temp1_ = (fftw_complex *)malloc(2*size*sizeof(fftw_complex));
	    temp2_ = (fftw_complex *)malloc(2*size*sizeof(fftw_complex));
#endif
#endif







	}
	return 1;
}


//////////////////////Temp memory///////////////////////////



#endif
