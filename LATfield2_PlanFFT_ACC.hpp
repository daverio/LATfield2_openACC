#ifndef LATFIELD2_PlanFFT_ACC_HPP
#define LATFIELD2_PlanFFT_ACC_HPP

/*! \file LATfield2_PlanFFT_ACC.hpp
 \brief FFT wrapper
 
 LATfield2_PlanFFT_ACC.hpp contain the class PlanFFT_ACC definition.
 
 */ 







/*! \class PlanFFT_ACC  
 
 
 */
template<class compType>
class PlanFFT_ACC
	{
	public:
        //! Constructor.
		PlanFFT_ACC();
		
	    ~PlanFFT_ACC();
#ifndef SINGLE
		
        /*!
         Constructor with initialization for complex to complex tranform.
         \sa initialize(Field<compType>*  rfield,Field<compType>*   kfield,const int mem_type = FFT_OUT_OF_PLACE);
         \param rfield : real space field
         \param kfield : fourier space field
         \param mem_type : memory type (FFT_OUT_OF_PLACE or FFT_IN_PLACE). In place mean that both fourier and real space field point to the same data array.
            
         */ 
		PlanFFT_ACC(Field<compType>* rfield, Field<compType>*  kfield, const int mem_type = FFT_OUT_OF_PLACE);
		/*!
         initialization for complex to complex tranform.
         For more detail see the QuickStart guide.
         \param rfield : real space field
         \param kfield : fourier space field
         \param mem_type : memory type (FFT_OUT_OF_PLACE or FFT_IN_PLACE). In place mean that both fourier and real space field point to the same data array.
         */
        void initialize(Field<compType>*  rfield, Field<compType>*   kfield, const int mem_type = FFT_OUT_OF_PLACE);
		
		/*!
         Constructor with initialization for real to complex tranform.
         \sa initialize(Field<compType>*  rfield,Field<compType>*   kfield,const int mem_type = FFT_OUT_OF_PLACE);
         \param rfield : real space field
         \param kfield : fourier space field
         \param mem_type : memory type (FFT_OUT_OF_PLACE or FFT_IN_PLACE). In place mean that both fourier and real space field point to the same data array.
         */
		PlanFFT_ACC(Field<double>* rfield, Field<compType>*  kfield,const int mem_type = FFT_OUT_OF_PLACE);
        /*!
         initialization for real to complex tranform.
         For more detail see the QuickStart guide.
         \param rfield : real space field
         \param kfield : fourier space field
         \param mem_type : memory type (FFT_OUT_OF_PLACE or FFT_IN_PLACE). In place mean that both fourier and real space field point to the same data array.
         */
		void initialize(Field<double>*  rfield,Field<compType>*   kfield,const int mem_type = FFT_OUT_OF_PLACE);
        
        /*!
         Execute the fourier transform.
         \param fft_type: dirrection of the transform. Can be FFT_BACKWARD or FFT_FORWARD.
         */
		void execute(int fft_type);

	    void execute_r2c_forward_h2d_send(int comp);
	    void execute_r2c_forward_d2h_send(int comp);
	    void execute_r2c_forward_dim0();


#endif		
        
    private:
        bool status_;
		bool type_;
        int mem_type_;
        
        static bool R2C;
		static bool C2C;
		static bool initialized;
        
        
        int components_;
		int rSize_[3];
		int kSize_[3];
		int rJump_[3];
		int kJump_[3];
		int rSizeLocal_[3];
		int kSizeLocal_[3];
		int rHalo_;
		int kHalo_;

	    long rSitesLocal_;
	    long kSitesLocal_;
        
#ifndef SINGLE
	    double * temp_r2c_real_;
	    fftw_complex * temp_r2c_complex_;

	    double * rData_;
	    fftw_complex * kData_;


	    cufftHandle cufPlan_real_i_;
#endif
	    

};

//constants
template<class compType>
bool PlanFFT_ACC<compType>::initialized = true;
template<class compType>
bool PlanFFT_ACC<compType>::R2C=false;
template<class compType>
bool PlanFFT_ACC<compType>::C2C=true;




template<class compType>
PlanFFT_ACC<compType>::PlanFFT_ACC()
{
	status_ = false;
}
template<class compType>
PlanFFT_ACC<compType>::~PlanFFT_ACC()
{
    free(temp_r2c_real_);
    free(temp_r2c_complex_);

    cufftDestroy(cufPlan_real_i_);

}


#ifndef SINGLE

template<class compType>
PlanFFT_ACC<compType>::PlanFFT_ACC(Field<compType>*  rfield,Field<compType>* kfield,const int mem_type)
{
	status_ = false;
	initialize(rfield,kfield,mem_type);
}

template<class compType>
void PlanFFT_ACC<compType>::initialize(Field<compType>*  rfield,Field<compType>*  kfield,const int mem_type )
{
	type_ = C2C;
	mem_type_=mem_type;
    
	//general variable
	if(rfield->components() != kfield->components())
	{
		cerr<<"Latfield2d::PlanFFT_ACC::initialize : fft curently work only for fields with same number of components"<<endl;
		cerr<<"Latfield2d::PlanFFT_ACC::initialize : coordinate and fourier space fields have not the same number of components"<<endl;
		cerr<<"Latfield2d : Abort Process Requested"<<endl;
		
	}
	else components_ = rfield->components();
	
	for(int i = 0; i<3; i++)
	{
		rSize_[i]=rfield->lattice().size(i);
		kSize_[i]=kfield->lattice().size(i);
		rSizeLocal_[i]=rfield->lattice().sizeLocal(i);
		kSizeLocal_[i]=kfield->lattice().sizeLocal(i);
		rJump_[i]=rfield->lattice().jump(i);
		kJump_[i]=kfield->lattice().jump(i);
	}
	
	rHalo_ = rfield->lattice().halo();
	kHalo_ = kfield->lattice().halo();
    
    
	if(rfield->lattice().dim()!=3)
	{
		if(parallel.isRoot())
		{
			cerr<<"Latfield2d::PlanFFT_ACC::initialize : fft curently work only for 3d cubic lattice"<<endl;
			cerr<<"Latfield2d::PlanFFT_ACC::initialize : real lattice have not 3 dimensions"<<endl;
			cerr<<"Latfield2d : Abort Process Requested"<<endl;
			
		}
		parallel.abortForce();
	}
	
	
	if(rSize_[0]!=rSize_[1] | rSize_[1]!=rSize_[2])
	{
		if(parallel.isRoot())
		{
			cerr<<"Latfield2d::PlanFFT_ACC::initialize : fft curently work only for 3d cubic lattice"<<endl;
			cerr<<"Latfield2d::PlanFFT_ACC::initialize : real lattice is not cubic"<<endl;
			cerr<<"Latfield2d : Abort Process Requested"<<endl;
			
 		}
		parallel.abortForce();
	}
    
    long rfield_size = rfield->lattice().sitesLocalGross();
	long kfield_size = kfield->lattice().sitesLocalGross(); 
	
    if(mem_type_ == FFT_IN_PLACE)
	{
		if(rfield_size>=kfield_size)
		{		    
			rfield->alloc();
			kfield->data() = (Imag*)rfield->data();
		}
		else
		{
			kfield->alloc();
			rfield->data() = (Imag*)kfield->data();
		}
	}
	if(mem_type_ == FFT_OUT_OF_PLACE)
	{
		rfield->alloc();
		kfield->alloc();
	}

	

//	temp_r2c_real_ = (double*)malloc(rfield->lattice().sitesLocal() * sizeof(double));

    
    //initialization of fft planer....
    
    ////
  	
}



template<class compType>
PlanFFT_ACC<compType>::PlanFFT_ACC(Field<double>* rfield, Field<compType>*  kfield,const int mem_type )
{
	status_ = false;
	initialize(rfield,kfield,mem_type);
}



template<class compType>
void PlanFFT_ACC<compType>::initialize(Field<double>*  rfield,Field<compType>*   kfield,const int mem_type )
{
	type_ = R2C;
	mem_type_=mem_type;
	
	//general variable
	

	components_ = rfield->components();
	
	for(int i = 0; i<3; i++)
	{
		rSize_[i]=rfield->lattice().size(i);
		kSize_[i]=kfield->lattice().size(i);
		rSizeLocal_[i]=rfield->lattice().sizeLocal(i);
		kSizeLocal_[i]=kfield->lattice().sizeLocal(i);
		rJump_[i]=rfield->lattice().jump(i);
		kJump_[i]=kfield->lattice().jump(i);
	}
	rHalo_ = rfield->lattice().halo();
	kHalo_ = kfield->lattice().halo();

	rSitesLocal_ = rfield->lattice().sitesLocal();
	kSitesLocal_ = kfield->lattice().sitesLocal();

	if(rfield->components() != kfield->components())
	{
        if(parallel.isRoot())
		{
            cerr<<"Latfield2d::PlanFFT_ACC::initialize : fft curently work only for fields with same number of components"<<endl;
            cerr<<"Latfield2d::PlanFFT_ACC::initialize : coordinate and fourier space fields have not the same number of components"<<endl;
            cerr<<"Latfield2d : Abort Process Requested"<<endl;
        
            parallel.abortForce();
		}
	}
    if(rfield->lattice().dim()!=3)
	{
		if(parallel.isRoot())
		{
			cerr<<"Latfield2d::PlanFFT_ACC::initialize : fft curently work only for 3d cubic lattice"<<endl;
			cerr<<"Latfield2d::PlanFFT_ACC::initialize : real lattice have not 3 dimensions"<<endl;
			cerr<<"Latfield2d : Abort Process Requested"<<endl;
			
		}
		parallel.abortForce();
	}
    if(rSize_[0]!=rSize_[1] | rSize_[1]!=rSize_[2])
    {
        if(parallel.isRoot())
        {
            cerr<<"Latfield2d::PlanFFT_ACC::initialize : fft curently work only for 3d cubic lattice"<<endl;
            cerr<<"Latfield2d::PlanFFT_ACC::initialize : real lattice is not cubic"<<endl;
            cerr<<"Latfield2d : Abort Process Requested"<<endl;
            
        }
        parallel.abortForce();
    }
    
	
	

    long rfield_size = rfield->lattice().sitesLocalGross();
	long kfield_size = kfield->lattice().sitesLocalGross();
    
    if(mem_type_ == FFT_IN_PLACE)
	{
		if(rfield_size>=kfield_size)
		{		    
			rfield->alloc();
			kfield->data() = (Imag*)rfield->data();
		}
		else
		{
			kfield->alloc();
			rfield->data() = (double*)kfield->data();
		}
	}
	if(mem_type_ == FFT_OUT_OF_PLACE)
	{
		rfield->alloc();
		kfield->alloc();
	}


	//temp_r2c_real_ = (double*)malloc(rfield->lattice().sitesLocal() * sizeof(double));
	//temp_r2c_complex_ = (Imag*)malloc(kfield->lattice().sitesLocal() * sizeof(Imag));
	tempMemory.setTempReal(rfield->lattice().sitesLocal());
	temp_r2c_real_ = tempMemory.tempReal();
	tempMemory.setTemp(kfield->lattice().sitesLocal());
	temp_r2c_complex_ = tempMemory.temp1();

//initialization of fft planer....
       
    ////
	
	rData_ = rfield->data(); 
	kData_ = (fftw_complex*)kfield->data();
	//cufftHandle cufPlan_i_;
	//cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
	//		  int istride, int idist, int *onembed, int ostride,
//			  int odist, cufftType type, int batch, size_t *workSize);
	
//        cufftMakePlanMany(cufPlan_i_,1, &rSize[0],  &rSize[0],
//			  1, 1, int *onembed, int ostride,
//			  int odist, cufftType type, int batch, size_t *workSize);

	if (cufftPlan1d(&cufPlan_real_i_, rSize_[0], CUFFT_R2C, rSizeLocal_[1]*rSizeLocal_[2]) != CUFFT_SUCCESS){
	    fprintf(stderr, "CUFFT error: Plan r2c forward i,  creation failed");
	    MPI_Abort(MPI_COMM_WORLD,11);
	}

}

#endif

template<class compType>
void PlanFFT_ACC<compType>::execute(int fft_type)
{
    temp_r2c_real_ = tempMemory.tempReal();
    temp_r2c_complex_ = tempMemory.temp1();

    int comp;


    //#ifdef SINGLE
	if(type_ == R2C)
	{
		if(fft_type == FFT_FORWARD)
		{
		    for(comp=0;comp<components_;comp++)
		    {
			this->execute_r2c_forward_h2d_send(comp);
			this->execute_r2c_forward_dim0();
			this->execute_r2c_forward_d2h_send(comp);
		    }
		}
		if(fft_type == FFT_BACKWARD)
		{
            
            
		}
	}
	if(type_ == C2C)
	{
        if(fft_type == FFT_FORWARD)
		{
            
			
		}
		if(fft_type == FFT_BACKWARD)
		{
           
            
		}
	}
}


template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_h2d_send(int comp)
{
    
    temp_r2c_real_ = tempMemory.tempReal();
    temp_r2c_complex_ = tempMemory.temp1();


    Lattice lat(3,rSize_,rHalo_);
    Site x(lat);
    long i=0;

    for(x.first();x.test();x.next(),i++)
    {
	temp_r2c_real_[i]=rData_[x.index()*components_ + comp] ;
    }


#pragma acc update device(temp_r2c_real_[0:rSitesLocal_])
  

    
}
template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_d2h_send(int comp)
{

    Lattice lat(3,kSize_,kHalo_);
    rKSite k(lat);
    long i;


#pragma acc update host(temp_r2c_complex_[0:kSitesLocal_][0:2])    

    for(k.first();k.test();k.next(),i++)
    {
        kData_[k.index()*components_ + comp][0] = temp_r2c_complex_[i][0] ;
        kData_[k.index()*components_ + comp][1] = temp_r2c_complex_[i][1] ;

    }


}


template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim0()
{

    double * p_in = temp_r2c_real_;
    fftw_complex * p_out = temp_r2c_complex_;




#pragma acc data present(p_in[0:rSitesLocal_],p_out[0:kSitesLocal_][0:2])
    {
        #pragma acc host_data use_device(p_in,p_out)
        {
            cufftExecD2Z(cufPlan_real_i_,(cufftDoubleReal *)p_in,(cufftDoubleComplex *)temp_r2c_complex_);
        }
    }

}
#endif
