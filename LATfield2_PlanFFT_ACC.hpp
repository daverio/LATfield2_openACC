
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
    
    //initialization of fft planer....
       
    ////
	
	
		
	
}

#endif

template<class compType>
void PlanFFT_ACC<compType>::execute(int fft_type)
{
	
    //#ifdef SINGLE
	if(type_ == R2C)
	{
		if(fft_type == FFT_FORWARD)
		{
						
			
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


#endif
