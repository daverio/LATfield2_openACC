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


//forward function, nice logical way...
	    void execute_r2c_forward_h2d_send(int comp);

	    void execute_r2c_forward_dim0_fft();
	    void execute_r2c_forward_dim0_a2a();
	    void execute_r2c_forward_dim0_tsp();

	    void execute_r2c_forward_dim1_fft();
            void execute_r2c_forward_dim1_a2a();
            void execute_r2c_forward_dim1_tsp();

	    void execute_r2c_forward_dim2_fft();
            void execute_r2c_forward_dim2_a2a();
            void execute_r2c_forward_dim2_tsp(int comp);

	    void execute_r2c_forward_d2h_send(int comp);


//backward function, a nice ilogical way... going backward for every call... !!! maybe a2a and tsp should be glued in only one method... this will be more intuitive.
// function name are completely crapy ... 0 for first meaning k, 1 second meaning j, 2 third meaning i

	    // does prepare the data for the first alltoall
	    void execute_r2c_backward_h2d_send(int comp);


            void execute_r2c_backward_dim0_a2a();
            void execute_r2c_backward_dim0_tsp();
	    void execute_r2c_backward_dim0_fft();

	    void execute_r2c_backward_dim1_a2a();
            void execute_r2c_backward_dim1_tsp();
            void execute_r2c_backward_dim1_fft();

	    void execute_r2c_backward_dim2_a2a();
            void execute_r2c_backward_dim2_tsp();
            void execute_r2c_backward_dim2_fft();

	    void execute_r2c_backward_d2h_send(int comp);

#endif

	double timing;

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

	    int r2cSize_;  //of 0 dimension
	    int r2cSizeLocal_; //of r2csize on proc_dim[1]
	    int r2cSizeLocal_as_;

	    long rSitesLocal_;
	    long kSitesLocal_;

#ifndef SINGLE
	    double * temp_r2c_real_;
	    fftw_complex * temp_r2c_complex_;
	    fftw_complex * temp2_r2c_complex_;



	    double * rData_;
	    fftw_complex * kData_;

	    cufftHandle fPlan_i_;
	    cufftHandle fPlan_j_;
	    cufftHandle fPlan_k_;
	    cufftHandle fPlan_k_real_;

	    cufftHandle bPlan_i_;
	    cufftHandle bPlan_j_;
	    cufftHandle bPlan_j_real_;
	    cufftHandle bPlan_k_;


	    void transpose_0_2( fftw_complex * in, fftw_complex * out,int dim_i,int dim_j ,int dim_k);
	    void transpose_0_2_last_proc( fftw_complex * in, fftw_complex * out,int dim_i,int dim_j ,int dim_k);
	    void implement_local_0_last_proc( fftw_complex * in, fftw_complex * out,int proc_dim_i,int proc_dim_j,int proc_dim_k,int proc_size);
	    // second transposition
	    void transpose_1_2(fftw_complex * in , fftw_complex * out  ,int dim_i,int dim_j ,int dim_k);
	    //third transposition
	    void transpose_back_0_3(fftw_complex * in, fftw_complex * out,int r2c,int local_r2c,
				    int local_size_j,int local_size_k,int proc_size,int halo,int components, int comp);
	    void implement_0(fftw_complex * in, fftw_complex * out,int r2c_size,int local_size_j,int local_size_k,int halo,int components,int comp);
	    ////backward real to complex
	    void b_arrange_data_0(fftw_complex *in, fftw_complex * out,int dim_i,int dim_j ,int dim_k, int khalo, int components, int comp);
	    void b_transpose_back_0_1(fftw_complex * in, fftw_complex * out,int r2c,int local_r2c,int local_size_j,int local_size_k,int proc_size);
	    void b_implement_0(fftw_complex * in, fftw_complex * out,int r2c_size,int local_size_j,int local_size_k);
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

    cufftDestroy(fPlan_i_);

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

	//Calculate index jumps
	for(int i = 0; i<3; i++)
	{
		rSize_[i]=rfield->lattice().size(i);
		kSize_[i]=kfield->lattice().size(i);
		rSizeLocal_[i]=rfield->lattice().sizeLocal(i);
		kSizeLocal_[i]=kfield->lattice().sizeLocal(i);
		// rJump_[i]=rfield->lattice().jump(i);



		kJump_[i]=kfield->lattice().jump(i);
	}
	rJump_[0]=1;
	for(int i=1;i<3;i++)
	{
		rJump_[i]=rJump_[i-1]*(rSizeLocal_[i-1]);
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
		// rJump_[i]=rfield->lattice().jump(i);
		kJump_[i]=kfield->lattice().jump(i);
	}
	rJump_[0]=1;
	for(int i=1;i<3;i++)
	{
		rJump_[i]=rJump_[i-1]*(rSizeLocal_[i-1]);
	}

	r2cSize_ = rfield->lattice().size(0)/2 + 1;
	r2cSizeLocal_as_ = rfield->lattice().sizeLocal(0)/(2*parallel.grid_size()[1]);
	if(parallel.last_proc()[1]) r2cSizeLocal_ = r2cSizeLocal_as_ + 1;
	else r2cSizeLocal_ = r2cSizeLocal_as_;



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
	temp2_r2c_complex_= tempMemory.temp2();
//initialization of fft planer....

    ////

	rData_ = rfield->data();
	kData_ = (fftw_complex*)kfield->data();

/*
	if (cufftPlan1d(&fPlan_i_, rSize_[0], CUFFT_D2Z, rSizeLocal_[1]*rSizeLocal_[2]) != CUFFT_SUCCESS){
	    fprintf(stderr, "CUFFT error: Plan r2c forward i,  creation failed");
	    MPI_Abort(MPI_COMM_WORLD,11);
	}
*/


	int istride,idist,ostride,odist;
	//size_t workSize=rSize_[0];

	int a=rSizeLocal_[1]*rSizeLocal_[2];
	int b = rSizeLocal_[1];

	//forward

	idist = rJump_[1];
	// idist = rSize_[0];
	odist= 1;
	if(cufftPlanMany(&fPlan_i_, 1, rSize_, &idist,
			     1, idist, &odist, rSizeLocal_[1]*rSizeLocal_[2],
			     odist, CUFFT_D2Z, rSizeLocal_[1]) != CUFFT_SUCCESS){
	    fprintf(stderr, "CUFFT error: Plan forward  i,  creation failed");
	    MPI_Abort(MPI_COMM_WORLD,11);
	}

/*
	idist = 1;
	odist= 1;
	int result;
	result=cufftMakePlanMany(fPlan_i_, 1, rSize_, NULL,
			  1, 0, NULL, 1,
			  0, CUFFT_D2Z, 1,  &workSize);
	if(result != CUFFT_SUCCESS){
	    fprintf(stderr, "CUFFT error: Plan forward  i,  creation failed %d",result);
	    MPI_Abort(MPI_COMM_WORLD,11);
	}
*/

	idist = 1;
	odist= 1;
	if (cufftPlanMany(&fPlan_j_, 1, &rSize_[0], &idist,
			      rSizeLocal_[2]*r2cSizeLocal_, idist, &odist, rSizeLocal_[2]*r2cSizeLocal_,
			      odist, CUFFT_Z2Z, rSizeLocal_[2]*r2cSizeLocal_) != CUFFT_SUCCESS){
	    fprintf(stderr, "CUFFT error: Plan forward   j,  creation failed");
	    MPI_Abort(MPI_COMM_WORLD,11);
	}

	idist = 1;
	odist= 1;
	if (cufftPlanMany(&fPlan_k_, 1, &rSize_[0], &idist,
                              rSizeLocal_[2]*r2cSizeLocal_,idist, &odist, rSizeLocal_[2]*r2cSizeLocal_as_,
                              odist, CUFFT_Z2Z, r2cSizeLocal_as_) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan  forward  k,  creation failed");
            MPI_Abort(MPI_COMM_WORLD,11);
        }

	idist = r2cSizeLocal_;
	odist= 1;
	if (cufftPlanMany(&fPlan_k_real_, 1, &rSize_[0], &idist,
                              rSizeLocal_[2]*r2cSizeLocal_, idist, &odist, rSizeLocal_[2],
                              odist, CUFFT_Z2Z, rSizeLocal_[2]) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan forward real k,  creation failed");
            MPI_Abort(MPI_COMM_WORLD,11);
        }

	///backward



	idist = r2cSize_;
        odist= rJump_[1];
	if (cufftPlanMany(&bPlan_i_, 1, &rSize_[0], &idist,
                              1, idist, &odist, 1,
                              odist, CUFFT_Z2D, rSizeLocal_[1]) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan r2c backward i,  creation failed");
            MPI_Abort(MPI_COMM_WORLD,11);
        }

	idist = 1;
        odist= 1;
	if (cufftPlanMany(&bPlan_j_, 1, &rSize_[0], &idist,
                              rSizeLocal_[2]*r2cSizeLocal_, idist, &odist, rSizeLocal_[2]*r2cSizeLocal_as_,
                              odist, CUFFT_Z2Z, r2cSizeLocal_as_) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan r2c backward i,  creation failed");
            MPI_Abort(MPI_COMM_WORLD,11);
        }

	idist = r2cSizeLocal_;
        odist= 1;
	if (cufftPlanMany(&bPlan_j_real_, 1, &rSize_[0], &idist,
                              rSizeLocal_[2]*r2cSizeLocal_, idist, &odist, rSizeLocal_[2],
                              odist, CUFFT_Z2Z, rSizeLocal_[2]) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan r2c backward i,  creation failed");
            MPI_Abort(MPI_COMM_WORLD,11);
        }

	idist = 1;
        odist= 1;
	if (cufftPlanMany(&bPlan_k_, 1, &rSize_[0], &idist,
                              kSizeLocal_[2]*r2cSizeLocal_, idist, &odist, kSizeLocal_[2]*r2cSizeLocal_,
                              odist, CUFFT_Z2Z, kSizeLocal_[2]*r2cSizeLocal_) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan r2c backward i,  creation failed");
            MPI_Abort(MPI_COMM_WORLD,11);
        }


        void * cuStream;
	cuStream = acc_get_cuda_stream(acc_async_sync);
        cufftSetStream(fPlan_i_,(cudaStream_t) cuStream);
	 cufftSetStream(fPlan_j_,(cudaStream_t) cuStream);
        cufftSetStream(fPlan_k_,(cudaStream_t) cuStream);
        cufftSetStream(fPlan_k_real_,(cudaStream_t) cuStream);
        cufftSetStream(bPlan_i_,(cudaStream_t) cuStream);
        cufftSetStream(bPlan_j_,(cudaStream_t) cuStream);
        cufftSetStream(bPlan_j_real_,(cudaStream_t) cuStream);
        cufftSetStream(bPlan_k_,(cudaStream_t) cuStream);

}

#endif

template<class compType>
void PlanFFT_ACC<compType>::execute(int fft_type)
{
    temp_r2c_real_ = tempMemory.tempReal();
    temp_r2c_complex_ = tempMemory.temp1();

    int comp;

    double time_comp;
    timing = 0.;

    //#ifdef SINGLE
	if(type_ == R2C)
	{
		if(fft_type == FFT_FORWARD)
		{
		    for(comp=0;comp<components_;comp++)
		    {
				this->execute_r2c_forward_h2d_send(comp);
				time_comp = MPI_Wtime();
				this->execute_r2c_forward_dim0_fft();

			// /* 2015-7-9: This is only here for debug reasons to see what happens in phase 1 */
			// for(int l = 0; l < kSizeLocal_[0]*kSizeLocal_[1]*kSizeLocal_[2]; l++)
			// {
			// 	kData_[l][0] = temp_r2c_complex_[l][0];
			// 	kData_[l][1] = temp_r2c_complex_[l][1];
			// }

				this->execute_r2c_forward_dim0_a2a(); //this currently also sends the data back to the host
				this->execute_r2c_forward_dim0_tsp();
				this->execute_r2c_forward_dim1_fft();
				this->execute_r2c_forward_dim1_a2a();
            	this->execute_r2c_forward_dim1_tsp();
				this->execute_r2c_forward_dim2_fft();
            	this->execute_r2c_forward_dim2_a2a();
            	this->execute_r2c_forward_dim2_tsp(comp);
				timing += MPI_Wtime() - time_comp;
                        /*
			/// no need to use it as execute_r2c_forward_dim2_tsp on host and already do the work!!!
			this->execute_r2c_forward_d2h_send(comp);
			*/

		    }
		}
		if(fft_type == FFT_BACKWARD)
		{

			for(comp=0;comp<components_;comp++)
		    {

				this->execute_r2c_backward_h2d_send(comp);


            	this->execute_r2c_backward_dim0_a2a();
            	this->execute_r2c_backward_dim0_tsp();
	    		this->execute_r2c_backward_dim0_fft();

	    		this->execute_r2c_backward_dim1_a2a();
            	this->execute_r2c_backward_dim1_tsp();
            	this->execute_r2c_backward_dim1_fft();

	    		this->execute_r2c_backward_dim2_a2a();
            	this->execute_r2c_backward_dim2_tsp();
            	this->execute_r2c_backward_dim2_fft();

	    		this->execute_r2c_backward_d2h_send(comp);
	    	}
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
    temp2_r2c_complex_ = tempMemory.temp2();

    Lattice lat(3,rSize_,rHalo_);
    Site x(lat);
    long i=0;

    for(x.first();x.test();x.next(),i++)
    {
	temp_r2c_real_[i]=rData_[x.index()*components_ + comp] ;
    }


#pragma acc update device(temp_r2c_real_[0:rSitesLocal_])
    /*
#pragma acc update host(temp_r2c_real_[0:rSitesLocal_])



    for(int proc=0;proc<parallel.size();proc++)
    {
	MPI_Barrier(MPI_COMM_WORLD);
	if(proc== parallel.rank())
	{
	    for(int l=0;l<rSitesLocal_;l++)
	    {
		cout<< l%rSize_[0]<<"  "<<temp_r2c_real_[l]<<endl;
	    }
	}
	MPI_Barrier(MPI_COMM_WORLD);
    }
    */


}
template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_d2h_send(int comp)
{

    Lattice lat(3,kSize_,kHalo_);
    rKSite k(lat);
    long i=0;


#pragma acc update host(temp_r2c_complex_[0:kSitesLocal_][0:2])

    for(k.first();k.test();k.next(),i++)
    {
        kData_[k.index()*components_ + comp][0] = temp_r2c_complex_[i][0] ;
        kData_[k.index()*components_ + comp][1] = temp_r2c_complex_[i][1] ;

    }


}


template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim0_fft()
{

    double * p_in = temp_r2c_real_;
    fftw_complex * p_out = temp_r2c_complex_;
/*
#pragma acc update host(p_in[0:rSitesLocal_])

    for(int proc=0;proc<parallel.size();proc++)
    {
	MPI_Barrier(MPI_COMM_WORLD);
	if(proc== parallel.rank())
	{
	    for(int l=0;l<rSitesLocal_;l++)
	    {
		cout<< l%rSize_[0]<<"  "<<p_in[l]<<endl;
	    }
	}
	MPI_Barrier(MPI_COMM_WORLD);
    }

*/
#pragma acc data present(p_in[0:rSitesLocal_])
    {
        #pragma acc host_data use_device(p_in,p_out)
        {
		    for(int l = 0;l< rSizeLocal_[2] ;l++)
		    {
				cufftResult result;
				result = cufftExecD2Z(
					fPlan_i_,
					(cufftDoubleReal *)(p_in + (rJump_[2]*l)),
					(cufftDoubleComplex *)(p_out + (rSizeLocal_[1]*l))
				);
				if (result != CUFFT_SUCCESS) {
					printf("cuFFT result for fPlan_i_unsuccessful: %d\n", (int)result);
					MPI_Abort(MPI_COMM_WORLD,result);
				}
		    }
		}
    }

/*
    for(int proc=0;proc<parallel.size();proc++)
    {
	MPI_Barrier(MPI_COMM_WORLD);
	if(proc== parallel.rank())
	{
	    cout<< "rank: " << proc << endl;
	    for(int l=0;l<rSitesLocal_;l++)
	    {
		cout<< l%rSize_[0]<<"  "<<p_out[l][0]<<" , "<< p_out[l][1] <<endl;
	    }
	}
	MPI_Barrier(MPI_COMM_WORLD);
    }
*/

}
template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim0_a2a()
{
//ok now all proc have same amount of data as it is cubic rSize[1]=rSize[2] and therefor as multiplication commute rSitesLocal_ is safe.
#pragma acc  update host(temp_r2c_complex_[0:rSitesLocal_][0:2])
    MPI_Alltoall(temp_r2c_complex_, 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, temp2_r2c_complex_, 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Gather(&temp_r2c_complex_[(r2cSize_-1)*rSizeLocal_[1]*rSizeLocal_[2]][0], 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC, &temp2_r2c_complex_[rSize_[0]/2*rSizeLocal_[1]*rSizeLocal_[2]][0] , 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC ,parallel.grid_size()[1]-1, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Barrier(parallel.dim1_comm()[parallel.grid_rank()[0]]);
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim0_tsp()
{
// !!!! there after the transpose not anymore same number of site in each core> last one have 1 addition layer, but r2cSizeLocal_ take care of it!!!

#pragma acc update device(temp2_r2c_complex_[0:(r2cSizeLocal_ * rSize_[1]* rSizeLocal_[2]) ][0:2])
#pragma acc data present(temp2_r2c_complex_[0:1], temp_r2c_complex_[0:1])
{
    if(parallel.last_proc()[1])
    {
	for(int i=0;i<parallel.grid_size()[1];i++)transpose_0_2_last_proc(&temp2_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_],&temp_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_],rSizeLocal_[1],rSizeLocal_[2],r2cSizeLocal_as_);
	implement_local_0_last_proc(&temp2_r2c_complex_[rSize_[0]/2*rSizeLocal_[1]*rSizeLocal_[2]],temp_r2c_complex_,rSizeLocal_[1],rSizeLocal_[2],r2cSizeLocal_as_,parallel.grid_size()[1]);
    }
    else for(int i=0;i<parallel.grid_size()[1];i++)transpose_0_2(&temp2_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_],&temp_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_],rSizeLocal_[1],rSizeLocal_[2],r2cSizeLocal_as_);
}
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim1_fft()
{
    fftw_complex * p_in = temp_r2c_complex_;
    fftw_complex * p_out = temp2_r2c_complex_;


#pragma acc data present(p_in[0:(r2cSizeLocal_ * rSize_[1]* rSizeLocal_[2]) ][0:2])
    {
        #pragma acc host_data use_device(p_in,p_out)
        {
	    cufftExecZ2Z(fPlan_j_,(cufftDoubleComplex *)p_in,(cufftDoubleComplex *)p_out,CUFFT_FORWARD);
        }
    }


}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim1_a2a()
{
#pragma acc update host(temp2_r2c_complex_[0:(r2cSizeLocal_ * rSize_[1]* rSizeLocal_[2]) ][0:2])

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Alltoall(temp2_r2c_complex_, (2*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_), MPI_DATA_PREC,
		 temp_r2c_complex_, (2*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_),
		 MPI_DATA_PREC, parallel.dim0_comm()[parallel.grid_rank()[1]]);
    MPI_Barrier(parallel.dim0_comm()[parallel.grid_rank()[1]]);

}
template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim1_tsp()
{

//with have dim2 in 0, dim0 in 1, dim1 in 2: so size = dim2size(rSize_[2]) * r2cSizeLocal_ * dim1size/procsize0(rSizeLocal_[2])
// be extremly carefull!!!! as cubic we have dim1size/procsize0 = rSizeLocal_[2]

	#pragma acc update device(temp_r2c_complex_[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2]) ][0:2])
	#pragma acc data present(temp2_r2c_complex_[0:1], temp_r2c_complex_[0:1])
	{
    for(int i=0;i<parallel.grid_size()[0];i++)transpose_1_2(&temp_r2c_complex_[i*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_],
							&temp2_r2c_complex_[i*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_],
							r2cSizeLocal_,rSizeLocal_[2],rSizeLocal_[2]);
    }


}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim2_fft()
{
    fftw_complex * p_in = temp2_r2c_complex_;
    fftw_complex * p_out = temp_r2c_complex_;

    bool flag_last_proc=false;
    if(parallel.last_proc()[1])flag_last_proc=true;

#pragma acc data present(p_in[0:(r2cSizeLocal_ * rSize_[1]* rSizeLocal_[2]) ][0:2])
    {
#pragma acc host_data use_device(p_in,p_out)
        {
	    for(int l=0;l<rSizeLocal_[2];l++)
	    {
		cufftExecZ2Z(fPlan_k_,(cufftDoubleComplex *)(p_in+(l*r2cSizeLocal_)),(cufftDoubleComplex *)(p_out+(l*r2cSizeLocal_as_)),CUFFT_FORWARD);
	    }
	    if(flag_last_proc)cufftExecZ2Z(fPlan_k_real_,(cufftDoubleComplex *)(p_in+r2cSizeLocal_as_),
					   (cufftDoubleComplex *)(p_out+(r2cSizeLocal_as_*rSizeLocal_[2]*rSize_[0])),CUFFT_FORWARD);
        }
    }
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim2_a2a()
{
#pragma acc update host(temp_r2c_complex_[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2]) ][0:2])

    MPI_Alltoall(temp_r2c_complex_, 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, temp2_r2c_complex_,
		 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Scatter(&temp_r2c_complex_[r2cSizeLocal_as_*rSizeLocal_[2]*rSize_[0]][0], 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC,
		&temp2_r2c_complex_[(r2cSize_-1)*rSizeLocal_[1]*rSizeLocal_[2]][0] , 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC,
		parallel.grid_size()[1]-1, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Barrier(parallel.dim1_comm()[parallel.grid_rank()[0]]);

}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_forward_dim2_tsp(int comp)
{
//final rearengment should be on host or device -- now on host as we transfert data for ffts part only

//host verion
    transpose_back_0_3(temp2_r2c_complex_, kData_,r2cSize_,r2cSizeLocal_as_,rSizeLocal_[2],rSizeLocal_[1],parallel.grid_size()[1],kHalo_,components_,comp);
    implement_0(&temp2_r2c_complex_[(r2cSize_-1)*rSizeLocal_[1]*rSizeLocal_[2]], kData_,r2cSize_,rSizeLocal_[2],rSizeLocal_[1],kHalo_,components_,comp);
/*
//device version:
    transpose_back_0_3(temp2_r2c_complex_, temp_r2c_complex_,r2cSize_,r2cSizeLocal_as_,rSizeLocal_[2],rSizeLocal_[1],parallel.grid_size()[1],0,1,0);
    implement_0(&temp2_r2c_complex_[(r2cSize_-1)*rSizeLocal_[1]*rSizeLocal_[2]], temp2_r2c_complex_, r2cSize_,rSizeLocal_[2],rSizeLocal_[1],0,1,0);
    #pragma acc update device(temp_r2c_complex_[0:kSitesLocal_][0:2])
*/
}


template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_h2d_send(int comp)
{
    temp_r2c_real_ = tempMemory.tempReal();
    temp_r2c_complex_ = tempMemory.temp1();
    temp2_r2c_complex_ = tempMemory.temp2();

    b_arrange_data_0(kData_, temp_r2c_complex_,kSizeLocal_[0],kSizeLocal_[1] ,kSizeLocal_[2], kHalo_, components_, comp);
    MPI_Barrier(MPI_COMM_WORLD);


//#pragma acc update device(temp_r2c_real_[0:rSitesLocal_])
}


template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim0_a2a()
{

    MPI_Alltoall(temp_r2c_complex_,2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, temp2_r2c_complex_,
		 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Gather(&temp_r2c_complex_[rSize_[0]/2*rSizeLocal_[1]*rSizeLocal_[2]][0], 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC,
	       &temp2_r2c_complex_[rSize_[0]/2*rSizeLocal_[1]*rSizeLocal_[2]][0] , 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC ,
	       parallel.grid_size()[1]-1, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Barrier(parallel.dim1_comm()[parallel.grid_rank()[0]]);

//#pragma acc update device(temp2_r2c_real_[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2])])

}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim0_tsp()
{
    if(parallel.last_proc()[1])
    {
	for(int i=0;i<parallel.grid_size()[1];i++)transpose_0_2_last_proc(&temp2_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_],
								      &temp_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_],rSizeLocal_[1],
								      rSizeLocal_[2],r2cSizeLocal_as_);
	implement_local_0_last_proc(&temp2_r2c_complex_[rSize_[0]/2*rSizeLocal_[1]*rSizeLocal_[2]],temp_r2c_complex_,rSizeLocal_[1],
				    rSizeLocal_[2],r2cSizeLocal_as_,parallel.grid_size()[1]);
    }
    else for(int i=0;i<parallel.grid_size()[1];i++)transpose_0_2(&temp2_r2c_complex_[i*rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_],
							     &temp_r2c_complex_[i*rSizeLocal_[1]* rSizeLocal_[2]*r2cSizeLocal_as_],
							     rSizeLocal_[1],rSizeLocal_[2],r2cSizeLocal_as_);

#pragma acc update device(temp_r2c_real_[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2])])
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim0_fft()
{
    fftw_complex * p_in = temp_r2c_complex_;
    fftw_complex * p_out = temp2_r2c_complex_;


#pragma acc data present(p_in[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2]) ][0:2])
    {
#pragma acc host_data use_device(p_in,p_out)
        {
            cufftExecZ2Z(bPlan_k_,(cufftDoubleComplex *)p_in,(cufftDoubleComplex *)p_out,CUFFT_INVERSE);
        }
    }
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim1_a2a()
{

#pragma acc update host(temp2_r2c_complex_[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2]) ][0:2])

    MPI_Alltoall(temp2_r2c_complex_, (2*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_), MPI_DATA_PREC,
		 temp_r2c_complex_, (2*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_)
		 , MPI_DATA_PREC, parallel.dim0_comm()[parallel.grid_rank()[1]]);
    MPI_Barrier(parallel.dim0_comm()[parallel.grid_rank()[1]]);
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim1_tsp()
{

    for(int i=0;i<parallel.grid_size()[0];i++)transpose_1_2(&temp_r2c_complex_[i*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_],
							&temp2_r2c_complex_[i*rSizeLocal_[2]*rSizeLocal_[2]*r2cSizeLocal_],
							r2cSizeLocal_,rSizeLocal_[2],rSizeLocal_[2]);

#pragma acc update host(temp2_r2c_complex_[0:(r2cSizeLocal_ * rSize_[2]* rSizeLocal_[2]) ][0:2])
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim1_fft()
{
    fftw_complex * p_in = temp2_r2c_complex_;
    fftw_complex * p_out = temp_r2c_complex_;

    bool flag_last_proc=false;
    if(parallel.last_proc()[1])flag_last_proc=true;

#pragma acc data present(p_in[0:(r2cSizeLocal_ * rSize_[1]* rSizeLocal_[2]) ][0:2])
    {
#pragma acc host_data use_device(p_in,p_out)
        {
            for(int l=0;l<rSizeLocal_[2];l++)
            {
                cufftExecZ2Z(bPlan_j_,(cufftDoubleComplex *)(p_in+(l*r2cSizeLocal_)),(cufftDoubleComplex *)(p_out+(l*r2cSizeLocal_as_)),CUFFT_FORWARD);
            }
            if(flag_last_proc)cufftExecZ2Z(bPlan_j_real_,(cufftDoubleComplex *)(p_in+r2cSizeLocal_as_),
                                           (cufftDoubleComplex *)(p_out+(r2cSizeLocal_as_*rSizeLocal_[2]*rSize_[0])),CUFFT_FORWARD);
        }
    }

}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim2_a2a()
{
#pragma acc update host(temp_r2c_complex_[0:(r2cSizeLocal_ * rSize_[1]* rSizeLocal_[2])][0:2])
    MPI_Alltoall(temp_r2c_complex_, 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, temp2_r2c_complex_,
		 2* rSizeLocal_[1]*rSizeLocal_[2]*r2cSizeLocal_as_, MPI_DATA_PREC, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Scatter(&temp_r2c_complex_[r2cSizeLocal_as_*rSizeLocal_[2]*rSize_[0]][0], 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC,
		&temp2_r2c_complex_[(r2cSize_-1)*rSizeLocal_[1]*rSizeLocal_[2]][0] , 2*rSizeLocal_[1]*rSizeLocal_[2], MPI_DATA_PREC ,
		parallel.grid_size()[1]-1, parallel.dim1_comm()[parallel.grid_rank()[0]]);
    MPI_Barrier(parallel.dim1_comm()[parallel.grid_rank()[0]]);
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim2_tsp()
{
    b_transpose_back_0_1(temp2_r2c_complex_, temp_r2c_complex_, r2cSize_,r2cSizeLocal_as_,rSizeLocal_[2],rSizeLocal_[1],parallel.grid_size()[1]);
    b_implement_0(&temp2_r2c_complex_[(r2cSize_-1)*rSizeLocal_[1]*rSizeLocal_[2]], temp_r2c_complex_,r2cSize_,rSizeLocal_[2],rSizeLocal_[1]);
#pragma acc update device(temp_r2c_complex_[0:rSitesLocal_][0:2])
}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_dim2_fft()
{
    fftw_complex * p_in = temp_r2c_complex_;
    double * p_out = temp_r2c_real_;

#pragma acc data present(p_in[0:kSitesLocal_])
    {
#pragma acc host_data use_device(p_in,p_out)
        {
            for(int l = 0;l< rSizeLocal_[2] ;l++)
            {
                cufftExecZ2D(bPlan_i_,(cufftDoubleComplex *)(p_in + (l*r2cSize_*rSizeLocal_[1] )),
                             (cufftDoubleReal *)(p_out + (l*rJump_[2])));
            }
        }
    }

}

template<class compType>
void PlanFFT_ACC<compType>::execute_r2c_backward_d2h_send(int comp)
{

#pragma acc update host(temp_r2c_real_[0:rSitesLocal_])

    Lattice lat(3,rSize_,rHalo_);
    Site x(lat);
    long i=0;

    for(x.first();x.test();x.next(),i++)
    {
	// temp_r2c_real_[i]=rData_[x.index()*components_ + comp] ;
	rData_[x.index()*components_ + comp] = temp_r2c_real_[i];

    }

}

//transposition methods

#ifndef SINGLE

#define DATA_SIZE_SEEN_BY_OPENACC 1
#define DATA_SIZE_SEEN_BY_OPENACC_LAST 1

/////
template<class compType>
void PlanFFT_ACC<compType>::transpose_0_2( fftw_complex * in, fftw_complex * out,int dim_i,int dim_j ,int dim_k) //we need offset_in, offset_out everywhere
{
	int i,j,k;

	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC], out[0:DATA_SIZE_SEEN_BY_OPENACC])
	{
		#pragma acc loop independent
		for(i=0;i<dim_i;i++)
		{
			#pragma acc loop independent
			for(j=0;j<dim_j;j++)
			{
				#pragma acc loop independent
				for(k=0;k<dim_k;k++)
				{

					out[k+dim_k*(j+i*dim_j)][0]=in[i+dim_i*(j+k*dim_j)][0];
					out[k+dim_k*(j+i*dim_j)][1]=in[i+dim_i*(j+k*dim_j)][1];

				}
			}
		}
	}

}

template<class compType>
void PlanFFT_ACC<compType>::transpose_0_2_last_proc( fftw_complex * in, fftw_complex * out,int dim_i,int dim_j ,int dim_k)
{
	int i,j,k;
	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC_LAST], out[0:DATA_SIZE_SEEN_BY_OPENACC_LAST])
	{
		#pragma acc loop independent
		for(i=0;i<dim_i;i++)
		{
			#pragma acc loop independent
			for(j=0;j<dim_j;j++)
			{
				#pragma acc loop independent
				for(k=0;k<dim_k;k++)
				{
					out[k+(dim_k+1)*(j+i*dim_j)][0]=in[i+dim_i*(j+k*dim_j)][0];
					out[k+(dim_k+1)*(j+i*dim_j)][1]=in[i+dim_i*(j+k*dim_j)][1];
				}
			}
		}
	}
}

template<class compType>
void PlanFFT_ACC<compType>::implement_local_0_last_proc( fftw_complex * in, fftw_complex * out,int proc_dim_i,int proc_dim_j,int proc_dim_k,int proc_size)
{

	int i_in,i_out,j,rank;
	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC_LAST], out[0:DATA_SIZE_SEEN_BY_OPENACC_LAST])
	{
		#pragma acc loop independent
		for(i_in=0;i_in<proc_dim_i;i_in++)
		{
			#pragma acc loop independent
			for(j=0;j<proc_dim_j;j++)
			{
				#pragma acc loop independent
				for(rank=0;rank<proc_size;rank++)
				{
					i_out=i_in+rank*proc_dim_i;
					out[proc_dim_k + (proc_dim_k+1)*(j+i_out*proc_dim_j)][0]=in[i_in+proc_dim_i*(j+proc_dim_j*rank)][0];
					out[proc_dim_k + (proc_dim_k+1)*(j+i_out*proc_dim_j)][1]=in[i_in+proc_dim_i*(j+proc_dim_j*rank)][1];
				}
			}
		}
	}
}


template<class compType>
void PlanFFT_ACC<compType>::transpose_1_2(fftw_complex * in , fftw_complex * out ,int dim_i,int dim_j ,int dim_k )
{
	int i,j,k;
	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC], out[0:DATA_SIZE_SEEN_BY_OPENACC])
	{
		#pragma acc loop independent
		for(i=0;i<dim_i;i++)
		{
			#pragma acc loop independent
			for(j=0;j<dim_j;j++)
			{
				#pragma acc loop independent
				for(k=0;k<dim_k;k++)
				{
					out[i+dim_i*(k+j*dim_k)][0]=in[i+dim_i*(j+k*dim_j)][0];
					out[i+dim_i*(k+j*dim_k)][1]=in[i+dim_i*(j+k*dim_j)][1];
				}
			}
		}
	}
}

template<class compType>
void PlanFFT_ACC<compType>::transpose_back_0_3( fftw_complex * in, fftw_complex * out,int r2c,int local_r2c,int local_size_j,int local_size_k,int proc_size,int halo,int components, int comp)
{
	int i,j,k,l, i_t, j_t, k_t;
	int r2c_halo = r2c + 2*halo;
	int local_size_k_halo = local_size_k + 2*halo;
	// #pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC]), present(out[0:DATA_SIZE_SEEN_BY_OPENACC])
	// {
	// 	#pragma acc loop independent
		for (i=0;i<local_r2c;i++)
		{
			// #pragma acc loop independent
			for(k=0;k<local_size_k;k++)
			{
				// #pragma acc loop independent
				for(j=0;j<local_size_j;j++)
				{
					// #pragma acc loop independent
					for(l=0;l<proc_size;l++)
					{
						i_t = i + l*local_r2c;
						j_t = j ;
						k_t = k ;
						out[comp+components*(i_t + r2c_halo * (k_t + local_size_k_halo * j_t))][0]=in[i + local_r2c * (j + local_size_j * (k + local_size_k *l)) ][0];
						out[comp+components*(i_t + r2c_halo * (k_t + local_size_k_halo * j_t))][1]=in[i + local_r2c * (j + local_size_j * (k + local_size_k *l)) ][1];
					}
				}
			}
		}
	// }
}

template<class compType>
void PlanFFT_ACC<compType>::implement_0(fftw_complex * in, fftw_complex * out,int r2c_size,int local_size_j,int local_size_k, int halo,int components, int comp)
{
	int i,j,k;
	i=r2c_size-1;
	int r2c_halo = r2c_size + 2*halo;
	int local_size_k_halo = local_size_k + 2*halo;

	// #pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC]), present(out[0:DATA_SIZE_SEEN_BY_OPENACC])
	// {
		// #pragma acc loop independent
		for(j=0;j<local_size_j;j++)
		{
			// #pragma acc loop independent
			for(k=0;k<local_size_k;k++)
			{
				out[comp+components*(i + r2c_halo * (k + local_size_k_halo *j))][0]=in[j + local_size_j *k][0];
				out[comp+components*(i + r2c_halo * (k + local_size_k_halo *j))][1]=in[j + local_size_j *k][1];
			}
		}
	// }
}


template<class compType>
void PlanFFT_ACC<compType>::b_arrange_data_0(fftw_complex *in, fftw_complex * out,int dim_i,int dim_j ,int dim_k, int khalo, int components, int comp)
{
	int i,j,k;
	int jump_i=(dim_i+ 2 *khalo);
	int jump_j=dim_j+ 2 *khalo;
	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC]), present(out[0:DATA_SIZE_SEEN_BY_OPENACC])
	{
		#pragma acc loop independent
		for(i=0;i<dim_i;i++)
		{
			#pragma acc loop independent
			for(j=0;j<dim_j;j++)
			{
				#pragma acc loop independent
				for(k=0;k<dim_k;k++)
				{
					out[j + dim_j * (k + dim_k * i)][0]=in[comp+components*(i + jump_i * (j + jump_j*k))][0];
					out[j + dim_j * (k + dim_k * i)][1]=in[comp+components*(i + jump_i * (j + jump_j*k))][1];
				}
			}
		}
	}
}

template<class compType>
void PlanFFT_ACC<compType>::b_transpose_back_0_1( fftw_complex * in, fftw_complex * out,int r2c,int local_r2c,int local_size_j,int local_size_k,int proc_size)
{
	int i,j,k,l, i_t, j_t, k_t;

	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC]), present(out[0:DATA_SIZE_SEEN_BY_OPENACC])
	{
		#pragma acc loop independent
		for (i=0;i<local_r2c;i++)
		{
			#pragma acc loop independent
			for(k=0;k<local_size_k;k++)
			{
				#pragma acc loop independent
				for(j=0;j<local_size_j;j++)
				{
					#pragma acc loop independent
					for(l=0;l<proc_size;l++)
					{
						i_t = i + l*local_r2c;
						j_t = j ;
						k_t = k ;
						out[i_t + r2c * (k_t + local_size_k * j_t)][0]=in[i + local_r2c * (j + local_size_j * (k + local_size_k *l)) ][0];
						out[i_t + r2c * (k_t + local_size_k * j_t)][1]=in[i + local_r2c * (j + local_size_j * (k + local_size_k *l)) ][1];
					}
				}
			}
		}
	}
}

template<class compType>
void PlanFFT_ACC<compType>::b_implement_0(fftw_complex * in, fftw_complex * out,int r2c_size,int local_size_j,int local_size_k)
{
	int i,j,k;
	i=r2c_size-1;


	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC]), present(out[0:DATA_SIZE_SEEN_BY_OPENACC])
	{
		#pragma acc loop independent
		for(j=0;j<local_size_j;j++)
		{
			#pragma acc loop independent
			for(k=0;k<local_size_k;k++)
			{
				out[i + r2c_size * (k + local_size_k *j)][0]=in[j + local_size_j *k][0];
				out[i + r2c_size * (k + local_size_k *j)][1]=in[j + local_size_j *k][1];
			}
		}
	}

}

#endif

#endif
