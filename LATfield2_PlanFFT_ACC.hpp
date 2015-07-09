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

	    /// forward real to complex
		// first transopsition
		void transpose_0_2( fftw_complex * in, fftw_complex * out,int dim_i,int dim_j ,int dim_k);
		void transpose_0_2_last_proc( fftw_complex * in, fftw_complex * out,int dim_i,int dim_j ,int dim_k);
		void implement_local_0_last_proc( fftw_complex * in, fftw_complex * out,int proc_dim_i,int proc_dim_j,int proc_dim_k,int proc_size);
		// second transposition
		void transpose_1_2(fftw_complex * in , fftw_complex * out  ,int dim_i,int dim_j ,int dim_k);
		//third transposition
		void transpose_back_0_3(fftw_complex * in, fftw_complex * out,int r2c,int local_r2c,int local_size_j,int local_size_k,int proc_size,int halo,int components,int comp);
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

//	if (cufftPlan1d(&cufPlan_real_i_, rSize_[0], CUFFT_D2Z, rSizeLocal_[1]*rSizeLocal_[2]) != CUFFT_SUCCESS){
//	    fprintf(stderr, "CUFFT error: Plan r2c forward i,  creation failed");
//	    MPI_Abort(MPI_COMM_WORLD,11);
//	}



	int istride,idist,ostride,odist;

	idist = rJump_[1]*components_;
	odist= 1;
	size_t workSize;


	if(cufftMakePlanMany(cufPlan_real_i_, 1, &rSize_[0], &idist,
			     components_, idist, &odist, rSizeLocal_[1]*rSizeLocal_[2],
			     odist, CUFFT_D2Z, rSizeLocal_[1],  &workSize) != CUFFT_SUCCESS){
	    fprintf(stderr, "CUFFT error: Plan r2c forward i,  creation failed");
	    MPI_Abort(MPI_COMM_WORLD,11);
	}




        void * cuStream;
	cuStream = acc_get_cuda_stream(acc_async_sync);
        cufftSetStream(cufPlan_real_i_,(cudaStream_t) cuStream);


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
void PlanFFT_ACC<compType>::execute_r2c_forward_dim0()
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
#pragma acc data present(p_in[0:rSitesLocal_],p_out[0:kSitesLocal_][0:2])
    {
        #pragma acc host_data use_device(p_in,p_out)
        {
            cufftExecD2Z(cufPlan_real_i_,(cufftDoubleReal *)p_in,(cufftDoubleComplex *)p_out);
        }
    }
/*
#pragma acc update host(p_out[0:rSitesLocal_][0:2])

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

#ifndef SINGLE

#define DATA_SIZE_SEEN_BY_OPENACC I_WILL_FAIL
#define DATA_SIZE_SEEN_BY_OPENACC_LAST I_WILL_FAIL

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
						out[comp+components*(i_t + r2c_halo * (k_t + local_size_k_halo * j_t))][0]=in[i + local_r2c * (j + local_size_j * (k + local_size_k *l)) ][0];
						out[comp+components*(i_t + r2c_halo * (k_t + local_size_k_halo * j_t))][1]=in[i + local_r2c * (j + local_size_j * (k + local_size_k *l)) ][1];
					}
				}
			}
		}
}

template<class compType>
void PlanFFT_ACC<compType>::implement_0(fftw_complex * in, fftw_complex * out,int r2c_size,int local_size_j,int local_size_k, int halo,int components, int comp)
{
	int i,j,k;
	i=r2c_size-1;
	int r2c_halo = r2c_size + 2*halo;
	int local_size_k_halo = local_size_k + 2*halo;

	#pragma acc kernels present(in[0:DATA_SIZE_SEEN_BY_OPENACC]), present(out[0:DATA_SIZE_SEEN_BY_OPENACC])
	{
		#pragma acc loop independent
		for(j=0;j<local_size_j;j++)
		{
			#pragma acc loop independent
			for(k=0;k<local_size_k;k++)
			{
				out[comp+components*(i + r2c_halo * (k + local_size_k_halo *j))][0]=in[j + local_size_j *k][0];
				out[comp+components*(i + r2c_halo * (k + local_size_k_halo *j))][1]=in[j + local_size_j *k][1];
			}
		}
	}
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

