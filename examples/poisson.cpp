/*! file poissonSolver.cpp
    Created by David Daverio.

    A simple example of LATfield2d usage. This exec solve the poisson equation in fourier space.


 */



#include <iostream>
#include "LATfield2.hpp"
#ifdef _OPENACC
#include <accelmath.h>
#endif

using namespace LATfield2;



int main(int argc, char **argv)
{
    int n,m;
    int BoxSize = 64;
    int halo = 1;
    int khalo =0;
    int dim = 3;
    int comp = 1;
    double sigma2 = 1.0;
    double res = 0.5;
    int temporary_datastructure_mode = 0;


    for (int i=1 ; i < argc ; i++ ){
		if ( argv[i][0] != '-' )
			continue;
		switch(argv[i][1]) {
			case 'n':
				n = atoi(argv[++i]);
				break;
			case 'm':
				m =  atoi(argv[++i]);
				break;
            case 'b':
                BoxSize = atoi(argv[++i]);
                break;
            case 't':
                temporary_datastructure_mode = atoi(argv[++i]);
                break;
		}
	}

    if (temporary_datastructure_mode != 0 && temporary_datastructure_mode != 1) {
        COUT<<"invalid datastructure mode";
        exit(100);
    }

	parallel.initialize(n,m);


    double res2 =res*res;
    double renormFFT;



    Lattice lat(dim,BoxSize,halo);
    Lattice latK;
    latK.initializeRealFFT(lat, khalo);

    Site x(lat);
    rKSite k(latK);

    Field<Real> phi;
    phi.initialize(lat,comp);
    Field<Imag> phiK;
    phiK.initialize(latK,comp);
    PlanFFT<Imag> planPhi(&phi,&phiK);


    Field<Real> rhoVerif(lat,comp);
    Field<Real> rho;
    rho.initialize(lat,comp);
    Field<Imag> rhoK;
    rhoK.initialize(latK,comp);
    PlanFFT<Imag> planRho(&rho,&rhoK);


    renormFFT=(double)lat.sites();

    //fill rho with a gaussian:

    double mean = 0.;

    sigma2 = BoxSize*BoxSize/9.;

    double* temp_rho;
    double* temp_x_coord_0;
    double* temp_x_coord_1;
    double* temp_x_coord_2;

#ifdef GPU
    if (temporary_datastructure_mode == 1)
    {
        double ctime_start_setup_time_openacc = MPI_Wtime();
        long temp_size = 0;
        for(x.first();x.test();x.next())
        {
            temp_size += 1;
        }

        temp_rho = new double[temp_size];
        temp_x_coord_0 = new double[temp_size];
        temp_x_coord_1 = new double[temp_size];
        temp_x_coord_2 = new double[temp_size];

        {
            long site_ix = 0;
            for(x.first();x.test();x.next())
            {
                temp_x_coord_0[site_ix] = x.coord(0);
                temp_x_coord_1[site_ix] = x.coord(1);
                temp_x_coord_2[site_ix] = x.coord(2);
                site_ix += 1;
            }
        }
        parallel.barrier();
        if(parallel.rank() == 0) printf("Elapsed Setup Time (CTime,OpenACC)= %9.3e [sec]\n",(MPI_Wtime() - ctime_start_setup_time_openacc));
        {
            COUT<<"Running with gaussian on gpu\n";
            double ctime_start_gaussian_kernel_time = MPI_Wtime();
            long half_lat_size_0 = lat.size(0)/2;
            long half_lat_size_1 = lat.size(1)/2;
            long half_lat_size_2 = lat.size(2)/2;
            #pragma acc kernels
            {
                #pragma acc loop independent
                for(long site_ix = 0; site_ix < temp_size; site_ix++)
                {
                    double x2 = pow(0.5 + temp_x_coord_0[site_ix] - half_lat_size_0,2.);
                    x2 += pow(0.5 + temp_x_coord_1[site_ix] - half_lat_size_1,2.);
                    x2 += pow(0.5 + temp_x_coord_2[site_ix] - half_lat_size_2,2.);
                    temp_rho[site_ix]= 1.0 * exp(-x2/sigma2) + 0.1;
                mean += temp_rho[site_ix];
                }
            }
            parallel.barrier();
            if(parallel.rank() == 0) printf("Elapsed Gaussian Time (CTime,OpenACC)= %9.3e [sec]\n",(MPI_Wtime() - ctime_start_gaussian_kernel_time));
        }
    }
    else
    {
        COUT<<"Running with gaussian on gpu using the original data structure\n";
        double ctime_start_gaussian_kernel_time_original_data_structure = MPI_Wtime();
        #pragma acc kernels
        {
            #pragma acc loop independent
            for(x.first();x.test();x.next())
            {
                double x2 = pow(0.5 + x.coord(0) - lat.size(0)/2,2.0);
                x2 += pow(0.5 + x.coord(1) - lat.size(1)/2,2.0);
                x2 += pow(0.5 + x.coord(2) - lat.size(2)/2,2.0);
                rho(x)= 1.0 * exp(-x2/sigma2) + 0.1;
            mean += rho(x);
            }
        }
        parallel.barrier();
        if(parallel.rank() == 0) printf("Elapsed Gaussian Time (CTime,OpenACC,Original Structure)= %9.3e [sec]\n",(MPI_Wtime() - ctime_start_gaussian_kernel_time_original_data_structure));
    }
#else
    double ctime_original_time = MPI_Wtime();
    for(x.first();x.test();x.next())
    {
        double x2 = pow(0.5l + x.coord(0) - lat.size(0)/2,2);
        x2 += pow(0.5l + x.coord(1) - lat.size(1)/2,2);
        x2 += pow(0.5l + x.coord(2) - lat.size(2)/2,2);
        rho(x)= 1.0 * exp(-x2/sigma2) + 0.1;
	mean += rho(x);
    }
    parallel.barrier();
    if(parallel.grid_rank()[0]==0 && parallel.grid_rank()[1]==0) printf("Elapsed Gaussian Time (CTime,Original)= %9.3e [sec]\n",(MPI_Wtime() - ctime_original_time));
#endif
#ifdef GPU
    if (temporary_datastructure_mode == 1)
    {
        long site_ix = 0;
        for(x.first();x.test();x.next()) {
            rho(x) = temp_rho[site_ix];
            site_ix += 1;
        }
        delete [] temp_rho;
        delete [] temp_x_coord_0;
        delete [] temp_x_coord_1;
        delete [] temp_x_coord_2;
    }
#endif

    parallel.sum(mean);
    mean /= (double) lat.sites();

    planRho.execute(FFT_FORWARD);


    k.first();
    if(parallel.isRoot())
    {
        phiK(k)=0.0;
        k.next();
    }
    for(;k.test();k.next())
    {
        phiK(k)= rhoK(k) /
        ( 2.0 *(cos(2.0*M_PI*k.coord(0)/BoxSize)
                + cos(2.0*M_PI*k.coord(1)/BoxSize)
                + cos(2.0*M_PI*k.coord(2)/BoxSize)-3.0)/res2 );
    }

    planPhi.execute(FFT_BACKWARD);

    phi.updateHalo();

    for(x.first();x.test();x.next())
    {
        rhoVerif(x) =  (phi(x+0) - 2 * phi(x) + phi(x-0))/res2;
        rhoVerif(x) += (phi(x+1) - 2 * phi(x) + phi(x-1))/res2;
        rhoVerif(x) += (phi(x+2) - 2 * phi(x) + phi(x-2))/res2;
    }





    double error;

    double maxError=0;
    double minError=20;
    double averageError=0;

    for(x.first();x.test();x.next())
    {
        error =  (fabs( rho(x)- (mean + rhoVerif(x)/renormFFT) ) ) / fabs(rho(x));
        averageError+=error;
        if(minError>error)minError=error;
        if(maxError<error)maxError=error;

    }

    parallel.max(maxError);
    parallel.min(minError);
    parallel.sum(averageError);

    averageError/=renormFFT;

    parallel.barrier();

    COUT<<"Min error: "<<minError<<", Max error: "<<maxError<<" , Average error: "<<averageError<<endl;

#ifdef SINGLE
#define TOLERANCE 1.0e-6
#else
#define TOLERANCE 1.0e-10
#endif
    if (maxError > TOLERANCE) exit(max(1, 1 + (int) fabs(log10(maxError))));
    else exit(0);
}
