#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <cmath>
#include <curand_kernel.h>
#include <sys/time.h>

#define Nsol 1024
#define Nvar 6
#define CR 0.3
#define CG 0.5
#define CP 0.7 
#define Xmin -36
#define Xmax 36
#define THREADSPERBLOCK 128
#define BLOCKSPERGRID Nsol/THREADSPERBLOCK  
#define LOCAL_COUNT Nsol/THREADSPERBLOCK 

__global__ void MISSO(float *X_gpu,float *u_j_gpu,float partialGbest[]);

void random(float* a, int n){
	for(int k=0; k<n; k++){                               /* initialized solution*/
		a[k]=(float) rand()*(Xmax-Xmin)/RAND_MAX+Xmin;
	}
}


double wallclock(void)
{
  struct timeval tv;
  struct timezone tz;
  double t;

  gettimeofday(&tv, &tz);

  t = (double)tv.tv_sec*1000;
  t += ((double)tv.tv_usec)/1000.0;

  return t;
}// millisecond
   
int main(void)
{ 
   
   int i;
   //float Pbest[Nsol] ,best_try[Nvar]; /* initialize */
   //float random_number,random_number2 ,Pbest_try ,X_try;
   float *X;
   float *partialGbest_gpu;
   //clock_t start_time, end_time;
   //float total_time = 0;
   double t2,t4;
   
     srand(time(NULL));
     
    X = (float*) malloc(sizeof(float) * Nvar * Nsol);
    partialGbest_gpu = (float*)malloc(sizeof(float)*LOCAL_COUNT);

	
	random(X,Nvar * Nsol);   
     
   			
    /*for(i=0;i<Nsol;i++)                   initialized value
   		Pbest[i]=fitness_function(X,i);
		   
	float min=Pbest[0];
	   		
   	for(i=1;i<Nsol;i++)
        if(Pbest[i]<min){
            min=Pbest[i];
			Gbest=i;
		}	   			
   	*/	
   	
	float *X_gpu,*u_j_gpu,*partialGbest;
	 
    cudaMalloc((void**)&X_gpu,sizeof(float)* Nsol* Nvar );
	cudaMalloc((void**)&partialGbest,sizeof(float)*LOCAL_COUNT ); 
    cudaMalloc((void**)&u_j_gpu,sizeof(float));
    

    t4 = wallclock();
    // Copy Data from Host to Device
    
    cudaMemcpy(X_gpu,X,sizeof(float)*(Nsol*Nvar),cudaMemcpyHostToDevice);
    //cudaMemcpy(u_j_gpu,u_j,sizeof(float),cudaMemcpyHostToDevice);
    
    
    MISSO<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(X_gpu,u_j_gpu,partialGbest);
	cudaThreadSynchronize();
	
	cudaMemcpy(partialGbest_gpu,partialGbest,sizeof(float)*LOCAL_COUNT,cudaMemcpyDeviceToHost);
	t2 = wallclock();

	for(i=0;i<LOCAL_COUNT;i++)
		printf("%f ,",partialGbest_gpu[i]);

	printf("\n Elapsed time = %10.3f(ms)\n",t2-t4);
	
 
	free(X);
	free(partialGbest_gpu);

	
	cudaFree(X_gpu);
	cudaFree(partialGbest);
	cudaFree(u_j_gpu);
	
    /*start_time = clock();	#CPU main_coding
    
    for(t=1;t<=Ngen;t++){
   	  
   	    for(i=1;i<=Nsol;i++){
   	    		flag=1;
   	  	    for(j=1;j<=Nvar;j++){
   	  	    	
   	  	    	random_number  = (float) rand()/RAND_MAX;      random 0~1  
   	  	    	random_number2 = (float) rand()/RAND_MAX-0.5;  random -0.5~0.5 
   	  	    	
   	  	    	//random_number2=0.5*random_number2;
                if (random_number < CR ||  X[i][j]==X[Gbest][j] )
                	best_try[0][j]=X[i][j]+random_number2*u_j;
                else if (random_number < CG &&  X[i][j]!=X[Gbest][j] )
                	best_try[0][j]=X[Gbest][j]+random_number2*u_j;
                else if (random_number < CP &&  X[i][j]!=X[Gbest][j] )    
                	best_try[0][j]=X[i][j]+random_number2*(X[i][j]-X[Gbest][j]);     
   	  	    	else
   	  	    		best_try[0][j]=X[i][j];
   	  	        
			}
					
   	  	    		Pbest_try=fitness_function2(best_try); 
   	  	        	X_try=fitness_function(X,i);
   	  	        
   	  	    	if(Pbest_try<X_try){
   	  	        	for(j=0;j<Nvar;j++)
						X[i][j]=best_try[0][j];
   	  	    	}
   	  	    
		}
		
		for(i=0;i<Nsol;i++)
			Pbest[i]=fitness_function(X,i);
		
		float min1=Pbest[0];
		       
		
    	for(i=1;i<Nsol;i++)
        	if(Pbest[i]<min1){
        	    min1=Pbest[i];
				Gbest=i;
		    }
		
		while(flag){
			for(j=1;j<=Nvar;j++){ 
			 
				random_number2 = (double) rand()/RAND_MAX-0.5;
				 
				best_try[0][j]=X[Gbest][j]+random_number2*u_j;
				
			}
				Pbest_try=fitness_function2(best_try); 
			    X_try=fitness_function(X,Gbest);
			    
			    if(Pbest_try<X_try){
	
			    	for(j=1;j<=Nvar;j++)
						X[Gbest][j]=best_try[0][j];				
			    }
			    else{
			    	flag=0;
			    	break;
				}
	    } 
		
		if( t%200==0 )
			printf("%f \n",Pbest[Gbest]);	
		
		end_time = clock();
		total_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;
		
		if(total_time>=0.5){
			printf("%d ",t);
			printf("Time : %f \n", total_time);
			break;
		}
		       
   	    /*if (P[Gbest][1] ==  && P[Gbest][2] == 0){
   	    	//printf("%f  %f\n",P[Gbest][1],P[Gbest][2]);
   	    	fprintf(fp, "%lf,%lf\n", P[Gbest][1],P[Gbest][2]);
   	    	//printf("both 0\n");
   	    	break;
		}
    }*/
       
   		
    return 0;
}



	__device__ float fitness_function(float *X_gpu,int id){
        	float value,sum1=0,sum2=0;
			 
   	        for(int i=0;i<Nvar;i++)
   	        	sum1+=(X_gpu[id*Nvar+i]-1)*(X_gpu[id*Nvar+i]-1);

		for(int i=1;i<Nvar;i++)
			sum2+=X_gpu[id*Nvar+i]*X_gpu[id*Nvar+i-1];

   	        
   	     	value=sum1-sum2;
		
		 return value;	   
	}
	
	__device__ float fitness_function2(float X[][Nvar],int tid){
        float value,sum1=0,sum2=0;
			 
   	        for(int i=0;i<Nvar;i++)
   	        	sum1+=(X[tid][i]-1)*(X[tid][i]-1);

		for(int i=1;i<Nvar;i++)
			sum2+=X[tid][i]*X[tid][i-1];
   	        
   	     value=sum1-sum2;
		
		 return value;	   
	}

__global__ void MISSO(float *X_gpu,float *u_j_gpu,float partialGbest[]){
		int id = blockIdx.x * blockDim.x + threadIdx.x; 
		int tid = threadIdx.x;
		float Pbest_try;
		float X_try;
		float random_number;
		float random_number2;
		
		__shared__ float min;
		__shared__ int Gbest; 
		__shared__ float Pbest[THREADSPERBLOCK]; 	
		__shared__ float best_try[THREADSPERBLOCK][Nvar];
	
	*u_j_gpu= (float) (Xmax-Xmin)/(10*Nvar);
	
		
	Gbest=0;
		
	if( tid < THREADSPERBLOCK){
				curandState State;
				curand_init(1, id, 0, &State);
		
	for(int t=0;t<5000;t++){

		
		
		for(int j=0;j<Nvar;j++){
			
   	  	    
   	  	    	random_number  = curand_uniform(&State);      
   	  	    	random_number2 = curand_uniform(&State)-0.5;                   
   	  	    	
   	  	    	//random_number2=0.5*random_number2;
                if (random_number < CR ||  X_gpu[id*Nvar+j]==X_gpu[(blockIdx.x * blockDim.x+Gbest)*Nvar+j] )
                	best_try[tid][j]=X_gpu[id*Nvar+j]+random_number2 * *u_j_gpu;
                else if (random_number < CG && X_gpu[id*Nvar+j]!=X_gpu[(blockIdx.x * blockDim.x+Gbest)*Nvar+j] )
                	best_try[tid][j]=X_gpu[(blockIdx.x * blockDim.x+Gbest)*Nvar+j]+random_number2* *u_j_gpu;
                else if (random_number < CP && X_gpu[id*Nvar+j]!=X_gpu[(blockIdx.x * blockDim.x+Gbest)*Nvar+j] )    
                	best_try[tid][j]=X_gpu[id*Nvar+j]+random_number2*(X_gpu[id*Nvar+j]-X_gpu[(blockIdx.x * blockDim.x+Gbest)*Nvar+j]);     
   	  	    	else
   	  	    		best_try[tid][j]=X_gpu[id*Nvar+j];
				
   	  	        
		}
				__syncthreads();
				
		
		Pbest_try=fitness_function2(best_try,tid); 
   	  	X_try=fitness_function(X_gpu,id);
		
		
        if(Pbest_try<X_try){
   	  	    for(int j=0;j<Nvar;j++)
				X_gpu[id*Nvar+j]=best_try[tid][j];
				
		} 
       
	    
	Pbest[tid]=fitness_function(X_gpu,id);
	
		for(unsigned int s=blockDim.x/2;s>0;s>>=1){
			
			
			if(tid<s){
				if( Pbest[tid+s]<Pbest[tid] )
					Pbest[tid]=Pbest[tid+s];
			}
				__syncthreads();
		}
		 
		
		min=Pbest[0];
		
	Pbest[tid]=fitness_function(X_gpu,id);	
	
	if( min==Pbest[tid] )		
		Gbest=tid;	
		
	}
	 
 	        

}



		partialGbest[blockIdx.x]=fitness_function(X_gpu,blockIdx.x * blockDim.x+Gbest);

}
