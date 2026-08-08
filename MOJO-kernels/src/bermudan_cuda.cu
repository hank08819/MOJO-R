/* Bermudan max-call (d=5) optimal-stopping INFERENCE -- CUDA C.
 * One path per thread: simulate 5 GBM assets over 9 dates, apply the PyTorch-trained
 * Longstaff-Schwartz exercise policy (BETA), accumulate discounted cashflow. The policy
 * is trained offline (productivity); inference runs on the GPU (performance). */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define BLOCK 128
#define NT (1<<18)
#define GRID (NT/BLOCK)
#define D 5
#define N 9
#define K 100.f
__device__ const float BETA[91]={39.334827f,-24.232960f,-30.312733f,-30.686251f,-28.654774f,-25.382729f,19.704042f,22.673143f,22.793228f,21.636700f,20.137524f,5.654584f,9.446648f,21.960005f,-19.810450f,-24.112822f,-22.714302f,-23.395575f,-20.043325f,16.534216f,18.733135f,17.921757f,18.126127f,16.544779f,8.421432f,11.905353f,2.858793f,-19.511061f,-17.618572f,-17.208374f,-18.153769f,-16.728020f,15.433704f,14.452720f,14.205855f,14.673250f,14.017172f,20.548666f,10.617363f,-20.735924f,-12.321731f,-14.007333f,-13.265647f,-14.728768f,-14.322659f,10.971556f,11.890816f,11.330043f,12.071499f,11.908772f,39.661861f,6.429322f,-33.445992f,-11.130936f,-11.520391f,-11.558201f,-11.732652f,-11.121455f,9.594664f,9.788746f,9.746010f,9.898325f,9.614931f,48.805805f,5.455533f,-47.144600f,-10.724863f,-10.596046f,-9.988170f,-10.143642f,-10.488173f,8.689857f,8.582464f,8.232291f,8.188211f,8.609746f,63.780602f,2.327287f,-63.072121f,-9.514313f,-6.557866f,-8.470887f,-8.393569f,-7.818264f,7.161981f,5.770020f,6.639585f,6.590458f,6.357351f,77.070389f,0.158136f};
__device__ const float DRIFT=-0.02333333f, VOL=0.11547005f, DISC=0.98347145f;
__device__ __forceinline__ unsigned lcg(unsigned s){return s*1664525u+1013904223u;}
__device__ __forceinline__ float uf(unsigned s){return __uint_as_float((s>>9)|0x3F800000u)-1.f;}
__device__ __forceinline__ float nrm(unsigned&st){unsigned s1=lcg(st),s2=lcg(s1);st=s2;
  float u1=uf(s1)+1e-7f,u2=uf(s2);return sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2);}
__global__ void berm(float* part,long long M){
  int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=NT)return;
  float acc=0.f;
  for(long long o=tid;o<M;o+=NT){
    unsigned st=((unsigned)o*2654435761u)^0xBEEF1234u;
    float S[D]; for(int i=0;i<D;i++)S[i]=K;
    float cash=0.f; bool stop=false;
    for(int k=0;k<N;k++){
      float mx=-1e30f;
      for(int i=0;i<D;i++){ S[i]*=expf(DRIFT+VOL*nrm(st)); mx=fmaxf(mx,S[i]); }
      float pay=fmaxf(mx-K,0.f);
      if(!stop){
        if(k>=1&&k<=7&&pay>0.f){
          const float* b=&BETA[(k-1)*13];
          float C=b[0]+b[11]*(mx/K)+b[12]*(mx/K)*(mx/K);
          for(int i=0;i<D;i++){float m=S[i]/K; C+=b[1+i]*m+b[6+i]*m*m;}
          if(pay>C){ cash=powf(DISC,(float)k)*pay; stop=true; }
        } else if(k==N-1){ cash=powf(DISC,(float)k)*pay; stop=true; }
      }
    }
    acc+=cash;
  }
  part[tid]=acc;
}
double wms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
int main(){
  float*dP,*hP=(float*)malloc(NT*sizeof(float)); cudaMalloc(&dP,NT*sizeof(float));
  long long sizes[]={1000000,4000000,16000000};
  printf("Bermudan max-call d=5 INFERENCE (CUDA)\n%12s %10s %10s\n","Paths","median(ms)","Price");
  for(int s=0;s<3;s++){long long M=sizes[s];
    for(int w=0;w<5;w++)berm<<<GRID,BLOCK>>>(dP,M); cudaDeviceSynchronize();
    double best=1e9;for(int rr=0;rr<30;rr++){double t0=wms();berm<<<GRID,BLOCK>>>(dP,M);cudaDeviceSynchronize();double e=wms()-t0;if(e<best)best=e;}
    cudaMemcpy(hP,dP,NT*sizeof(float),cudaMemcpyDeviceToHost);
    double tot=0;for(int j=0;j<NT;j++)tot+=hP[j];
    printf("%12lld %10.4f %10.4f\n",M,best,tot/(double)M);
  }
  return 0;
}
