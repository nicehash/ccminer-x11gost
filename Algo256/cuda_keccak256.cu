/*
	--KECCAK-256 CUDA implementation for CCMINER--

	BASED UPON: djm's work 
	Improved from: SP-hash (350Mh/s to 380Mh/s for GTX970 @1278-1290MHz)

	April-2016
	Alexis Provos
	Optimized for maxwell GPUS (380Mh/s to 392Mh/s for GTX970 @1278-1290MHz)
*/

#include "miner.h"

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include "cuda_helper.h"

static const uint64_t host_keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

uint32_t *d_nounce[MAX_GPUS];
static uint32_t *h_nounce[MAX_GPUS];

__constant__ uint2 c_PaddedMessage80[ 6]; // padded message (80 bytes + padding?)

__constant__ uint2 c_mid[17];

__constant__ uint2 keccak_round_constants[24];


__global__
void keccak256_gpu_hash_80(uint32_t threads, uint32_t startNounce,uint32_t *resNounce,const uint64_t highTarget){
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 s[25],t[5], u[5], v, w;
	if (thread < threads){
		uint32_t nounce = startNounce + thread;

		s[9]= c_PaddedMessage80[0];
		s[9].y = cuda_swab32(nounce);
		s[10] = make_uint2(1, 0);

		t[ 4] = c_PaddedMessage80[ 1]^s[ 9];
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[ 0]=t[ 4]^c_mid[ 0];
		u[ 1]=c_mid[ 1]^ROL2(t[ 4],1);
		u[ 2]=c_mid[ 2];
		/* thetarho pi: b[..] = rotl(a[..] ^ d[...], ..) //There's no need to perform theta and -store- the result since it's unique for each a[..]*/
		s[ 7] = ROL2(s[10]^u[ 0], 3);
		s[10] = c_mid[ 3];
		    w = c_mid[ 4];
		s[20] = c_mid[ 5];
		s[ 6] = ROL2(s[ 9]^u[ 2],20);
		s[ 9] = c_mid[ 6];
		s[22] = c_mid[ 7];
		s[14] = ROL2(u[ 0],18);
		s[ 2] = c_mid[ 8];
		s[12] = ROL2(u[ 1],25);
		s[13] = c_mid[ 9];
		s[19] = ROL2(u[ 1],56);
		s[23] = ROL2(u[ 0],41);
		s[15] = c_mid[10];
		s[ 4] = c_mid[11];
		s[24] = c_mid[12];
		s[21] = ROL2(c_PaddedMessage80[ 2]^u[ 1],55);
		s[ 8] = c_mid[13];
		s[16] = ROL2(c_PaddedMessage80[ 3]^u[ 0],36);
		s[ 5] = ROL2(c_PaddedMessage80[ 4]^u[ 1],28);
		s[ 3] = ROL2(u[ 1],21);
		s[18] = c_mid[14];
		s[17] = c_mid[15];
		s[11] = c_mid[16];

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = c_PaddedMessage80[ 5]^u[ 0];
		s[ 0] = chi(v,w,s[ 2]);
		s[ 1] = chi(w,s[ 2],s[ 3]);
		s[ 2] = chi(s[ 2],s[ 3],s[ 4]);
		s[ 3] = chi(s[ 3],s[ 4],v);
		s[ 4] = chi(s[ 4],v,w);
		v = s[ 5];w = s[ 6];s[ 5] = chi(v,w,s[ 7]);s[ 6] = chi(w,s[ 7],s[ 8]);s[ 7] = chi(s[ 7],s[ 8],s[ 9]);s[ 8] = chi(s[ 8],s[ 9],v);s[ 9] = chi(s[ 9],v,w);
		v = s[10];w = s[11];s[10] = chi(v,w,s[12]);s[11] = chi(w,s[12],s[13]);s[12] = chi(s[12],s[13],s[14]);s[13] = chi(s[13],s[14],v);s[14] = chi(s[14],v,w);
		v = s[15];w = s[16];s[15] = chi(v,w,s[17]);s[16] = chi(w,s[17],s[18]);s[17] = chi(s[17],s[18],s[19]);s[18] = chi(s[18],s[19],v);s[19] = chi(s[19],v,w);
		v = s[20];w = s[21];s[20] = chi(v,w,s[22]);s[21] = chi(w,s[22],s[23]);s[22] = chi(s[22],s[23],s[24]);s[23] = chi(s[23],s[24],v);s[24] = chi(s[24],v,w);
		s[ 0]^=keccak_round_constants[0];
		#pragma unroll 10
		for (size_t i = 1; i < 23; i++) {
			/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
			#pragma unroll
			for(size_t j=0;j<5;j++){
				t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
			}
			/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
			u[0]=t[4]^ROL2(t[1],1);			u[1]=t[0]^ROL2(t[2],1);
			u[2]=t[1]^ROL2(t[3],1);			u[3]=t[2]^ROL2(t[4],1);
			u[4]=t[3]^ROL2(t[0],1);
	
			/* thetarho pi: b[..] = rotl(a[..] ^ d[...], ..) //There's no need to perform theta and -store- the result since it's unique for each a[..]*/
			v = s[1]^u[ 1];
			s[ 1] = ROL2(s[ 6]^u[ 1],44);		s[ 6] = ROL2(s[ 9]^u[ 4],20);
			s[ 9] = ROL2(s[22]^u[ 2],61);		s[22] = ROL2(s[14]^u[ 4],39);
			s[14] = ROL2(s[20]^u[ 0],18);		s[20] = ROL2(s[ 2]^u[ 2],62);
			s[ 2] = ROL2(s[12]^u[ 2],43);		s[12] = ROL2(s[13]^u[ 3],25);
			s[13] = ROL2(s[19]^u[ 4], 8);		s[19] = ROL2(s[23]^u[ 3],56);
			s[23] = ROL2(s[15]^u[ 0],41);		s[15] = ROL2(s[ 4]^u[ 4],27);
			s[ 4] = ROL2(s[24]^u[ 4],14);		s[24] = ROL2(s[21]^u[ 1], 2);
			s[21] = ROL2(s[ 8]^u[ 3],55);		s[ 8] = ROL2(s[16]^u[ 1],45);
			s[16] = ROL2(s[ 5]^u[ 0],36);		s[ 5] = ROL2(s[ 3]^u[ 3],28);
			s[ 3] = ROL2(s[18]^u[ 3],21);		s[18] = ROL2(s[17]^u[ 2],15);
			s[17] = ROL2(s[11]^u[ 1],10);		s[11] = ROL2(s[ 7]^u[ 2], 6);
			s[ 7] = ROL2(s[10]^u[ 0], 3);		s[10] = ROL2(v, 1);

			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			v = s[ 0]^u[ 0];w = s[ 1];
			s[ 0] = chi(v,w,s[ 2]);
			s[ 1] = chi(w,s[ 2],s[ 3]);s[ 2] = chi(s[ 2],s[ 3],s[ 4]);s[ 3] = chi(s[ 3],s[ 4],v);s[ 4] = chi(s[ 4],v,w);
			v = s[ 5];w = s[ 6];s[ 5] = chi(v,w,s[ 7]);s[ 6] = chi(w,s[ 7],s[ 8]);s[ 7] = chi(s[ 7],s[ 8],s[ 9]);s[ 8] = chi(s[ 8],s[ 9],v);s[ 9] = chi(s[ 9],v,w);
			v = s[10];w = s[11];s[10] = chi(v,w,s[12]);s[11] = chi(w,s[12],s[13]);s[12] = chi(s[12],s[13],s[14]);s[13] = chi(s[13],s[14],v);s[14] = chi(s[14],v,w);
			v = s[15];w = s[16];s[15] = chi(v,w,s[17]);s[16] = chi(w,s[17],s[18]);s[17] = chi(s[17],s[18],s[19]);s[18] = chi(s[18],s[19],v);s[19] = chi(s[19],v,w);
			v = s[20];w = s[21];s[20] = chi(v,w,s[22]);s[21] = chi(w,s[22],s[23]);s[22] = chi(s[22],s[23],s[24]);s[23] = chi(s[23],s[24],v);s[24] = chi(s[24],v,w);
			s[ 0]^= keccak_round_constants[i];
		}
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		#pragma unroll
		for(size_t j=0;j<5;j++){
			t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
		}
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		s[ 3] = ROL2(xor3x(s[18],t[2],ROL2(t[4],1)),21);
		s[ 4] = ROL2(xor3x(s[24],t[3],ROL2(t[0],1)),14);
		s[ 3] = chi(s[ 3],s[ 4],xor3x(s[ 0],t[4],ROL2(t[1],1)));
		if (devectorize(s[3]) <= highTarget){
			resNounce[0] = nounce;
		}
	}
}

__host__
uint32_t keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce,const uint64_t highTarget){

	const uint32_t threadsperblock = 320;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	keccak256_gpu_hash_80<<<grid, block>>>(threads, startNounce, d_nounce[thr_id],highTarget);
	cudaMemcpy(h_nounce[thr_id], d_nounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_nounce[thr_id][0];
}

__global__
void keccak256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash){
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){
		uint2 t[5], u[5], v, w;

		uint2 s[25];
		#pragma unroll
		for (size_t i = 0; i<25; i++) {
			if (i<4) s[i] = vectorize(outputHash[i*threads+thread]);
			else     s[i] = make_uint2(0, 0);
		}
		s[4]  = make_uint2(1, 0);
		s[16] = make_uint2(0, 0x80000000);
		
		#pragma unroll 10
		for (size_t i = 0; i < 24; i++) {
			/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
			#pragma unroll
			for(size_t j=0;j<5;j++){
				t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
			}
			/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
			u[0]=t[4]^ROL2(t[1],1);			u[1]=t[0]^ROL2(t[2],1);
			u[2]=t[1]^ROL2(t[3],1);			u[3]=t[2]^ROL2(t[4],1);
			u[4]=t[3]^ROL2(t[0],1);

			/* thetarho pi: b[..] = rotl(a[..] ^ d[...], ..) //There's no need to perform theta and -store- the result since it's unique for each a[..]*/
			v = s[1]^u[ 1];
			s[ 1] = ROL2(s[ 6]^u[ 1],44);		s[ 6] = ROL2(s[ 9]^u[ 4],20);
			s[ 9] = ROL2(s[22]^u[ 2],61);		s[22] = ROL2(s[14]^u[ 4],39);
			s[14] = ROL2(s[20]^u[ 0],18);		s[20] = ROL2(s[ 2]^u[ 2],62);
			s[ 2] = ROL2(s[12]^u[ 2],43);		s[12] = ROL2(s[13]^u[ 3],25);
			s[13] = ROL2(s[19]^u[ 4], 8);		s[19] = ROL2(s[23]^u[ 3],56);
			s[23] = ROL2(s[15]^u[ 0],41);		s[15] = ROL2(s[ 4]^u[ 4],27);
			s[ 4] = ROL2(s[24]^u[ 4],14);		s[24] = ROL2(s[21]^u[ 1], 2);
			s[21] = ROL2(s[ 8]^u[ 3],55);		s[ 8] = ROL2(s[16]^u[ 1],45);
			s[16] = ROL2(s[ 5]^u[ 0],36);		s[ 5] = ROL2(s[ 3]^u[ 3],28);
			s[ 3] = ROL2(s[18]^u[ 3],21);		s[18] = ROL2(s[17]^u[ 2],15);
			s[17] = ROL2(s[11]^u[ 1],10);		s[11] = ROL2(s[ 7]^u[ 2], 6);
			s[ 7] = ROL2(s[10]^u[ 0], 3);		s[10] = ROL2(v, 1);

			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			v = s[ 0]^u[ 0];w = s[ 1];
			s[ 0] = chi(v,w,s[ 2]);
			s[ 1] = chi(w,s[ 2],s[ 3]);s[ 2] = chi(s[ 2],s[ 3],s[ 4]);s[ 3] = chi(s[ 3],s[ 4],v);s[ 4] = chi(s[ 4],v,w);
			v = s[ 5];w = s[ 6];s[ 5] = chi(v,w,s[ 7]);s[ 6] = chi(w,s[ 7],s[ 8]);s[ 7] = chi(s[ 7],s[ 8],s[ 9]);s[ 8] = chi(s[ 8],s[ 9],v);s[ 9] = chi(s[ 9],v,w);
			v = s[10];w = s[11];s[10] = chi(v,w,s[12]);s[11] = chi(w,s[12],s[13]);s[12] = chi(s[12],s[13],s[14]);s[13] = chi(s[13],s[14],v);s[14] = chi(s[14],v,w);
			v = s[15];w = s[16];s[15] = chi(v,w,s[17]);s[16] = chi(w,s[17],s[18]);s[17] = chi(s[17],s[18],s[19]);s[18] = chi(s[18],s[19],v);s[19] = chi(s[19],v,w);
			v = s[20];w = s[21];s[20] = chi(v,w,s[22]);s[21] = chi(w,s[22],s[23]);s[22] = chi(s[22],s[23],s[24]);s[23] = chi(s[23],s[24],v);s[24] = chi(s[24],v,w);
			s[ 0]^= keccak_round_constants[i];
		}
		#pragma unroll
		for (int i=0; i<4; i++)
			outputHash[i*threads+thread] = devectorize(s[i]);
	}
}

__host__
void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = 320;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	keccak256_gpu_hash_32 <<<grid, block>>> (threads, startNounce, d_outputHash);
	MyStreamSynchronize(NULL, order, thr_id);
}

__host__
void keccak256_setBlock_80(uint64_t *PaddedMessage80){

	uint64_t s[25],t[5],u[5],midstate[17];
	
	s[10] = 1;//(uint64_t)make_uint2(1, 0);
	s[16] = (uint64_t)1<<63;//(uint64_t)make_uint2(0, 0x80000000);

	t[ 0] = PaddedMessage80[ 0]^PaddedMessage80[ 5]^s[10];
	t[ 1] = PaddedMessage80[ 1]^PaddedMessage80[ 6]^s[16];
	t[ 2] = PaddedMessage80[ 2]^PaddedMessage80[ 7];
	t[ 3] = PaddedMessage80[ 3]^PaddedMessage80[ 8];
	
	midstate[ 0] = ROTL64(t[ 1],1);		//u[0] -partial
	       u[ 1] = t[ 0]^ROTL64(t[ 2],1);	//u[1]
	       u[ 2] = t[ 1]^ROTL64(t[ 3],1);	//u[2]
	midstate[ 1] = t[ 2];			//u[3]; -partial
	midstate[ 2] = t[ 3]^ROTL64(t[ 0],1);	//u[4];
	midstate[ 3] = ROTL64(PaddedMessage80[ 1]^u[ 1],1); //v
	midstate[ 4] = ROTL64(PaddedMessage80[ 6]^u[ 1],44);
	midstate[ 5] = ROTL64(PaddedMessage80[ 2]^u[ 2],62);
	midstate[ 6] = ROTL64(u[ 2],61);
	midstate[ 7] = ROTL64(midstate[ 2],39);
	midstate[ 8] = ROTL64(u[ 2],43);
	midstate[ 9] = ROTL64(midstate[ 2], 8);
	midstate[10] = ROTL64(PaddedMessage80[ 4]^midstate[ 2],27);
	midstate[11] = ROTL64(midstate[ 2],14);
	midstate[12] = ROTL64(u[ 1], 2);
	midstate[13] = ROTL64(s[16]^u[ 1],45);
	midstate[14] = ROTL64(u[ 2],15);
	midstate[15] = ROTL64(u[ 1],10);
	midstate[16] = ROTL64(PaddedMessage80[ 7]^u[ 2], 6);
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mid, midstate,17*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
	
	//rearrange PaddedMessage80, pass only what's needed
	uint64_t PaddedMessage[ 6];
	PaddedMessage[ 0] = PaddedMessage80[ 9];
	PaddedMessage[ 1] = PaddedMessage80[ 4];
	PaddedMessage[ 2] = PaddedMessage80[ 8];
	PaddedMessage[ 3] = PaddedMessage80[ 5];
	PaddedMessage[ 4] = PaddedMessage80[ 3];
	PaddedMessage[ 5] = PaddedMessage80[ 0];	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 6*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void keccak256_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(keccak_round_constants, host_keccak_round_constants,sizeof(host_keccak_round_constants), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(&d_nounce[thr_id], sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMallocHost(&h_nounce[thr_id], sizeof(uint32_t)));
}

__host__
void keccak256_cpu_free(int thr_id)
{
	cudaFree(d_nounce[thr_id]);
	cudaFreeHost(h_nounce[thr_id]);
}
