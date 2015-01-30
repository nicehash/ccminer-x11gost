
#include <memory.h>

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))

#include "cuda_helper.h"

__constant__ uint32_t echo512_padding[16] = {
	0x00000080, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x02000000,
	0x00000200, 0x00000000, 0x00000000, 0x00000000
};

#include "x11/cuda_x11_aes.cu"

static uint32_t *d_nonce[MAX_GPUS];
static int *d_hashidx[MAX_GPUS];
__constant__ uint32_t pTarget[8];

__device__ __forceinline__ void AES_2ROUND(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
	uint32_t &k0)
{
	uint32_t y0, y1, y2, y3;

	aes_round(sharedMemory,
		x0, x1, x2, x3,
		k0++,
		y0, y1, y2, y3);

	aes_round(sharedMemory,
		y0, y1, y2, y3,
		x0, x1, x2, x3);
}

__device__ __forceinline__ void cuda_echo_round(
	const uint32_t * __restrict__ sharedMemory,
	uint32_t &k0,
	uint32_t *W)
{
	// W hat 16*4 als Abmaﬂe

	// Big Sub Words
#pragma unroll 16
	for(int i=0;i<64;i += 4)
	{
		AES_2ROUND(sharedMemory,
			W[i+0], W[i+1], W[i+2], W[i+3],
			k0);
	}

	// Shift Rows
#pragma unroll 4
	for(int i=0;i<4;i++)
	{
		uint32_t t;

		/// 1, 5, 9, 13
		t = W[4 + i];
		W[4 + i] = W[20 + i];
		W[20 + i] = W[36 + i];
		W[36 + i] = W[52 + i];
		W[52 + i] = t;

		// 2, 6, 10, 14
		t = W[8 + i];
		W[8 + i] = W[40 + i];
		W[40 + i] = t;
		t = W[24 + i];
		W[24 + i] = W[56 + i];
		W[56 + i] = t;

		// 15, 11, 7, 3
		t = W[60 + i];
		W[60 + i] = W[44 + i];
		W[44 + i] = W[28 + i];
		W[28 + i] = W[12 + i];
		W[12 + i] = t;
	}

	// Mix Columns
#pragma unroll 4
	for(int i=0;i<4;i++) // Schleife ¸ber je 2*uint32_t
	{
#pragma unroll 4
		for(int j=0;j<16;j += 4) // Schleife ¸ber die elemnte
		{
			uint32_t a = W[ ((j + 0)<<2) + i];
			uint32_t b = W[ ((j + 1)<<2) + i];
			uint32_t c = W[ ((j + 2)<<2) + i];
			uint32_t d = W[ ((j + 3)<<2) + i];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;

			uint32_t t;
			t = ((ab & 0x80808080) >> 7);
			uint32_t abx = t<<4 ^ t<<3 ^ t<<1 ^ t;
			t = ((bc & 0x80808080) >> 7);
			uint32_t bcx = t<<4 ^ t<<3 ^ t<<1 ^ t;
			t = ((cd & 0x80808080) >> 7);
			uint32_t cdx = t<<4 ^ t<<3 ^ t<<1 ^ t;

			abx ^= ((ab & 0x7F7F7F7F) << 1);
			bcx ^= ((bc & 0x7F7F7F7F) << 1);
			cdx ^= ((cd & 0x7F7F7F7F) << 1);

			W[ ((j + 0)<<2) + i] = abx ^ bc ^ d;
			W[ ((j + 1)<<2) + i] = bcx ^ a ^ cd;
			W[ ((j + 2)<<2) + i] = cdx ^ ab ^ d;
			W[ ((j + 3)<<2) + i] = abx ^ bcx ^ cdx ^ ab ^ c;
		}
	}
}

__global__ __launch_bounds__(128) void x11_echo512_gpu_hash_64_final(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector, uint32_t *d_nonce, int *d_hashidx)
{
	__shared__ uint32_t sharedMemory[1024];

	aes_gpu_init(sharedMemory);

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		uint32_t W[64];
		uint32_t k0 = 512; // K0 = bitlen
		/* Initialisierung */
#pragma unroll 8
		for(int i=0;i<32;i+=4)
		{
			W[i + 0] = k0;
			W[i + 1] = 0;
			W[i + 2] = 0;
			W[i + 3] = 0;
		}

		// kopiere 32-byte groﬂen hash
#pragma unroll 8
		for(int i=0;i<16;i++) {
			W[i+32] = Hash[i];
			W[i+48] = echo512_padding[i];
		}

		for(int i=0;i<10;i++)
		{
			cuda_echo_round(sharedMemory, k0, W);
		}

#pragma unroll 2
		for(int i=0;i<8;i+=4)
		{
			W[i  ] ^= W[32 + i    ] ^ 512;
			W[i+1] ^= W[32 + i + 1];
			W[i+2] ^= W[32 + i + 2];
			W[i+3] ^= W[32 + i + 3];
		}

#pragma unroll 8
		for(int i=0;i<8;i++)
			Hash[i] ^= W[i];

		int position = -1;
		bool rc = true;

#pragma unroll 8
		for (int i = 7; i >= 0; i--) {
			if (Hash[i] > pTarget[i]) {
				if(position < i) {
					position = i;
					rc = false;
				}
			}
			if (Hash[i] < pTarget[i]) {
				if(position < i) {
					position = i;
					rc = true;
				}
			}
		}

		if(rc == true) {
			d_nonce[0] = nounce;
			d_hashidx[0] = thread;
		}
	}
}

__host__
void spreadx11_echo512_cpu_init(int thr_id, int threads)
{
	cudaMalloc(&d_nonce[thr_id], sizeof(uint32_t));
	cudaMalloc(&d_hashidx[thr_id], sizeof(int));
	aes_cpu_init(thr_id);
}

__host__
void spreadx11_echo512_cpu_setTarget(void *ptarget)
{
	cudaMemcpyToSymbol( pTarget, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__
uint32_t spreadx11_echo512_cpu_hash_64_final(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int *hashidx)
{
	const int threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	cudaMemset(d_nonce[thr_id], 0xffffffff, sizeof(uint32_t));

	x11_echo512_gpu_hash_64_final<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector, d_nonce[thr_id], d_hashidx[thr_id]);

	cudaThreadSynchronize();

	uint32_t res;
	cudaMemcpy(&res, d_nonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(hashidx, d_hashidx[thr_id], sizeof(int), cudaMemcpyDeviceToHost);
	return res;
}
