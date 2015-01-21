#include <stdio.h>

#include "cuda_helper.h"

#define SWAP64(x) cuda_swab64(x)

__constant__ uint64_t c_PaddedMessage[32];
__constant__ uint32_t c_sigma[16][16];
static const uint32_t host_sigma[16][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

__constant__ uint64_t c_u512[16];
const uint64_t host_u512[16] = {
	0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
	0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
	0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
	0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};

__forceinline__ __device__
uint64_t ROTR(const uint64_t value, const int offset) {
	uint64_t result;
#if __CUDA_ARCH__ >= 320
	if(offset < 32) {
		asm("{\n\t"
		" .reg .u32 tl,th,vl,vh; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"shf.r.wrap.b32 vl,tl,th,%2; \n\t"
		"shf.r.wrap.b32 vh,th,tl,%2; \n\t"
		"mov.b64 %0,{vl,vh}; \n\t"
		"}"
		: "=l"(result) : "l"(value) , "r"(offset));
	} else {
		asm("{\n\t"
		" .reg .u32 tl,th,vl,vh; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"shf.r.wrap.b32 vh,tl,th,%2; \n\t"
		"shf.r.wrap.b32 vl,th,tl,%2; \n\t"
		"mov.b64 %0,{vl,vh}; \n\t"
		"}"
		: "=l"(result) : "l"(value) , "r"(offset));
	}
#else
	result = (value >> offset) | (value << (64-offset));
#endif
	return  result;
}

__device__ __forceinline__
uint64_t ROTR64_32( uint64_t val )
{
	asm("{\n\t"
		".reg .u32 tl,th; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"mov.b64 %0,{th,tl}; \n\t"
		"}"
		: "=l"(val) : "l"(val));

	return val;
}

__device__ __forceinline__ uint64_t
ROTR64_16( uint64_t val )
{
	asm("{\n\t"
		".reg .u16 b0, b1, b2, b3; \n\t"
		"mov.b64 { b0, b1, b2, b3 }, %1; \n\t"
		"mov.b64 %0, {b1, b2, b3, b0}; \n\t"
		"}"
		: "=l"(val) : "l"(val));

	return val;
}

#define G(a,b,c,d,e) \
	v[a] += (m[c_sigma[i][e]] ^ c_u512[c_sigma[i][e+1]]) + v[b]; \
	v[d] = ROTR64_32(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR( v[b] ^ v[c],25); \
	v[a] += (m[c_sigma[i][e+1]] ^ c_u512[c_sigma[i][e]])+v[b]; \
	v[d] = ROTR64_16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR( v[b] ^ v[c],11); \


__device__
void blake_compress( uint64_t *h, const uint64_t *m, const int bits )
{
	uint64_t v[16];

	#pragma unroll 8
	for(int i = 0; i < 8; ++i)
		v[i] = h[i];

	v[ 8] = c_u512[0];
	v[ 9] = c_u512[1];
	v[10] = c_u512[2];
	v[11] = c_u512[3];
	v[12] = c_u512[4] ^ bits;
	v[13] = c_u512[5] ^ bits;
	v[14] = c_u512[6];
	v[15] = c_u512[7];

	#pragma unroll 2
	for( int i = 0; i < 16; ++i )
	{
		/* column step */
		G( 0, 4, 8, 12, 0 );
		G( 1, 5, 9, 13, 2 );
		G( 2, 6, 10, 14, 4 );
		G( 3, 7, 11, 15, 6 );
		/* diagonal step */
		G( 0, 5, 10, 15, 8 );
		G( 1, 6, 11, 12, 10 );
		G( 2, 7, 8, 13, 12 );
		G( 3, 4, 9, 14, 14 );
	}

	#pragma unroll 8
	for( int i = 0; i < 8; ++i )  h[i] ^= v[i] ^ v[i+8];
}

static __device__ uint32_t cuda_swap32(uint32_t x)
{
	return __byte_perm(x, 0, 0x0123);
}

static __constant__ uint64_t d_IV[8];
static const uint64_t h_IV[8] = {
	0x6a09e667f3bcc908ULL,
	0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL,
	0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL,
	0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL,
	0x5be0cd19137e2179ULL
};

__global__
void blake_gpu_hash_185(int threads, uint32_t startNonce, uint32_t *outputHash, uint32_t *g_signature, uint32_t *g_hashwholeblock)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		uint32_t nonce = startNonce + thread;
		uint32_t idx64 = thread >> 6;
		uint64_t h[8];
		uint64_t buf[16];
		uint64_t *hashwholeblock = (uint64_t *)&g_hashwholeblock[idx64 << 3];
		uint64_t *signature = (uint64_t *)&g_signature[idx64 << 3];

		#pragma unroll 8
		for(int i=0;i<8;i++)
			h[i] = d_IV[i];

		#pragma unroll 11
		for (int i=0; i < 11; ++i) buf[i] = SWAP64(c_PaddedMessage[i]);

		((uint32_t *)buf)[20] = cuda_swap32(nonce);

		#pragma unroll 4
		for (int i=0; i < 4; ++i) buf[i+11] = SWAP64(hashwholeblock[i]);

		buf[15] = SWAP64(c_PaddedMessage[15]);

		blake_compress( h, buf, 1024 );

		#pragma unroll 16
		for (int i=0; i < 16; ++i) buf[i] = c_PaddedMessage[i+16];

		#pragma unroll 32
		for (int i=0; i < 32; ++i) ((unsigned char *)buf)[25+i] = ((unsigned char *)signature)[i];//SWAP64(signature[i]);

		for (int i=0; i < 16; ++i) buf[i] = SWAP64(buf[i]);

		blake_compress( h, buf, 1480 );

		uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;

		#pragma unroll 8
		for (int i=0; i < 8; ++i) {
			outHash[2*i+0] = cuda_swap32( _HIWORD(h[i]) );
			outHash[2*i+1] = cuda_swap32( _LOWORD(h[i]) );
		}
	}
}

__host__
void blake_cpu_init(int thr_id, int threads)
{
	cudaMemcpyToSymbol(c_sigma, host_sigma, sizeof(host_sigma), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_u512, host_u512, sizeof(host_u512), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_IV, h_IV, sizeof(h_IV), 0, cudaMemcpyHostToDevice));
}

__host__
void blake_cpu_setBlock_185(void *pdata)
{
	unsigned char PaddedMessage[256];
	memcpy(PaddedMessage, pdata, 185);
	memset(PaddedMessage+185, 0, 71);
	PaddedMessage[185] = 0x80;
	PaddedMessage[239] = 1;
	PaddedMessage[254] = 0x05;
	PaddedMessage[255] = 0xC8;

	cudaMemcpyToSymbol(c_PaddedMessage, PaddedMessage, 32*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__
void blake_cpu_hash_185( int thr_id, int threads, uint32_t startNonce, uint32_t *d_outputHash, uint32_t *d_signature, uint32_t *d_hashwholeblock )
{
	const int threadsperblock = 64;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	blake_gpu_hash_185<<<grid, block, shared_size>>>(threads, startNonce, d_outputHash, d_signature, d_hashwholeblock);

	MyStreamSynchronize(NULL, 0, thr_id);
}
