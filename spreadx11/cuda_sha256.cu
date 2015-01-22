#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"

__constant__ uint32_t c_PaddedMessage88[32]; // padded message, 88 bytes + padding to 128 bytes

// precalculated state for sha256double_88
static __constant__ uint32_t d_state[8];

// init vector
static __constant__ uint32_t d_IV[8];
static const uint32_t h_IV[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};
// round constants
static __constant__ uint32_t d_K[64];
static const uint32_t h_K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
// padding for 400000 bytes of input data
static __constant__ uint32_t padding400000[16] = {
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x0030d400
};

// padding for 32 bytes of input data
static __constant__ uint32_t padding32[16] = {
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000100
};

__device__ __host__ __forceinline__
uint32_t ROTR32( const uint32_t val, const size_t offset )
{
#if __CUDA_ARCH__ >= 320
	return __funnelshift_r(val, val, offset);
#else
	return (val >> offset) | (val << (32-offset));
#endif
}
/*
static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}
*/
__host__
uint32_t bswap32( const uint32_t x )
{
	return
		((x & 0xff000000) >> 24 ) |
		((x & 0x00ff0000) >> 8 ) |
		((x & 0x0000ff00) << 8 ) |
		((x & 0x000000ff) << 24 );
}

__device__ __host__ __forceinline__
uint32_t BSG2_0( const uint32_t x )
{
	return
		ROTR32(x, 2) ^
		ROTR32(x, 13) ^
		ROTR32(x, 22);
}

__device__ __host__ __forceinline__
uint32_t BSG2_1( const uint32_t x )
{
	return
		ROTR32(x, 6) ^
		ROTR32(x, 11) ^
		ROTR32(x, 25);
}

__device__ __host__ __forceinline__
uint32_t SSG2_0( const uint32_t x )
{
	return
		ROTR32(x, 7) ^
		ROTR32(x, 18) ^
		(x >> 3);
}

__device__ __host__ __forceinline__
uint32_t SSG2_1( const uint32_t x )
{
	return
		ROTR32(x, 17) ^
		ROTR32(x, 19) ^
		(x >> 10);
}

__device__ __host__ __forceinline__
uint32_t CH( const uint32_t x, const uint32_t y, const uint32_t z )
{
	return ((y ^ z) & x) ^ z;
}

__device__ __host__ __forceinline__
uint32_t MAJ( const uint32_t x, const uint32_t y, const uint32_t z )
{
	return (x & y) | ((x | y) & z);
}


__device__ __host__ __forceinline__
void sha2_step1( uint32_t *W, const uint32_t *K, const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d,
	const uint32_t e, const uint32_t f, const uint32_t g, uint32_t &h, const uint32_t *in, const uint32_t pc, const uint32_t pcount )
{
	W[pc] = in[pc];
	uint32_t t1 =
		h + BSG2_1(e) + CH(e, f, g) +
		K[pcount + pc] + W[pc];
	d += t1;
	h = t1 + BSG2_0(a) + MAJ(a, b, c);
}

__device__ __host__ __forceinline__
void sha2_step2( uint32_t *W, const uint32_t *K, const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d,
	const uint32_t e, const uint32_t f, const uint32_t g, uint32_t &h, const uint32_t *in, const uint32_t pc, const uint32_t pcount )
{
	uint32_t t1;
	W[pc] = SSG2_1(W[(pc - 2) & 0x0F])
		+ W[(pc - 7) & 0x0F]
		+ SSG2_0(W[(pc - 15) & 0x0F]) + W[pc];
	t1 = h + BSG2_1(e) + CH(e, f, g)
		+ K[pcount + (pc)] + W[(pc)];
	d += t1;
	h = t1 + BSG2_0(a) + MAJ(a, b, c);
}

__device__
void sha2_transform( const uint32_t *in, uint32_t *state )
{
	uint32_t a, b, c, d, e, f, g, h;
	uint32_t w[16];
	uint32_t pcount;

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	e = state[4];
	f = state[5];
	g = state[6];
	h = state[7];
	pcount = 0;

	sha2_step1(w, d_K, a, b, c, d, e, f, g, h, in,  0, pcount);
	sha2_step1(w, d_K, h, a, b, c, d, e, f, g, in,  1, pcount);
	sha2_step1(w, d_K, g, h, a, b, c, d, e, f, in,  2, pcount);
	sha2_step1(w, d_K, f, g, h, a, b, c, d, e, in,  3, pcount);
	sha2_step1(w, d_K, e, f, g, h, a, b, c, d, in,  4, pcount);
	sha2_step1(w, d_K, d, e, f, g, h, a, b, c, in,  5, pcount);
	sha2_step1(w, d_K, c, d, e, f, g, h, a, b, in,  6, pcount);
	sha2_step1(w, d_K, b, c, d, e, f, g, h, a, in,  7, pcount);
	sha2_step1(w, d_K, a, b, c, d, e, f, g, h, in,  8, pcount);
	sha2_step1(w, d_K, h, a, b, c, d, e, f, g, in,  9, pcount);
	sha2_step1(w, d_K, g, h, a, b, c, d, e, f, in, 10, pcount);
	sha2_step1(w, d_K, f, g, h, a, b, c, d, e, in, 11, pcount);
	sha2_step1(w, d_K, e, f, g, h, a, b, c, d, in, 12, pcount);
	sha2_step1(w, d_K, d, e, f, g, h, a, b, c, in, 13, pcount);
	sha2_step1(w, d_K, c, d, e, f, g, h, a, b, in, 14, pcount);
	sha2_step1(w, d_K, b, c, d, e, f, g, h, a, in, 15, pcount);

	for (pcount = 16; pcount < 64; pcount += 16) {

		sha2_step2(w, d_K, a, b, c, d, e, f, g, h, in,  0, pcount);
		sha2_step2(w, d_K, h, a, b, c, d, e, f, g, in,  1, pcount);
		sha2_step2(w, d_K, g, h, a, b, c, d, e, f, in,  2, pcount);
		sha2_step2(w, d_K, f, g, h, a, b, c, d, e, in,  3, pcount);
		sha2_step2(w, d_K, e, f, g, h, a, b, c, d, in,  4, pcount);
		sha2_step2(w, d_K, d, e, f, g, h, a, b, c, in,  5, pcount);
		sha2_step2(w, d_K, c, d, e, f, g, h, a, b, in,  6, pcount);
		sha2_step2(w, d_K, b, c, d, e, f, g, h, a, in,  7, pcount);
		sha2_step2(w, d_K, a, b, c, d, e, f, g, h, in,  8, pcount);
		sha2_step2(w, d_K, h, a, b, c, d, e, f, g, in,  9, pcount);
		sha2_step2(w, d_K, g, h, a, b, c, d, e, f, in, 10, pcount);
		sha2_step2(w, d_K, f, g, h, a, b, c, d, e, in, 11, pcount);
		sha2_step2(w, d_K, e, f, g, h, a, b, c, d, in, 12, pcount);
		sha2_step2(w, d_K, d, e, f, g, h, a, b, c, in, 13, pcount);
		sha2_step2(w, d_K, c, d, e, f, g, h, a, b, in, 14, pcount);
		sha2_step2(w, d_K, b, c, d, e, f, g, h, a, in, 15, pcount);
	}
	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

__host__
void host_sha2_transform( const uint32_t *in, uint32_t *state )
{
	uint32_t a, b, c, d, e, f, g, h;
	uint32_t w[16];
	uint32_t pcount;

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	e = state[4];
	f = state[5];
	g = state[6];
	h = state[7];
	pcount = 0;

	sha2_step1(w, h_K, a, b, c, d, e, f, g, h, in,  0, pcount);
	sha2_step1(w, h_K, h, a, b, c, d, e, f, g, in,  1, pcount);
	sha2_step1(w, h_K, g, h, a, b, c, d, e, f, in,  2, pcount);
	sha2_step1(w, h_K, f, g, h, a, b, c, d, e, in,  3, pcount);
	sha2_step1(w, h_K, e, f, g, h, a, b, c, d, in,  4, pcount);
	sha2_step1(w, h_K, d, e, f, g, h, a, b, c, in,  5, pcount);
	sha2_step1(w, h_K, c, d, e, f, g, h, a, b, in,  6, pcount);
	sha2_step1(w, h_K, b, c, d, e, f, g, h, a, in,  7, pcount);
	sha2_step1(w, h_K, a, b, c, d, e, f, g, h, in,  8, pcount);
	sha2_step1(w, h_K, h, a, b, c, d, e, f, g, in,  9, pcount);
	sha2_step1(w, h_K, g, h, a, b, c, d, e, f, in, 10, pcount);
	sha2_step1(w, h_K, f, g, h, a, b, c, d, e, in, 11, pcount);
	sha2_step1(w, h_K, e, f, g, h, a, b, c, d, in, 12, pcount);
	sha2_step1(w, h_K, d, e, f, g, h, a, b, c, in, 13, pcount);
	sha2_step1(w, h_K, c, d, e, f, g, h, a, b, in, 14, pcount);
	sha2_step1(w, h_K, b, c, d, e, f, g, h, a, in, 15, pcount);

	for (pcount = 16; pcount < 64; pcount += 16) {

		sha2_step2(w, h_K, a, b, c, d, e, f, g, h, in,  0, pcount);
		sha2_step2(w, h_K, h, a, b, c, d, e, f, g, in,  1, pcount);
		sha2_step2(w, h_K, g, h, a, b, c, d, e, f, in,  2, pcount);
		sha2_step2(w, h_K, f, g, h, a, b, c, d, e, in,  3, pcount);
		sha2_step2(w, h_K, e, f, g, h, a, b, c, d, in,  4, pcount);
		sha2_step2(w, h_K, d, e, f, g, h, a, b, c, in,  5, pcount);
		sha2_step2(w, h_K, c, d, e, f, g, h, a, b, in,  6, pcount);
		sha2_step2(w, h_K, b, c, d, e, f, g, h, a, in,  7, pcount);
		sha2_step2(w, h_K, a, b, c, d, e, f, g, h, in,  8, pcount);
		sha2_step2(w, h_K, h, a, b, c, d, e, f, g, in,  9, pcount);
		sha2_step2(w, h_K, g, h, a, b, c, d, e, f, in, 10, pcount);
		sha2_step2(w, h_K, f, g, h, a, b, c, d, e, in, 11, pcount);
		sha2_step2(w, h_K, e, f, g, h, a, b, c, d, in, 12, pcount);
		sha2_step2(w, h_K, d, e, f, g, h, a, b, c, in, 13, pcount);
		sha2_step2(w, h_K, c, d, e, f, g, h, a, b, in, 14, pcount);
		sha2_step2(w, h_K, b, c, d, e, f, g, h, a, in, 15, pcount);
	}
	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

__global__
void spreadx11_sha256_gpu_hash_wholeblock( int threads, uint32_t startNonce, uint32_t *g_input, uint32_t *g_hash, uint32_t *g_signature )
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t state[8];
		uint32_t input[16];
		uint32_t *outputHash = (uint32_t *)&g_hash[thread<<3];
		uint8_t *signature = (uint8_t *)&g_signature[thread<<3];

#pragma unroll 8
		for( int i=0; i < 8; i++ ) state[i] = d_IV[i];

/*
	Input is as follows, we need to replace nonce and MinerSignature for each hash.
	This means the first two blocks need to be handled separately.

	4 bytes     (nNonce & ~NONCE_MASK)
	8 bytes     nTime;
	65 bytes    MinerSignature;
	4 bytes     nVersion;
	and so on....
*/

		// we process the same 200000 bytes of data twice
		for( int round = 0; round < 2; round++ ) {

			// insert nonce
			input[0] = startNonce + (thread << 6);

			// read the rest of the block
#pragma unroll 15
			for( int i=1; i < 16; i++ ) input[i] = g_input[i];

			// first 19 bytes of signature go into the first block
#pragma unroll 19
			for( int i = 0; i < 19; i++ ) ((uint8_t *)input)[45+i] = signature[i];

			// endian swap the block and transform
#pragma unroll 16
			for( int i=0; i < 16; i++ ) input[i] = cuda_swab32(input[i]);
			sha2_transform(input, state);

			// read the second block
#pragma unroll 16
			for( int i=0; i < 16; i++ ) input[i] = g_input[i+16];

			// replace in the remaining bytes of the signature
#pragma unroll 13
			for( int i = 0; i < 13; i++ ) ((uint8_t *)input)[i] = signature[i+19];

			// endian swap the second block and transform
#pragma unroll 16
			for( int i=0; i < 16; i++ ) input[i] = cuda_swab32(input[i]);
			sha2_transform(input, state);

			// transform the remaining blocks as they are
			for( int block = 2; block < 3125; block++ ) {

#pragma unroll 16
				for( int i=0; i < 16; i++ ) input[i] = cuda_swab32(g_input[i+(block << 4)]);

				sha2_transform(input, state);
			}
		}

		// load and transform the padding block
#pragma unroll 16
		for( int i = 0; i < 16; i++ ) input[i] = padding400000[i];
		sha2_transform(input, state);

#pragma unroll 8
		for( int i=0; i < 8; i++ ) outputHash[i] = cuda_swab32(state[i]);
	}
}

__global__
void spreadx11_sha256double_gpu_hash_88( int threads, uint32_t startNonce, uint32_t *g_hash )
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t state[8];
		uint32_t input[16];
		uint32_t nonce = startNonce + (thread << 6);
		uint32_t *outputHash = (uint32_t *)&g_hash[thread<<3];

// load precalculated state from the first block
#pragma unroll 8
		for( int i = 0; i < 8; i++ ) state[i] = d_state[i];

#pragma unroll 16
		for( int i = 0; i < 16; i++ ) input[i] = cuda_swab32(c_PaddedMessage88[i+16]);
		input[5] = cuda_swab32(nonce);
		sha2_transform(input, state);

// finished hashing the input data, next step: hash the first result hash

#pragma unroll 8
		for( int i = 0; i < 8; i++ ) {
			input[i] = state[i];
			input[i+8] = padding32[i];
			state[i] = d_IV[i];
		}

		sha2_transform(input, state);

#pragma unroll 8
		for( int i = 0; i < 8; i++ ) outputHash[i] = cuda_swab32(state[i]);
	}
}

__host__
void spreadx11_sha256_cpu_init( int thr_id, int threads )
{
	cudaMemcpyToSymbol(d_K, h_K, sizeof(h_K), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_IV, h_IV, sizeof(h_IV), 0, cudaMemcpyHostToDevice);
}

__host__
void spreadx11_sha256double_cpu_hash_88(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	spreadx11_sha256double_gpu_hash_88<<<grid, block>>>(threads, startNonce, d_hash);

	cudaDeviceSynchronize();
}

__host__
void spreadx11_sha256double_setBlock_88( void *data )
{
	unsigned char PaddedMessage[128];
	uint32_t tmpstate[8], tmpinput[16];

	memcpy(PaddedMessage, data, 88);
	memset(PaddedMessage+88, 0, 40);
	PaddedMessage[ 88] = 0x80;
	PaddedMessage[126] = 0x02;
	PaddedMessage[127] = 0xC0;

	// we can transform the first block here and start the CUDA kernel with the precalculated state
	for( int i = 0; i < 8; i++ ) tmpstate[i] = h_IV[i];
	for( int i = 0; i < 16; i++ ) tmpinput[i] = bswap32(((uint32_t *)PaddedMessage)[i]);
	host_sha2_transform(tmpinput, tmpstate);

	cudaMemcpyToSymbol(d_state, tmpstate, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_PaddedMessage88, PaddedMessage, 128, 0, cudaMemcpyHostToDevice);
}

__host__
void spreadx11_sha256_setBlock_wholeblock(struct work *work, uint32_t *d_wholeblock)
{
	unsigned char paddedmessage[200000];

/*
	We need a hash of:
		offset  size    field
		0       4       nNonce & 0xffffffc0
		4       8       nTime
		12      65      MinerSignature
		77      4       nVersion
		81      32      hashPrevBlock
		113     32      hashMerkleRoot
		145     4       nBits
		149     4       nHeight
		153     x       transactions, variable length
		padding calculated backwards from the copy of hashPrevBlock at the end
		199968  32      hashPrevBlock

	We have in work->longdata:
		offset  size    field           data
		0	    4	    nVersion	    02000000
		4	    32	    hashPrevBlock   83ac9d7e96c4bb15d1a9a2fe3dbc81747768125009efc14e3824fbacaec65fad
		36	    32	    HashMerkleRoot	4e06a8366a959df705ea34dfe7eafea0d8a9e6fe3ba6b83afddb145b27da2362
		68	    8	    nTime		    ed71585400000000
		76	    4	    nBits		    19850d1d
		80	    4	    nHeight		    33d20100
		84	    4	    nNonce		    00000000
		88	    32	    hashWholeBlock	0000000000000000000000000000000000000000000000000000000000000000
		120	    33	    MinerSignature1	1cad95a3e3e3598b74a1b8e2e3c013ababf6c45c83534029dbe8b4698916f78255
		153	    32	    MinerSignature2	0000000000000000000000000000000000000000000000000000000000000000
*/

	memcpy(paddedmessage+0, work->longdata+84, 4); // nNonce
	memcpy(paddedmessage+4, work->longdata+68, 8); // nTime
	memcpy(paddedmessage+12, work->longdata+120, 65); // MinerSignature
	memcpy(paddedmessage+77, work->longdata+0, 4); // nVersion
	memcpy(paddedmessage+81, work->longdata+4, 32); // hashPrevBlock
	memcpy(paddedmessage+113, work->longdata+36, 32); // HashMerkleRoot
	memcpy(paddedmessage+145, work->longdata+76, 4); // nBits
	memcpy(paddedmessage+149, work->longdata+80, 4); // nHeight
	memcpy(paddedmessage+153, work->tx, work->txsize); // tx

	// the total amount of bytes in our data
	int blocksize = 153 + work->txsize;

	// pad the block with 0x07 bytes to make it a multiple of uint32_t
	while( blocksize & 3 ) paddedmessage[blocksize++] = 0x07;

	// figure out the offsets for the padding
	uint32_t *pFillBegin = (uint32_t*)&paddedmessage[blocksize];
	uint32_t *pFillEnd = (uint32_t*)&paddedmessage[SPREAD_MAX_BLOCK_SIZE]; // FIXME: isn't this out of bounds by one... but it seems to work out...
	uint32_t *pFillFooter = pFillBegin > pFillEnd - 8 ? pFillBegin : pFillEnd - 8;

	memcpy(pFillFooter, work->longdata+4, (pFillEnd - pFillFooter)*4);
	for (uint32_t *pI = pFillFooter; pI < pFillEnd; pI++)
		*pI |= 1;

	for (uint32_t *pI = pFillFooter - 1; pI >= pFillBegin; pI--)
		pI[0] = pI[3]*pI[7];

	// copy the whole monstrosity into device memory for processing
	cudaMemcpy(d_wholeblock, paddedmessage, 200000, cudaMemcpyHostToDevice);
}

__host__
void spreadx11_sha256_cpu_hash_wholeblock(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_wholeblock)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	spreadx11_sha256_gpu_hash_wholeblock<<<grid, block>>>(threads, startNonce, d_wholeblock, d_hash, d_signature);

	cudaDeviceSynchronize();
}
