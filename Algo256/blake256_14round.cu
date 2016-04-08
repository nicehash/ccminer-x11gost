/**
 * 14-round Blake-256 Cuda Kernel (Tested on SM 5.2) for SaffronCoin
 * Provos Alexis - April 2016
 *
 * Based on blake256 ccminer implementation of
 * Tanguy Pruvot / SP - Jan 2016
 *
 *
 * April 2016: +10.2% speed increase: 1396Mh/s -> 1539Mh/s on GTX970 at 1252MHz
 */

#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

/* threads per block and nonces per thread */
#define TPB 768
#define NPT 192
/* max count of found nonces in one call */
#define NBN 1

/* hash by cpu with blake 256 */
extern "C" void blake256_14roundHash(void *output, const void *input)
{
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

__constant__ uint32_t _ALIGN(16) d_data[21];

/* 8 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

#define GS(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ z[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c],12); \
	v[a] += (m[y] ^ z[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
	}

#define GS2(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1) { \
	v[ a]+= (m[ x] ^ z[ y]) + v[ b];		v[a1]+= (m[x1] ^ z[y1]) + v[b1];\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x1032);	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x1032);\
	v[ c]+= v[ d];					v[c1]+= v[d1];\
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);		v[b1] = ROTR32(v[b1] ^ v[c1], 12);\
	v[ a]+= (m[ y] ^ z[ x]) + v[ b];		v[a1]+= (m[y1] ^ z[x1]) + v[b1];\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x0321);	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x0321);\
	v[ c]+= v[ d];					v[c1]+= v[d1];\
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);		v[b1] = ROTR32(v[b1] ^ v[c1], 7);\
}

#define hostGS(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ z[y]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ z[x]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}
	
/* ############################################################################################################################### */
/* Precalculated 1st 64-bytes block (midstate) method */

__global__ __launch_bounds__(TPB,1)
void blake256_14round_gpu_hash_16(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint32_t highTarget){
	uint32_t v[16];
	uint4 tmp[16];

	const uint64_t thread   = blockDim.x * blockIdx.x + threadIdx.x;
	const uint64_t step     = gridDim.x * blockDim.x;
	const uint64_t maxNonce = startNonce + threads;

	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

//PREFETCH
	#pragma unroll 4
	for(int i=0;i<4;i++){
		tmp[i] = *(uint4*)&d_data[i<<2];
	}

	uint32_t m[16] = {
				  d_data[16],	d_data[17],	d_data[18],	0,
				  0x80000000,	0,		0,		0,
				  0,		0,		0,		0,
				  0,		1,		0,		640
	};		

	const uint32_t h7 = d_data[19];
	const uint32_t h6 = d_data[20];
//END OF PREFETCH
	for(uint64_t m3 = startNonce + thread; m3<maxNonce;m3+=step){

		m[3]  = m3;
		
		#pragma unroll 4
		for(int i=0;i<4;i++){
			*(uint4*)&v[i<<2] = tmp[ i];
		}
		v[ 1]+= (m[3] ^ z[2]) + v[ 5];
		v[13] = __byte_perm(v[13] ^ v[ 1],0, 0x0321);
		v[ 9]+= v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[ 9], 7);
		
		GS2(	0, 5,10,15, 8, 9,	1, 6,11,12,10,11);GS2(	2, 7, 8,13,12,13,	3, 4, 9,14,14,15);
		GS2(	0, 4, 8,12,14,10,	1, 5, 9,13, 4, 8);GS2(	2, 6,10,14, 9,15,	3, 7,11,15,13, 6);
		GS2(	0, 5,10,15, 1,12,	1, 6,11,12, 0, 2);GS2(	2, 7, 8,13,11, 7,	3, 4, 9,14, 5, 3);
		GS2(	0, 4, 8,12,11, 8,	1, 5, 9,13,12, 0);GS2(	2, 6,10,14, 5, 2,	3, 7,11,15,15,13);
		GS2(	0, 5,10,15,10,14,	1, 6,11,12, 3, 6);GS2(	2, 7, 8,13, 7, 1,	3, 4, 9,14, 9, 4);
		GS2(	0, 4, 8,12, 7, 9,	1, 5, 9,13, 3, 1);GS2(	2, 6,10,14,13,12,	3, 7,11,15,11,14);
		GS2(	0, 5,10,15, 2, 6,	1, 6,11,12, 5,10);GS2(	2, 7, 8,13, 4, 0,	3, 4, 9,14,15, 8);
		GS2(	0, 4, 8,12, 9, 0,	1, 5, 9,13, 5, 7);GS2(	2, 6,10,14, 2, 4,	3, 7,11,15,10,15);
		GS2(	0, 5,10,15,14, 1,	1, 6,11,12,11,12);GS2(	2, 7, 8,13, 6, 8,	3, 4, 9,14, 3,13);
		GS2(	0, 4, 8,12, 2,12,	1, 5, 9,13, 6,10);GS2(	2, 6,10,14, 0,11,	3, 7,11,15, 8, 3);
		GS2(	0, 5,10,15, 4,13,	1, 6,11,12, 7, 5);GS2(	2, 7, 8,13,15,14,	3, 4, 9,14, 1, 9);
		GS2(	0, 4, 8,12,12, 5,	1, 5, 9,13, 1,15);GS2(	2, 6,10,14,14,13,	3, 7,11,15, 4,10);
		GS2(	0, 5,10,15, 0, 7,	1, 6,11,12, 6, 3);GS2(	2, 7, 8,13, 9, 2,	3, 4, 9,14, 8,11);
		GS2(	0, 4, 8,12,13,11,	1, 5, 9,13, 7,14);GS2(	2, 6,10,14,12, 1,	3, 7,11,15, 3, 9);
		GS2(	0, 5,10,15, 5, 0,	1, 6,11,12,15, 4);GS2(	2, 7, 8,13, 8, 6,	3, 4, 9,14, 2,10);
		GS2(	0, 4, 8,12, 6,15,	1, 5, 9,13,14, 9);GS2(	2, 6,10,14,11, 3,	3, 7,11,15, 0, 8);
		GS2(	0, 5,10,15,12, 2,	1, 6,11,12,13, 7);GS2(	2, 7, 8,13, 1, 4,	3, 4, 9,14,10, 5);
		GS2(	0, 4, 8,12,10, 2,	1, 5, 9,13, 8, 4);GS2(	2, 6,10,14, 7, 6,	3, 7,11,15, 1, 5);
		GS2(	0, 5,10,15,15,11,	1, 6,11,12, 9,14);GS2(	2, 7, 8,13, 3,12,	3, 4, 9,14,13, 0);
		GS2(	0, 4, 8,12, 0, 1,	1, 5, 9,13, 2, 3);GS2(	2, 6,10,14, 4, 5,	3, 7,11,15, 6, 7);
		GS2(	0, 5,10,15, 8, 9,	1, 6,11,12,10,11);GS2(	2, 7, 8,13,12,13,	3, 4, 9,14,14,15);
		GS2(	0, 4, 8,12,14,10,	1, 5, 9,13, 4, 8);GS2(	2, 6,10,14, 9,15,	3, 7,11,15,13, 6);
		GS2(	0, 5,10,15, 1,12,	1, 6,11,12, 0, 2);GS2(	2, 7, 8,13,11, 7,	3, 4, 9,14, 5, 3);
		GS2(	0, 4, 8,12,11, 8,	1, 5, 9,13,12, 0);GS2(	2, 6,10,14, 5, 2,	3, 7,11,15,15,13);
		GS2(	0, 5,10,15,10,14,	1, 6,11,12, 3, 6);GS2(	2, 7, 8,13, 7, 1,	3, 4, 9,14, 9, 4);
		GS2(	0, 4, 8,12, 7, 9,	1, 5, 9,13, 3, 1);GS2(	2, 6,10,14,13,12,	3, 7,11,15,11,14);
//		GS2(	0, 5,10,15, 2, 6,	1, 6,11,12, 5,10);
		v[ 0]+= (m[ 2] ^ z[ 6]) + v[ 5];
		v[ 1]+= (m[ 5] ^ z[10]) + v[ 6];
		v[ 2] += (m[4] ^ z[ 0]) + v[ 7];

		v[15] = __byte_perm(v[15] ^ v[ 0], 0, 0x1032);
		v[12] = __byte_perm(v[12] ^ v[ 1], 0, 0x1032);
		v[13] = __byte_perm(v[13] ^ v[ 2], 0, 0x1032);
		v[10]+= v[15];
		v[11]+= v[12];
		v[ 8] += v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 6] = ROTR32(v[ 6] ^ v[11], 12);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
		v[ 0]+= (m[ 6] ^ z[ 2]) + v[ 5];
		v[ 1]+= (m[10] ^ z[ 5]) + v[ 6];
		v[ 2]+= (m[ 0] ^ z[ 4]) + v[ 7];
		v[15] = __byte_perm(v[15] ^ v[ 0], 0, 0x0321);
		v[12] = __byte_perm(v[12] ^ v[ 1], 0, 0x0321);
		v[13] = __byte_perm(v[13] ^ v[ 2], 0, 0x0321);
		v[11]+= v[12];
		v[ 8] += v[13];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 7);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);
		// only compute h6 & 7
		if ((v[15]^h7) == v[ 7]){
			v[3] += (m[15] ^ z[8]) + v[4];
			v[14] = __byte_perm(v[14] ^ v[3], 0, 0x1032);
			v[9] += v[14];
			v[4] = ROTR32(v[4] ^ v[9], 12);
			v[3] += (m[ 8] ^ z[15]) + v[4];
			v[14] = __byte_perm(v[14] ^ v[3], 0, 0x0321);			
			if(cuda_swab32((v[ 6]^h6^v[14])) <= highTarget) {
#if NBN == 2
				if (resNonce[0] != UINT32_MAX)
					resNonce[1] = m[ 3];
				else
					resNonce[0] = m[ 3];
#else
				resNonce[0] = m[ 3];
#endif
				return;//<-- this may cause a problem on extranonce if the extranonce is on position current_nonce + X * step where X=[1,2,3..,N]
			}
		}
	}
}

__host__
void blake256_14round_cpu_setBlock_16(const uint32_t *pendd, const uint32_t *input){

	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};
		
	uint32_t _ALIGN(64) v[16];
	sph_blake256_context ctx;
	sph_blake256_set_rounds(14);
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 64);
	//memcpy(data, (void*)ctx.H, 32);
	v[ 0] = ctx.H[ 0];
	v[ 1] = ctx.H[ 1];
	v[ 2] = ctx.H[ 2];
	v[ 3] = ctx.H[ 3];
	v[ 4] = ctx.H[ 4];
	v[ 5] = ctx.H[ 5];
	v[ 6] = ctx.H[ 6];
	v[ 7] = ctx.H[ 7];
	v[ 8] = z[ 0];
	v[ 9] = z[ 1];
	v[10] = z[ 2];
	v[11] = z[ 3];
	v[12] = z[ 4] ^ 640;
	v[13] = z[ 5] ^ 640;
	v[14] = z[ 6];
	v[15] = z[ 7];	

	const uint32_t h[2]	= { 	v[ 7],		v[ 6]};
	
	const uint32_t m[16] 	= { 	pendd[ 0],	pendd[ 1], pendd[ 2],	0,
					0x80000000,	0,		0,		0,
					0,		0,		0,		0,
					0,		1,		0,		640
				};

	hostGS(	0, 4, 8,12, 0, 1);	hostGS(	2, 6,10,14, 4, 5);	hostGS(	3, 7,11,15, 6, 7);
	
	v[ 1]+= (m[ 2] ^ z[ 3]) + v[ 5];
	v[13] = ROTR32(v[13] ^ v[ 1],16);
	v[ 9] += v[13];
	v[ 5] = ROTR32(v[ 5] ^ v[ 9],12);
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, v, 16*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, m, 3*sizeof(uint32_t), 16*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, h, 2*sizeof(uint32_t), 19*sizeof(uint32_t), cudaMemcpyHostToDevice));
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake256_14round(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	uint64_t targetHigh = ((uint64_t*)ptarget)[3];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 30 : 26;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);
	
	int rc = 0;

	if (opt_benchmark) {
		targetHigh = 0x1ULL << 32;
		ptarget[6] = swab32(0xff);
	}

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_14round_cpu_setBlock_16(&pdata[16], endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	do {
		// GPU HASH (second block only, first is midstate)
		blake256_14round_gpu_hash_16  <<<grid, block>>> (throughput, pdata[19], d_resNonce[thr_id], targetHigh);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (h_resNonce[thr_id][0] != UINT32_MAX){
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[6];

			for (int k=16; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			be32enc(&endiandata[19], h_resNonce[thr_id][0]);
			blake256_14roundHash(vhashcpu, endiandata);

			if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget)){
				rc = 1;
				work_set_target_ratio(work, vhashcpu);
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = h_resNonce[thr_id][0];
#if NBN > 1
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					be32enc(&endiandata[19], h_resNonce[thr_id][1]);
					blake256_14roundHash(vhashcpu, endiandata);
					if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget)) {
						pdata[21] = h_resNonce[thr_id][1];
						if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio) {
							work_set_target_ratio(work, vhashcpu);
							xchg(pdata[21], pdata[19]);
						}
						rc = 2;
					}
					applog(LOG_BLUE, "1:%x 2:%x", h_resNonce[thr_id][0], h_resNonce[thr_id][1]);					
				}
#endif
				return rc;
			}
			else{
				applog_hash((uchar*)ptarget);
				applog_compare_hash((uchar*)vhashcpu, (uchar*)ptarget);
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + pdata[19]);

	*hashes_done = pdata[19] - first_nonce;

	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return rc;
}

// cleanup
extern "C" void free_blake256_14round(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

