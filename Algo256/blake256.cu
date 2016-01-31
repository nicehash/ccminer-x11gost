/**
 * Faster Blake-256 8round Cuda Kernel TEMPLATE
 * Based upon Blake-256 implementation of
 * Tanguy Pruvot - Nov. 2014
 */

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
#include <stdint.h>
#include <memory.h>
}

/* threads per block */
#define TPB 512

/* hash by cpu with blake 256 */
extern "C" void blake256hash(void *output, const void *input){

}

#include "cuda_helper.h"

__constant__ uint32_t d_data[14];

/* 8 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];


__global__
void blake256_gpu_hash_16_8(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce,const uint32_t highTarget){

}


__host__
static uint32_t blake256_cpu_hash_16(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint32_t highTarget){

}

__host__
static void blake256mid(uint32_t *output, const uint32_t *input){

}

__host__
void blake256_cpu_setBlock_16(uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget){

}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){
	return 0;
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const uint32_t targetHigh = ptarget[6];
	uint32_t _ALIGN(32) endiandata[20];
	uint32_t _ALIGN(32) midstate[8];
	uint32_t intensity = (device_sm[device_map[thr_id]] > 500) ? 30 : 28;
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1U << intensity);
	
	throughput = min(throughput, max_nonce - first_nonce);

	int rc = 0;

	if (!init[thr_id]){
		CUDA_CALL_OR_RET_X(cudaSetDevice(device_map[thr_id]),0);
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], sizeof(uint32_t)), 0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], sizeof(uint32_t)), 0);
		init[thr_id] = true;
	}

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);
	blake256mid(midstate, endiandata);
	blake256_cpu_setBlock_16(&pdata[16], midstate, ptarget);

	uint32_t foundNonce = UINT32_MAX;
	cudaMemset(d_resNonce[thr_id], 0xffffffff, sizeof(uint32_t));
	do {
		foundNonce = blake256_cpu_hash_16(thr_id, throughput, pdata[19], targetHigh);
		if (foundNonce != UINT32_MAX){
			rc = 1;
			*hashes_done = pdata[19] - first_nonce + throughput;
			pdata[19] = foundNonce;
			return rc;
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce;
	return rc;
}

// cleanup
extern "C" void free_blake256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
