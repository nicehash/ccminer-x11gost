#include "miner.h"
#include "bignum_fp.h"
#include "cuda_helper.h"

__constant__ uint32_t d_privkey_data[8];
__constant__ uint32_t d_kinv_data[8];

__constant__ uint32_t d_m_data[8] = {
	0xd0364141, 0xbfd25e8c, 0xaf48a03b, 0xbaaedce6,
	0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff
};

__constant__ uint32_t d_mu_data[9] = {
	0x2fc9bec0, 0x402da173, 0x50b75fc4, 0x45512319,
	0x00000001, 0x00000000, 0x00000000, 0x00000000,
	0x00000001
};

__global__
void spreadx11_sign_gpu(int threads, uint32_t startNonce, uint32_t *g_sha256hash, uint32_t *g_signature )
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		fp_int hash, result, m, mu, privkey, kinv;
		unsigned char *sha256hash = (unsigned char *)&g_sha256hash[thread<<3];
		uint32_t *signature = (uint32_t *)&g_signature[thread<<3];

		fp_zero(&result);
		fp_read_unsigned_bin(&hash, sha256hash, 32);

		privkey.used = 8;
		kinv.used = 8;
		m.used = 8;
		mu.used = 9;

		#pragma unroll 8
		for( int i = 0; i < 8; i++ ) {
			privkey.dp[i] = d_privkey_data[i];
			kinv.dp[i] = d_kinv_data[i];
			m.dp[i] = d_m_data[i];
			mu.dp[i] = d_mu_data[i];
			signature[i] = 0;
		}
		mu.dp[8] = d_mu_data[8];
		m.dp[8] = 0;

		#pragma unroll
		for( int i = 9; i < FP_SIZE; i++ ) {

			m.dp[i] = 0;
			mu.dp[i] = 0;
		}

		fp_add(&hash, &privkey, &result);
		fp_mul(&result, &kinv, &result);
		fp_reduce(&result, &m, &mu);
		fp_to_unsigned_bin(&result, &((unsigned char *)signature)[32-(fp_count_bits(&result)+7)/8]);
	}
}

__host__
void spreadx11_sign_cpu_init(int thr_id, int throughput) { }

__host__
void spreadx11_sign_cpu_setInput(struct work *work)
{
	fp_int fp_privkey, fp_kinv;

	fp_read_unsigned_bin(&fp_privkey, work->privkey, 32);
	fp_read_unsigned_bin(&fp_kinv, work->kinv, 32);

	cudaMemcpyToSymbol(d_privkey_data, fp_privkey.dp, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_kinv_data, fp_kinv.dp, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__
void spreadx11_sign_cpu(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	spreadx11_sign_gpu<<<grid, block>>>(threads, startNonce, d_hash, d_signature);

	MyStreamSynchronize(NULL, 0, thr_id);
}

