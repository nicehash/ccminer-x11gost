#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "uint256.h"

extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"

#define PROFILE 0
#if PROFILE == 1
#define PRINTTIME(s) do { \
	double duration; \
	gettimeofday(&tv_end, NULL); \
	duration = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec); \
	printf("%s: %.2f sec, %.2f MH/s\n", s, duration, (double)throughput / 1000000.0 / duration); \
	} while(0)
#else
#define PRINTTIME(s)
#endif

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_sha256hash[MAX_GPUS];
static uint32_t *d_signature[MAX_GPUS];
static uint32_t *d_hashwholeblock[MAX_GPUS];
static uint32_t *d_wholeblockdata[MAX_GPUS];

extern void spreadx11_sha256double_cpu_hash_88(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash);
extern void spreadx11_sha256double_setBlock_88(void *data);
extern void spreadx11_sha256_cpu_init( int thr_id, int throughput );

extern void spreadx11_sha256_cpu_hash_wholeblock(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_wholeblock);
extern void spreadx11_sha256_setBlock_wholeblock( struct work *work, uint32_t *d_wholeblock );

extern void spreadx11_sign_cpu_init( int thr_id, int throughput );
extern void spreadx11_sign_cpu_setInput( struct work *work );
extern void spreadx11_sign_cpu(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature);

extern void blake_cpu_init(int thr_id, int threads);
extern void blake_cpu_setBlock_185(void *pdata);
extern void blake_cpu_hash_185(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_hashwholeblock);

extern void quark_bmw512_cpu_init(int thr_id, int threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_groestl512_cpu_init(int thr_id, int threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, int threads);
extern void quark_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_keccak512_cpu_init(int thr_id, int threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_jh512_cpu_init(int thr_id, int threads);
extern void quark_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_luffaCubehash512_cpu_init(int thr_id, int threads);
extern void x11_luffaCubehash512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_shavite512_cpu_init(int thr_id, int threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern int x11_simd512_cpu_init(int thr_id, int threads);
extern void x11_simd512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void spreadx11_echo512_cpu_init(int thr_id, int threads);
extern void spreadx11_echo512_cpu_setTarget(void *ptarget);
extern uint32_t spreadx11_echo512_cpu_hash_64_final(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int *hashidx);

void hextobin(unsigned char *p, const char *hexstr, size_t len)
{
	char hex_byte[3];
	char *ep;

	hex_byte[2] = '\0';

	while (*hexstr && len) {
		if (!hexstr[1]) {
			applog(LOG_ERR, "hex2bin str truncated");
			return;
		}
		hex_byte[0] = hexstr[0];
		hex_byte[1] = hexstr[1];
		*p = (unsigned char) strtol(hex_byte, &ep, 16);
		if (*ep) {
			applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
			return;
		}
		p++;
		hexstr += 2;
		len--;
	}
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_spreadx11(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uchar *blocktemplate = work->longdata;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = (uint32_t *) &blocktemplate[84];
	uint32_t nonce = *pnonce;
	uint32_t first_nonce = nonce;

	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	// note: keep multiple of 64 to keep things simple with signatures
	int throughput = (int) device_intensity(thr_id, __func__, 1 << intensity); // 19=256*256*8;
	throughput = min(throughput, (int)(max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x000000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		// sha256 hashes used for signing, 32 bytes for every 64 nonces
		cudaMalloc(&d_sha256hash[thr_id], 32*(throughput>>6));
		// changing part of MinerSignature, 32 bytes for every 64 nonces
		cudaMalloc(&d_signature[thr_id], 32*(throughput>>6));
		// sha256 hashes for the whole block, 32 bytes for every 64 nonces
		cudaMalloc(&d_hashwholeblock[thr_id], 32*(throughput>>6));
		// single buffer to hold the padded whole block data
		cudaMalloc(&d_wholeblockdata[thr_id], 200000);
		// a 512-bit buffer for every nonce to hold the x11 intermediate hashes
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);

		spreadx11_sha256_cpu_init(thr_id, throughput);
		spreadx11_sign_cpu_init(thr_id, throughput);
		blake_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		//x11_luffa512_cpu_init(thr_id, throughput);
		//x11_cubehash512_cpu_init(thr_id, throughput);
		x11_luffaCubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		spreadx11_echo512_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	struct timeval tv_start;
#if PROFILE == 1
	struct timeval tv_end;
#endif

	spreadx11_sign_cpu_setInput(work);
	spreadx11_sha256_setBlock_wholeblock(work, d_wholeblockdata[thr_id]);
	spreadx11_sha256double_setBlock_88((void *)blocktemplate);

	blake_cpu_setBlock_185((void *)blocktemplate);

	spreadx11_echo512_cpu_setTarget(ptarget);

	do {
		int order = 0;

		gettimeofday(&tv_start, NULL);
		spreadx11_sha256double_cpu_hash_88(thr_id, throughput>>6, nonce, d_sha256hash[thr_id]);
		PRINTTIME("sha256 for signature");

		gettimeofday(&tv_start, NULL);
		spreadx11_sign_cpu(thr_id, throughput>>6, nonce, d_sha256hash[thr_id], d_signature[thr_id]);
		PRINTTIME("signing");

		gettimeofday(&tv_start, NULL);
		spreadx11_sha256_cpu_hash_wholeblock(thr_id, throughput>>6, nonce, d_hashwholeblock[thr_id], d_signature[thr_id], d_wholeblockdata[thr_id]);
		PRINTTIME("hashwholeblock");

		gettimeofday(&tv_start, NULL);
		blake_cpu_hash_185(thr_id, throughput, nonce, d_hash[thr_id], d_signature[thr_id], d_hashwholeblock[thr_id]);
		PRINTTIME("blake");

		gettimeofday(&tv_start, NULL);
		quark_bmw512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("bmw");

		gettimeofday(&tv_start, NULL);
		quark_groestl512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("groestl");

		gettimeofday(&tv_start, NULL);
		quark_skein512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("skein");

		gettimeofday(&tv_start, NULL);
		quark_jh512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("jh");

		gettimeofday(&tv_start, NULL);
		quark_keccak512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("keccak");

		gettimeofday(&tv_start, NULL);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("luffa-cube");

		gettimeofday(&tv_start, NULL);
		x11_shavite512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("shavite");

		gettimeofday(&tv_start, NULL);
		x11_simd512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		PRINTTIME("simd");

		gettimeofday(&tv_start, NULL);
		int winnerthread = 0;
		uint32_t foundNonce = spreadx11_echo512_cpu_hash_64_final(thr_id, throughput, nonce, NULL, d_hash[thr_id], &winnerthread);
		PRINTTIME("echo");

		if (foundNonce != UINT32_MAX)
		{
			uint32_t hash[8];
			char hexbuffer[SPREAD_MAX_BLOCK_SIZE*2];
			memset(hexbuffer, 0, sizeof(hexbuffer));

			applog(LOG_BLUE, "foundNonce=%x", foundNonce);
			for(size_t i = 0; i < work->txsize && i < SPREAD_MAX_BLOCK_SIZE; i++)
				sprintf(&hexbuffer[i*2], "%02x", work->tx[i]);

			uint32_t *resnonce = (uint32_t *)&work->longdata[84];
			uint32_t *reshashwholeblock = (uint32_t *)&work->longdata[88];
			uint32_t *ressignature = (uint32_t *)&work->longdata[153];
			uint32_t idx64 = winnerthread >> 6;

			applog(LOG_DEBUG,
				"Thread %d found a solution\n"
				"First nonce : %08x\n"
				"Found nonce : %08x\n"
				"Threadidx   : %d\n"
				"Threadidx64 : %d\n"
				"VTX         : %s\n",
				thr_id, first_nonce, foundNonce, winnerthread, idx64, hexbuffer);

			*resnonce = foundNonce;
			cudaMemcpy(reshashwholeblock, d_hashwholeblock[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
			cudaMemcpy(ressignature, d_signature[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
			cudaMemcpy(hash, d_hash[thr_id] + winnerthread * 16, 32, cudaMemcpyDeviceToHost);

			memset(hexbuffer, 0, sizeof(hexbuffer));
			for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)hash)[i]);
			applog(LOG_DEBUG, "Final hash 256 : %s", hexbuffer);

			memset(hexbuffer, 0, sizeof(hexbuffer));
			for( int i = 0; i < 185; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)work->longdata)[i]);
			applog(LOG_DEBUG, "Submit data    : %s", hexbuffer);

			memset(hexbuffer, 0, sizeof(hexbuffer));
			for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)reshashwholeblock)[i]);
			applog(LOG_DEBUG, "HashWholeBlock : %s", hexbuffer);

			memset(hexbuffer, 0, sizeof(hexbuffer));
			for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)ressignature)[i]);
			applog(LOG_DEBUG, "MinerSignature : %s", hexbuffer);

			// FIXME: should probably implement a CPU version to check the hash before submitting
			if (fulltest(hash, ptarget)) {

				//*hashes_done = foundNonce - first_nonce + 1;
				*hashes_done = nonce - first_nonce + throughput + 1;
				(*pnonce) = nonce;
				return 1;
			}
		}

		nonce += throughput;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = nonce - first_nonce + 1;
	(*pnonce) = nonce;
	return 0;
}

// SpreadX11 CPU Hash
extern "C" void spreadx11_hash(void *output, void* pbegin, void* pend) /* begin/end = input range */
{
	// blake1-bmw2-grs3-skein4-jh5-keccak6-luffa7-cubehash8-shavite9-simd10-echo11

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	static const uchar pblank[1] = { 0 };

	uchar input[1024] = { 0 };
	uchar hash[128];
	memset(hash, 0, sizeof hash);

	int len = (int) ((uintptr_t) pend - (uintptr_t) pbegin) * 32;

	memcpy(input, (void*) pbegin, len);

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, pbegin == pend ? pblank : input, len);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close (&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(output, hash, 32);
}
