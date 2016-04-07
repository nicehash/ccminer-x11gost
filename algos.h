#ifndef ALGOS_H
#define ALGOS_H

#include <string.h>
#include "compat.h"

enum sha_algos {
	ALGO_BLAKE,
	ALGO_VCASH,
	ALGO_BLAKECOIN,
	ALGO_WHIRLPOOLX,
	ALGO_KECCAK,
	ALGO_LYRA2,
	ALGO_LYRA2v2,
	ALGO_AUTO,
	ALGO_COUNT
};

extern volatile enum sha_algos opt_algo;

static const char *algo_names[] = {
	"blake",
	"vcash",
	"blakecoin",
	"whirlpoolx",
	"keccak",
	"lyra2",
	"lyra2v2",
	"auto", /* reserved for multi algo */
	""
};

// string to int/enum
static inline int algo_to_int(char* arg)
{
	int i;

	for (i = 0; i < ALGO_COUNT; i++) {
		if (algo_names[i] && !strcasecmp(arg, algo_names[i])) {
			return i;
		}
	}

	return -1;
}

#endif
