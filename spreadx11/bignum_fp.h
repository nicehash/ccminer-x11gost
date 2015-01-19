/*

Based on libtommath and tomsfastmath by Tom St Denis

This is basically Barret reduction and it's setup from libtommath and
basic bignum types and functions from tomsfastmath with some application
specific optimizations like dropping sign handling and branches that are
never reached, hard coding for a specific digit size (replacing slow ops like
x % DIGIT_BIT with x & 31 etc).

Could probably do a lot better with CUDA, this code spills to local memory
like a mofo due to dynamic array indexes.

*/

#include <stdint.h>

#ifndef CHAR_BIT
	#define CHAR_BIT 8
#endif

#ifndef MIN
	#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

#ifndef MAX
	#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#define DIGIT_BIT   32
#define FP_SIZE     20

typedef uint32_t fp_digit;
typedef uint64_t fp_word;
typedef fp_digit fp_min_u32;

/* return codes */
#define FP_OKAY     0
#define FP_VAL      1
#define FP_MEM      2

/* equalities */
#define FP_LT      -1   /* less than */
#define FP_EQ       0   /* equal to */
#define FP_GT       1   /* greater than */

/* replies */
#define FP_YES      1   /* yes response */
#define FP_NO       0   /* no response */

typedef int fp_err;

typedef struct {
	int used;
	fp_digit dp[FP_SIZE];
} fp_int;

#define USED(m)    ((m)->used)
#define DIGIT(m,k) ((m)->dp[(k)])

#define fp_iszero(a) (((a)->used == 0) ? FP_YES : FP_NO)
#define fp_iseven(a) (((a)->used > 0 && (((a)->dp[0] & 1) == 0)) ? FP_YES : FP_NO)
#define fp_isodd(a)  (((a)->used > 0 && (((a)->dp[0] & 1) == 1)) ? FP_YES : FP_NO)
#define fp_init(a)    fp_zero((a))
#define fp_mul(a,b,c) fp_mul_comba((a), (b), (c))

/* clamp digits */
#define fp_clamp(a)   { while ((a)->used && (a)->dp[(a)->used-1] == 0) --((a)->used); }

__device__ __host__
void fp_zero( fp_int *a )
{
	for( int i = 0; i < FP_SIZE; i++ ) a->dp[i] = 0;
	a->used = 0;
}

__device__ __host__
void fp_set( fp_int *a, fp_digit b )
{
	fp_zero(a);
	a->dp[0] = b;
	a->used  = a->dp[0] ? 1 : 0;
}

__device__ __host__
int fp_cmp_mag( fp_int *a, fp_int *b )
{
	if( a->used > b->used ) return FP_GT;
	else if( a->used < b->used ) return FP_LT;

	for( int x = a->used - 1; x >= 0; x-- ) {

		if( a->dp[x] > b->dp[x] ) return FP_GT;
		else if( a->dp[x] < b->dp[x] ) return FP_LT;
	}

	return FP_EQ;
}

/* unsigned addition */
__device__ __host__
void s_fp_add(fp_int *a, fp_int *b, fp_int *c)
{
	int     x, y, oldused;
	fp_word t;

	y       = MAX(a->used, b->used);
	oldused = c->used;
	c->used = y;

	t = 0;
	for (x = 0; x < y; x++) {
		t         += ((fp_word)a->dp[x]) + ((fp_word)b->dp[x]);
		c->dp[x]   = (fp_digit)t;
		t        >>= DIGIT_BIT;
	}
	if (t != 0 && x < FP_SIZE) {
		c->dp[c->used++] = (fp_digit)t;
		++x;
	}

	c->used = x;
	for (; x < oldused; x++) {
		c->dp[x] = 0;
	}
	fp_clamp(c);
}

/* unsigned subtraction ||a|| >= ||b|| ALWAYS! */
__device__ __host__
void s_fp_sub(fp_int *a, fp_int *b, fp_int *c)
{
	int      x, oldbused, oldused;
	fp_word  t;

	oldused  = c->used;
	oldbused = b->used;
	c->used  = a->used;
	t       = 0;
	for (x = 0; x < oldbused; x++) {
		t         = ((fp_word)a->dp[x]) - (((fp_word)b->dp[x]) + t);
		c->dp[x]  = (fp_digit)t;
		t         = (t >> DIGIT_BIT)&1;
	}
	for (; x < a->used; x++) {
		t         = ((fp_word)a->dp[x]) - t;
		c->dp[x]  = (fp_digit)t;
		t         = (t >> DIGIT_BIT)&1;
	 }
	for (; x < oldused; x++) {
		c->dp[x] = 0;
	}
	fp_clamp(c);
}

__device__ __host__
void fp_add( fp_int *a, fp_int *b, fp_int *c )
{
	s_fp_add (a, b, c);
}

/* c = a - b */
__device__ __host__
void fp_sub(fp_int *a, fp_int *b, fp_int *c)
{
	// int     sa, sb;

	/* The first has a larger or equal magnitude */
	s_fp_sub (a, b, c);
}

__device__ __host__
int fp_cmp(fp_int *a, fp_int *b)
{
	return fp_cmp_mag(a, b);
}

__device__ __host__
void fp_copy( fp_int *a, fp_int *b )
{
	for( int i = 0; i < FP_SIZE; i++ )
		b->dp[i] = a->dp[i];
	b->used = a->used;
}

// TODO: use clz, return a->used - clz(a->dp[a->used-1])
__device__ __host__
int fp_count_bits( fp_int *a )
{
	int     r;
	fp_digit q;

	/* shortcut */
	if (a->used == 0) return 0;

	/* get number of digits and add that */
	r = (a->used - 1) << 5;

	/* take the last digit and count the bits in it */
	q = a->dp[a->used - 1];
	while (q > ((fp_digit) 0)) {
			++r;
			q >>= ((fp_digit) 1);
	}

	return r;
}

__device__ __host__
void fp_rshd( fp_int *a, int x )
{
	int y;

	/* too many digits just zero and return */
	if (x >= a->used) {

			fp_zero(a);
			return;
	}

	/* shift */
	for (y = 0; y < a->used - x; y++) a->dp[y] = a->dp[y+x];

	/* zero rest */
	for (; y < a->used; y++) a->dp[y] = 0;

	/* decrement count */
	a->used -= x;
	fp_clamp(a);
}

__device__ __host__
void fp_lshd(fp_int *a, int x)
{
	int y;

	/* move up and truncate as required */
	y = MIN(a->used + x - 1, (int)(FP_SIZE-1));

	/* store new size */
	a->used = y + 1;

	/* move digits */
	for (; y >= x; y--) a->dp[y] = a->dp[y-x];

	/* zero lower digits */
	for (; y >= 0; y--) a->dp[y] = 0;

	/* clamp digits */
	fp_clamp(a);
}

/* c = a * 2**d */
__device__ __host__
void fp_mul_2d( fp_int *a, int b, fp_int *c )
{
	fp_digit carry, carrytmp, shift;
	int x;

	/* copy it */
	fp_copy(a, c);

	/* handle whole digits */
	if (b >= DIGIT_BIT) {
			fp_lshd(c, b >> 5);
	}
	b &= 31;

	/* shift the digits */
	if (b != 0) {
		carry = 0;
		shift = DIGIT_BIT - b;
		for (x = 0; x < c->used; x++) {
				carrytmp = c->dp[x] >> shift;
				c->dp[x] = (c->dp[x] << b) + carry;
				carry = carrytmp;
		}

		/* store last carry if room */
		if (carry && x < FP_SIZE) c->dp[c->used++] = carry;
	}
	fp_clamp(c);
}

/* c = a * b */
__device__ __host__
void fp_mul_d( fp_int *a, fp_digit b, fp_int *c )
{
	fp_word w;
	int     x, oldused;

	oldused = c->used;
	c->used = a->used;
	w       = 0;

	for (x = 0; x < a->used; x++) {
		w         = ((fp_word)a->dp[x]) * ((fp_word)b) + w;
		c->dp[x]  = (fp_digit)w;
		w         = w >> DIGIT_BIT;
	}

	if (w != 0 && (a->used != FP_SIZE)) {
		c->dp[c->used++] = w;
		++x;
	}

	for (; x < oldused; x++) c->dp[x] = 0;

	fp_clamp(c);
}

/* c = a mod 2**d */
__device__ __host__
void fp_mod_2d( fp_int *a, int b, fp_int *c )
{
	int x;

	/* zero if count less than or equal to zero */
	if (b <= 0) {
		fp_zero(c);
		return;
	}

	/* get copy of input */
	fp_copy(a, c);

	/* if 2**d is larger than we just return */
	if (b >= (a->used << 5)) return;

	/* zero digits above the last digit of the modulus */
	for (x = (b >> 5) + ((b & 31) == 0 ? 0 : 1); x < c->used; x++) {
		c->dp[x] = 0;
	}

	/* clear the digit that is not completely outside/inside the modulus */
	c->dp[b >> 5] &= ~((fp_digit)0) >> (DIGIT_BIT - b);
	fp_clamp (c);
}

/* c = a / 2**b */
__device__ __host__
void fp_div_2d( fp_int *a, int b, fp_int *c, fp_int *d )
{
	fp_digit D, r, rr;
	int      x;
	fp_int   t;

	/* if the shift count is <= 0 then we do no work */
	if (b <= 0) {
		fp_copy (a, c);
		if (d != NULL) fp_zero (d);
		return;
	}

	fp_zero(&t);

	/* get the remainder */
	if (d != NULL) fp_mod_2d (a, b, &t);

	/* copy */
	fp_copy(a, c);

	/* shift by as many digits in the bit count */
	if (b >= (int)DIGIT_BIT) fp_rshd (c, b >> 5);

	/* shift any bit count < DIGIT_BIT */
	D = (fp_digit) (b & 31);
	if (D != 0) {
		fp_digit *tmpc, mask, shift;

		/* mask */
		mask = (((fp_digit)1) << D) - 1;

		/* shift for lsb */
		shift = DIGIT_BIT - D;

		/* alias */
		tmpc = c->dp + (c->used - 1);

		/* carry */
		r = 0;
		for (x = c->used - 1; x >= 0; x--) {
			/* get the lower  bits of this word in a temp */
			rr = *tmpc & mask;

			/* shift the current word and mix in the carry bits from the previous word */
			*tmpc = (*tmpc >> D) | (r << shift);
			--tmpc;

			/* set the carry to the carry bits of the current word found above */
			r = rr;
		}
	}
	fp_clamp (c);

	if (d != NULL) fp_copy (&t, d);
}

/* a/b => cb + d == a */
__device__ __host__
void fp_div(fp_int *a, fp_int *b, fp_int *c, fp_int *d)
{
	fp_int q, x, y, t1, t2;
	int n, t, i, norm;

	/* if a < b then q=0, r = a */
	if( fp_cmp_mag(a, b) == FP_LT ) {

		if (d != NULL) fp_copy(a, d);
		if (c != NULL) fp_zero(c);

		return;
	}

	fp_init(&q);
	q.used = a->used + 2;
	fp_init(&t1);
	fp_init(&t2);
	fp_copy(a, &x);
	fp_copy(b, &y);

	/* normalize both x and y, ensure that y >= b/2, [b == 2**DIGIT_BIT] */
	norm = fp_count_bits(&y) & 31;
	if (norm < (int)(DIGIT_BIT-1)) {
		norm = (DIGIT_BIT-1) - norm;
		fp_mul_2d (&x, norm, &x);
		fp_mul_2d (&y, norm, &y);
	}
	else {
		norm = 0;
	}

	/* note hac does 0 based, so if used==5 then its 0,1,2,3,4, e.g. use 4 */
	n = x.used - 1;
	t = y.used - 1;

	/* while (x >= y*b**n-t) do { q[n-t] += 1; x -= y*b**{n-t} } */
	fp_lshd (&y, n - t);                                             /* y = y*b**{n-t} */

	while (fp_cmp (&x, &y) != FP_LT) {
			++(q.dp[n - t]);
			fp_sub (&x, &y, &x);
	}

	/* reset y by shifting it back down */
	fp_rshd (&y, n - t);

	/* step 3. for i from n down to (t + 1) */
	for (i = n; i >= (t + 1); i--) {

		if (i > x.used) continue;

		/* step 3.1 if xi == yt then set q{i-t-1} to b-1,
		* otherwise set q{i-t-1} to (xi*b + x{i-1})/yt */
		if (x.dp[i] == y.dp[t]) {
			q.dp[i - t - 1] = ((((fp_word)1) << DIGIT_BIT) - 1);
		}
		else {
			fp_word tmp;
			tmp = ((fp_word) x.dp[i]) << ((fp_word) DIGIT_BIT);
			tmp |= ((fp_word) x.dp[i - 1]);
			tmp /= ((fp_word) y.dp[t]);
			q.dp[i - t - 1] = (fp_digit) (tmp);
		}

		/* while (q{i-t-1} * (yt * b + y{t-1})) >
		xi * b**2 + xi-1 * b + xi-2

		do q{i-t-1} -= 1;
		*/
		q.dp[i - t - 1] = (q.dp[i - t - 1] + 1);
		do {
			q.dp[i - t - 1] = (q.dp[i - t - 1] - 1);

			/* find left hand */
			fp_zero (&t1);
			t1.dp[0] = (t - 1 < 0) ? 0 : y.dp[t - 1];
			t1.dp[1] = y.dp[t];
			t1.used = 2;
			fp_mul_d (&t1, q.dp[i - t - 1], &t1);

			/* find right hand */
			t2.dp[0] = (i - 2 < 0) ? 0 : x.dp[i - 2];
			t2.dp[1] = (i - 1 < 0) ? 0 : x.dp[i - 1];
			t2.dp[2] = x.dp[i];
			t2.used = 3;

		} while (fp_cmp_mag(&t1, &t2) == FP_GT);

		/* step 3.3 x = x - q{i-t-1} * y * b**{i-t-1} */
		fp_mul_d(&y, q.dp[i - t - 1], &t1);
		fp_lshd (&t1, i - t - 1);
		fp_sub  (&x, &t1, &x);
	}

	/* now q is the quotient and x is the remainder
	 * [which we have to normalize]
	 */

	if (c != NULL) {
		fp_clamp(&q);
		fp_copy (&q, c);
	}

	if (d != NULL) {
		fp_div_2d(&x, norm, &x, NULL);

		/* the following is a kludge, essentially we were seeing the right remainder but
		with excess digits that should have been zero
		*/
		for (i = b->used; i < x.used; i++) x.dp[i] = 0;

		fp_clamp(&x);
		fp_copy (&x, d);
	}
}

__device__ __host__
void fp_2expt( fp_int *a, int b )
{
	int z;

	fp_zero(a);

	z = b >> 5;

	/* set the used count of where the bit will go */
	a->used = z + 1;

	/* put the single bit in its place */
	a->dp[z] = ((fp_digit)1) << (b & 31);
}

/* generic PxQ multiplier */
__device__ __host__
void fp_mul_comba(fp_int *A, fp_int *B, fp_int *C)
{
	int       ix, iy, iz, tx, ty, pa;
	fp_digit  c0, c1, c2, *tmpx, *tmpy;
	fp_int    tmp, *dst;

	c0 = c1 = c2 = 0;

	/* get size of output and trim */
	pa = A->used + B->used;
	if (pa >= FP_SIZE) pa = FP_SIZE-1;

	if (A == C || B == C) {
			fp_zero(&tmp);
			dst = &tmp;
	}
	else {
			fp_zero(C);
			dst = C;
	}

	for (ix = 0; ix < pa; ix++) {

		/* get offsets into the two bignums */
		ty = MIN(ix, B->used-1);
		tx = ix - ty;

		/* setup temp aliases */
		tmpx = A->dp + tx;
		tmpy = B->dp + ty;

		/* this is the number of times the loop will iterrate, essentially its
		while (tx++ < a->used && ty-- >= 0) { ... }
		*/
		iy = MIN(A->used-tx, ty+1);

		/* execute loop */
		c0 = c1; c1 = c2; c2 = 0;

		for (iz = 0; iz < iy; ++iz) {

				fp_word t;
				t = (fp_word)c0 + ((fp_word)*tmpx++) * ((fp_word)*tmpy--);
				c0 = t;
				t = (fp_word)c1 + (t >> DIGIT_BIT);
				c1 = t;
				c2 += t >> DIGIT_BIT;
		}

		/* store term */
		dst->dp[ix] = c0;
	}

	dst->used = pa;
	fp_clamp(dst);
	fp_copy(dst, C);
}

__device__ __host__
void fp_read_unsigned_bin(fp_int *a, unsigned char *b, int c)
{
	/* zero the int */
	fp_zero (a);

	unsigned char *pd = (unsigned char *)a->dp;

	if ((unsigned)c > (FP_SIZE * sizeof(fp_digit))) {
		int excess = c - (FP_SIZE * sizeof(fp_digit));
		c -= excess;
		b += excess;
	}
	a->used = (c + sizeof(fp_digit) - 1)/sizeof(fp_digit);
	/* read the bytes in */
	for (c -= 1; c >= 0; c -= 1) {
		pd[c] = *b++;
	}

	fp_clamp (a);
}

__device__ __host__
void fp_reverse( unsigned char *s, int len )
{
	int     ix, iy;
	unsigned char t;

	ix = 0;
	iy = len - 1;
	while (ix < iy) {
		t     = s[ix];
		s[ix] = s[iy];
		s[iy] = t;
		++ix;
		--iy;
	}
}

__device__ __host__
void fp_to_unsigned_bin( fp_int *a, unsigned char *b )
{
	int    x;
	fp_int t;

	fp_copy(a, &t);

	x = 0;
	while (fp_iszero (&t) == FP_NO) {
		b[x++] = (unsigned char) (t.dp[0] & 255);
		fp_div_2d (&t, 8, &t, NULL);
	}
	fp_reverse(b, x);
}

__device__ __host__
void fp_reduce_setup( fp_int *a, fp_int *b )
{
	fp_2expt(a, b->used << 6);
	fp_div(a, b, a, NULL);
}

/* compare against a single digit */
__device__ __host__
int fp_cmp_d( fp_int *a, fp_digit b )
{
	/* compare based on magnitude */
	if (a->used > 1) return FP_GT;

	/* compare the only digit of a to b */
	if (a->dp[0] > b) return FP_GT;
	else if (a->dp[0] < b) return FP_LT;

	return FP_EQ;
}

__device__ __host__
void fp_reduce( fp_int *x, fp_int *m, fp_int *mu )
{
	fp_int q;
	int um = m->used;

	if( fp_cmp(x, m) == FP_LT ) return;

	/* q = x */
	fp_copy(x, &q);

	/* q1 = x / b**(k-1)  */
	fp_rshd(&q, um - 1);

	fp_mul(&q, mu, &q);

	/* q3 = q2 / b**(k+1) */
	fp_rshd(&q, um + 1);

	/* x = x mod b**(k+1), quick (no division) */
	fp_mod_2d(x, ((um + 1) << 5), x);

	/* q = q * m mod b**(k+1), quick (no division) */
	//s_fp_mul_digs (&q, m, &q, um + 1);
	// FIXME: actually multiply up to required digits instead of full multiply + truncation
	fp_mul(&q, m, &q);
	q.used = um + 1;
	for(int i = q.used; i < FP_SIZE; i++)
		q.dp[i] = 0;

	/* x = x - q */
	fp_sub(x, &q, x);

	/* If x < 0, add b**(k+1) to it */
	// FIXME: does this ever happen..?
	if (fp_cmp_d (x, 0) == FP_LT) {
		fp_set (&q, 1);
		fp_lshd(&q, um + 1);
		fp_add (x, &q, x);
	}

	int count = 0;
	/* Back off if it's too big */
	while( fp_cmp(x, m) != FP_LT) {
		s_fp_sub(x, m, x);
		count++;
		if( count > 100 ) {
			//printf("FUCK ME I GOT STUCK IN THE REDUCE WHILE LOOP, AGAIN!!!!\n");
			break;
		}
	}
}
