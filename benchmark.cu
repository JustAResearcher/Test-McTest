// MeowPoW v2 CUDA Benchmark — RTX 5090 targeted
// Tests the core algorithm loop on GPU to measure hashrate and verify functionality.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>

// ============================================================================
// Device-side MeowPoW v2 constants
// ============================================================================
#define MEOWPOW_PERIOD       3
#define MEOWPOW_NUM_REGS     32
#define MEOWPOW_NUM_LANES    16
#define MEOWPOW_CACHE_ACCESSES 14
#define MEOWPOW_MATH_OPS     24
#define MEOWPOW_L1_CACHE_SIZE (128u * 1024u)
#define MEOWPOW_L1_CACHE_ITEMS (MEOWPOW_L1_CACHE_SIZE / 4u)
#define MEOWPOW_ROUNDS        64
#define MEOWPOW_NUM_MATH_OPS  15   // switch cases in random_math
#define MEOWPOW_NUM_MERGE_OPS  6   // switch cases in random_merge

// FNV constants
#define FNV_PRIME        0x01000193u
#define FNV_OFFSET_BASIS 0x811c9dc5u

// ============================================================================
// Device helpers
// ============================================================================

__host__ __device__ __forceinline__ uint32_t d_fnv1a(uint32_t u, uint32_t v)
{
    return (u ^ v) * FNV_PRIME;
}

__device__ __forceinline__ uint32_t d_rotl32(uint32_t n, uint32_t c)
{
    c &= 31;
    return (n << c) | (n >> ((uint32_t)(-(int)c) & 31));
}

__device__ __forceinline__ uint32_t d_rotr32(uint32_t n, uint32_t c)
{
    c &= 31;
    return (n >> c) | (n << ((uint32_t)(-(int)c) & 31));
}

__device__ __forceinline__ uint32_t d_clz32(uint32_t x)
{
    return x ? __clz(x) : 32;
}

__device__ __forceinline__ uint32_t d_popcount32(uint32_t x)
{
    return __popc(x);
}

__device__ __forceinline__ uint32_t d_mul_hi32(uint32_t a, uint32_t b)
{
    return __umulhi(a, b);
}

__device__ __forceinline__ uint32_t d_byte_perm(uint32_t a, uint32_t b, uint32_t selector)
{
    // Use CUDA's native __byte_perm intrinsic!
    // __byte_perm(a, b, selector) selects 4 bytes from the 8-byte {b, a} pair
    return __byte_perm(a, b, selector & 0x7777u);
}

__device__ __forceinline__ uint32_t d_brev32(uint32_t x)
{
    return __brev(x);
}

__device__ __forceinline__ uint32_t d_funnelshift_l(uint32_t a, uint32_t b, uint32_t c)
{
    return __funnelshift_l(b, a, c);
}

__device__ __forceinline__ uint32_t d_mad_lo32(uint32_t a, uint32_t b, uint32_t c)
{
    return a * b + c;
}

// ============================================================================
// KISS99 RNG (device)
// ============================================================================
struct kiss99_state {
    uint32_t z, w, jsr, jcong;
};

__device__ __forceinline__ uint32_t kiss99_next(kiss99_state& st)
{
    st.z = 36969 * (st.z & 0xffff) + (st.z >> 16);
    st.w = 18000 * (st.w & 0xffff) + (st.w >> 16);
    uint32_t mwc = (st.z << 16) + st.w;
    st.jsr ^= (st.jsr << 17);
    st.jsr ^= (st.jsr >> 13);
    st.jsr ^= (st.jsr << 5);
    st.jcong = 69069 * st.jcong + 1234567;
    return (mwc ^ st.jcong) + st.jsr;
}

// ============================================================================
// Keccak-f[800] (25 x 32-bit words)
// ============================================================================
__device__ void d_keccakf800(uint32_t st[25])
{
    static const uint32_t rc[22] = {
        0x00000001,0x00008082,0x0000808A,0x80008000,0x0000808B,0x80000001,
        0x80008081,0x00008009,0x0000008A,0x00000088,0x80008009,0x8000000A,
        0x8000808B,0x0000008B,0x00008089,0x00008003,0x00008002,0x00000080,
        0x0000800A,0x8000000A,0x80008081,0x00008080
    };

    for (int round = 0; round < 22; round++) {
        // Theta
        uint32_t c[5], d[5];
        for (int i = 0; i < 5; i++)
            c[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            d[i] = c[(i+4)%5] ^ d_rotl32(c[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j+i] ^= d[i];
        }
        // Rho & Pi
        uint32_t tmp = st[1];
        static const int piln[24] = {
            10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
        };
        static const int rotc[24] = {
            1,3,6,10,15,21,28,4,13,23,2,14,27,9,24,8,25,11,7,20,12,18,5,31
        };
        for (int i = 0; i < 24; i++) {
            uint32_t j = piln[i];
            uint32_t bc = st[j];
            st[j] = d_rotl32(tmp, rotc[i]);
            tmp = bc;
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
            uint32_t t[5];
            for (int i = 0; i < 5; i++) t[i] = st[j+i];
            for (int i = 0; i < 5; i++)
                st[j+i] = t[i] ^ ((~t[(i+1)%5]) & t[(i+2)%5]);
        }
        // Iota
        st[0] ^= rc[round];
    }
}

// ============================================================================
// random_math — 15 operations
// ============================================================================
__device__ __forceinline__ uint32_t d_random_math(uint32_t a, uint32_t b, uint32_t selector)
{
    switch (selector % 15) {
    default:
    case 0:  return a + b;
    case 1:  return a * b;
    case 2:  return d_mul_hi32(a, b);
    case 3:  return min(a, b);
    case 4:  return d_rotl32(a, b);
    case 5:  return d_rotr32(a, b);
    case 6:  return a & b;
    case 7:  return a | b;
    case 8:  return a ^ b;
    case 9:  return d_clz32(a) + d_clz32(b);
    case 10: return d_popcount32(a) + d_popcount32(b);
    case 11: return d_byte_perm(a, b, selector >> 16);
    case 12: return d_brev32(a) ^ b;
    case 13: return d_funnelshift_l(a, b, selector);
    case 14: return d_mad_lo32(a, b, selector >> 16);
    }
}

// ============================================================================
// random_merge — 6 operations
// ============================================================================
__device__ __forceinline__ void d_random_merge(uint32_t& a, uint32_t b, uint32_t selector)
{
    uint32_t x = (selector >> 16) % 31 + 1;
    switch (selector % 6) {
    case 0: a = (a * 33) + b; break;
    case 1: a = (a ^ b) * 33; break;
    case 2: a = d_rotl32(a, x) ^ b; break;
    case 3: a = d_rotr32(a, x) ^ b; break;
    case 4: a = (a + b) ^ d_rotl32(b, x); break;
    case 5: a = d_mul_hi32(a, b) ^ b; break;
    }
}

// ============================================================================
// MeowPoW v2 core benchmark kernel
//
// This kernel runs the full MeowPoW v2 inner loop (init_mix + 64 rounds of
// random math/merge/cache access + lane reduction) for each nonce.
// It uses a synthetic L1 cache and synthetic DAG item to exercise the
// algorithm without needing the real multi-GB DAG.
// ============================================================================
__global__ void meowpow_v2_benchmark(
    const uint32_t* __restrict__ l1_cache,  // 128 KB L1 cache
    uint32_t* __restrict__ results,         // output hashes
    uint64_t start_nonce,
    uint32_t block_number,
    uint32_t num_hashes)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hashes) return;

    uint64_t nonce = start_nonce + tid;

    // Fake header hash from nonce for benchmark
    uint32_t header_hash[8];
    for (int i = 0; i < 8; i++)
        header_hash[i] = d_fnv1a(0xdeadbeef + i, (uint32_t)(nonce >> (i & 1 ? 32 : 0)));

    // === Initial Keccak-f[800] ===
    uint32_t kstate[25];
    for (int i = 0; i < 8; i++) kstate[i] = header_hash[i];
    kstate[8] = (uint32_t)nonce;
    kstate[9] = (uint32_t)(nonce >> 32);
    // Meowcoin constants
    static const uint32_t meow_const[15] = {
        0x4D,0x45,0x4F,0x57,0x43,0x4F,0x49,0x4E,
        0x4D,0x45,0x4F,0x57,0x50,0x4F,0x57
    };
    for (int i = 10; i < 25; i++) kstate[i] = meow_const[i-10];
    d_keccakf800(kstate);

    uint32_t seed[2] = {kstate[0], kstate[1]};
    uint32_t state2[8];
    for (int i = 0; i < 8; i++) state2[i] = kstate[i];

    // === Init mix (32 regs x 16 lanes = 512 uint32s) ===
    uint32_t mix[MEOWPOW_NUM_LANES][MEOWPOW_NUM_REGS];
    {
        uint32_t z = d_fnv1a(FNV_OFFSET_BASIS, seed[0]);
        uint32_t w = d_fnv1a(z, seed[1]);
        for (uint32_t l = 0; l < MEOWPOW_NUM_LANES; l++) {
            uint32_t jsr = d_fnv1a(w, l);
            uint32_t jcong = d_fnv1a(jsr, l);
            kiss99_state rng = {z, w, jsr, jcong};
            for (int r = 0; r < MEOWPOW_NUM_REGS; r++)
                mix[l][r] = kiss99_next(rng);
        }
    }

    // === RNG state for program generation ===
    uint64_t prog_number = (uint64_t)block_number / MEOWPOW_PERIOD;
    uint32_t prog_seed[2] = {(uint32_t)prog_number, (uint32_t)(prog_number >> 32)};

    uint32_t rng_z = d_fnv1a(FNV_OFFSET_BASIS, prog_seed[0]);
    uint32_t rng_w = d_fnv1a(rng_z, prog_seed[1]);
    uint32_t rng_jsr = d_fnv1a(rng_w, prog_seed[0]);
    uint32_t rng_jcong = d_fnv1a(rng_jsr, prog_seed[1]);
    kiss99_state prog_rng = {rng_z, rng_w, rng_jsr, rng_jcong};

    // Build dst/src permutations
    uint32_t dst_seq[MEOWPOW_NUM_REGS], src_seq[MEOWPOW_NUM_REGS];
    for (uint32_t i = 0; i < MEOWPOW_NUM_REGS; i++) {
        dst_seq[i] = i;
        src_seq[i] = i;
    }
    for (uint32_t i = MEOWPOW_NUM_REGS; i > 1; i--) {
        uint32_t j;
        j = kiss99_next(prog_rng) % i;
        uint32_t tmp = dst_seq[i-1]; dst_seq[i-1] = dst_seq[j]; dst_seq[j] = tmp;
        j = kiss99_next(prog_rng) % i;
        tmp = src_seq[i-1]; src_seq[i-1] = src_seq[j]; src_seq[j] = tmp;
    }
    uint32_t dst_ctr = 0, src_ctr = 0;

    // === 64 rounds ===
    for (uint32_t round = 0; round < MEOWPOW_ROUNDS; round++) {
        // Synthetic DAG item: derive from mix state (exercises same code paths)
        uint32_t dag_item[64]; // hash2048 = 64 x uint32
        uint32_t item_index = mix[round % MEOWPOW_NUM_LANES][0];
        for (int i = 0; i < 64; i++)
            dag_item[i] = d_fnv1a(item_index, i);

        constexpr int max_ops = MEOWPOW_CACHE_ACCESSES > MEOWPOW_MATH_OPS
                                ? MEOWPOW_CACHE_ACCESSES : MEOWPOW_MATH_OPS;

        for (int i = 0; i < max_ops; i++) {
            if (i < MEOWPOW_CACHE_ACCESSES) {
                uint32_t src = src_seq[(src_ctr++) % MEOWPOW_NUM_REGS];
                uint32_t dst = dst_seq[(dst_ctr++) % MEOWPOW_NUM_REGS];
                uint32_t sel = kiss99_next(prog_rng);
                for (int l = 0; l < MEOWPOW_NUM_LANES; l++) {
                    uint32_t offset = mix[l][src] % MEOWPOW_L1_CACHE_ITEMS;
                    d_random_merge(mix[l][dst], l1_cache[offset], sel);
                }
            }
            if (i < MEOWPOW_MATH_OPS) {
                uint32_t src_rnd = kiss99_next(prog_rng) % (MEOWPOW_NUM_REGS * (MEOWPOW_NUM_REGS - 1));
                uint32_t src1 = src_rnd % MEOWPOW_NUM_REGS;
                uint32_t src2 = src_rnd / MEOWPOW_NUM_REGS;
                if (src2 >= src1) ++src2;
                uint32_t sel1 = kiss99_next(prog_rng);
                uint32_t dst = dst_seq[(dst_ctr++) % MEOWPOW_NUM_REGS];
                uint32_t sel2 = kiss99_next(prog_rng);
                for (int l = 0; l < MEOWPOW_NUM_LANES; l++) {
                    uint32_t data = d_random_math(mix[l][src1], mix[l][src2], sel1);
                    d_random_merge(mix[l][dst], data, sel2);
                }
            }
        }

        // DAG merge
        constexpr int words_per_lane = 64 / MEOWPOW_NUM_LANES; // = 4
        uint32_t dag_dsts[words_per_lane], dag_sels[words_per_lane];
        for (int i = 0; i < words_per_lane; i++) {
            dag_dsts[i] = (i == 0) ? 0 : dst_seq[(dst_ctr++) % MEOWPOW_NUM_REGS];
            dag_sels[i] = kiss99_next(prog_rng);
        }
        for (int l = 0; l < MEOWPOW_NUM_LANES; l++) {
            int offset = ((l ^ round) % MEOWPOW_NUM_LANES) * words_per_lane;
            for (int i = 0; i < words_per_lane; i++)
                d_random_merge(mix[l][dag_dsts[i]], dag_item[offset + i], dag_sels[i]);
        }
    }

    // === Lane reduction (FNV1a) ===
    uint32_t lane_hash[MEOWPOW_NUM_LANES];
    for (int l = 0; l < MEOWPOW_NUM_LANES; l++) {
        lane_hash[l] = FNV_OFFSET_BASIS;
        for (int r = 0; r < MEOWPOW_NUM_REGS; r++)
            lane_hash[l] = d_fnv1a(lane_hash[l], mix[l][r]);
    }

    // Reduce to 256-bit mix_hash
    uint32_t mix_hash[8];
    for (int i = 0; i < 8; i++) mix_hash[i] = FNV_OFFSET_BASIS;
    for (int l = 0; l < MEOWPOW_NUM_LANES; l++)
        mix_hash[l % 8] = d_fnv1a(mix_hash[l % 8], lane_hash[l]);

    // === Final Keccak-f[800] ===
    uint32_t fstate[25] = {0};
    for (int i = 0; i < 8; i++) fstate[i] = state2[i];
    for (int i = 0; i < 8; i++) fstate[8+i] = mix_hash[i];
    for (int i = 16; i < 25; i++) fstate[i] = meow_const[i-16];
    d_keccakf800(fstate);

    // Write first word of final hash as proof of work
    results[tid] = fstate[0];
}

// ============================================================================
// Host main
// ============================================================================
int main()
{
    printf("=== MeowPoW v2 CUDA Benchmark ===\n");
    printf("Algorithm: MeowPoW v2.0.0\n");
    printf("Target: RTX 5090 (sm_120)\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Shared mem/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);

    printf("Parameters:\n");
    printf("  Registers:       %d\n", MEOWPOW_NUM_REGS);
    printf("  Lanes:           %d\n", MEOWPOW_NUM_LANES);
    printf("  Cache accesses:  %d\n", MEOWPOW_CACHE_ACCESSES);
    printf("  Math operations: %d\n", MEOWPOW_MATH_OPS);
    printf("  Math op types:   %d\n", MEOWPOW_NUM_MATH_OPS);
    printf("  Merge functions: %d\n", MEOWPOW_NUM_MERGE_OPS);
    printf("  L1 cache:        %d KB\n", MEOWPOW_L1_CACHE_SIZE / 1024);
    printf("  Rounds:          %d\n", MEOWPOW_ROUNDS);
    printf("  Period:          %d\n\n", MEOWPOW_PERIOD);

    // Allocate synthetic L1 cache (128 KB)
    uint32_t* d_l1_cache;
    cudaMalloc(&d_l1_cache, MEOWPOW_L1_CACHE_SIZE);
    {
        uint32_t* h_cache = (uint32_t*)malloc(MEOWPOW_L1_CACHE_SIZE);
        for (uint32_t i = 0; i < MEOWPOW_L1_CACHE_ITEMS; i++)
            h_cache[i] = d_fnv1a(0xcafebabe, i);  // host-side fnv1a is just inline
        // Manual FNV1a for host init
        for (uint32_t i = 0; i < MEOWPOW_L1_CACHE_ITEMS; i++)
            h_cache[i] = (0xcafebabe ^ i) * FNV_PRIME;
        cudaMemcpy(d_l1_cache, h_cache, MEOWPOW_L1_CACHE_SIZE, cudaMemcpyHostToDevice);
        free(h_cache);
    }

    // Benchmark parameters
    const uint32_t block_number = 100000;
    const uint64_t start_nonce = 0;

    // Warmup + benchmark at different batch sizes
    uint32_t batch_sizes[] = {1024, 4096, 16384, 65536, 262144};

    for (int b = 0; b < 5; b++) {
        uint32_t num_hashes = batch_sizes[b];
        uint32_t* d_results;
        cudaMalloc(&d_results, num_hashes * sizeof(uint32_t));

        int threads = 256;
        int blocks = (num_hashes + threads - 1) / threads;

        // Warmup
        meowpow_v2_benchmark<<<blocks, threads>>>(d_l1_cache, d_results, start_nonce, block_number, num_hashes);
        cudaDeviceSynchronize();

        // Timed run
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        int iterations = 5;
        cudaEventRecord(t0);
        for (int i = 0; i < iterations; i++) {
            meowpow_v2_benchmark<<<blocks, threads>>>(d_l1_cache, d_results, start_nonce + i * num_hashes, block_number, num_hashes);
        }
        cudaEventRecord(t1);
        cudaDeviceSynchronize();

        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);

        double total_hashes = (double)num_hashes * iterations;
        double hashrate = total_hashes / (ms / 1000.0);

        // Read back first result as sanity check
        uint32_t first_result;
        cudaMemcpy(&first_result, d_results, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        printf("Batch %6u: %.2f ms (%d iters) | %.2f KH/s | %.2f MH/s | sample=0x%08x\n",
               num_hashes, ms, iterations, hashrate / 1000.0, hashrate / 1000000.0, first_result);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(d_results);
    }

    cudaFree(d_l1_cache);

    printf("\nBenchmark complete.\n");
    printf("Press Enter to exit...");
    getchar();
    return 0;
}
