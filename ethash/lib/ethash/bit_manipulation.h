/* ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
 * Copyright 2018-2019 Pawel Bylica.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "builtins.h"
#include "../support/attributes.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t rotl32(uint32_t n, unsigned int c)
{
    const unsigned int mask = 31;

    c &= mask;
    unsigned int neg_c = (unsigned int)(-(int)c);
    return (n << c) | (n >> (neg_c & mask));
}

static inline uint32_t rotr32(uint32_t n, unsigned int c)
{
    const unsigned int mask = 31;

    c &= mask;
    unsigned int neg_c = (unsigned int)(-(int)c);
    return (n >> c) | (n << (neg_c & mask));
}

static inline uint32_t clz32(uint32_t x)
{
    return x ? (uint32_t)__builtin_clz(x) : 32;
}

static inline uint32_t popcount32(uint32_t x)
{
    return (uint32_t)__builtin_popcount(x);
}

static inline uint32_t mul_hi32(uint32_t x, uint32_t y)
{
    return (uint32_t)(((uint64_t)x * (uint64_t)y) >> 32);
}

/// Byte-level permutation: select 4 bytes from the 8-byte concatenation of a and b.
/// Each 2-bit field in `selector` picks a byte from {b[0], b[1], a[0], a[1], ...}.
/// Maps to CUDA __byte_perm intrinsic on GPUs, expensive to replicate in ASICs.
static inline uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t selector)
{
    const uint8_t* src = (const uint8_t*)&a;
    const uint8_t src_bytes[8] = {
        ((const uint8_t*)&b)[0], ((const uint8_t*)&b)[1],
        ((const uint8_t*)&b)[2], ((const uint8_t*)&b)[3],
        ((const uint8_t*)&a)[0], ((const uint8_t*)&a)[1],
        ((const uint8_t*)&a)[2], ((const uint8_t*)&a)[3]
    };
    uint32_t result = 0;
    result |= (uint32_t)src_bytes[selector & 7];
    result |= (uint32_t)src_bytes[(selector >> 4) & 7] << 8;
    result |= (uint32_t)src_bytes[(selector >> 8) & 7] << 16;
    result |= (uint32_t)src_bytes[(selector >> 12) & 7] << 24;
    return result;
}

/// Bit reversal of a 32-bit integer.
/// Maps to CUDA __brev intrinsic, requires dedicated hardware or many shifts in an ASIC.
static inline uint32_t brev32(uint32_t x)
{
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}

/// Funnel shift left: concatenate (a, b) and shift left by (c % 32), return upper 32 bits.
/// Maps to CUDA __funnelshift_l, a single PTX instruction on modern GPUs.
static inline uint32_t funnelshift_l(uint32_t a, uint32_t b, uint32_t c)
{
    c &= 31;
    if (c == 0)
        return a;
    return (a << c) | (b >> (32 - c));
}

/// Multiply-add low: (a * b + c) truncated to 32 bits.
/// Maps to GPU MAD instruction, single cycle on modern GPUs.
NO_SANITIZE("unsigned-integer-overflow")
static inline uint32_t mad_lo32(uint32_t a, uint32_t b, uint32_t c)
{
    return a * b + c;
}


/** FNV 32-bit prime. */
static const uint32_t fnv_prime = 0x01000193;

/** FNV 32-bit offset basis. */
static const uint32_t fnv_offset_basis = 0x811c9dc5;

/**
 * The implementation of FNV-1 hash.
 *
 * See https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1_hash.
 */
NO_SANITIZE("unsigned-integer-overflow")
static inline uint32_t fnv1(uint32_t u, uint32_t v) noexcept
{
    return (u * fnv_prime) ^ v;
}

/**
 * The implementation of FNV-1a hash.
 *
 * See https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash.
 */
NO_SANITIZE("unsigned-integer-overflow")
static inline uint32_t fnv1a(uint32_t u, uint32_t v) noexcept
{
    return (u ^ v) * fnv_prime;
}

#ifdef __cplusplus
}
#endif
