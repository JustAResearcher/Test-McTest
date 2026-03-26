// ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
// Copyright 2018-2019 Pawel Bylica.
// Licensed under the Apache License, Version 2.0.

/// @file
///
/// ProgPoW API
///
/// This file provides the public API for ProgPoW as the Ethash API extension.

#include <crypto/ethash/include/ethash/ethash.hpp>

namespace meowpow
{
using namespace ethash;  // Include ethash namespace.

/// The MeowPoW v2 algorithm — maximum ASIC resistance targeting RTX 50 series GPUs
constexpr auto revision = "2.0.0";

constexpr int period_length = 3;           // More frequent program changes (was 6)
constexpr uint32_t num_regs = 32;          // Full GPU register utilization (was 16)
constexpr size_t num_lanes = 16;           // Keep 16 lanes (warp half on NVIDIA)
constexpr int num_cache_accesses = 14;     // Heavy random memory pressure (was 6)
constexpr int num_math_operations = 24;    // Diverse ALU saturation (was 9)
constexpr size_t l1_cache_size = 128 * 1024;  // Fill RTX 50 shared memory (was 16KB)
constexpr size_t l1_cache_num_items = l1_cache_size / sizeof(uint32_t);

result hash(const epoch_context& context, int block_number, const hash256& header_hash,
    uint64_t nonce) noexcept;

result hash(const epoch_context_full& context, int block_number, const hash256& header_hash,
    uint64_t nonce) noexcept;

bool verify(const epoch_context& context, int block_number, const hash256& header_hash,
    const hash256& mix_hash, uint64_t nonce, const hash256& boundary) noexcept;

hash256 hash_no_verify(const int& block_number, const hash256& header_hash,
    const hash256& mix_hash, const uint64_t& nonce) noexcept;

search_result search_light(const epoch_context& context, int block_number,
    const hash256& header_hash, const hash256& boundary, uint64_t start_nonce,
    size_t iterations) noexcept;

search_result search(const epoch_context_full& context, int block_number,
    const hash256& header_hash, const hash256& boundary, uint64_t start_nonce,
    size_t iterations) noexcept;

}  // namespace progpow
