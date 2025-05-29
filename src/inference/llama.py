# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 
# This file has been modified by Shiwei Gao and Ruwen Fan 
# to enable LLM inference with NKI framework on NeuronX devices.
# Updated for compatibility with latest AWS Neuron SDK 2.23


"""PyTorch LLaMA model for NXD inference."""
import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
import math
import warnings
from typing import (
    Optional, Tuple, Union, Any, Type
)

import torch.nn.functional as F

from neuronx_distributed.parallel_layers.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    _gather_along_first_dim,  # Added for latest SDK
)
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.utils import cpu_mode  # Added for latest SDK

# Updated imports for latest SDK - includes quantized kernels
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel
)

from torch import nn, ones
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    transpose_parallel_linear_layer,
)

# Updated import for latest SDK
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

# No longer using direct RmsNorm from torch_neuronx.xla_impl.ops
# from torch_neuronx.xla_impl.ops import RmsNorm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa  # Note: Some functions may have moved or changed
import neuronxcc.nki.typing as nt
import numpy as np

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")

NKI_ENABLED = True
CONFIG_FUSE_MLP = True

CONFIG_MLP_FUSE_NONE = False
CONFIG_MLP_FUSE_NORM_ONLY = False
USE_FLASH = False

def cdiv(a, b):
    return (a + b - 1) // b

@nki.jit
def nki_gemm(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    if N <= 512:
        TILE_N = N
    else:
        TILE_N = 512 
    assert K % TILE_K == 0

    mask_m = nl.arange(TILE_M)[None, :]  # shape: [TILE_M, 1]
    mask_n = nl.arange(TILE_N)[None, :]  # shape: [1, TILE_N]

    for m in nl.affine_range(cdiv(M, TILE_M)):
        for n in nl.affine_range(cdiv(N, TILE_N)):
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):

                lhsT_tile = nl.load(
                    lhsT[k * TILE_K : (k + 1) * TILE_K, m * TILE_M : (m + 1) * TILE_M],
                    mask=(mask_m + m * TILE_M) < M,
                )

                rhs_tile = nl.load(
                    rhs[k * TILE_K : (k + 1) * TILE_K, n * TILE_N : (n + 1) * TILE_N],
                    mask=(mask_n + n * TILE_N) < N,
                )

                res_psum += nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)
            mask_m2 = nl.arange(TILE_M)[:,None]  # shape: [TILE_M, 1]
            nl.store(
                result[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
                value=res_psum,
                mask=((mask_m2 + m * TILE_M) < M) & ((mask_n + n * TILE_N) < N)
            )

    return result

@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=4,
    # Meta-parameters
):
    """NKI kernel to compute a matrix multiplication operation while blocking the
       free dimensions of the LHS and RHS to improve memory access pattern.

    Args:
        lhsT: an input tensor of shape [K,M], where both K and M are multiples for
          128.  It is the left-hand-side argument of the matrix multiplication,
          delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
          is a multiple of 512.  It is the right-hand-side argument of the matrix
          multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    if N <= 512:
        TILE_N = N
    else:
        TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Define the indices (shape) of the tiles
    i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

    # Configuring the blocking size for the free dimensions
    # TILES_IN_BLOCK_M = 2
    # TILES_IN_BLOCK_N = 4
    
    if TILES_IN_BLOCK_N > N // TILE_N:
        TILES_IN_BLOCK_N = N // TILE_N

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

    # the size has to be multiple of block size
    # assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0

    # Loop over blocks over the M dimension
    for m in nl.affine_range(cdiv(M,BLOCK_M)):
        # Load TILES_IN_BLOCK_M columns tiles from lhsT
        lhsT_tiles = nl.ndarray(
            (TILES_IN_BLOCK_M, K // TILE_K, nl.par_dim(TILE_K), TILE_M),
            dtype=lhsT.dtype,
            buffer=nl.sbuf,
        )
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            for k in nl.affine_range(K // TILE_K):
                lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[
                        k * TILE_K + i_lhsT.p,
                        (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_lhsT.x
                    ],mask = ((m * TILES_IN_BLOCK_M + bm) * TILE_M + i_lhsT.x < M)
                )

        for n in nl.affine_range(N // BLOCK_N):
            # Load TILES_IN_BLOCK_N columns from rhs
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf,
            )
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for k in nl.affine_range(K // TILE_K):
                    rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
                        rhs[
                            k * TILE_K + i_rhs.p,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x,
                        ]
                    )

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    # Allocate a tensor in PSUM
                    res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        # Accumulate partial-sums into PSUM
                        res_psum += nl.matmul(
                            lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x],
                            rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                            transpose_x=True,
                        )

                    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                    res_sb = nl.copy(res_psum, dtype=result.dtype)
                    nl.store(
                        result[
                            (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_res.p,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_res.x
                        ],
                        mask = (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_res.p < M ,
                        value=res_sb,
                    )

    return result


@nki.jit
def rms_norm_nki_thin_gemm(lhsT, rhs, g_tensor, eps, residual=None):
    M, K = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    if residual is not None:
        residual_result = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.shared_hbm)

    iw = nl.arange(1)[:, None]
    iy = nl.arange(K)[None, :]
    
    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])
    if M != 1:
        g_bcast = g_tile.broadcast_to((M, g_tensor.shape[0]))
    else:
        g_bcast = g_tile
    def micron_kernel(
        TILE_M, TILE_K, TILE_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, n_start, n_end
    ):
        # TILE_M = M  # 128
        # TILE_K = 128  # 128
        # TILE_N = 512  # 512

        # Define the indices (shape) of the tiles
        i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

        # Configuring the blocking size for the free dimensions
        # TILES_IN_BLOCK_M = 1
        # TILES_IN_BLOCK_N = 4

        BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
        BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

        # the size has to be multiple of block size
        assert M % BLOCK_M == 0
        # assert N % BLOCK_N == 0

        # Load TILES_IN_BLOCK_M columns tiles from lhsT
        a_tile = nl.load(lhsT)
        if residual is not None:
            res_tile = nl.load(residual)
            a_tile = nl.add(a_tile, res_tile)
            nl.store(residual_result, value=a_tile)
        
        in_square = nl.square(a_tile)
        square_sum = nl.sum(in_square, axis=[1])

        # Scale and get a reciprocal
        mean = square_sum / K

        rms_reciprocal = nl.rsqrt(mean + eps)

        out_tile = nl.multiply(a_tile, rms_reciprocal)
        out_tile[...] = nl.multiply(out_tile, g_bcast)
        
        lhsT_tiles = nl.ndarray(
            (K // TILE_K, nl.par_dim(TILE_K), TILE_M),
            dtype=lhsT.dtype,
            buffer=nl.sbuf,
        )
        
        for k in nl.affine_range(K // TILE_K):
            lhsT_tiles[k, :, :] = nisa.nc_transpose(
                out_tile[:, k * TILE_K: (k + 1) * TILE_K],
            )
        for n in nl.affine_range((n_end - n_start) // BLOCK_N):
            # Load TILES_IN_BLOCK_N columns from rhs

            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf,
            )
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for k in nl.affine_range(K // TILE_K):
                    rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
                        rhs[
                            k * TILE_K + i_rhs.p,
                            n_start + (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x,
                        ]
                    )
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                res_psum = nl.zeros((M, TILE_N), nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    # Accumulate partial-sums into PSUM
                    res_psum += nisa.nc_matmul(
                        stationary=lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                        moving=rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                    )

                # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                nl.store(
                    result[
                        0:M,
                        n_start
                        + (n * TILES_IN_BLOCK_N + bn)
                        * TILE_N : n_start+(n * TILES_IN_BLOCK_N + bn + 1)
                        * TILE_N,
                    ],
                    value=res_psum,
                )
    micron_kernel(M, 128, 512, 1, 4, 0, N)
    if residual is not None:
        return result, residual_result
    return result


@nki.jit
def nki_thin_gemm(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    def micron_kernel(
        TILE_M, TILE_K, TILE_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, n_start, n_end
    ):
        # TILE_M = M  # 128
        # TILE_K = 128  # 128
        # TILE_N = 512  # 512

        # Define the indices (shape) of the tiles
        i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

        # Configuring the blocking size for the free dimensions
        # TILES_IN_BLOCK_M = 1
        # TILES_IN_BLOCK_N = 4

        BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
        BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

        # Process M in chunks of TILE_M to avoid exceeding partition limits
        for m_block in nl.affine_range(cdiv(M, TILE_M)):
            m_start = m_block * TILE_M
            m_size = min(TILE_M, M - m_start)
            
            # Load TILES_IN_BLOCK_M columns tiles from lhsT
            lhsT_tiles = nl.ndarray(
                (K // TILE_K, nl.par_dim(TILE_K), TILE_M),
                dtype=lhsT.dtype,
                buffer=nl.sbuf,
            )
            for k in nl.affine_range(K // TILE_K):
                # Create mask for loading
                m_mask = i_lhsT.x < m_size
                lhsT_tiles[k, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[
                        k * TILE_K + i_lhsT.p,
                        m_start + i_lhsT.x,
                    ],
                    mask=m_mask
                )
            
            for n in nl.affine_range((n_end - n_start) // BLOCK_N):
                # Load TILES_IN_BLOCK_N columns from rhs
                rhs_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                    dtype=rhs.dtype,
                    buffer=nl.sbuf,
                )
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for k in nl.affine_range(K // TILE_K):
                        rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
                            rhs[
                                k * TILE_K + i_rhs.p,
                                n_start + (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x,
                            ]
                        )
                
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        # Accumulate partial-sums into PSUM
                        res_psum += nisa.nc_matmul(
                            stationary=lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                            moving=rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                        )

                    # Create mask for storing
                    store_mask = i_res.p < m_size
                    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                    nl.store(
                        result[
                            m_start + i_res.p,
                            n_start + (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_res.x,
                        ],
                        value=res_psum,
                        mask=store_mask
                    )
            
            # Handle remainder columns if (n_end - n_start) is not divisible by BLOCK_N
            remainder_n = (n_end - n_start) % BLOCK_N
            if remainder_n > 0:
                n_offset = n_start + ((n_end - n_start) // BLOCK_N) * BLOCK_N
                remaining_tiles = cdiv(remainder_n, TILE_N)
                
                for bn in nl.affine_range(remaining_tiles):
                    tile_start = n_offset + bn * TILE_N
                    tile_width = min(TILE_N, n_end - tile_start)
                    
                    # Load remaining rhs tiles
                    rhs_tile = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                    for k in nl.affine_range(K // TILE_K):
                        n_mask = i_rhs.x < tile_width
                        rhs_tile[k, i_rhs.p, i_rhs.x] = nl.load(
                            rhs[k * TILE_K + i_rhs.p, tile_start + i_rhs.x],
                            mask=n_mask
                        )
                    
                    res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        res_psum += nisa.nc_matmul(
                            stationary=lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                            moving=rhs_tile[k, i_rhs.p, i_rhs.x],
                        )
                    
                    # Store with combined mask
                    store_mask = (i_res.p < m_size) & (i_res.x < tile_width)
                    nl.store(
                        result[m_start + i_res.p, tile_start + i_res.x],
                        value=res_psum,
                        mask=store_mask
                    )
                    
    # Use appropriate tile size based on M
    if M <= 128:
        TILE_M = M
    else:
        TILE_M = 128
        
    if N >=2048:
        first_n_end = (N // (512 * 4)) * (512 * 4)
        micron_kernel(TILE_M, 128, 512, 1, 4, 0, first_n_end)
        # print("first_n_end", first_n_end)
        # print("N", N)
        if first_n_end < N:
            micron_kernel(TILE_M, 128, 128, 1, 1, first_n_end, N)
    elif N == 256:
        micron_kernel(TILE_M, 128, N, 1, 1, 0, N)
    elif N == 512:
        micron_kernel(TILE_M, 128, 512, 1, 1, 0, N)
    elif N == 1024:
        micron_kernel(TILE_M, 128, 512, 1, 2, 0, N)
    elif N == 1536:
        micron_kernel(TILE_M, 128, 512, 1, 3, 0, N)
    else:
        # Handle other N dimensions by using appropriate tile sizes
        if N < 256:
            micron_kernel(TILE_M, 128, 128, 1, 1, 0, N)
        else:
            # Process in chunks of 512
            n_processed = 0
            while n_processed < N:
                chunk_size = min(512, N - n_processed)
                if chunk_size == 512:
                    micron_kernel(TILE_M, 128, 512, 1, 1, n_processed, n_processed + chunk_size)
                else:
                    micron_kernel(TILE_M, 128, 128, 1, 1, n_processed, n_processed + chunk_size)
                n_processed += chunk_size
    return result

configs = {(256, 2048, 1024): (1, 1), (384, 2048, 1024): (2, 2), (512, 2048, 1024): (2, 2), (640, 2048, 1024): (4, 1), (256, 2048, 256): (1, 1), (384, 2048, 256): (1, 1), (512, 2048, 256): (1, 1), (640, 2048, 256): (1, 1), (256, 2048, 8192): (4, 1), (384, 2048, 8192): (4, 2), (512, 2048, 8192): (4, 1), (640, 2048, 8192): (4, 2), (256, 4096, 2048): (2, 4), (384, 4096, 2048): (4, 4), (512, 4096, 2048): (4, 1), (640, 4096, 2048): (4, 4), (256, 1024, 2048): (1, 1), (384, 1024, 2048): (2, 1), (512, 1024, 2048): (2, 1), (640, 1024, 2048): (4, 1)}
def custom_gemx_implement(input_parallel, weight):
    
    
    origin_shape = input_parallel.shape
    bsz = input_parallel.shape[1] * input_parallel.shape[0]
    dim_input = input_parallel.shape[-1]
    dim_output = weight.shape[-1]
    # print(input_parallel.shape,weight.shape,flush=True)
    # print(input_parallel.shape,weight.shape,flush=True)
    input_parallel = input_parallel.view(-1,dim_input)
    assert input_parallel.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    
    # Use nki_thin_gemm for small batch sizes or when dimensions don't align well
    if bsz <= 128:
        output_parallel = nki_thin_gemm(input_parallel.T,weight)
        # print(output_parallel.shape)
    else:
        bsz = input_parallel.shape[0]
        # iterate through the bsz and add to find a config, try 100 times, otherwise use the default config
        test_bsz = (bsz + 127) // 128 * 128
        # Default values
        m = 2
        n = 4
        k = 4  # This was defined but never used, keeping for compatibility
        found_config = False
        
        for i in range(100):
            if (test_bsz,dim_input,dim_output) in configs:
                config = configs[(test_bsz,dim_input,dim_output)]
                m = config[0]
                n = config[1]
                found_config = True
                break
            test_bsz = test_bsz + 128
        
        # Check if the dimensions are suitable for the optimized kernel
        TILE_N = 512 if dim_output > 512 else dim_output
        BLOCK_N = TILE_N * n
        
        # If dimensions don't align well or no config found, use nki_thin_gemm
        if not found_config or dim_output % BLOCK_N != 0 or dim_output < 256:
            output_parallel = nki_thin_gemm(input_parallel.T, weight)
        else:
            # print("dim_input,dim_output,bsz",dim_input,dim_output,test_bsz,m,n,k,flush=True)
            output_parallel = nki_matmul_fully_optimized_(input_parallel.T,weight, TILES_IN_BLOCK_M=m,TILES_IN_BLOCK_N=n)
    
    output = output_parallel.view(origin_shape[0],origin_shape[1],dim_output)
    return output


class CustomColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        global NKI_ENABLED
        if NKI_ENABLED:
            self.weight = transpose_parallel_linear_layer(self.weight)

    def forward(self, input: torch.Tensor, *_: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input, process_group=self.tensor_parallel_group)
        global NKI_ENABLED
        # Matrix multiply.
        if not NKI_ENABLED:
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group,
                reduce_dtype = self.reduce_dtype,
            )
        else:
            # print(input_parallel.shape,self.weight.shape)
            output_parallel = custom_gemx_implement(input_parallel, self.weight)
        # print(input_parallel.squeeze(0).T.shape,self.weight.T.shape)
        # print(output_parallel.shape)
        # print(input_parallel.shape,self.weight.shape,output_parallel.shape)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel, process_group=self.tensor_parallel_group)
            if self.pad and self.pad_size > 0:
                output = torch.narrow(output, -1, 0, self.output_size - self.pad_size)
        else:
            output = output_parallel
        if self.skip_bias_add:
            return output, self.bias
        output = (output + self.bias) if self.bias is not None else output
        return output


class CustomFusedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        global NKI_ENABLED
        if NKI_ENABLED:
            self.weight = transpose_parallel_linear_layer(self.weight)
        self.act_fn = torch.nn.SiLU()

    def forward(self, input: torch.Tensor, rmsnorm, residual, *_: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input, process_group=self.tensor_parallel_group)
        global NKI_ENABLED
        # Matrix multiply.
        if not NKI_ENABLED:
            if residual is not None:
                input_parallel = residual + input_parallel
            residual = input_parallel
            if rmsnorm is not None:
                input_parallel = rmsnorm(input_parallel)
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group,
                reduce_dtype = self.reduce_dtype,
            )
            dim_output = output_parallel.shape[-1]
            gate_output = output_parallel[:, :, : dim_output // 2]
            up_output = output_parallel[:, :, dim_output // 2 :]
            output = self.act_fn(gate_output) * up_output
        else:
            dim_output = self.weight.shape[-1]
            bsz = input_parallel.shape[1] * input_parallel.shape[0]
            if bsz <= 128 and not CONFIG_MLP_FUSE_NONE and residual is not None and rmsnorm is not None:

                origin_shape = input_parallel.shape
                dim_input = input_parallel.shape[-1]
                dim_output = self.weight.shape[-1]
                input_parallel = input_parallel.view(-1,dim_input)
                assert input_parallel.dtype == torch.bfloat16
                assert self.weight.dtype == torch.bfloat16
                residual = residual.view(-1,dim_input)
                if CONFIG_MLP_FUSE_NORM_ONLY:
                    input_parallel = input_parallel + residual
                    residual = input_parallel
                    output_parallel = rms_norm_nki_thin_gemm(input_parallel, self.weight, rmsnorm.weight, rmsnorm.variance_epsilon, None)
                    residual = residual.view(origin_shape[0],origin_shape[1],dim_input)
                    output_parallel = output_parallel.view(origin_shape[0],origin_shape[1],dim_output)
                else:
                    output_parallel, residual = rms_norm_nki_thin_gemm(input_parallel, self.weight, rmsnorm.weight, rmsnorm.variance_epsilon, residual)
                    residual = residual.view(origin_shape[0],origin_shape[1],dim_input)
                    output_parallel = output_parallel.view(origin_shape[0],origin_shape[1],dim_output)
            else:
                # Handle case where residual might be None
                if residual is not None:
                    input_parallel = residual + input_parallel
                residual = input_parallel
                if rmsnorm is not None:
                    input_parallel = nki_rmsnorm_kernel(input_parallel, rmsnorm.weight, rmsnorm.variance_epsilon)
                output_parallel = custom_gemx_implement(input_parallel, self.weight)
            gate_output = output_parallel[:, :, : dim_output // 2]
            up_output = output_parallel[:, :, dim_output// 2 :]
            output = self.act_fn(gate_output) * up_output
       
        if self.skip_bias_add:
            return output, self.bias
        output = (output + self.bias) if self.bias is not None else output
        return output, residual


class CustomRowParallelLinear(RowParallelLinear):

    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        global NKI_ENABLED
        if NKI_ENABLED:
            self.weight = transpose_parallel_linear_layer(self.weight)
            
    def forward(self, input_: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            if self.pad and self.pad_size > 0:
                input_ = torch.nn.functional.pad(input_, (0, self.pad_size))
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_, process_group=self.tensor_parallel_group)
        global NKI_ENABLED
        # Matrix multiply.
        if not NKI_ENABLED:
        # Matrix multiply.
            output_ = self._forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=None,
                async_grad_allreduce=False,
                sequence_parallel_enabled=False,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group,
                reduce_dtype = self.reduce_dtype,
            )
        else:
            output_ = custom_gemx_implement(input_parallel, self.weight)

            
        if self.reduce_output:
            # All-reduce across all the partitions.
            original_dtype = output_.dtype

            output_ = output_.to(self.reduce_dtype)

            if self.sequence_parallel_enabled:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_, self.sequence_dimension, process_group=self.tensor_parallel_group,
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_, process_group=self.tensor_parallel_group,
                )

            output_ = output_.to(original_dtype)

        if self.skip_bias_add:
            return output_, self.bias
        output = (output_ + self.bias) if self.bias is not None else output_
        return output

        
@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[2] == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[2])[None, :]

    num_rows = a_tensor.shape[1]
    
    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently

    for b in range(a_tensor.shape[0]):
        for i in range(math.ceil(a_tensor.shape[1]/128)):
            # Load input data from external memory to on-chip memory
            a_tile = nl.zeros([128, a_tensor.shape[2]], a_tensor.dtype)
            a_tile[...] = nl.load(a_tensor[b, i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

            # Compute element-wise square of a_tensor
            in_square = nl.square(a_tile)

            # Calculate sum of squared elements, along last dimension
            square_sum = nl.sum(in_square, axis=[1])

            # Scale and get a reciprocal
            mean = square_sum / a_tensor.shape[2]

            # Take square root of mean and then reciprocal with
            # rsqrt API (one ISA instruction)
            rms_reciprocal = nl.rsqrt(mean + eps)

            # Scale the input tensor
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            # Broadcast weight along first axis to match tensor shape
            # num_rows_active = min(num_rows - i * 128, 128)
            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

            # Multiply with the RMSNorm weight
            out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

            # store the addition results back to external memory (out_tensor)
            nl.store(out_tensor[b, i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows))

    return out_tensor


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, nki_enabled=False):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps
        self.nki_enabled = nki_enabled

    def forward(self, hidden_states):
        if self.nki_enabled:
            out_tensor = nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
            return out_tensor

        # Updated for latest SDK - use CustomRMSNorm from neuronx_distributed_inference
        # instead of direct RmsNorm from torch_neuronx.xla_impl.ops
        # The CustomRMSNorm in the latest SDK handles this properly
        return CustomRMSNorm.forward(self, hidden_states)


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    # Updated to use cpu_mode() from latest SDK
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def get_modules_to_not_convert(neuron_config: NeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a Llama model.

    This function creates a list of InferenceConfig objects, one for each layer. It
    modifies the configurations for certain layers based on which modules should not
    be converted to quantized format. The function uses get_modules_to_not_convert()
    to determine which modules should not be converted.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version
                           with "quantized_mlp_kernel_enabled" as False for that specific layer.
    """
    updated_configs = []
    modules_to_not_convert = get_modules_to_not_convert(config.neuron_config)
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for i in range(config.num_hidden_layers):
        # If any of the MLP modules for this layer are in modules_to_not_convert
        module_pattern = f"layers.{i}.mlp"
        if any(module_pattern in module for module in modules_to_not_convert):
            non_quant_config = copy.deepcopy(config)
            non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
            non_quant_config.neuron_config.activation_quantization_type = None
            non_quant_config.neuron_config.quantize_clamp_bound = float("inf")
            updated_configs.append(non_quant_config)
        else:
            updated_configs.append(config)
    return updated_configs


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


def _register_module(key: str, cls: Type[nn.Module]):
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner


def _helper_concat_and_delete_qkv(llama_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        llama_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    llama_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(llama_state_dict, l, "weight")
        if (
            getattr(cfg.neuron_config, 'quantized_mlp_kernel_enabled', False) or getattr(cfg.neuron_config, 'quantized', False)
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            # Only try to concatenate scale if it exists
            if f"layers.{l}.self_attn.q_proj.scale" in llama_state_dict:
                _helper_concat_and_delete_qkv(llama_state_dict, l, "scale")

    gc.collect()

    return llama_state_dict


class NeuronConfigNKI(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nki_enabled = kwargs.pop("enable_nki", False)
        
        # Add missing attributes for compatibility with latest SDK
        # These are attributes expected by the NeuronLlamaMLP class
        if not hasattr(self, 'quantized_kernel_lower_bound'):
            self.quantized_kernel_lower_bound = kwargs.get('quantized_kernel_lower_bound', 0.0)
        
        if not hasattr(self, 'logical_neuron_cores'):
            self.logical_neuron_cores = kwargs.get('logical_neuron_cores', 1)
            
        if not hasattr(self, 'logical_nc_config'):
            self.logical_nc_config = kwargs.get('logical_nc_config', 1)
            
        if not hasattr(self, 'activation_quantization_type'):
            self.activation_quantization_type = kwargs.get('activation_quantization_type', None)
            
        if not hasattr(self, 'quantize_clamp_bound'):
            self.quantize_clamp_bound = kwargs.get('quantize_clamp_bound', float('inf'))
            
        if not hasattr(self, 'fused_rmsnorm_skip_gamma'):
            self.fused_rmsnorm_skip_gamma = kwargs.get('fused_rmsnorm_skip_gamma', False)
            
        if not hasattr(self, 'mlp_kernel_fuse_residual_add'):
            self.mlp_kernel_fuse_residual_add = kwargs.get('mlp_kernel_fuse_residual_add', False)
            
        if not hasattr(self, 'qkv_kernel_fuse_residual_add'):
            self.qkv_kernel_fuse_residual_add = kwargs.get('qkv_kernel_fuse_residual_add', False)
            
        if not hasattr(self, 'is_prefill_stage'):
            self.is_prefill_stage = kwargs.get('is_prefill_stage', False)
            
        if not hasattr(self, 'attn_tkg_builtin_kernel_enabled'):
            self.attn_tkg_builtin_kernel_enabled = kwargs.get('attn_tkg_builtin_kernel_enabled', False)
            
    def is_mlp_quantized(self):
        """Check if MLP is quantized"""
        return getattr(self, 'quantized_mlp_kernel_enabled', False) or getattr(self, 'quantized', False)


class WeightGatheredColumnParallel(ColumnParallelLinear):
    """
    A specialized column-parallel linear layer that implements weight gathering optimization
    for efficient processing of long sequences in transformer models during eagle speculation.

    This layer provides two forward paths:
    1. Standard column-parallel forward (inherited from parent)
    2. Weight-gathered forward for long sequences
    """
    def forward_wg(self, input: torch.Tensor, weight_gather: bool = False):
        """
        Performs the forward pass with optional weight gathering optimization.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len/TP, 2*hidden_size)
            weight_gather (bool): Whether to use weight gathering optimization.
                                Typically True for sequences >= 32K

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If skip_bias_add is False: Output tensor of shape (batch_size, seq_len, hidden_size)
                - If skip_bias_add is True: Tuple of (output tensor, bias)
        """
        if weight_gather:
            weight = _gather_along_first_dim(self.weight, process_group=self.tensor_parallel_group)
            output = self._forward_impl(
                input=input,
                weight=weight,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group
            )

            output = gather_from_sequence_parallel_region(
                output,
                self.sequence_dimension,
                process_group=self.tensor_parallel_group,
            )
            if self.skip_bias_add:
                return output, self.bias

            output = (output + self.bias) if self.bias is not None else output
            return output
        else:
            return self.forward(input)


class LlamaInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfigNKI


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = getattr(self.neuron_config, "mlp_kernel_enabled", False)
        self.fused_rmsnorm_skip_gamma = getattr(self.config.neuron_config, "fused_rmsnorm_skip_gamma", False)
        self.quantized_mlp_kernel_enabled = getattr(self.neuron_config, "quantized_mlp_kernel_enabled", False)
        self.rmsnorm_quantize_kernel_enabled = getattr(self.neuron_config, "rmsnorm_quantize_kernel_enabled", False)
        self.quantized_kernel_lower_bound = getattr(self.neuron_config, "quantized_kernel_lower_bound", 0.0)
        self.quantize_clamp_bound = getattr(self.neuron_config, "quantize_clamp_bound", float('inf'))
        self.logical_neuron_cores = getattr(self.neuron_config, "logical_neuron_cores", 1)
        self.logical_nc_config = getattr(self.neuron_config, "logical_nc_config", 1)
        self.activation_quantization_type = getattr(self.neuron_config, "activation_quantization_type", None)
        mlp_bias = getattr(config, "mlp_bias", False)

        if self.neuron_config.quantized_mlp_kernel_enabled and self.quantize_clamp_bound == float(
            "inf"
        ):
            logging.warning(
                "quantize_clamp_bound is not specified in NeuronConfig. We will use the default value of 1200 for llama models in quantized kernels."
            )
            self.quantize_clamp_bound = 1200.0
        if parallel_state.model_parallel_is_initialized():
            if CONFIG_FUSE_MLP:
                self.gateup = CustomFusedColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = CustomRowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias=mlp_bias,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=self.sequence_parallel_enabled,
                    sequence_dimension=self.sequence_dimension,
                    tensor_model_parallel_group=get_tp_group(config),
                    reduce_dtype=config.neuron_config.rpl_reduce_dtype,
                )
            else:
                self.gate_proj = CustomColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = CustomColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = CustomRowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias=mlp_bias,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=self.sequence_parallel_enabled,
                    sequence_dimension=self.sequence_dimension,
                    tensor_model_parallel_group=get_tp_group(config),
                    reduce_dtype=config.neuron_config.rpl_reduce_dtype,
                )
           

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def _kernel_enabled_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

        # Choose which kernel to call
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        grid = (self.logical_neuron_cores,)

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                up_w,  # up_w
                down_w,  # down_w
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,
                up_w,
                down_w,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(
                output_tensor, process_group=get_tp_group(self.config)
            )

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, rmsnorm, adapter_ids=None, residual=None):
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        if CONFIG_FUSE_MLP:
            gateup_output, residual = self.gateup(x, rmsnorm, residual)
            output = self.down_proj(gateup_output)
            
        else:
            if residual is not None:
                x = residual + x
                residual = x
            if rmsnorm is not None:
                x = rmsnorm(x)
            gate_proj_output = (
                self.gate_proj(x)
                if not is_lora_module(self.gate_proj)
                else self.gate_proj(x, adapter_ids)
            )
            up_proj_output = (
                self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
            )
            down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
            output = (
                self.down_proj(down_proj_input)
                if not is_lora_module(self.down_proj)
                else self.down_proj(down_proj_input, adapter_ids)
            )
        logger.debug(f"MLP output shape {output.shape}")
        return output, residual

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """
        if self.mlp_kernel_enabled:
            fused_rmsnorm = not self.sequence_parallel_enabled
            # Quantized MLP kernel
            # MLP kernel
            return self._kernel_enabled_mlp(
                x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
            )
        else:
            # No kernel
            return self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids, residual=residual)

def smallest_multiple(k, n):
    if k % n == 0:
        return k
    else:
        return (k // n + 1) * n

@nki.jit
def _flash_attention_core(q_local_tile, k, v, 
                            o_buffer, l_buffer, m_buffer,
                            q_tile_idx,
                            local_k_large_tile_idx,
                            kernel_dtype, acc_type,
                            LARGE_TILE_SZ,
                            initialize,
                            B_P_SIZE=128, B_F_SIZE=512, B_D_SIZE=128):
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

    qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        qk_psum = nl.ndarray((nl.par_dim(B_P_SIZE), B_F_SIZE),
                                                dtype=np.float32, buffer=nl.psum)  # (128, 512)

        multiplication_required_selection = q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE


        if multiplication_required_selection:
            qk_psum[:, :] = nl.matmul(q_local_tile, k[:, k_i_b_f_slice], transpose_x=True) # (p(128), 512)
        else:
            qk_psum[:, :] = 0


        left_diagonal_selection = q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
        diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE)

        i_q_p, i_q_f = nl.mgrid[0:B_P_SIZE, 0:B_F_SIZE]
        q_pos = q_tile_idx * B_P_SIZE + i_q_p
        k_pos = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
        pred = q_pos >= k_pos

        qk_select_tmp = nl.ndarray(qk_psum.shape, dtype=qk_psum.dtype, buffer=nl.sbuf)

        # For tiles on and to the right of the diagonal, need to do affine_select.
        if diagonal_and_right_selection:
            qk_select_tmp[...] = qk_psum

            qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                pred=pred,
                on_true_tile=qk_select_tmp, on_false_value=-9984.0, dtype=acc_type)

        qk_res_buf[:, k_i_b_f_slice] = \
            nl.copy(qk_psum, dtype=acc_type, mask=left_diagonal_selection)

        max_local[:, k_i] = nisa.tensor_reduce(
            np.max, qk_res_buf[:, k_i_b_f_slice], axis=(1,), dtype=acc_type,
            negate=False)

    max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1, ),
                                                        dtype=acc_type, negate=False)

    o_previous_scaled = nl.ndarray((nl.par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

    if initialize:
        m_buffer[:, 0] = nl.copy(max_)
        m_current = max_
    else:
        m_previous = nl.copy(m_buffer[:, 0])
        m_buffer[:, 0] = nl.maximum(m_previous, max_) # (128,1)

        m_current = m_buffer[:, 0]
        # Compute scaling factor
        alpha = nisa.activation(np.exp, m_current, bias=m_previous, scale=-1.0)
        o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

    p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)

    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        p_local[:, k_r_i_reduce_slice] = \
            nisa.activation_reduce(np.exp, qk_res_buf[:, k_r_i_reduce_slice],
                                                            bias=-1 * m_current, scale=1.0,
                                                            reduce_op=nl.add, reduce_res=p_partial_sum[:, k_r_i],
                                                            dtype=kernel_dtype)

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    for j in nl.affine_range(LARGE_TILE_SZ // 128):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            p_local_transposed[:, nl.ds(j * 128, 128)] = nisa.dma_transpose(
                p_local[:, nl.ds(j * 128, 128)])
        else:
            p_local_transposed[:, nl.ds(j * 128, 128)] = nisa.nc_transpose(
                p_local[:, nl.ds(j * 128, 128)])

    pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32,
                                         buffer=nl.psum, lazy_initialization=True)
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
                                                             v[k_i, :, :], transpose_x=True) # (128, 128) (p(Br), d)

    if initialize:
        o_buffer[:, :] = nl.copy(pv_psum[:, :])
        l_buffer[:, 0] = nl.add(nl.log(ps), max_)
    else:
        o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

        exp = nisa.activation(nl.exp, m_current, bias=l_buffer[:, 0], scale=-1.0)
        l_buffer[:, 0] = nl.add(m_current, nisa.activation(nl.log, exp, bias=ps))


@nki.jit
def flash_attention_fwd(q, k, v, LARGE_TILE_SZ):
    B_F_SIZE=LARGE_TILE_SZ
    B_P_SIZE=128
    b, h, d, seqlen_q  = q.shape
    B_D_SIZE=d
    _, k_h, _, seqlen_k = k.shape

    assert tuple(v.shape) == (b, k_h, seqlen_k, d), f"Expect shape of V to be {(b, k_h, seqlen_k, d)} (batch, heads, seqlen_k, d_head) but got {v.shape}"
    assert tuple(k.shape) == (b, k_h, d, seqlen_k), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
    kernel_dtype = nl.bfloat16
    acc_type = np.dtype(np.float32)

    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    assert nl.program_ndim() == 2,\
        f'Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!'
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)

    softmax_scale = 1.0 / (d ** 0.5)

    n_tile_q = seqlen_q // B_P_SIZE # since q will be loaded on tensor engine

    PAR_LEN = 512

    assert seqlen_k % LARGE_TILE_SZ == 0, f"Need seqlen_k to be divisible by {LARGE_TILE_SZ} but got {seqlen_k}"
    num_large_k_tile = seqlen_k // LARGE_TILE_SZ

    q_h_per_k_h = h // k_h

    PAR_LEN = min(n_tile_q, PAR_LEN)
    n_remat = cdiv(n_tile_q, PAR_LEN)

    for i_q_h in nl.affine_range(q_h_per_k_h):
        l_buffer = nl.zeros((nl.par_dim(B_P_SIZE), n_tile_q), dtype=acc_type,
                                                buffer=nl.sbuf, lazy_initialization=False)

        for i0 in nl.sequential_range(n_remat):
            o_buffer = nl.zeros((PAR_LEN, nl.par_dim(B_P_SIZE), d), dtype=acc_type,
                                                    buffer=nl.sbuf, lazy_initialization=False)
            m_buffer = nl.zeros((PAR_LEN, nl.par_dim(B_P_SIZE), 1), dtype=acc_type,
                                                    buffer=nl.sbuf, lazy_initialization=False)

            for j in nl.sequential_range(0, num_large_k_tile):
                cur_k_tile = nl.ndarray((nl.par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
                cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, nl.par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
                # print(k.shape, int(batch_id), int(head_id), j, nl.ds(j*LARGE_TILE_SZ, LARGE_TILE_SZ))
                cur_k_tile[:, :] = nl.load(k[batch_id, head_id, :, nl.ds(j*LARGE_TILE_SZ, LARGE_TILE_SZ)])

                load_tile_size = B_P_SIZE

                v_calc = v[batch_id, head_id]
                for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
                    cur_v_tile[v_i, :, :] = nl.load(
                        v_calc[nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :],
                        dtype=cur_v_tile.dtype)
                for i1 in nl.affine_range(PAR_LEN):
                    i = i0 * PAR_LEN + i1

                    forward_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ

                    if (i < n_tile_q) & forward_mask:
                        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                        q_hbm_tile = q[batch_id, head_id * q_h_per_k_h + i_q_h]
                        q_sbuf_tile = nl.load(q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                                                                    dtype=kernel_dtype) # load (d, 128) tile in SBUF
                        q_tile[:, :] = q_sbuf_tile * softmax_scale

                        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                                                o_buffer=o_buffer[i1], l_buffer=l_buffer[:, i], m_buffer=m_buffer[i1],
                                                q_tile_idx=i, local_k_large_tile_idx=j,
                                                kernel_dtype=kernel_dtype, acc_type=acc_type,
                                                LARGE_TILE_SZ=LARGE_TILE_SZ,
                                                initialize=(j == 0),
                                                B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE)

            for i1 in nl.affine_range(PAR_LEN):
                i = i0 * PAR_LEN + i1

                if i < n_tile_q:
                    exp = nisa.activation(np.exp, l_buffer[:, i], bias=m_buffer[i1, :, :],
                                                                scale=-1.0)
                    out = nl.multiply(o_buffer[i1, :, :], exp,
                                                        dtype=kernel_dtype)

                    nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h,
                                         nl.ds(i*B_P_SIZE, B_P_SIZE), :], value=out)
    return o


def flash_decode_core(qk_raw, v_tile, kernel_dtype, B_P_SIZE, calc_mask, acc_type, l_buffer, o_buffer, m_buffer, q_head_per_k):
    qk = nl.where(calc_mask, qk_raw, -9984.0)
    
    qk_max = nisa.tensor_reduce(np.max, qk, axis=(1,)) # (1, 1)

    qk_reduce = nl.ndarray((q_head_per_k, 1), dtype=acc_type, buffer=nl.sbuf, lazy_initialization=True)
    qk_soft = nisa.activation_reduce(np.exp, qk,
                                bias=-1 * qk_max, scale=1.0,
                                reduce_op=nl.add, reduce_res=qk_reduce,
                                dtype=kernel_dtype)

    qk_soft_transposed = nisa.nc_transpose(qk_soft[:, nl.ds(0, B_P_SIZE)])
    
    o_tile = nl.matmul(qk_soft_transposed, v_tile, transpose_x=True)

    o_buffer[...] = o_tile
    m_buffer[...] = qk_max
    l_buffer[...] = qk_reduce

@nki.jit()
def flash_decode(q, k, v, mask):
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)
    bsz, h, d = q.shape
    k_h = k.shape[1]
    q_h_per_k_h = h // k_h
    kernel_dtype = nl.bfloat16
    acc_type = nl.float32
    softmax_scale = 1.0 / (d ** 0.5)
    PAR_LEN = 128

    o = nl.ndarray((bsz, h, d), dtype=kernel_dtype, buffer=nl.shared_hbm)

    k_total_len = k.shape[-1]
    PARTITION_CNT = cdiv(k_total_len, PAR_LEN)
    FULL_PARTITION_CNT = k_total_len // PAR_LEN

    k_calc = k[batch_id, head_id]
    v_calc = v[batch_id, head_id]
    q_calc = q[batch_id, head_id * q_h_per_k_h: (head_id + 1) * q_h_per_k_h]

    o_buffer = nl.ndarray((q_h_per_k_h, PARTITION_CNT, d), dtype=acc_type, buffer=nl.sbuf, lazy_initialization=True)
    m_buffer = nl.ndarray((q_h_per_k_h, PARTITION_CNT), dtype=acc_type, buffer=nl.sbuf, lazy_initialization=True)
    l_buffer = nl.ndarray((q_h_per_k_h, PARTITION_CNT), dtype=acc_type, buffer=nl.sbuf, lazy_initialization=True)

    q_sbuf_tile = nisa.nc_transpose(nl.load(q_calc, dtype=kernel_dtype)) # load (d, 1) tile in SBUF
    q_tile = q_sbuf_tile * softmax_scale

    for par in nl.affine_range(FULL_PARTITION_CNT):
        k_tile = nl.load(k_calc[:, nl.ds(par * PAR_LEN, PAR_LEN)], dtype=kernel_dtype)
        qk = nl.matmul(q_tile, k_tile, transpose_x=True)
        v_tile = nl.load(v_calc[nl.ds(par * PAR_LEN, PAR_LEN), :], dtype=kernel_dtype)
        calc_mask = nl.load(mask[nl.ds(batch_id, 1), par * PAR_LEN: par * PAR_LEN + PAR_LEN]).broadcast_to((q_h_per_k_h, PAR_LEN))
        flash_decode_core(qk, v_tile, kernel_dtype, PAR_LEN, calc_mask, acc_type, l_buffer[:, par], o_buffer[:, par, :], m_buffer[:, par], q_h_per_k_h)

    if FULL_PARTITION_CNT != PARTITION_CNT:
        len_remain = k_total_len - FULL_PARTITION_CNT * PAR_LEN
        k_tile = nl.load(k_calc[:, nl.ds(FULL_PARTITION_CNT * PAR_LEN, len_remain)], dtype=kernel_dtype)
        qk = nl.matmul(q_tile, k_tile, transpose_x=True)
        v_tile = nl.load(v_calc[nl.ds(FULL_PARTITION_CNT * PAR_LEN, len_remain), :], dtype=kernel_dtype)
        calc_mask = nl.load(mask[nl.ds(batch_id, 1), FULL_PARTITION_CNT * PAR_LEN: FULL_PARTITION_CNT * PAR_LEN + len_remain]).broadcast_to((q_h_per_k_h, len_remain))
        flash_decode_core(qk, v_tile, kernel_dtype, len_remain, calc_mask, acc_type, l_buffer[:, FULL_PARTITION_CNT], o_buffer[:, FULL_PARTITION_CNT, :], m_buffer[:, FULL_PARTITION_CNT], q_h_per_k_h)

    qk_new_max = nisa.tensor_reduce(np.max, m_buffer, axis=(1, ), dtype=acc_type, negate=True)
    qk_exp = nisa.activation(np.exp, m_buffer, bias=qk_new_max)

    for par in nl.affine_range(PARTITION_CNT):
        o_buffer[:, par, :] = nl.multiply(o_buffer[:, par, :], qk_exp[:, par])

    l_buffer = nl.multiply(l_buffer, qk_exp)
    scales = nisa.tensor_reduce(nl.add, l_buffer, axis=(1,), dtype=acc_type)

    o_buffer_reduced = nisa.tensor_reduce(nl.add, o_buffer, axis=(1,), dtype=acc_type)

    scales = (1 / scales).broadcast_to((q_h_per_k_h, d))
    out = nl.multiply(o_buffer_reduced, scales)

    nl.store(o[batch_id, head_id * q_h_per_k_h: (head_id + 1) * q_h_per_k_h], value=out)
    return o

def get_suitable_len(size):
    if size <= 128:
        return 128
    return 256

@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(tensor_model_parallel_group=tensor_model_parallel_group)

        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype
        self.is_medusa = config.neuron_config.is_medusa
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.num_cores_per_group = config.num_cores_per_group
        self.bias = getattr(config, "attention_bias", False)
        self.rpl_reduce_dtype = config.neuron_config.rpl_reduce_dtype
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.rms_norm_eps = config.rms_norm_eps

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = self.config.neuron_config.tp_degree
        else:
            self.tp_degree = 1

        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = None

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        logger.debug(
            f"Hello from NeuronLlamaAttention init! Is SP enabled? {self.sequence_parallel_enabled}. Dim? {self.sequence_dimension}"
        )

        self.init_gqa_properties()

        self.init_rope()
        global NKI_ENABLED
        if NKI_ENABLED:
            self.qkv_proj.q_proj = CustomColumnParallelLinear(
                self.qkv_proj.hidden_size,
                self.qkv_proj.num_attention_heads * self.qkv_proj.head_dim,
                bias=self.qkv_proj.bias,
                gather_output=self.qkv_proj.gather_output,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=self.qkv_proj.tensor_model_parallel_group,
            )
            self.qkv_proj.k_proj = CustomColumnParallelLinear(
                self.qkv_proj.hidden_size,
                self.qkv_proj.num_key_value_heads * self.qkv_proj.head_dim,
                bias=self.qkv_proj.bias,
                gather_output=self.qkv_proj.gather_output,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=self.qkv_proj.tensor_model_parallel_group,
            )
            self.qkv_proj.v_proj = CustomColumnParallelLinear(
                self.qkv_proj.hidden_size,
                self.qkv_proj.num_key_value_heads * self.qkv_proj.head_dim,
                bias=self.qkv_proj.bias,
                gather_output=self.qkv_proj.gather_output,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=self.qkv_proj.tensor_model_parallel_group,
            )
            self.o_proj.o_proj = CustomRowParallelLinear(
                self.o_proj.num_attention_heads * self.o_proj.head_dim,
                self.o_proj.hidden_size,
                bias=self.o_proj.bias,
                input_is_parallel=self.o_proj.input_is_parallel,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=False,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=self.o_proj.tensor_model_parallel_group,
                reduce_dtype=self.rpl_reduce_dtype,
            )
            

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        seq_ids: Optional[torch.LongTensor] = None,  # Added for latest SDK
        **kwargs,  # Catch any other unexpected kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        if USE_FLASH:
            if not past_key_value:
                Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
                    position_ids,
                    hidden_states,
                    past_key_value,
                    adapter_ids=adapter_ids,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    rmsnorm=rmsnorm,
                )
                raw_k = K
                raw_v = V
                bsz, h, seq_len, dim = Q.shape
                Q = Q.permute(0, 1, 3, 2)
                K = K.permute(0, 1, 3, 2)
                LARGE_TILE_SZ = get_suitable_len(seq_len)
                q_pad_size = smallest_multiple(seq_len, 128) - seq_len
                kv_pad_size = smallest_multiple(seq_len, LARGE_TILE_SZ) - seq_len
                K = F.pad(K, (0, kv_pad_size))
                V = F.pad(V, (0, 0, 0, kv_pad_size))
                Q = F.pad(Q, (0, q_pad_size))
                
                # Q [bsz, h, dim, seq_len]
                # K [bsz, h, dim, seq_len]
                # V [bsz, h, seq_len, dim]
                
                o = flash_attention_fwd[bsz, self.num_key_value_heads](Q, K, V, LARGE_TILE_SZ)
                o = o.permute(0, 2, 1, 3).view(bsz, -1, h * dim)[:,:seq_len,:]
                o = self.o_proj(o, adapter_ids=adapter_ids)
                return o, (raw_k, raw_v), cos_cache, sin_cache
            else:
                Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
                    position_ids,
                    hidden_states,
                    past_key_value,
                    adapter_ids=adapter_ids,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    rmsnorm=rmsnorm,
                )
                
                raw_k = K
                raw_v = V
                bsz, h, seq_len, dim = Q.shape
                
                k_cache = past_key_value[0]
                v_cache = past_key_value[1]
                Q = Q.squeeze(2)
                K = torch.cat([k_cache, K], dim=-2)
                V = torch.cat([v_cache, V], dim=-2)
                
                K = K.permute(0, 1, 3, 2)
                
                mask = F.pad(attention_mask.squeeze(1).squeeze(1), (0, 1), value=True)
                
                o = flash_decode[bsz, self.num_key_value_heads](Q, K, V, mask)
                
                o = o.view(bsz, 1, self.num_heads * self.head_dim)
                o = self.o_proj(o, adapter_ids=adapter_ids)
                
                return o, (raw_k, raw_v), cos_cache, sin_cache

        # Pass seq_ids to parent class if needed
        o, past, cos, sin = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            active_mask=active_mask,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            seq_ids=seq_ids,  # Pass seq_ids to parent
            **kwargs,  # Pass any other kwargs
        )
        return o, past, cos, sin
        
# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )

            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / self.factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                        self.high_freq_factor - self.low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _LLAMA_MODULE_MAP[config.neuron_config.attn_cls](
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )
        self.mlp = NeuronLlamaMLP(config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls()(
                config.hidden_size,
                eps=config.rms_norm_eps,
                nki_enabled=config.neuron_config.nki_enabled,
            )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
            nki_enabled=config.neuron_config.nki_enabled,
        )
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = getattr(config.neuron_config, 'quantized_mlp_kernel_enabled', False)
        self.rmsnorm_quantize_kernel_enabled = getattr(config.neuron_config, 'rmsnorm_quantize_kernel_enabled', False)
        self.mlp_kernel_fuse_residual_add = getattr(config.neuron_config, 'mlp_kernel_fuse_residual_add', False)
        self.qkv_kernel_fuse_residual_add = getattr(config.neuron_config, 'qkv_kernel_fuse_residual_add', False)
        self.sequence_parallel_enabled = getattr(config.neuron_config, 'sequence_parallel_enabled', False)
        self.is_prefill_stage = getattr(config.neuron_config, 'is_prefill_stage', False)
        self.config = config
        
        if self.is_prefill_stage and hasattr(config.neuron_config, 'is_mlp_quantized') and config.neuron_config.is_mlp_quantized():
            # for CTE, quantized MLP kernel does not support fused rmsnorm
            self.mlp_kernel_fused_rmsnorm = False
        else:
            self.mlp_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,  # Added for latest SDK
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        entry_hidden_states = hidden_states
        
        # Initialize residual if not provided (first layer)
        if residual is None:
            residual = hidden_states
        
        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            rotary_position_ids=rotary_position_ids,
            residual=residual if self.qkv_kernel_fuse_residual_add else None,
            seq_ids=seq_ids,  # Pass seq_ids to attention
            **kwargs,
        )
        
        # Handle different return formats from attention
        if hasattr(attn_output, 'hidden_states'):
            # NamedTuple or similar structure
            hidden_states = attn_output.hidden_states
            present_key_value = attn_output.present_key_value
            cos_cache = getattr(attn_output, 'cos_cache', None)
            sin_cache = getattr(attn_output, 'sin_cache', None)
            attn_residual = getattr(attn_output, 'residual', None)
        else:
            # Tuple format
            hidden_states = attn_output[0]
            present_key_value = attn_output[1]
            cos_cache = attn_output[2] if len(attn_output) > 2 else None
            sin_cache = attn_output[3] if len(attn_output) > 3 else None
            attn_residual = attn_output[4] if len(attn_output) > 4 else None

        # Update residual if returned from attention
        if attn_residual is not None:
            residual = attn_residual

        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            # First residual add handled in the MLP kernel
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )
        else:
            # Add residual for attention output
            hidden_states = residual + hidden_states
            residual = hidden_states
            
            # RMSNorm (fused with MLP kernel when conditions are met)
            if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
                rmsnorm = self.post_attention_layernorm
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                rmsnorm = None
                
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=rmsnorm,
                residual=None,  # Don't pass residual if not fusing
                adapter_ids=adapter_ids,
            )

        # Final residual connection (unless using qkv_kernel_fuse_residual_add for next layer)
        if not self.qkv_kernel_fuse_residual_add:
            hidden_states = residual + hidden_states
            residual = None  # set to None to prevent it from being used again

        # Return 5 elements as expected by the model base class
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class NeuronLlamaModel(NeuronBaseModel):
    """
    The neuron version of the LlamaModel
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        global NKI_ENABLED
        NKI_ENABLED = config.neuron_config.nki_enabled
        
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            self.lm_head = CustomColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        # Use get_updated_configs to handle per-layer configurations
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(conf) for conf in updated_configs])
        
        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps, nki_enabled=config.neuron_config.nki_enabled)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            # replicate fc weights since activations are sequence sharded
            self.fc = WeightGatheredColumnParallel(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True, sequence_dimension=1
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = CustomColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)

def fuse_mlp(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    tp_size = cfg.neuron_config.tp_degree
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        dummpy_concat = torch.cat(
            [
                llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"],
                llama_state_dict[f"layers.{l}.mlp.up_proj.weight"],
            ],
        )
        hidden = llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"].shape[0]
        per_tp_output = hidden // tp_size
        output = torch.zeros_like(dummpy_concat)
        for i in range(tp_size):
            output[2 * i * per_tp_output : (2 * i + 1) * per_tp_output,:] = llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"][i * per_tp_output : (i + 1) * per_tp_output,:]
            output[(2 * i + 1) * per_tp_output : (2 * i + 2) * per_tp_output,:] = llama_state_dict[f"layers.{l}.mlp.up_proj.weight"][i * per_tp_output : (i + 1) * per_tp_output,:]
        llama_state_dict[f"layers.{l}.mlp.gateup.weight"] = output
        del llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"]
        del llama_state_dict[f"layers.{l}.mlp.up_proj.weight"]
        del dummpy_concat

    gc.collect()
    # print(llama_state_dict.keys())
    return llama_state_dict
class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return LlamaForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config
        
        # Handle fused_rmsnorm_skip_gamma transformation
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            """
            for every layer do the following transformations
            gate_w_prime = (gate_w.T * gamma).T
            up_w_prime = (up_w.T * gamma).T
            """
            if (
                getattr(neuron_config, 'fused_rmsnorm_skip_gamma', False)
                and not getattr(neuron_config, 'sequence_parallel_enabled', False)
            ):
                if getattr(neuron_config, 'mlp_kernel_enabled', False):
                    # MLP
                    state_dict[f"layers.{i}.mlp.gate_proj.weight"] = state_dict[
                        f"layers.{i}.mlp.gate_proj.weight"
                    ] * state_dict[f"layers.{i}.post_attention_layernorm.weight"].unsqueeze(0)
                    state_dict[f"layers.{i}.mlp.up_proj.weight"] = state_dict[
                        f"layers.{i}.mlp.up_proj.weight"
                    ] * state_dict[f"layers.{i}.post_attention_layernorm.weight"].unsqueeze(0)

                if getattr(neuron_config, 'qkv_kernel_enabled', False):
                    # QKV
                    state_dict[f"layers.{i}.self_attn.q_proj.weight"] = state_dict[
                        f"layers.{i}.self_attn.q_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
                    state_dict[f"layers.{i}.self_attn.k_proj.weight"] = state_dict[
                        f"layers.{i}.self_attn.k_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
                    state_dict[f"layers.{i}.self_attn.v_proj.weight"] = state_dict[
                        f"layers.{i}.self_attn.v_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
        
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)
        if CONFIG_FUSE_MLP:
            state_dict = fuse_mlp(state_dict, config)
        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict
    def get_compiler_args(self):
        res = super().get_compiler_args()
        
        
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig