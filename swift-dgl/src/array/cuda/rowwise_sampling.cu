/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/rowwise_sampling.cu
 * \brief uniform rowwise sampling
 */

#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <curand_kernel.h>
#include <numeric>
#include <cmath>

#include "./dgl_cub.cuh"
#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"


using namespace dgl::aten::cuda;

namespace dgl {
namespace aten {
namespace impl {

namespace {

constexpr int BLOCK_SIZE = 128;

/**
* @brief Compute the size of each row in the sampled CSR, without replacement.
*
* @tparam IdType The type of node and edge indexes.
* @param num_picks The number of non-zero entries to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The index where each row's edges start.
* @param out_deg The size of each row in the sampled matrix, as indexed by
* `in_rows` (output).
*/
template<typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(static_cast<IdType>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

template<typename IdType>
__global__ void _CSRRowWiseSampleDegreeWithEdgeKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x*blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(static_cast<IdType>(num_picks), in_ptr[in_row+1]-in_ptr[in_row]);

    if (out_row == num_rows-1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
* @brief Compute the size of each row in the sampled CSR, with replacement.
*
* @tparam IdType The type of node and edge indexes.
* @param num_picks The number of non-zero entries to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The index where each row's edges start.
* @param out_deg The size of each row in the sampled matrix, as indexed by
* `in_rows` (output).
*/
template<typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row + 1] - in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
* @brief Perform row-wise uniform sampling on a CSR matrix,
* and generate a COO matrix, without replacement.
*
* @tparam IdType The ID type used for matrices.
* @tparam TILE_SIZE The number of rows covered by each threadblock.
* @param rand_seed The random seed to use.
* @param num_picks The number of non-zeros to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The indptr array of the input CSR.
* @param in_index The indices array of the input CSR.
* @param data The data array of the input CSR.
* @param out_ptr The offset to write each row to in the output COO.
* @param out_rows The rows of the output COO (output).
* @param out_cols The columns of the output COO (output).
* @param out_idxs The data array of the output COO (output).
*/
template<typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}

template<typename IdType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleWithEdgeKernel(
    const uint64_t rand_seed,
    int64_t num_picks,      // fanout
    int64_t num_rows,       // seedNUM
    IdType * in_rows, // seeds
    IdType * in_ptr,
    IdType * in_index,  
    IdType * out_ptr, // write place
    IdType * out_rows,      // dst
    IdType * out_cols) {    // src
  // we assign one warp per row
 
  assert(BLOCK_CTAS == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   
  
  curandStateXORWOW_t local_state;
  curand_init(rand_seed+idx,0,0,&local_state);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_CTAS) {
      if (index < num_rows) {
        const int64_t row = in_rows[index]; 
        const int64_t in_row_start = in_ptr[row]; 
        const int64_t deg = in_ptr[row+1] - in_row_start; 
        const int64_t out_row_start = out_ptr[index]; 
        if (deg <= num_picks) { 
            size_t j = 0;
            for (; j < deg; ++j) {
                out_rows[out_row_start + j] = row;
                out_cols[out_row_start + j] = in_index[in_row_start + j];
            }
        } else {
          for (int j = 0; j < num_picks; ++j) {
              int selected_j = curand(&local_state) % (deg - j);
              int selected_node_id = in_index[in_row_start + selected_j];
              out_rows[out_row_start + j] = row;
              out_cols[out_row_start + j] = selected_node_id;
              in_index[in_row_start + selected_j] = in_index[in_row_start+deg-j-1];
              in_index[in_row_start+deg-j-1] = selected_node_id;
          }
        } 
      }
    }
}

template<typename IdType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleWithTsRecentKernel(
    const uint64_t rand_seed,
    int64_t num_picks,      // fanout
    int64_t num_rows,       // seedNUM
    IdType * in_rows, // seeds
    IdType * in_ptr,
    IdType * in_index,  
    IdType * sample_mask,  
    IdType * cur_ts,
    IdType * dts,
    IdType * total_ts,  
    IdType * total_eid,  
    IdType * out_rows,      
    IdType * out_cols,
    IdType * out_ts,      
    IdType * out_eid) {    
  // we assign one warp per row
 
  assert(BLOCK_CTAS == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   
  
  curandStateXORWOW_t local_state;
  curand_init(rand_seed+idx,0,0,&local_state);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_CTAS) {
      if (index < num_rows) {
        const int64_t row = in_rows[index];
        const int64_t mask_row = sample_mask[index];
        if (mask_row > 0){
          continue;
        }
        const int64_t in_row_start = in_ptr[row];
        int64_t deg = in_ptr[row+1] - in_row_start;
        const int64_t out_row_start = index * num_picks;

        const double ts = cur_ts[index];

        int64_t left = in_row_start;
        int64_t right = in_ptr[row+1] - 1;
        if (left > right){
          continue;
        }

        int64_t res = -1;
        while(left < right){
            int64_t middle = (left+right)/2;

            if(total_ts[middle] < ts && total_ts[middle + 1] >= ts){
              res = middle;
              break;
            }else{
              if(total_ts[middle] < ts)
                left = middle+1;
              else
                right = middle-1;
            }
        }
        if (res == -1){
          if (total_ts[left] < ts){
            res = left;
          }else{
            res = in_row_start - 1;
          }
        }
            

        
        deg = res - in_row_start + 1;

        int64_t picks = deg < num_picks ? deg : num_picks;

        // out_rows[out_row_start] = mask_row;
        for (int j = 0; j < picks; ++j) {
          
          dts[out_row_start + j] = ts;
          out_rows[out_row_start + j] = index;
          out_cols[out_row_start + j] = in_index[res - j];
          out_eid[out_row_start + j] = total_eid[res - j];
          out_ts[out_row_start + j] = total_ts[res - j];
        }






        // if (deg <= num_picks){
        //   size_t j = 0;
        //   for (; j < deg; ++j) {
        //       out_rows[out_row_start + j] = row;
        //       out_cols[out_row_start + j] = in_index[in_row_start + j];
        //       out_eid[out_row_start + j] = total_eid[in_row_start + j];
        //       out_ts[out_row_start + j] = total_ts[in_row_start + j];
        //   }
        // }else{
        //   for (int j = 0; j < num_picks; ++j) {
        //       out_rows[out_row_start + j] = row;
        //       out_cols[out_row_start + j] = in_index[res - j - 1];
        //       out_eid[out_row_start + j] = total_eid[res - j - 1];
        //       out_ts[out_row_start + j] = total_ts[res - j - 1];
        //   }
        // }

        
        // if (deg <= num_picks) {
        //     size_t j = 0;
        //     for (; j < deg; ++j) {
        //         out_rows[out_row_start + j] = row;
        //         out_cols[out_row_start + j] = in_index[in_row_start + j];
        //         out_eid[out_row_start + j] = total_eid[in_row_start + j];
        //         total_ts[out_row_start + j] = ts;
        //     }
        // } else {
        //   for (int j = 0; j < num_picks; ++j) {
        //       int selected_j = curand(&local_state) % (deg - j);
        //       int selected_node_id = in_index[in_row_start + selected_j];
        //       out_rows[out_row_start + j] = row;
        //       out_cols[out_row_start + j] = selected_node_id;
        //       // in_index[in_row_start + selected_j] = in_index[in_row_start+deg-j-1];
        //       // in_index[in_row_start+deg-j-1] = selected_node_id;
        //   }
        // } 
      }
    }
}

template<typename IdType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleWithExpiredKernel(
    const uint64_t rand_seed,
    int64_t num_picks,      // fanout
    int64_t num_rows,       // seedNUM
    int64_t cur_block,       
    int64_t zombie_block,   
    IdType * in_rows, // seeds
    IdType * in_ptr,
    IdType * in_index,  
    IdType * expired,
    IdType * cur_ts,
    IdType * total_ts,  
    IdType * total_eid,  
    IdType * out_ptr, // write place
    IdType * out_rows,      
    IdType * out_cols,
    IdType * out_ts,      
    IdType * out_eid) {    
  // we assign one warp per row
 
  assert(BLOCK_CTAS == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   
  
  curandStateXORWOW_t local_state;
  curand_init(rand_seed+idx,0,0,&local_state);

  
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_CTAS) {
      if (index < num_rows) {
        const int64_t row = in_rows[index]; 
       
        const int64_t in_row_start = in_ptr[row]; 
        int64_t deg = in_ptr[row+1] - in_row_start;
        const int64_t out_row_start = index * num_picks; 

        const double ts = cur_ts[index];

        int64_t left = in_row_start;
        int64_t right = in_ptr[row+1] - 1;

        int64_t res = -1;
        // while(left < right){
        //     int64_t middle = (left+right) / 2;

        //     if(total_ts[middle] <= ts && total_ts[middle + 1] > ts){
        //       res = middle;
        //       break;
        //     }else{
        //       if(total_ts[middle] <= ts)
        //         left = middle+1;
        //       else
        //         right = middle-1;
        //     }
        // }
        // if (res == -1){
        //   if (total_ts[left] <= ts){
        //     res = left;
        //   }else{
        //     res = in_row_start - 1; 
        //   }
        // }

        if (left > right){
          continue;
        }
        
        while(left < right){
            int64_t middle = (left+right)/2;

            if(total_ts[middle] < ts && total_ts[middle + 1] >= ts){
              res = middle;
              break;
            }else{
              if(total_ts[middle] < ts)
                left = middle+1;
              else
                right = middle-1;
            }
        }
        if (res == -1){
          if (total_ts[left] < ts){
            res = left;
          }else{
            res = in_row_start - 1; 
          }
        }
            

        
        deg = res - in_row_start + 1; 

        int64_t picks = deg < num_picks ? deg : num_picks;

        // out_rows[out_row_start] = mask_row;
        int64_t j = 0;

        for (; j < picks; ++j) {

        }

        while(j < deg && (res - j) >= in_row_start){
          if (expired[res - j] == 1){
            break;
          }
          expired[res - j] = 1;
          ++j;
        }


      }
    }
}

template<typename IdType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleWithEdgeAndMapKernel(
    const uint64_t rand_seed,
    int64_t num_picks,      // fanout
    int64_t num_rows,       // seedNUM
    IdType * in_rows,       // seeds
    IdType * in_ptr,
    IdType * in_index,  
    // IdType * out_ptr,       // write place
    IdType * out_rows,      // dst
    IdType * out_cols,      // src
    IdType * in_mapTable) {    

  assert(BLOCK_CTAS == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   
  
  curandStateXORWOW_t local_state;
  curand_init(rand_seed+idx,0,0,&local_state);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_CTAS) {
      if (index < num_rows) {
        const int64_t row = in_rows[index];
        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row+1] - in_row_start;
        // const int64_t out_row_start = out_ptr[index];
        const int64_t out_row_start = index * num_picks;
        if (deg <= num_picks) {
            size_t j = 0;
            int picked = 0;
            for (; j < deg; ++j) {
                int srcid = in_index[in_row_start + j];
                if(in_mapTable[srcid] < 0)
                  continue;
                else{
                  out_rows[out_row_start + picked] = row;
                  out_cols[out_row_start + picked] = srcid;
                  picked++;
                }
            }
            for (; picked < num_picks; ++picked) {
              out_rows[out_row_start + picked] = -1;
              out_cols[out_row_start + picked] = -1;
            }

        } else {
          int picked = 0;
          for (int j = 0; j < deg && picked < num_picks  ; ++j) {
              int selected_j = curand(&local_state) % (deg - j);
              int selected_node_id = in_index[in_row_start + selected_j];
              in_index[in_row_start + selected_j] = in_index[in_row_start+deg-j-1];
              in_index[in_row_start+deg-j-1] = selected_node_id;
              if(in_mapTable[selected_node_id] < 0)
                  continue;
              else {
                out_rows[out_row_start + picked] = row;
                out_cols[out_row_start + picked] = selected_node_id;
                picked++;
              }
          }

          for (; picked < num_picks; ++picked) {
            out_rows[out_row_start + picked] = -1;
            out_cols[out_row_start + picked] = -1;
          }
        } 
      }
    }
}

template <typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(const IdType running_total)
      : running_total_(running_total) {}

  __device__ IdType operator()(const IdType block_aggregate) {
    const IdType old_prefix = running_total_;
    running_total_ += block_aggregate;
    return old_prefix;
  }
};

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_compact_edge(
  const int *tmp_src, 
  const int *tmp_dst,
  int *out_src, 
  int *out_dst, 
  size_t *num_out,
  const size_t *item_prefix, 
  const int num_input,
  const int fanout) {
  assert(BLOCK_SIZE == blockDim.x);
  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;
  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  __shared__ typename BlockScan::TempStorage temp_space;
  const size_t offset = item_prefix[blockIdx.x];
  BlockPrefixCallbackOp<size_t> prefix_op(0);
  for (int i = 0; i < VALS_PER_THREAD; ++i) {
    const size_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    size_t item_per_thread = 0;
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (tmp_src[index * fanout + j] != -1) {
          item_per_thread++;
        }
      }
    }

    size_t item_prefix_per_thread = item_per_thread;
    BlockScan(temp_space)
        .ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread,
                      prefix_op);
    __syncthreads();
    
    for (size_t j = 0; j < item_per_thread; j++) {
      out_src[offset + item_prefix_per_thread + j] =
          tmp_src[index * fanout + j];
      out_dst[offset + item_prefix_per_thread + j] =
          tmp_dst[index * fanout + j];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_out = item_prefix[gridDim.x];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_count_edge(
  int *edge_src, 
  size_t *item_prefix,
  const size_t num_input, 
  const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);
  using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (edge_src[index * fanout + j] != -1) {
          ++count;
        }
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}


/**
* @brief Perform row-wise uniform sampling on a CSR matrix,
* and generate a COO matrix, with replacement.
*
* @tparam IdType The ID type used for matrices.
* @tparam TILE_SIZE The number of rows covered by each threadblock.
* @param rand_seed The random seed to use.
* @param num_picks The number of non-zeros to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The indptr array of the input CSR.
* @param in_index The indices array of the input CSR.
* @param data The data array of the input CSR.
* @param out_ptr The offset to write each row to in the output COO.
* @param out_rows The rows of the output COO (output).
* @param out_cols The columns of the output COO (output).
* @param out_idxs The data array of the output COO (output).
*/
template<typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
        out_idxs[out_idx] = data ? data[in_row_start + edge] : in_row_start + edge;
      }
    }
    out_row += 1;
  }
}
}  // namespace


///////////////////////////// CSR sampling //////////////////////////

template <DLDeviceType XPU, typename IdType>
int32_t CSRSamplingWithEdgeUniform(
    IdArray& indptr ,IdArray& indices,
    IdArray& sampleIDs ,int seedNUM, int num_picks,
    IdArray& outSRC, IdArray& outDST
) {
  //std::cout << "cuda CSRSamplingWithEdgeUniform func success..." << std::endl;
  const auto& ctx = indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = 0;

  int64_t num_rows = sampleIDs->shape[0];
  IdType * slice_rows = static_cast<IdType*>(sampleIDs->data);

  IdType* in_ptr = static_cast<IdType*>(indptr->data);
  IdType* in_cols = static_cast<IdType*>(indices->data);
  IdType* out_rows = static_cast<IdType*>(outDST->data);
  IdType* out_cols = static_cast<IdType*>(outSRC->data);

  // compute degree
  IdType * out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));

  
  const dim3 block_(512);
  const dim3 grid_((num_rows+block_.x-1)/block_.x);
  _CSRRowWiseSampleDegreeWithEdgeKernel<<<grid_, block_, 0, stream>>>(
      num_picks, num_rows, slice_rows, in_ptr, out_deg);
  

  
  // fill out_ptr
  // because the elements number picked per row (out_deg info) is given already, so we can determine the ind_ptr now
  IdType * out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  void * prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_temp, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  IdType new_len;
  device->CopyDataFromTo(out_ptr, num_rows*sizeof(new_len), &new_len, 0,
        sizeof(new_len),
        ctx,
        DGLContext{kDLCPU, 0},
        indptr->dtype);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (num_rows + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);
  _CSRRowWiseSampleWithEdgeKernel<IdType, blockSize, slice><<<grid, block, 0, stream>>>(
      random_seed,
      num_picks,
      num_rows,
      slice_rows,
      in_ptr,
      in_cols,
      out_ptr,
      out_rows,
      out_cols);

  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  outDST = outDST.CreateView({new_len}, outDST->dtype);
  outSRC = outSRC.CreateView({new_len}, outSRC->dtype);
  return new_len;
}

template int32_t CSRSamplingWithEdgeUniform<kDLGPU, int32_t>(
    IdArray& ,IdArray&,IdArray& ,int, int, IdArray& ,IdArray&);
template int32_t CSRSamplingWithEdgeUniform<kDLGPU, int64_t>(
    IdArray&  ,IdArray& ,IdArray& ,int , int ,IdArray& , IdArray&);
    
 
template <DLDeviceType XPU, typename IdType>
int32_t CSRSamplingWithTsRecentUniform(
    IdArray& indptr ,IdArray& indices,IdArray& curts ,IdArray& dts ,IdArray& totalts,IdArray& totaleid,
    IdArray& sampleIDs ,IdArray& sampleMask ,int seedNUM, int num_picks,
    IdArray& outSRC, IdArray& outDST,IdArray& outts, IdArray& outeid
) {
  const auto& ctx = indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = 0;

  int64_t num_rows = sampleIDs->shape[0];
  IdType * slice_rows = static_cast<IdType*>(sampleIDs->data);

  IdType* in_ptr = static_cast<IdType*>(indptr->data);
  IdType* in_cols = static_cast<IdType*>(indices->data);
  IdType* cur_ts = static_cast<IdType*>(curts->data);
  IdType* d_ts = static_cast<IdType*>(dts->data);
  IdType* total_ts = static_cast<IdType*>(totalts->data);
  IdType* sample_mask = static_cast<IdType*>(sampleMask->data);
  IdType* total_eid = static_cast<IdType*>(totaleid->data);
  IdType* out_rows = static_cast<IdType*>(outDST->data);
  IdType* out_cols = static_cast<IdType*>(outSRC->data);
  IdType* out_ts = static_cast<IdType*>(outts->data);
  IdType* out_eid = static_cast<IdType*>(outeid->data);

  // // compute degree
  // IdType * out_deg = static_cast<IdType*>(
  //     device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));

  
  // const dim3 block_(512);
  // const dim3 grid_((num_rows+block_.x-1)/block_.x);
  // _CSRRowWiseSampleDegreeWithEdgeKernel<<<grid_, block_, 0, stream>>>(
  //     num_picks, num_rows, slice_rows, in_ptr, out_deg);
  

  
  // // fill out_ptr
  // // because the elements number picked per row (out_deg info) is given already, so we can determine the ind_ptr now
  // IdType * out_ptr = static_cast<IdType*>(
  //     device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  // size_t prefix_temp_size = 0;
  // CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
  //     out_deg,
  //     out_ptr,
  //     num_rows+1,
  //     stream));
  // void * prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  // CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_temp, prefix_temp_size,
  //     out_deg,
  //     out_ptr,
  //     num_rows+1,
  //     stream));
  // device->FreeWorkspace(ctx, prefix_temp);
  // device->FreeWorkspace(ctx, out_deg);

  // cudaEvent_t copyEvent;
  // CUDA_CALL(cudaEventCreate(&copyEvent));

  // IdType new_len;
  // device->CopyDataFromTo(out_ptr, num_rows*sizeof(new_len), &new_len, 0,
  //       sizeof(new_len),
  //       ctx,
  //       DGLContext{kDLCPU, 0},
  //       indptr->dtype);
  // CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (num_rows + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);
  _CSRRowWiseSampleWithTsRecentKernel<IdType, blockSize, slice><<<grid, block, 0, stream>>>(
      random_seed,
      num_picks,
      num_rows,
      slice_rows,
      in_ptr,
      in_cols,
      sample_mask,
      cur_ts,
      d_ts,
      total_ts,
      total_eid,
      out_rows,
      out_cols,
      out_ts,
      out_eid);

  // device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  // CUDA_CALL(cudaEventSynchronize(copyEvent));
  // CUDA_CALL(cudaEventDestroy(copyEvent));
  // outDST = outDST.CreateView({new_len}, outDST->dtype);
  // outSRC = outSRC.CreateView({new_len}, outSRC->dtype);
  // return new_len;
}

   
template int32_t CSRSamplingWithTsRecentUniform<kDLGPU, int32_t>(
    IdArray& ,IdArray& ,IdArray& ,IdArray& ,IdArray& ,IdArray&,IdArray&,IdArray&,int , int ,IdArray& , IdArray&,IdArray& , IdArray&);


 
template <DLDeviceType XPU, typename IdType>
int32_t CSRSamplingWithExpiredUniform(
    IdArray& indptr ,IdArray& indices,IdArray& curts ,IdArray& totalts,IdArray& totaleid,
    IdArray& sampleIDs ,IdArray& sampleMask ,int seedNUM, int num_picks,int cur_block, int zombie_block,
    IdArray& outSRC, IdArray& outDST,IdArray& outts, IdArray& outeid
) {
  const auto& ctx = indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = 0;

  int64_t num_rows = sampleIDs->shape[0];
  IdType * slice_rows = static_cast<IdType*>(sampleIDs->data);

  IdType* in_ptr = static_cast<IdType*>(indptr->data);
  IdType* in_cols = static_cast<IdType*>(indices->data);
  IdType* cur_ts = static_cast<IdType*>(curts->data);
  IdType* total_ts = static_cast<IdType*>(totalts->data);
  IdType* sample_mask = static_cast<IdType*>(sampleMask->data);
  IdType* total_eid = static_cast<IdType*>(totaleid->data);
  IdType* out_rows = static_cast<IdType*>(outDST->data);
  IdType* out_cols = static_cast<IdType*>(outSRC->data);
  IdType* out_ts = static_cast<IdType*>(outts->data);
  IdType* out_eid = static_cast<IdType*>(outeid->data);

  // compute degree
  IdType * out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));

  
  const dim3 block_(512);
  const dim3 grid_((num_rows+block_.x-1)/block_.x);
  _CSRRowWiseSampleDegreeWithEdgeKernel<<<grid_, block_, 0, stream>>>(
      num_picks, num_rows, slice_rows, in_ptr, out_deg);
  

  
  // fill out_ptr
  // because the elements number picked per row (out_deg info) is given already, so we can determine the ind_ptr now
  IdType * out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  void * prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_temp, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  IdType new_len;
  device->CopyDataFromTo(out_ptr, num_rows*sizeof(new_len), &new_len, 0,
        sizeof(new_len),
        ctx,
        DGLContext{kDLCPU, 0},
        indptr->dtype);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  
  const int slice = 1024;
  const int blockSize = 256;
  int steps = (num_rows + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);
  _CSRRowWiseSampleWithExpiredKernel<IdType, blockSize, slice><<<grid, block, 0, stream>>>(
      random_seed,
      num_picks,
      num_rows,
      cur_block,
      zombie_block,
      slice_rows,
      in_ptr,
      in_cols,
      sample_mask,
      cur_ts,
      total_ts,
      total_eid,
      out_ptr,
      out_rows,
      out_cols,
      out_ts,
      out_eid);

  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  outDST = outDST.CreateView({new_len}, outDST->dtype);
  outSRC = outSRC.CreateView({new_len}, outSRC->dtype);
  return new_len;
}

   
template int32_t CSRSamplingWithExpiredUniform<kDLGPU, int32_t>(
    IdArray& ,IdArray& ,IdArray& ,IdArray& ,IdArray& ,IdArray&,IdArray&,int,int,int,int,IdArray& , IdArray&,IdArray& , IdArray&);


template <DLDeviceType XPU, typename IdType>
int32_t CSRSamplingWithEdgeAndMapTable(
	IdArray& indptr ,IdArray& indices,
  IdArray& sampleIDs ,int seedNUM, int fanNUM,
  IdArray& outSRC, IdArray& outDST, IdArray& mapTable
){
  const auto& ctx = indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = 0;

  int64_t num_rows = sampleIDs->shape[0];
  IdType * slice_rows = static_cast<IdType*>(sampleIDs->data);
  IdType* in_ptr = static_cast<IdType*>(indptr->data);
  IdType* in_cols = static_cast<IdType*>(indices->data);
  IdType* out_rows = static_cast<IdType*>(outDST->data);
  IdType* out_cols = static_cast<IdType*>(outSRC->data);
  IdType* in_mapTable = static_cast<IdType*>(mapTable->data);
  IdType *tmp_src = static_cast<IdType *>(device->AllocWorkspace(ctx,sizeof(IdType) * seedNUM * fanNUM));
  IdType *tmp_dst = static_cast<IdType *>(device->AllocWorkspace(ctx,sizeof(IdType) * seedNUM * fanNUM));

  const int slice = 1024;
  const int blockSize = 256;
  int steps = (num_rows + slice - 1) / slice;
  dim3 grid(steps);
  dim3 block(blockSize);
  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  _CSRRowWiseSampleWithEdgeAndMapKernel<IdType, blockSize, slice><<<grid, block, 0, stream>>>(
      random_seed,
      fanNUM,
      num_rows,
      slice_rows,
      in_ptr,
      in_cols,
      tmp_dst,
      tmp_src,
      in_mapTable);

  CUDA_CALL(cudaDeviceSynchronize());

  size_t *item_prefix = static_cast<size_t *>(device->AllocWorkspace(ctx,sizeof(size_t) * (grid.x + 1)));
  sample_count_edge<blockSize, slice>
    <<<grid, block>>>(tmp_src, item_prefix, seedNUM, fanNUM);
  CUDA_CALL(cudaDeviceSynchronize());

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
      static_cast<size_t *>(nullptr), grid.x + 1));
  CUDA_CALL(cudaDeviceSynchronize());

  void *workspace = device->AllocWorkspace(ctx,workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1));
  CUDA_CALL(cudaDeviceSynchronize());

  size_t *num_out = static_cast<size_t *>(device->AllocWorkspace(ctx,sizeof(size_t) * 1));
  sample_compact_edge<blockSize, slice>
    <<<grid, block>>>(tmp_src, tmp_dst, out_cols, out_rows,
                                    num_out, item_prefix, seedNUM, fanNUM);
  CUDA_CALL(cudaDeviceSynchronize());

  IdType new_len;
  device->CopyDataFromTo(num_out, 0, &new_len, 0,
        sizeof(new_len),
        ctx,
        DGLContext{kDLCPU, 0},
        indptr->dtype);

  device->FreeWorkspace(ctx,workspace);
  device->FreeWorkspace(ctx,item_prefix);
  device->FreeWorkspace(ctx,tmp_src);
  device->FreeWorkspace(ctx,tmp_dst);

  return new_len;
}

template int32_t CSRSamplingWithEdgeAndMapTable<kDLGPU, int32_t>(
    IdArray& ,IdArray&,IdArray& ,int, int, IdArray& ,IdArray& ,IdArray&);
// template int32_t CSRSamplingWithEdgeAndMapTable<kDLGPU, int64_t>(
//     IdArray&  ,IdArray& ,IdArray& ,int , int ,IdArray& , IdArray& ,IdArray&);

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat,
                                    IdArray rows,
                                    const int64_t num_picks,
                                    const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType * const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row = NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col = NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx = NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  const IdType * const in_ptr = static_cast<const IdType*>(mat.indptr->data);
  const IdType * const in_cols = static_cast<const IdType*>(mat.indices->data);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* const data = CSRHasData(mat) ?
      static_cast<IdType*>(mat.data->data) : nullptr;

  // compute degree
  IdType * out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeReplaceKernel,
        grid, block, 0, stream,
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeKernel,
        grid, block, 0, stream,
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType * out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  void * prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_temp, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  // TODO(dlasalle): use pinned memory to overlap with the actual sampling, and wait on
  // a cudaevent
  IdType new_len;
  // copy using the internal current stream
  device->CopyDataFromTo(out_ptr, num_rows * sizeof(new_len), &new_len, 0,
      sizeof(new_len),
      ctx,
      DGLContext{kDLCPU, 0},
      mat.indptr->dtype);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>),
        grid, block, 0, stream,
        random_seed,
        num_picks,
        num_rows,
        slice_rows,
        in_ptr,
        in_cols,
        data,
        out_ptr,
        out_rows,
        out_cols,
        out_idxs);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>),
        grid, block, 0, stream,
        random_seed,
        num_picks,
        num_rows,
        slice_rows,
        in_ptr,
        in_cols,
        data,
        out_ptr,
        out_rows,
        out_cols,
        out_idxs);
  }
  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(mat.num_rows, mat.num_cols, picked_row,
      picked_col, picked_idx);
}

template COOMatrix CSRRowWiseSamplingUniform<kDLGPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDLGPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
