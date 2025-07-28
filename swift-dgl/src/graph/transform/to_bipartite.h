/*!
 *  Copyright 2021 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * \file graph/transform/to_bipartite.h
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

#ifndef DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_
#define DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

/**
 * @brief Create a graph block from the set of
 * src and dst nodes (lhs and rhs respectively).
 *
 * @tparam XPU The type of device to operate on.
 * @tparam IdType The type to use as an index.
 * @param graph The graph from which to extract the block.
 * @param rhs_nodes The destination nodes of the block.
 * @param include_rhs_in_lhs Whether or not to include the
 * destination nodes of the block in the sources nodes.
 * @param [in/out] lhs_nodes The source nodes of the block.
 *
 * @return The block and the induced edges.
 */
template<DLDeviceType XPU, typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlock(HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
        bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes);

template<DLDeviceType XPU, typename IdType>
std::tuple<IdArray, IdArray, IdArray>
Trans2ReMap(
IdArray &lhs_nodes,IdArray &rhs_nodes,IdArray &uni_nodes,bool include_rhs_in_lhs);

template<DLDeviceType XPU, typename IdType>
std::tuple<IdArray, IdArray, IdArray>
mapByNodeTable(
IdArray &lhsNode,IdArray &rhsNode,IdArray &uniTable,IdArray &srcList,IdArray &dstList,bool include_rhs_in_lhs); 

template<DLDeviceType XPU, typename IdType>
void
loadHalo2Graph(
IdArray &indptr,IdArray &indices,IdArray &edges,IdArray &bound,int gap);

template<DLDeviceType XPU, typename IdType>
void
FindNeighbor(
IdArray &nodeTable, IdArray &srcList, IdArray &dstList,int64_t flag,bool acc);

template<DLDeviceType XPU, typename IdType>
void
FindNeigEdge(
IdArray &nodeTable,IdArray &edgeTable,IdArray &srcList,IdArray &dstList,int64_t offset,int32_t loopFlag);

template<DLDeviceType XPU, typename IdType>
void
maplocalIds(
IdArray &nodeTable,IdArray &Gids,IdArray &Lids);

template<DLDeviceType XPU, typename IdType>
void
findSameNode(
IdArray &tensor1,IdArray &tensor2,IdArray &indexTable1,IdArray &indexTable2);

template<DLDeviceType XPU, typename IdType>
void
findSameIndex(
IdArray &tensor1,IdArray &tensor2,IdArray &indexTable1,IdArray &indexTable2);

template<DLDeviceType XPU, typename IdType>
void
sumDegree(
IdArray &InNodeTabel,IdArray &OutNodeTabel,IdArray &srcList,IdArray &dstList);

template<DLDeviceType XPU, typename IdType>
void
calculateP(
IdArray &DegreeTabel,IdArray &PTabel,IdArray &srcList,IdArray &dstList,int64_t fanout);

template<DLDeviceType XPU, typename IdType>
void
PPR(
IdArray &src,IdArray &dst,IdArray &nodeTable,IdArray &nodeValue,IdArray &nodeInfo,int64_t tableNUM); 

template<DLDeviceType XPU, typename IdType>
void
loss_csr(
IdArray &raw_ptr,IdArray &new_ptr,IdArray &raw_indice,IdArray &new_indice);

template<DLDeviceType XPU, typename IdType>
void
indptr_sort(
IdArray &raw_ptr,IdArray &raw_indices,IdArray &ts,IdArray &eid);

template<DLDeviceType XPU, typename IdType>
void
csr_to_dag(
IdArray &raw_ptr,IdArray &raw_indices,IdArray &eid,IdArray &out_src,IdArray &out_dst);

template<DLDeviceType XPU, typename IdType>
void
cooTocsr(
IdArray &inptr,IdArray &indice,IdArray &addr,IdArray &srcList,IdArray &dstList);

template<DLDeviceType XPU, typename IdType>
void
cooTocsrArg(
IdArray &inptr,IdArray &indice,IdArray &addr,IdArray &srcList,IdArray &dstList,IdArray &eid);

template<DLDeviceType XPU, typename IdType>
void
lpGraph(
IdArray &srcList,IdArray &dstList,IdArray &nodeTable,IdArray &InNodeTable,IdArray &OutNodeTable);

template<DLDeviceType XPU, typename IdType>
void
initBitmap(
IdArray &row,IdArray &col,IdArray &bitmap,int32_t bit_num);

template<DLDeviceType XPU, typename IdType>
void
matrixSim(
IdArray &matrix,IdArray &labels);

template<DLDeviceType XPU, typename IdType>
int32_t
matrixSimTh(
IdArray &matrix,IdArray &edges, double threshold);

template<DLDeviceType XPU, typename IdType>
void
bincount(
IdArray &nodelist,IdArray &nodeTable);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_
