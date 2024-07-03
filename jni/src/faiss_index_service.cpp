// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_index_service.h"
#include "faiss_methods.h"
#include "faiss/index_factory.h"
#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_io.h"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <faiss/impl/io.h>

namespace knn_jni {
namespace faiss_wrapper {

template<typename INDEX, typename IVF, typename HNSW>
void SetExtraParameters(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env,
                        const std::unordered_map<std::string, jobject>& parametersCpp, INDEX * index) {
    std::unordered_map<std::string,jobject>::const_iterator value;
    if (auto * indexIvf = dynamic_cast<IVF*>(index)) {
        if ((value = parametersCpp.find(knn_jni::NPROBES)) != parametersCpp.end()) {
            indexIvf->nprobe = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::COARSE_QUANTIZER)) != parametersCpp.end()
                && indexIvf->quantizer != nullptr) {
            auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, value->second);
            SetExtraParameters<INDEX, IVF, HNSW>(jniUtil, env, subParametersCpp, indexIvf->quantizer);
        }
    }

    if (auto * indexHnsw = dynamic_cast<HNSW*>(index)) {

        if ((value = parametersCpp.find(knn_jni::EF_CONSTRUCTION)) != parametersCpp.end()) {
            indexHnsw->hnsw.efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::EF_SEARCH)) != parametersCpp.end()) {
            indexHnsw->hnsw.efSearch = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }
    }
}

IndexService::IndexService(std::unique_ptr<FaissMethods> faissMethods) : faissMethods(std::move(faissMethods)) {}

void IndexService::createIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    std::unique_ptr<faiss::Index> indexWriter(faissMethods->indexFactory(dim, indexDescription.c_str(), metric));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexIDMap> idMap(faissMethods->indexIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    // Write the index to disk
    faissMethods->writeIndex(idMap.get(), indexPath.c_str());
}

void IndexService::createIndexFromTemplate(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::VectorIOReader vectorIoReader,
        std::vector<int64_t> idVector,
        int numVectors,
        std::vector<float> *inputVectors,
        std::string& indexPathCpp) {
    // Read vectors from memory address
    // Create faiss index
    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::read_index(&vectorIoReader, 0));

    faiss::IndexIDMap idMap =  faiss::IndexIDMap(indexWriter.get());

    idMap.add_with_ids(numVectors, inputVectors->data(), idVector.data());

    // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
    // This is not the ideal approach, please refer this gh issue for long term solution:
    // https://github.com/opensearch-project/k-NN/issues/1600
    delete inputVectors;

    faiss::write_index(&idMap, indexPathCpp.c_str());
}

BinaryIndexService::BinaryIndexService(std::unique_ptr<FaissMethods> faissMethods) : IndexService(std::move(faissMethods)) {}

void BinaryIndexService::createIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddress);

    if (dim % 8 != 0) {
        throw std::runtime_error("Dimensions should be multiply of 8");
    }
    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    std::unique_ptr<faiss::IndexBinary> indexWriter(faissMethods->indexBinaryFactory(dim, indexDescription.c_str()));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(jniUtil, env, parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    // Write the index to disk
    faissMethods->writeIndexBinary(idMap.get(), indexPath.c_str());
}

void BinaryIndexService::createIndexFromTemplate(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::VectorIOReader vectorIoReader,
        std::vector<int64_t> idVector,
        int numVectors,
        std::vector<float> *inputVectors,
        std::string& indexPathCpp) {
    // Read vectors from memory address
    // Create faiss index
    std::unique_ptr<faiss::IndexBinary> indexWriter;
    indexWriter.reset(dynamic_cast<faiss::IndexBinary*>(faiss::read_index(&vectorIoReader, 0)));

    // faiss::IndexBinaryIDMap idMap = faiss::IndexBinaryIDMap(indexWriter.get());

    // idMap.add_with_ids(numVectors, inputVectors->data(), idVector.data());
    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));

    auto* vectorData = reinterpret_cast<uint8_t*>(inputVectors->data());
    idMap->add_with_ids(numVectors, vectorData, idVector.data());

    // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
    // This is not the ideal approach, please refer this gh issue for long term solution:
    // https://github.com/opensearch-project/k-NN/issues/1600
    delete inputVectors;

    // Write the index to disk
    faissMethods->writeIndexBinary(idMap.get(), indexPathCpp.c_str());
}

} // namespace faiss_wrapper
} // namesapce knn_jni