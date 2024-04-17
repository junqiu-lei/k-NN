/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_opensearch_knn_jni_FaissService */

#ifndef _Included_org_opensearch_knn_jni_FaissService
#define _Included_org_opensearch_knn_jni_FaissService
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createIndex
 * Signature: ([IJILjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndex
        (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createIndexFromTemplate
 * Signature: ([IJILjava/lang/String;[BLjava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate
  (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jbyteArray, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    loadIndex
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndex
  (JNIEnv *, jclass, jstring);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    isSharedIndexStateRequired
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    initSharedIndexState
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initSharedIndexState
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    setSharedIndexState
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setSharedIndexState
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    queryIndex
 * Signature: (J[FI[I)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jintArray);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    queryIndexWithFilter
 * Signature: (J[FI[JI[I)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jlongArray, jint, jintArray);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    freeSharedIndexState
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    initLibrary
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_initLibrary
  (JNIEnv *, jclass);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    trainIndex
 * Signature: (Ljava/util/Map;IJ)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainIndex
  (JNIEnv *, jclass, jobject, jint, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    transferVectors
 * Signature: (J[[F)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_transferVectors
  (JNIEnv *, jclass, jlong, jobjectArray);

/*
* Class:     org_opensearch_knn_jni_FaissService
* Method:    rangeSearchIndex
* Signature: (J[F[F)J
*/
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex
  (JNIEnv *, jclass, jlong, jfloatArray, jfloat, jint);

#ifdef __cplusplus
}
#endif
#endif
