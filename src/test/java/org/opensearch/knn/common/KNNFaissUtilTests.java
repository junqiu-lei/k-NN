/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import junit.framework.TestCase;

import java.util.HashMap;
import java.util.Map;

public class KNNFaissUtilTests extends TestCase {
    public void testIsBinaryIndex_whenBinary_thenTrue() {
        KNNFaissUtil faissUtil = new KNNFaissUtil();
        Map<String, Object> binaryIndexParams = new HashMap<>();
        binaryIndexParams.put(KNNConstants.INDEX_DESCRIPTION_PARAMETER, "BHNSW");
        assertTrue(faissUtil.isBinaryIndex(binaryIndexParams));
    }

    public void testIsBinaryIndex_whenNonBinary_thenFalse() {
        KNNFaissUtil faissUtil = new KNNFaissUtil();
        Map<String, Object> nonBinaryIndexParams = new HashMap<>();
        nonBinaryIndexParams.put(KNNConstants.INDEX_DESCRIPTION_PARAMETER, "HNSW");
        assertFalse(faissUtil.isBinaryIndex(nonBinaryIndexParams));
    }

}
