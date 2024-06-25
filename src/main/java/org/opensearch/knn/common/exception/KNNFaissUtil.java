/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.knn.common.KNNConstants;

import java.util.Map;


public class KNNFaissUtil {
	private static final String FAISS_BINARY_INDEX_PREFIX = "B";

	public boolean isBinaryIndex(Map<String, Object> parameters) {
		return parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER) != null
			&& parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString().startsWith(FAISS_BINARY_INDEX_PREFIX);
	}
}
