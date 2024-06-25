/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * Class having methods to extract a value from field info
 */
public class FieldInfoExtractor {
    public static String getIndexDescription(FieldInfo fieldInfo) throws IOException {
        String parameters = fieldInfo.attributes().get(KNNConstants.PARAMETERS);
        if (parameters == null) {
            return null;
        }

        String indexDescription = (String) XContentHelper.createParser(
            NamedXContentRegistry.EMPTY,
            DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
            new BytesArray(parameters),
            MediaTypeRegistry.getDefaultMediaType()
        ).map().getOrDefault(INDEX_DESCRIPTION_PARAMETER, null);

        if (VectorDataType.BINARY.getValue()
            .equals(fieldInfo.attributes().getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue()))) {
            indexDescription = FAISS_BINARY_INDEX_DESCRIPTION_PREFIX + indexDescription;
        }

        return indexDescription;
    }
}
