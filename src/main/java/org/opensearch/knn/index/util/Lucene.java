/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.SpaceType;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * KNN Library for Lucene
 */
public class Lucene extends JVMLibrary {

    Map<SpaceType, Function<Float, Float>> distanceTranslation;

    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
        METHOD_HNSW,
        KNNMethod.Builder.builder(
            MethodComponent.Builder.builder(METHOD_HNSW)
                .addParameter(
                    METHOD_PARAMETER_M,
                    new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0)
                )
                .addParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    new Parameter.IntegerParameter(
                        METHOD_PARAMETER_EF_CONSTRUCTION,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                        v -> v > 0
                    )
                )
                .build()
        ).addSpaces(SpaceType.L2, SpaceType.COSINESIMIL).build()
    );

    private final static Map<SpaceType, Function<Float, Float>> DISTANCE_TRANSLATIONS = ImmutableMap.<SpaceType, Function<Float, Float>>builder()
            .put(SpaceType.COSINESIMIL, distance -> (2 - distance) / 2)
            .put(SpaceType.L2, distance -> 1/(1+distance))
            .build();

    final static Lucene INSTANCE = new Lucene(METHODS, Version.LATEST.toString(), DISTANCE_TRANSLATIONS);

    /**
     * Constructor
     *
     * @param methods Map of k-NN methods that the library supports
     * @param version String representing version of library
     */
    Lucene(Map<String, KNNMethod> methods, String version, Map<SpaceType, Function<Float, Float>> distanceTranslation) {
        super(methods, version);
        this.distanceTranslation = distanceTranslation;
    }

    @Override
    public String getExtension() {
        throw new UnsupportedOperationException("Getting extension for Lucene is not supported");
    }

    @Override
    public String getCompoundExtension() {
        throw new UnsupportedOperationException("Getting compound extension for Lucene is not supported");
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        // The score returned by Lucene follows the higher the score, the better the result convention. It will
        // actually invert the distance score so that a higher number is a better score. So, we can just return the
        // score provided.
        return rawScore;
    }

    @Override
    public float distanceTranslation(float distance, SpaceType spaceType) {
        if (this.distanceTranslation.containsKey(spaceType)) {
            return this.distanceTranslation.get(spaceType).apply(distance);
        } else {
            throw new UnsupportedOperationException("Distance translation for space type " + spaceType + " is not supported");
        }
    }

    @Override
    public List<String> mmapFileExtensions() {
        return List.of("vec", "vex");
    }
}
