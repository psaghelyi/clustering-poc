import { HDBSCAN } from 'hdbscan-ts';
import { EmbeddedDocument, Cluster, ClusteringConfig } from '../types.js';
import { EmbeddingService } from './embedding-service.js';

/**
 * Service for clustering documents using HDBSCAN algorithm
 * HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
 * is a density-based clustering algorithm that automatically determines the number of clusters
 */
export class ClusteringService {
  private config: ClusteringConfig;

  constructor(config: ClusteringConfig) {
    this.config = config;
  }

  /**
   * Cluster documents using HDBSCAN algorithm
   */
  clusterDocuments(docs: EmbeddedDocument[]): Cluster[] {
    if (docs.length === 0) {
      return [];
    }

    return this.clusterWithHDBSCAN(docs);
  }

  /**
   * Cluster using HDBSCAN algorithm (hierarchical density-based, automatic cluster count)
   * HDBSCAN automatically determines the number of clusters at varying densities.
   * Supports configurable distance metrics (euclidean, cosine).
   */
  private clusterWithHDBSCAN(docs: EmbeddedDocument[]): Cluster[] {
    const minClusterSize = this.config.minClusterSize ?? 2;
    const minSamples = this.config.minSamples ?? 2;
    const metric = this.config.metric ?? 'euclidean';

    // Extract embeddings as dataset
    const dataset = docs.map((doc) => doc.embedding);

    // Run HDBSCAN clustering with configurable distance metric
    // Note: metric parameter may not be in type definitions but is supported at runtime
    const hdbscan = new HDBSCAN({
      minClusterSize,
      minSamples,
      metric,
      debugMode: false,
    } as any);

    const labels = hdbscan.fit(dataset);

    // Convert labels to cluster indices format
    // Group document indices by cluster label
    const clusterMap = new Map<number, number[]>();

    labels.forEach((label, docIndex) => {
      // HDBSCAN uses -1 for noise points, skip them
      if (label === -1) return;

      if (!clusterMap.has(label)) {
        clusterMap.set(label, []);
      }
      clusterMap.get(label)!.push(docIndex);
    });

    // Convert map to array of index arrays
    const clusterIndices = Array.from(clusterMap.values());

    return this.indicesToClusters(docs, clusterIndices);
  }

  /**
   * Convert cluster indices to Cluster objects with documents
   */
  private indicesToClusters(
    docs: EmbeddedDocument[],
    clusterIndices: number[][]
  ): Cluster[] {
    return clusterIndices.map((indices, clusterId) => {
      const clusterDocs = indices.map((idx) => docs[idx]);
      const centroid = this.calculateCentroid(clusterDocs);

      return {
        clusterId,
        documents: clusterDocs,
        centroid,
      };
    });
  }

  /**
   * Calculate the centroid (average) of embeddings
   */
  private calculateCentroid(docs: EmbeddedDocument[]): number[] {
    if (docs.length === 0) return [];

    const dimensions = docs[0].embedding.length;
    const centroid = new Array(dimensions).fill(0);

    for (const doc of docs) {
      for (let i = 0; i < dimensions; i++) {
        centroid[i] += doc.embedding[i];
      }
    }

    return centroid.map((val) => val / docs.length);
  }

  /**
   * Get statistics about the clustering results
   */
  getClusteringStats(clusters: Cluster[]): {
    numClusters: number;
    avgClusterSize: number;
    minClusterSize: number;
    maxClusterSize: number;
    totalDocuments: number;
  } {
    const sizes = clusters.map((c) => c.documents.length);
    const totalDocs = sizes.reduce((sum, size) => sum + size, 0);

    return {
      numClusters: clusters.length,
      avgClusterSize: totalDocs / clusters.length || 0,
      minClusterSize: Math.min(...sizes),
      maxClusterSize: Math.max(...sizes),
      totalDocuments: totalDocs,
    };
  }
}
