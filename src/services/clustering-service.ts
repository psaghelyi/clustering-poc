import * as clustering from 'density-clustering';
import { EmbeddedDocument, Cluster, ClusteringConfig } from '../types.js';
import { EmbeddingService } from './embedding-service.js';

/**
 * Service for clustering documents based on embedding similarity
 */
export class ClusteringService {
  private config: ClusteringConfig;

  constructor(config: ClusteringConfig) {
    this.config = config;
  }

  /**
   * Cluster documents using the configured algorithm
   */
  clusterDocuments(docs: EmbeddedDocument[]): Cluster[] {
    if (docs.length === 0) {
      return [];
    }

    const { algorithm } = this.config;

    switch (algorithm) {
      case 'dbscan':
        return this.clusterWithDBSCAN(docs);
      case 'optics':
        return this.clusterWithOPTICS(docs);
      case 'kmeans':
        return this.clusterWithKMeans(docs);
      default:
        throw new Error(`Unsupported algorithm: ${algorithm}`);
    }
  }

  /**
   * Cluster using DBSCAN algorithm (density-based, automatic cluster count)
   */
  private clusterWithDBSCAN(docs: EmbeddedDocument[]): Cluster[] {
    const dbscan = new clustering.DBSCAN();
    const epsilon = this.config.epsilon ?? 0.3; // Cosine distance threshold
    const minPoints = this.config.minPoints ?? 2;

    // Extract embeddings as dataset
    const dataset = docs.map((doc) => doc.embedding);

    // Use cosine distance if specified
    const distanceFunction =
      this.config.distanceMetric === 'cosine'
        ? this.cosineDistance.bind(this)
        : undefined;

    // Run DBSCAN clustering
    const clusterIndices = dbscan.run(
      dataset,
      epsilon,
      minPoints,
      distanceFunction
    );

    // Convert cluster indices to Cluster objects
    return this.indicesToClusters(docs, clusterIndices);
  }

  /**
   * Cluster using OPTICS algorithm (hierarchical density-based)
   */
  private clusterWithOPTICS(docs: EmbeddedDocument[]): Cluster[] {
    const optics = new clustering.OPTICS();
    const epsilon = this.config.epsilon ?? 0.3;
    const minPoints = this.config.minPoints ?? 2;

    const dataset = docs.map((doc) => doc.embedding);

    const distanceFunction =
      this.config.distanceMetric === 'cosine'
        ? this.cosineDistance.bind(this)
        : undefined;

    const clusterIndices = optics.run(
      dataset,
      epsilon,
      minPoints,
      distanceFunction
    );

    return this.indicesToClusters(docs, clusterIndices);
  }

  /**
   * Cluster using K-Means algorithm (requires specifying k)
   */
  private clusterWithKMeans(docs: EmbeddedDocument[]): Cluster[] {
    const kmeans = new clustering.KMEANS();
    const k = this.config.k ?? Math.ceil(Math.sqrt(docs.length / 2));

    const dataset = docs.map((doc) => doc.embedding);

    const clusterIndices = kmeans.run(dataset, k);

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
   * Cosine distance function for clustering (1 - cosine similarity)
   * Lower distance means more similar vectors
   */
  private cosineDistance(a: number[], b: number[]): number {
    const similarity = EmbeddingService.cosineSimilarity(a, b);
    return 1 - similarity;
  }

  /**
   * Find similar documents to a query document
   */
  findSimilarDocuments(
    query: EmbeddedDocument,
    docs: EmbeddedDocument[],
    threshold: number = 0.8,
    limit?: number
  ): Array<{ document: EmbeddedDocument; similarity: number }> {
    const similarities = docs
      .filter((doc) => doc.id !== query.id)
      .map((doc) => ({
        document: doc,
        similarity: EmbeddingService.cosineSimilarity(
          query.embedding,
          doc.embedding
        ),
      }))
      .filter((item) => item.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity);

    return limit ? similarities.slice(0, limit) : similarities;
  }

  /**
   * Calculate pairwise similarity matrix
   */
  calculateSimilarityMatrix(docs: EmbeddedDocument[]): number[][] {
    const n = docs.length;
    const matrix: number[][] = Array(n)
      .fill(0)
      .map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1; // Self-similarity is 1
        } else {
          const similarity = EmbeddingService.cosineSimilarity(
            docs[i].embedding,
            docs[j].embedding
          );
          matrix[i][j] = similarity;
          matrix[j][i] = similarity; // Symmetric
        }
      }
    }

    return matrix;
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
