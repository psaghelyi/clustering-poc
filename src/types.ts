/**
 * Supported embedding model providers on AWS Bedrock
 */
export type EmbeddingProvider = 'nova' | 'titan' | 'cohere';

/**
 * Embedding model configuration
 */
export interface EmbeddingModelConfig {
  /** Provider name */
  provider: EmbeddingProvider;
  /** AWS Bedrock model ID */
  modelId: string;
  /** Output dimension (for models that support it) */
  dimensions?: number;
}

/**
 * Pre-configured embedding models using cross-region inference profiles where available
 */
export const EMBEDDING_MODELS: Record<EmbeddingProvider, EmbeddingModelConfig> = {
  nova: {
    provider: 'nova',
    modelId: 'amazon.nova-2-multimodal-embeddings-v1:0',
    dimensions: 1024, // Supports 256, 384, 1024
  },
  titan: {
    provider: 'titan',
    modelId: 'amazon.titan-embed-text-v2:0',
    dimensions: 1024, // Supports up to 1024
  },
  cohere: {
    provider: 'cohere',
    modelId: 'us.cohere.embed-v4:0', // Using cross-region inference profile
    dimensions: 1024, // Cohere v4 supports 1024 dimensions
  },
};

/**
 * Document with metadata
 */
export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, any>;
  timestamp?: Date;
}

/**
 * Document with embedding vector
 */
export interface EmbeddedDocument extends Document {
  embedding: number[];
}

/**
 * Cluster of similar documents
 */
export interface Cluster {
  clusterId: number;
  documents: EmbeddedDocument[];
  centroid?: number[];
}

/**
 * HDBSCAN clustering configuration
 */
export interface ClusteringConfig {
  /** Algorithm (only HDBSCAN is supported) */
  algorithm: 'hdbscan';
  /** Minimum cluster size - minimum number of documents to form a cluster (default: 2) */
  minClusterSize?: number;
  /** Minimum samples - minimum samples for core points (default: 2) */
  minSamples?: number;
  /** Distance metric for similarity calculation (default: 'euclidean') */
  metric?: 'euclidean' | 'cosine';
}

/**
 * S3 Vectors configuration
 */
export interface S3VectorsConfig {
  /** Vector bucket name */
  vectorBucket: string;
  /** Vector index name */
  indexName: string;
  /** AWS region */
  region: string;
  /** Number of dimensions (must match embedding model) */
  dimensions: number;
}
