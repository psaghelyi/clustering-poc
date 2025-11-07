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
 * Pre-configured embedding models
 */
export const EMBEDDING_MODELS: Record<EmbeddingProvider, EmbeddingModelConfig> = {
  nova: {
    provider: 'nova',
    modelId: 'amazon.nova-embed-text-v1',
    dimensions: 1024, // Supports 3072, 1024, 384, 256
  },
  titan: {
    provider: 'titan',
    modelId: 'amazon.titan-embed-text-v2:0',
    dimensions: 1024, // Supports up to 1024
  },
  cohere: {
    provider: 'cohere',
    modelId: 'cohere.embed-english-v3',
    dimensions: 1024,
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
 * Clustering algorithm configuration
 */
export interface ClusteringConfig {
  /** Algorithm to use */
  algorithm: 'dbscan' | 'optics' | 'kmeans';
  /** Epsilon (maximum distance between points for DBSCAN) */
  epsilon?: number;
  /** Minimum points in neighborhood for DBSCAN */
  minPoints?: number;
  /** Number of clusters for k-means */
  k?: number;
  /** Distance metric */
  distanceMetric?: 'euclidean' | 'cosine';
}

/**
 * S3 configuration
 */
export interface S3Config {
  bucket: string;
  region: string;
  endpoint?: string; // For localstack
  forcePathStyle?: boolean; // For localstack
}

/**
 * Application configuration
 */
export interface AppConfig {
  embeddingModel: EmbeddingModelConfig;
  s3: S3Config;
  clustering: ClusteringConfig;
}
