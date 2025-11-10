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
    modelId: 'cohere.embed-english-v3:0:512',
    dimensions: 512,
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
 * S3 configuration (for simple JSON storage)
 */
export interface S3Config {
  bucket: string;
  region: string;
  endpoint?: string; // For localstack
  forcePathStyle?: boolean; // For localstack
}

/**
 * Vector store backend types
 */
export type VectorStoreType = 'simple-s3' | 's3-vectors';

/**
 * S3 Vectors configuration (for native vector storage)
 */
export interface S3VectorsConfig {
  /** Vector bucket name */
  vectorBucket: string;
  /** Vector index name */
  indexName: string;
  /** AWS region */
  region: string;
  /** Optional endpoint (for testing) */
  endpoint?: string;
  /** Number of dimensions (must match embedding model) */
  dimensions: number;
}

/**
 * Vector store configuration (supports multiple backends)
 */
export interface VectorStoreConfig {
  /** Type of vector store backend */
  type: VectorStoreType;
  /** Configuration for simple S3 backend */
  s3Config?: S3Config;
  /** Configuration for S3 Vectors backend */
  s3VectorsConfig?: S3VectorsConfig;
}

/**
 * Vector store interface (abstraction for multiple backends)
 */
export interface VectorStore {
  /** Store a single embedding */
  storeEmbedding(doc: EmbeddedDocument): Promise<void>;
  /** Store multiple embeddings in batch */
  storeEmbeddings(docs: EmbeddedDocument[]): Promise<void>;
  /** Get a specific embedding by ID */
  getEmbedding(id: string): Promise<EmbeddedDocument | null>;
  /** Get all embeddings */
  getAllEmbeddings(): Promise<EmbeddedDocument[]>;
  /** Delete a specific embedding */
  deleteEmbedding(id: string): Promise<void>;
  /** Delete all embeddings */
  deleteAllEmbeddings(): Promise<void>;
  /** Query similar vectors (if supported) */
  querySimilar?(queryVector: number[], topK: number): Promise<EmbeddedDocument[]>;
}

/**
 * Application configuration
 */
export interface AppConfig {
  embeddingModel: EmbeddingModelConfig;
  vectorStore: VectorStoreConfig;
  clustering: ClusteringConfig;
}
