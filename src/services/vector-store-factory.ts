import type { VectorStore, VectorStoreConfig } from '../types.js';
import { S3Service } from './s3-service.js';
import { S3VectorsService } from './s3-vectors-service.js';

/**
 * Factory for creating vector store instances based on configuration
 *
 * Supports multiple backend types:
 * - simple-s3: JSON files in regular S3 buckets
 * - s3-vectors: Native AWS S3 Vectors with similarity search
 */
export class VectorStoreFactory {
  /**
   * Create a vector store instance based on configuration
   */
  static create(config: VectorStoreConfig): VectorStore {
    switch (config.type) {
      case 'simple-s3': {
        if (!config.s3Config) {
          throw new Error('S3Config is required for simple-s3 vector store type');
        }
        return new S3Service(config.s3Config);
      }

      case 's3-vectors': {
        if (!config.s3VectorsConfig) {
          throw new Error('S3VectorsConfig is required for s3-vectors vector store type');
        }
        return new S3VectorsService(config.s3VectorsConfig);
      }

      default: {
        const exhaustive: never = config.type;
        throw new Error(`Unknown vector store type: ${exhaustive}`);
      }
    }
  }

  /**
   * Create vector store from environment variables
   *
   * Environment variables:
   * - VECTOR_STORE_TYPE: 'simple-s3' | 's3-vectors'
   *
   * For simple-s3:
   * - S3_BUCKET: Bucket name
   * - AWS_REGION: AWS region
   *
   * For s3-vectors:
   * - S3_VECTOR_BUCKET: Vector bucket name
   * - S3_VECTOR_INDEX: Index name
   * - AWS_REGION: AWS region
   * - EMBEDDING_DIMENSIONS: Vector dimensions (e.g., 1024)
   */
  static createFromEnv(): VectorStore {
    const vectorStoreType = (process.env.VECTOR_STORE_TYPE || 'simple-s3') as 'simple-s3' | 's3-vectors';
    const region = process.env.AWS_REGION || 'us-east-1';

    if (vectorStoreType === 'simple-s3') {
      const bucket = process.env.S3_BUCKET;
      if (!bucket) {
        throw new Error('S3_BUCKET environment variable is required for simple-s3 vector store');
      }

      const config: VectorStoreConfig = {
        type: 'simple-s3',
        s3Config: {
          bucket,
          region,
        },
      };

      return VectorStoreFactory.create(config);
    }

    if (vectorStoreType === 's3-vectors') {
      const vectorBucket = process.env.S3_VECTOR_BUCKET;
      const indexName = process.env.S3_VECTOR_INDEX;
      const dimensions = process.env.EMBEDDING_DIMENSIONS;

      if (!vectorBucket) {
        throw new Error('S3_VECTOR_BUCKET environment variable is required for s3-vectors vector store');
      }
      if (!indexName) {
        throw new Error('S3_VECTOR_INDEX environment variable is required for s3-vectors vector store');
      }
      if (!dimensions) {
        throw new Error('EMBEDDING_DIMENSIONS environment variable is required for s3-vectors vector store');
      }

      const config: VectorStoreConfig = {
        type: 's3-vectors',
        s3VectorsConfig: {
          vectorBucket,
          indexName,
          region,
          dimensions: parseInt(dimensions, 10),
        },
      };

      return VectorStoreFactory.create(config);
    }

    throw new Error(`Unsupported VECTOR_STORE_TYPE: ${vectorStoreType}`);
  }
}
