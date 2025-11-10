import {
  S3VectorsClient,
  CreateVectorBucketCommand,
  CreateIndexCommand,
  PutVectorsCommand,
  GetVectorsCommand,
  QueryVectorsCommand,
  ListVectorsCommand,
  DeleteVectorsCommand,
  GetIndexCommand,
  GetVectorBucketCommand,
} from '@aws-sdk/client-s3vectors';
import type { EmbeddedDocument, S3VectorsConfig, VectorStore } from '../types.js';

/**
 * S3 Vectors service for native vector storage and similarity search
 *
 * Uses AWS S3 Vectors for efficient vector storage with built-in similarity search.
 * Provides up to 90% cost reduction compared to traditional vector databases.
 */
export class S3VectorsService implements VectorStore {
  private client: S3VectorsClient;
  private config: S3VectorsConfig;

  constructor(config: S3VectorsConfig) {
    this.config = config;
    this.client = new S3VectorsClient({
      region: config.region,
      ...(config.endpoint && { endpoint: config.endpoint }),
    });
  }

  /**
   * Initialize the vector bucket and index (call once during setup)
   */
  async initialize(): Promise<void> {
    try {
      // Check if vector bucket exists
      await this.client.send(
        new GetVectorBucketCommand({
          vectorBucketName: this.config.vectorBucket,
        })
      );
      console.log(`Vector bucket '${this.config.vectorBucket}' already exists`);
    } catch (error: any) {
      if (error.name === 'NoSuchVectorBucket' || error.$metadata?.httpStatusCode === 404) {
        // Create vector bucket if it doesn't exist
        console.log(`Creating vector bucket '${this.config.vectorBucket}'...`);
        await this.client.send(
          new CreateVectorBucketCommand({
            vectorBucketName: this.config.vectorBucket,
          })
        );
        console.log(`Vector bucket '${this.config.vectorBucket}' created successfully`);
      } else {
        throw error;
      }
    }

    try {
      // Check if index exists
      await this.client.send(
        new GetIndexCommand({
          vectorBucketName: this.config.vectorBucket,
          indexName: this.config.indexName,
        })
      );
      console.log(`Vector index '${this.config.indexName}' already exists`);
    } catch (error: any) {
      if (error.name === 'NoSuchIndex' || error.$metadata?.httpStatusCode === 404) {
        // Create index if it doesn't exist
        console.log(`Creating vector index '${this.config.indexName}'...`);
        await this.client.send(
          new CreateIndexCommand({
            vectorBucketName: this.config.vectorBucket,
            indexName: this.config.indexName,
            dataType: 'float32',
            dimension: this.config.dimensions,
            distanceMetric: 'cosine',
          })
        );
        console.log(`Vector index '${this.config.indexName}' created successfully`);
      } else {
        throw error;
      }
    }
  }

  /**
   * Store a single embedding as a vector
   */
  async storeEmbedding(doc: EmbeddedDocument): Promise<void> {
    await this.client.send(
      new PutVectorsCommand({
        vectorBucketName: this.config.vectorBucket,
        indexName: this.config.indexName,
        vectors: [
          {
            key: doc.id,
            data: {
              float32: Array.from(doc.embedding),
            },
            metadata: this.prepareMetadata(doc),
          },
        ],
      })
    );
  }

  /**
   * Store multiple embeddings in batch
   */
  async storeEmbeddings(docs: EmbeddedDocument[]): Promise<void> {
    // S3 Vectors supports batch operations, but we'll chunk for safety
    const BATCH_SIZE = 100;

    for (let i = 0; i < docs.length; i += BATCH_SIZE) {
      const batch = docs.slice(i, i + BATCH_SIZE);
      const vectors = batch.map(doc => ({
        key: doc.id,
        data: {
          float32: Array.from(doc.embedding),
        },
        metadata: this.prepareMetadata(doc),
      }));

      await this.client.send(
        new PutVectorsCommand({
          vectorBucketName: this.config.vectorBucket,
          indexName: this.config.indexName,
          vectors,
        })
      );
    }
  }

  /**
   * Get a specific embedding by ID
   */
  async getEmbedding(id: string): Promise<EmbeddedDocument | null> {
    try {
      const response = await this.client.send(
        new GetVectorsCommand({
          vectorBucketName: this.config.vectorBucket,
          indexName: this.config.indexName,
          keys: [id],
          returnData: true,
          returnMetadata: true,
        })
      );

      if (!response.vectors || response.vectors.length === 0) {
        return null;
      }

      const vector = response.vectors[0];
      return this.convertToEmbeddedDocument(vector);
    } catch (error: any) {
      if (error.name === 'NoSuchKey' || error.$metadata?.httpStatusCode === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get all embeddings from the index
   */
  async getAllEmbeddings(): Promise<EmbeddedDocument[]> {
    const embeddings: EmbeddedDocument[] = [];
    let nextToken: string | undefined;

    do {
      const response = await this.client.send(
        new ListVectorsCommand({
          vectorBucketName: this.config.vectorBucket,
          indexName: this.config.indexName,
          nextToken,
          returnData: true,
          returnMetadata: true,
        })
      );

      if (response.vectors) {
        for (const vector of response.vectors) {
          embeddings.push(this.convertToEmbeddedDocument(vector));
        }
      }

      nextToken = response.nextToken;
    } while (nextToken);

    return embeddings;
  }

  /**
   * Delete a specific embedding
   */
  async deleteEmbedding(id: string): Promise<void> {
    await this.client.send(
      new DeleteVectorsCommand({
        vectorBucketName: this.config.vectorBucket,
        indexName: this.config.indexName,
        keys: [id],
      })
    );
  }

  /**
   * Delete all embeddings from the index
   */
  async deleteAllEmbeddings(): Promise<void> {
    const embeddings = await this.getAllEmbeddings();
    const keys = embeddings.map(doc => doc.id);

    // Delete in batches
    const BATCH_SIZE = 100;
    for (let i = 0; i < keys.length; i += BATCH_SIZE) {
      const batch = keys.slice(i, i + BATCH_SIZE);
      await this.client.send(
        new DeleteVectorsCommand({
          vectorBucketName: this.config.vectorBucket,
          indexName: this.config.indexName,
          keys: batch,
        })
      );
    }
  }

  /**
   * Query similar vectors using native S3 Vectors similarity search
   * This is the key advantage of S3 Vectors - built-in similarity search!
   */
  async querySimilar(queryVector: number[], topK: number = 10): Promise<EmbeddedDocument[]> {
    const response = await this.client.send(
      new QueryVectorsCommand({
        vectorBucketName: this.config.vectorBucket,
        indexName: this.config.indexName,
        queryVector: {
          float32: Array.from(queryVector),
        },
        topK,
        returnMetadata: true,
      })
    );

    if (!response.vectors || response.vectors.length === 0) {
      return [];
    }

    return response.vectors.map(vector => this.convertToEmbeddedDocument(vector));
  }

  /**
   * Prepare metadata for S3 Vectors (AWS Document type - supports nested objects)
   */
  private prepareMetadata(doc: EmbeddedDocument): any {
    const metadata: any = {
      content: doc.content,
    };

    if (doc.metadata) {
      // S3 Vectors supports nested metadata as AWS Document type
      metadata.originalMetadata = doc.metadata;
    }

    if (doc.timestamp) {
      metadata.timestamp = doc.timestamp.toISOString();
    }

    return metadata;
  }

  /**
   * Convert S3 Vectors response to EmbeddedDocument
   */
  private convertToEmbeddedDocument(vector: any): EmbeddedDocument {
    const metadata: Record<string, any> = {};
    let content = '';
    let timestamp: Date | undefined;

    // Extract metadata
    if (vector.metadata) {
      content = vector.metadata.content || '';

      if (vector.metadata.timestamp) {
        timestamp = new Date(vector.metadata.timestamp);
      }

      if (vector.metadata.originalMetadata) {
        Object.assign(metadata, vector.metadata.originalMetadata);
      }
    }

    // Extract vector data
    const embedding: number[] = vector.data?.float32 ? Array.from(vector.data.float32 as number[]) : [];

    return {
      id: vector.key,
      content,
      embedding,
      metadata,
      ...(timestamp && { timestamp }),
    };
  }
}
