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
import type { EmbeddedDocument, S3VectorsConfig } from '../types.js';

/**
 * S3 Vectors service for native vector storage and similarity search
 *
 * Uses AWS S3 Vectors for efficient vector storage with built-in similarity search.
 * Provides up to 90% cost reduction compared to traditional vector databases.
 */
export class S3VectorsService {
  private client: S3VectorsClient;
  private config: S3VectorsConfig;

  constructor(config: S3VectorsConfig) {
    this.config = config;
    this.client = new S3VectorsClient({
      region: config.region,
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
      const indexInfo = await this.client.send(
        new GetIndexCommand({
          vectorBucketName: this.config.vectorBucket,
          indexName: this.config.indexName,
        })
      );
      console.log(`Vector index '${this.config.indexName}' already exists`);
      if (indexInfo.dimension && indexInfo.dimension !== this.config.dimensions) {
        console.log(`⚠️  Warning: Existing index has ${indexInfo.dimension} dimensions but configured for ${this.config.dimensions}`);
      }
    } catch (error: any) {
      if (error.name === 'NotFoundException' || error.name === 'NoSuchIndex' || error.$metadata?.httpStatusCode === 404) {
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
      const vectors = batch.map(doc => {
        if (!doc.embedding || !Array.isArray(doc.embedding)) {
          throw new Error(`Document ${doc.id} has invalid or missing embedding`);
        }

        return {
          key: doc.id,
          data: {
            float32: Array.from(doc.embedding),
          },
          metadata: this.prepareMetadata(doc),
        };
      });

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
   * Prepare metadata for S3 Vectors
   * Note: S3 Vectors only supports string, number, boolean, or array values (no nested objects)
   * Stores source document information including owner, description, sourceFile, and index
   */
  private prepareMetadata(doc: EmbeddedDocument): any {
    const metadata: any = {
      content: doc.content,
    };

    // Store source document metadata (owner, description, sourceFile, index, etc.)
    if (doc.metadata) {
      // Flatten common source document fields for easier querying
      if (doc.metadata.owner) metadata.owner = doc.metadata.owner;
      if (doc.metadata.description) metadata.description = doc.metadata.description;
      if (doc.metadata.sourceFile) metadata.sourceFile = doc.metadata.sourceFile;
      if (doc.metadata.index !== undefined) metadata.index = doc.metadata.index;
      
      // Store all metadata as JSON string (S3 Vectors doesn't support nested objects)
      metadata.originalMetadata = JSON.stringify(doc.metadata);
    }

    if (doc.timestamp) {
      metadata.timestamp = doc.timestamp.toISOString();
    }

    return metadata;
  }

  /**
   * Convert S3 Vectors response to EmbeddedDocument
   * Reconstructs source document information from stored metadata
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

      // Reconstruct source document metadata from flattened fields
      if (vector.metadata.owner) metadata.owner = vector.metadata.owner;
      if (vector.metadata.description) metadata.description = vector.metadata.description;
      if (vector.metadata.sourceFile) metadata.sourceFile = vector.metadata.sourceFile;
      if (vector.metadata.index !== undefined) metadata.index = vector.metadata.index;

      // Parse originalMetadata from JSON string
      if (vector.metadata.originalMetadata) {
        try {
          const parsed = JSON.parse(vector.metadata.originalMetadata);
          Object.assign(metadata, parsed);
        } catch (error) {
          console.warn('Failed to parse originalMetadata:', error);
        }
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
