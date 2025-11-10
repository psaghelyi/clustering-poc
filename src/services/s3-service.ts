import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  ListObjectsV2Command,
  DeleteObjectCommand,
} from '@aws-sdk/client-s3';
import { EmbeddedDocument, S3Config, VectorStore } from '../types.js';

/**
 * Service for storing and retrieving embeddings from S3 as JSON files
 * Implements VectorStore interface for simple S3-based storage
 */
export class S3Service implements VectorStore {
  private client: S3Client;
  private bucket: string;

  constructor(config: S3Config) {
    this.bucket = config.bucket;

    // Configure S3 client (supports localstack)
    this.client = new S3Client({
      region: config.region,
      endpoint: config.endpoint,
      forcePathStyle: config.forcePathStyle ?? false,
      // Credentials for localstack (will use AWS credentials in production)
      ...(config.endpoint && {
        credentials: {
          accessKeyId: 'test',
          secretAccessKey: 'test',
        },
      }),
    });
  }

  /**
   * Store an embedded document in S3
   */
  async storeEmbedding(doc: EmbeddedDocument): Promise<void> {
    const key = `embeddings/${doc.id}.json`;

    await this.client.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: JSON.stringify(doc),
        ContentType: 'application/json',
      })
    );
  }

  /**
   * Store multiple embedded documents
   */
  async storeEmbeddings(docs: EmbeddedDocument[]): Promise<void> {
    await Promise.all(docs.map((doc) => this.storeEmbedding(doc)));
  }

  /**
   * Retrieve a single embedded document by ID
   */
  async getEmbedding(id: string): Promise<EmbeddedDocument | null> {
    try {
      const key = `embeddings/${id}.json`;
      const response = await this.client.send(
        new GetObjectCommand({
          Bucket: this.bucket,
          Key: key,
        })
      );

      const body = await response.Body?.transformToString();
      if (!body) return null;

      return JSON.parse(body) as EmbeddedDocument;
    } catch (error: any) {
      if (error.name === 'NoSuchKey') {
        return null;
      }
      throw error;
    }
  }

  /**
   * Retrieve all embedded documents from S3
   */
  async getAllEmbeddings(): Promise<EmbeddedDocument[]> {
    const keys: string[] = [];
    let continuationToken: string | undefined;

    // List all objects in the embeddings prefix
    do {
      const response = await this.client.send(
        new ListObjectsV2Command({
          Bucket: this.bucket,
          Prefix: 'embeddings/',
          ContinuationToken: continuationToken,
        })
      );

      if (response.Contents) {
        keys.push(...response.Contents.map((obj) => obj.Key!));
      }

      continuationToken = response.NextContinuationToken;
    } while (continuationToken);

    // Fetch all documents in parallel
    const docs = await Promise.all(
      keys.map(async (key) => {
        const response = await this.client.send(
          new GetObjectCommand({
            Bucket: this.bucket,
            Key: key,
          })
        );

        const body = await response.Body?.transformToString();
        if (!body) return null;

        return JSON.parse(body) as EmbeddedDocument;
      })
    );

    return docs.filter((doc): doc is EmbeddedDocument => doc !== null);
  }

  /**
   * Delete an embedded document by ID
   */
  async deleteEmbedding(id: string): Promise<void> {
    const key = `embeddings/${id}.json`;

    await this.client.send(
      new DeleteObjectCommand({
        Bucket: this.bucket,
        Key: key,
      })
    );
  }

  /**
   * Delete all embeddings
   */
  async deleteAllEmbeddings(): Promise<void> {
    const keys: string[] = [];
    let continuationToken: string | undefined;

    do {
      const response = await this.client.send(
        new ListObjectsV2Command({
          Bucket: this.bucket,
          Prefix: 'embeddings/',
          ContinuationToken: continuationToken,
        })
      );

      if (response.Contents) {
        keys.push(...response.Contents.map((obj) => obj.Key!));
      }

      continuationToken = response.NextContinuationToken;
    } while (continuationToken);

    // Delete all in parallel
    await Promise.all(
      keys.map((key) =>
        this.client.send(
          new DeleteObjectCommand({
            Bucket: this.bucket,
            Key: key,
          })
        )
      )
    );
  }
}
