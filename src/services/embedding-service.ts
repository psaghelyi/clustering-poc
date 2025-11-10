import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from '@aws-sdk/client-bedrock-runtime';
import { Document, EmbeddedDocument, EmbeddingModelConfig } from '../types.js';

/**
 * Service for generating embeddings using AWS Bedrock
 */
export class EmbeddingService {
  private client: BedrockRuntimeClient;
  private config: EmbeddingModelConfig;

  constructor(config: EmbeddingModelConfig, region: string = 'us-east-1') {
    this.config = config;
    this.client = new BedrockRuntimeClient({ region });
  }

  /**
   * Generate embedding for a single document
   */
  async embedDocument(doc: Document): Promise<EmbeddedDocument> {
    const embedding = await this.generateEmbedding(doc.content);

    return {
      ...doc,
      embedding,
    };
  }

  /**
   * Generate embeddings for multiple documents using batch processing
   */
  async embedDocuments(docs: Document[]): Promise<EmbeddedDocument[]> {
    const { provider } = this.config;
    
    // Cohere supports native batch processing
    if (provider === 'cohere') {
      return this.embedDocumentsBatch(docs);
    }
    
    // For Titan and Nova, use parallel processing with concurrency limit
    return this.embedDocumentsParallel(docs, 10);
  }

  /**
   * Batch embed documents using Cohere's native batch API
   */
  private async embedDocumentsBatch(docs: Document[]): Promise<EmbeddedDocument[]> {
    if (docs.length === 0) return [];
    
    const batchSize = 96; // Cohere supports up to 96 texts per request
    const results: EmbeddedDocument[] = [];
    
    for (let i = 0; i < docs.length; i += batchSize) {
      const batch = docs.slice(i, i + batchSize);
      const texts = batch.map(doc => doc.content);
      
      const embeddings = await this.generateEmbeddingsBatch(texts);
      
      // Combine documents with their embeddings
      for (let j = 0; j < batch.length; j++) {
        results.push({
          ...batch[j],
          embedding: embeddings[j],
        });
      }
    }
    
    return results;
  }

  /**
   * Process documents in parallel with concurrency limit
   */
  private async embedDocumentsParallel(docs: Document[], concurrency: number): Promise<EmbeddedDocument[]> {
    const results: EmbeddedDocument[] = [];
    
    for (let i = 0; i < docs.length; i += concurrency) {
      const batch = docs.slice(i, i + concurrency);
      const batchResults = await Promise.all(
        batch.map(doc => this.embedDocument(doc))
      );
      results.push(...batchResults);
    }
    
    return results;
  }

  /**
   * Generate embeddings for multiple texts in one API call (Cohere only)
   */
  private async generateEmbeddingsBatch(texts: string[]): Promise<number[][]> {
    const { provider, modelId, dimensions } = this.config;
    
    if (provider !== 'cohere') {
      throw new Error('Batch embedding only supported for Cohere');
    }
    
    const requestBody = {
      texts,
      input_type: 'search_document',
      truncate: 'END',
      embedding_types: ['float'],
      ...(dimensions && { output_dimension: dimensions }),
    };
    
    const command = new InvokeModelCommand({
      modelId,
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify(requestBody),
    });
    
    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));
    
    // Cohere v4 returns embeddings in a nested structure
    return responseBody.embeddings?.float || responseBody.embeddings;
  }

  /**
   * Generate embedding vector for text using the configured model
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    const { provider, modelId, dimensions } = this.config;

    let requestBody: any;

    switch (provider) {
      case 'nova':
        // Nova multimodal embeddings support text input
        requestBody = {
          inputText: text,
          embeddingConfig: {
            outputEmbeddingLength: dimensions || 1024,
          },
        };
        break;

      case 'titan':
        requestBody = {
          inputText: text,
          dimensions: dimensions || 1024,
          normalize: true,
        };
        break;

      case 'cohere':
        requestBody = {
          texts: [text],
          input_type: 'search_document',
          truncate: 'END',
          embedding_types: ['float'],
          ...(dimensions && { output_dimension: dimensions }),
        };
        break;

      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }

    const command = new InvokeModelCommand({
      modelId,
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify(requestBody),
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    // Extract embedding from response based on provider
    let embedding: number[];

    switch (provider) {
      case 'nova':
        // Nova returns embedding in the response
        embedding = responseBody.embedding || responseBody.embeddings?.[0];
        break;

      case 'titan':
        embedding = responseBody.embedding;
        break;

      case 'cohere':
        // Cohere v4 returns embeddings in a nested structure
        embedding = responseBody.embeddings?.float?.[0] || responseBody.embeddings?.[0];
        break;

      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }

    if (!embedding) {
      throw new Error(`Failed to extract embedding from response: ${JSON.stringify(responseBody)}`);
    }

    return embedding;
  }

  /**
   * Normalize a vector to unit length (for cosine similarity)
   */
  static normalizeVector(vector: number[]): number[] {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return magnitude === 0 ? vector : vector.map((val) => val / magnitude);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  static cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }

    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }
}
