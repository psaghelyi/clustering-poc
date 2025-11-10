# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a TypeScript-based proof-of-concept for document clustering using embeddings stored in S3, supporting multiple AWS Bedrock embedding models (Nova, Titan, Cohere) with DBSCAN clustering.

## Development Commands

### Build and Run
```bash
# Install dependencies
npm install

# Development mode (runs src/example.ts with tsx)
npm run dev

# Build TypeScript to JavaScript
npm run build

# Run compiled JavaScript
npm start

# Clean build artifacts
npm run clean
```

### Document Clustering and Merging
```bash
# Merge similar documents using embeddings and Claude
# Requires aws-vault for AWS credentials
npm run merge:aws

# Alternative: run directly with tsx (requires AWS credentials in environment)
npm run merge
```

### Testing
```bash
# Run example/test with mock embeddings
npm test
```

## Architecture

The codebase follows a service-oriented architecture with seven core services and a factory pattern for vector storage:

### Core Services

1. **EmbeddingService** (`src/services/embedding-service.ts`):
   - Handles AWS Bedrock embedding generation
   - Supports Nova, Titan, and Cohere models
   - Includes mock mode for testing without AWS credentials
   - Normalizes vectors and handles batching

2. **S3Service** (`src/services/s3-service.ts`):
   - Manages simple S3 storage of embeddings as JSON files
   - Stores embeddings as individual JSON objects
   - Implements VectorStore interface
   - Ideal for debugging and simple use cases

3. **S3VectorsService** (`src/services/s3-vectors-service.ts`):
   - AWS S3 Vectors integration for native vector storage
   - Implements VectorStore interface with native similarity search
   - Auto-creates vector buckets and indexes
   - Provides `querySimilar()` for efficient vector retrieval
   - Up to 90% cost reduction vs traditional vector databases

4. **VectorStoreFactory** (`src/services/vector-store-factory.ts`):
   - Factory pattern for creating vector store instances
   - Supports multiple backends: simple-s3, s3-vectors
   - Creates stores from environment variables or config objects
   - Enables easy switching between storage backends

5. **ClusteringService** (`src/services/clustering-service.ts`):
   - Implements HDBSCAN (default), DBSCAN, OPTICS, and K-Means clustering
   - HDBSCAN: Hierarchical density-based clustering with automatic cluster detection
   - Uses cosine similarity for distance calculation
   - Provides cluster statistics and similarity search

6. **ClaudeService** (`src/services/claude-service.ts`):
   - Calls AWS Bedrock Claude models for document merging
   - Merges semantically similar documents into single consolidated entries
   - Handles batch merging of document clusters
   - Uses Claude 4.5 Haiku by default for cost-effectiveness

7. **DocumentLoader** (`src/services/document-loader.ts`):
   - Loads documents from JSON files
   - Tracks source file information for each document
   - Generates unique document IDs

### Key Design Patterns

- **Configuration-driven**: All services accept configuration objects from `types.ts`
- **Provider abstraction**: Embedding models are abstracted through `EmbeddingProvider` interface
- **Vector store abstraction**: Multiple storage backends (simple S3, S3 Vectors) through `VectorStore` interface
- **Mock support**: Built-in mock embeddings for testing without AWS

## Vector Storage Backends

The system supports two vector storage backends, configurable via `VECTOR_STORE_TYPE` environment variable:

### 1. Simple S3 (default)
Stores embeddings as JSON files in regular S3 buckets.

**Pros:**
- Simple setup, works with any S3 bucket
- Easy to inspect and debug (JSON format)
- Standard S3 bucket (no special configuration)

**Cons:**
- No native similarity search
- Manual clustering required via HDBSCAN/DBSCAN/OPTICS

**Configuration:**
```bash
VECTOR_STORE_TYPE=simple-s3
S3_BUCKET=clustering-poc-embeddings
```

### 2. S3 Vectors (AWS native, preview)
Uses AWS S3 Vectors for native vector storage with built-in similarity search.

**Pros:**
- Native similarity search via `QueryVectorsCommand`
- Up to 90% cost reduction vs traditional vector databases
- Sub-second query latency
- Scalable (10,000 indexes per bucket, millions of vectors per index)

**Cons:**
- Preview feature (as of July 2025)
- Limited regional availability (us-east-1, us-east-2, us-west-2, eu-central-1, ap-southeast-2)
- Requires vector bucket creation

**Configuration:**
```bash
VECTOR_STORE_TYPE=s3-vectors
S3_VECTOR_BUCKET=clustering-poc-vectors
S3_VECTOR_INDEX=embeddings-index
EMBEDDING_DIMENSIONS=1024  # Must match embedding model
```

**Setup:**
The S3VectorsService will automatically create the vector bucket and index on first run. You can also create them manually:
```bash
aws s3vectors create-vector-bucket --vector-bucket clustering-poc-vectors --region us-east-1
aws s3vectors create-index --vector-bucket clustering-poc-vectors --index embeddings-index --vector-dimensions 1024
```

## Configuration

The application uses environment variables and configuration objects:

### Environment Variables (.env)
- `AWS_REGION`: AWS region for Bedrock/S3
- `AWS_ACCESS_KEY_ID`: AWS credentials
- `AWS_SECRET_ACCESS_KEY`: AWS credentials
- `VECTOR_STORE_TYPE`: simple-s3 | s3-vectors (default: simple-s3)
- `S3_BUCKET`: S3 bucket name (for simple-s3 backend)
- `S3_VECTOR_BUCKET`: Vector bucket name (for s3-vectors backend)
- `S3_VECTOR_INDEX`: Vector index name (for s3-vectors backend)
- `EMBEDDING_DIMENSIONS`: Vector dimensions (for s3-vectors backend, e.g., 1024)
- `EMBEDDING_PROVIDER`: nova | titan | cohere
- `USE_MOCK_EMBEDDINGS`: true/false for mock mode

### Clustering Parameters
Configured in `src/example.ts` and `src/merge-documents.ts`:

**HDBSCAN (default, recommended):**
- `algorithm`: 'hdbscan'
- `minClusterSize`: Minimum documents to form a cluster (2-5 typical)
- `minSamples`: Minimum samples for core points (2-5 typical)

**DBSCAN:**
- `algorithm`: 'dbscan'
- `epsilon`: Distance threshold (0.1-0.5 for cosine)
- `minPoints`: Minimum points in neighborhood (2-5 typical)
- `distanceMetric`: 'cosine' | 'euclidean'

**Other algorithms:**
- `algorithm`: 'optics' | 'kmeans'

## TypeScript Configuration

- **ES Modules**: Uses modern ES modules (`"type": "module"` in package.json)
- **Target**: ES2022 with strict mode enabled
- **File extensions**: Import statements must include `.js` extension for local files

## Testing Strategy

1. **Mock Mode**: Use `USE_MOCK_EMBEDDINGS=true` for deterministic testing without AWS credentials
2. **Example Script**: `src/example.ts` serves as both demo and integration test
3. **AWS Testing**: Use aws-vault for production testing with real AWS credentials

## Document Merging Workflow

The `src/merge-documents.ts` script implements a complete workflow for clustering and merging similar documents:

1. **Load Documents**: Reads JSON files from `agent-input-output/outputs/` directory
2. **Generate Embeddings**: Creates vector embeddings for each document using AWS Bedrock
3. **Cluster Documents**: Groups semantically similar documents using HDBSCAN
4. **Merge Clusters**: Uses Claude 4.5 Haiku to merge documents in each cluster into single consolidated entries
5. **Output Results**: Saves merged documents to `agent-input-output/output.json`

### Workflow Configuration

Key parameters in `src/merge-documents.ts`:
- `embeddingProvider`: Embedding model to use (titan | nova | cohere)
- `claudeModelId`: Claude model for merging (default: claude-haiku-4-5)
- `algorithm`: Clustering algorithm (hdbscan default)
- `minClusterSize`: Minimum documents to form a cluster (2 default)
- `minSamples`: Minimum samples for core points (2 default)

### Embedding Cache

Embeddings are automatically cached to `agent-input-output/embeddings-cache.json`:
- First run: Generates embeddings and saves to cache (3-5 minutes for 385 docs)
- Subsequent runs: Loads embeddings from cache (instant)
- Delete the cache file to regenerate embeddings with different settings

### Expected Results

- Documents that are semantically similar get merged into single entries
- Unclustered documents (noise) are kept as-is
- Output includes `mergedFrom` field showing original descriptions
- Typical reduction: 20-40% fewer documents after merging

## Common Tasks

### Switching Vector Storage Backends

**To use Simple S3 (default):**
```bash
export VECTOR_STORE_TYPE=simple-s3
export S3_BUCKET=clustering-poc-embeddings
npm run dev
```

**To use S3 Vectors:**
```bash
export VECTOR_STORE_TYPE=s3-vectors
export S3_VECTOR_BUCKET=clustering-poc-vectors
export S3_VECTOR_INDEX=embeddings-index
export EMBEDDING_DIMENSIONS=1024
export USE_MOCK_EMBEDDINGS=false  # S3 Vectors requires real embeddings
npm run dev
```

### Using S3 Vectors Similarity Search

The S3 Vectors backend provides native similarity search:

```typescript
import { VectorStoreFactory } from './services/vector-store-factory.js';

const vectorStore = VectorStoreFactory.create({
  type: 's3-vectors',
  s3VectorsConfig: {
    vectorBucket: 'my-vectors',
    indexName: 'embeddings',
    region: 'us-east-1',
    dimensions: 1024,
  },
});

// Initialize (creates bucket/index if needed)
await vectorStore.initialize();

// Store embeddings
await vectorStore.storeEmbeddings(embeddedDocs);

// Native similarity search (only available with s3-vectors)
if (vectorStore.querySimilar) {
  const similar = await vectorStore.querySimilar(queryVector, 10);
  console.log('Top 10 similar documents:', similar);
}
```

### Adding a New Embedding Model
1. Add provider type to `EmbeddingProvider` in `types.ts`
2. Add configuration to `EMBEDDING_MODELS` in `types.ts`
3. Implement provider-specific logic in `EmbeddingService.embedDocument()`

### Adjusting Clustering Sensitivity

**HDBSCAN (recommended):**
Modify `minClusterSize` and `minSamples`:
- Lower minClusterSize (2-3): More, smaller clusters
- Higher minClusterSize (5-10): Fewer, larger clusters
- Lower minSamples: More sensitive to local density variations
- Higher minSamples: More robust to noise

**DBSCAN:**
Modify epsilon in clustering config:
- Lower epsilon (0.1-0.2): Stricter clusters, near-duplicates only
- Higher epsilon (0.3-0.5): Looser clusters, broader topics

### Using Different Embedding Models
Set `EMBEDDING_PROVIDER` environment variable:
- `titan`: Amazon Titan (default, 1024 dimensions)
- `nova`: Amazon Nova Multimodal (1024 dimensions)
- `cohere`: Cohere Embed English (512 dimensions)

**Note:** When using S3 Vectors, ensure `EMBEDDING_DIMENSIONS` matches your model's dimensions

### Debugging S3 Issues
```bash
# List S3 contents
aws s3 ls s3://clustering-poc-embeddings

# Check specific embedding
aws s3 cp s3://clustering-poc-embeddings/embeddings/doc-1.json -
```

## Performance Considerations

- **Embedding Dimensions**: Lower dimensions (256-384) for speed, higher (1024-3072) for accuracy
- **Batch Processing**: EmbeddingService handles batching automatically
- **Vector Normalization**: All embeddings are L2-normalized for consistent cosine similarity
- **Memory Usage**: Each 1024-dim embedding uses ~4KB; plan accordingly for large datasets