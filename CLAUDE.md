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

### Testing Infrastructure
```bash
# Start LocalStack (local S3)
npm run localstack:up

# Stop LocalStack
npm run localstack:down

# Run example/test
npm test
```

## Architecture

The codebase follows a service-oriented architecture with five main services:

### Core Services

1. **EmbeddingService** (`src/services/embedding-service.ts`):
   - Handles AWS Bedrock embedding generation
   - Supports Nova, Titan, and Cohere models
   - Includes mock mode for testing without AWS credentials
   - Normalizes vectors and handles batching

2. **S3Service** (`src/services/s3-service.ts`):
   - Manages S3 storage of embeddings
   - Supports both real S3 and LocalStack
   - Stores embeddings as JSON with metadata

3. **ClusteringService** (`src/services/clustering-service.ts`):
   - Implements DBSCAN, OPTICS, and K-Means clustering
   - Uses cosine similarity for distance calculation
   - Provides cluster statistics and similarity search

4. **ClaudeService** (`src/services/claude-service.ts`):
   - Calls AWS Bedrock Claude models for document merging
   - Merges semantically similar documents into single consolidated entries
   - Handles batch merging of document clusters
   - Uses Claude 4.5 Haiku by default for cost-effectiveness

5. **DocumentLoader** (`src/services/document-loader.ts`):
   - Loads documents from JSON files
   - Tracks source file information for each document
   - Generates unique document IDs

### Key Design Patterns

- **Configuration-driven**: All services accept configuration objects from `types.ts`
- **Provider abstraction**: Embedding models are abstracted through `EmbeddingProvider` interface
- **Mock support**: Built-in mock embeddings for testing without AWS
- **LocalStack integration**: Full S3 functionality without AWS costs

## Configuration

The application uses environment variables and configuration objects:

### Environment Variables (.env)
- `AWS_REGION`: AWS region for Bedrock/S3
- `AWS_ACCESS_KEY_ID`: AWS credentials
- `AWS_SECRET_ACCESS_KEY`: AWS credentials
- `S3_BUCKET`: S3 bucket name
- `EMBEDDING_PROVIDER`: nova | titan | cohere
- `USE_MOCK_EMBEDDINGS`: true/false for mock mode
- `USE_LOCALSTACK`: true/false for LocalStack S3

### Clustering Parameters
Configured in `src/example.ts`:
- `epsilon`: Distance threshold (0.1-0.5 for cosine)
- `minPoints`: Minimum cluster size (2-5 typical)
- `algorithm`: dbscan | optics | kmeans

## TypeScript Configuration

- **ES Modules**: Uses modern ES modules (`"type": "module"` in package.json)
- **Target**: ES2022 with strict mode enabled
- **File extensions**: Import statements must include `.js` extension for local files

## Testing Strategy

1. **Mock Mode**: Use `USE_MOCK_EMBEDDINGS=true` for deterministic testing
2. **LocalStack**: Use `USE_LOCALSTACK=true` for S3 testing without AWS
3. **Example Script**: `src/example.ts` serves as both demo and integration test

## Document Merging Workflow

The `src/merge-documents.ts` script implements a complete workflow for clustering and merging similar documents:

1. **Load Documents**: Reads JSON files from `agent-input-output/outputs/` directory
2. **Generate Embeddings**: Creates vector embeddings for each document using AWS Bedrock
3. **Cluster Documents**: Groups semantically similar documents using DBSCAN
4. **Merge Clusters**: Uses Claude 4.5 Haiku to merge documents in each cluster into single consolidated entries
5. **Output Results**: Saves merged documents to `agent-input-output/output.json`

### Workflow Configuration

Key parameters in `src/merge-documents.ts`:
- `embeddingProvider`: Embedding model to use (titan | nova | cohere)
- `claudeModelId`: Claude model for merging (default: claude-haiku-4-5)
- `epsilon`: Clustering threshold (0.15 default, lower = stricter)
- `minPoints`: Minimum documents to form a cluster (2 default)

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

### Adding a New Embedding Model
1. Add provider type to `EmbeddingProvider` in `types.ts`
2. Add configuration to `EMBEDDING_MODELS` in `types.ts`
3. Implement provider-specific logic in `EmbeddingService.embedDocument()`

### Adjusting Clustering Sensitivity
Modify epsilon in clustering config:
- Lower epsilon (0.1-0.2): Stricter clusters, near-duplicates only
- Higher epsilon (0.3-0.5): Looser clusters, broader topics

### Using Different Embedding Models
Set `EMBEDDING_PROVIDER` environment variable:
- `titan`: Amazon Titan (default, 1024 dimensions)
- `nova`: Amazon Nova Multimodal (1024 dimensions)
- `cohere`: Cohere Embed English (512 dimensions)

### Debugging S3 Issues
```bash
# Check LocalStack logs
docker logs clustering-localstack

# List S3 contents (with LocalStack)
aws s3 ls s3://clustering-poc-embeddings --endpoint-url http://localhost:4566
```

## Performance Considerations

- **Embedding Dimensions**: Lower dimensions (256-384) for speed, higher (1024-3072) for accuracy
- **Batch Processing**: EmbeddingService handles batching automatically
- **Vector Normalization**: All embeddings are L2-normalized for consistent cosine similarity
- **Memory Usage**: Each 1024-dim embedding uses ~4KB; plan accordingly for large datasets