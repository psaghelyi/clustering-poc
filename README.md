# Document Clustering with S3 Embeddings

A TypeScript-based proof-of-concept for document clustering using embeddings stored in S3. Supports multiple AWS Bedrock embedding models (Nova, Titan, Cohere) and uses DBSCAN clustering with cosine similarity.

## Features

- **Configurable Embedding Models**: Support for AWS Bedrock's Nova, Titan, and Cohere embedding models
- **S3 Storage**: Store and retrieve embeddings from S3 (or LocalStack for testing)
- **DBSCAN Clustering**: Automatic cluster discovery using density-based clustering with cosine similarity
- **Mock Mode**: Test without AWS credentials using mock embeddings
- **TypeScript**: Fully typed with modern ES modules

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Documents  │───▶│  Embeddings  │───▶│  S3 Storage  │
└─────────────┘    └──────────────┘    └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Clustering  │
                   │   (DBSCAN)   │
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   Clusters   │
                   └──────────────┘
```

## Prerequisites

- Node.js 18+
- Docker & Docker Compose (for LocalStack)
- AWS CLI (optional, for LocalStack bucket creation)
- AWS credentials (only if using real Bedrock/S3)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd clustering-poc
```

2. Install dependencies:
```bash
npm install
```

3. Copy environment configuration:
```bash
cp .env.example .env
```

## Configuration

### Environment Variables

Edit `.env` to configure the application:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

# S3 Configuration
S3_BUCKET=clustering-poc-embeddings

# Embedding Model: nova | titan | cohere
EMBEDDING_PROVIDER=nova

# Testing Options
USE_MOCK_EMBEDDINGS=true   # true = mock, false = real Bedrock
USE_LOCALSTACK=true        # true = localstack, false = real S3
```

### Embedding Models

| Provider | Model ID | Dimensions | Features |
|----------|----------|------------|----------|
| **Nova** | `amazon.nova-embed-text-v1` | 1024* | State-of-the-art, multimodal, 200 languages |
| **Titan** | `amazon.titan-embed-text-v2:0` | 1024 | Strong domain performance, multilingual |
| **Cohere** | `cohere.embed-english-v3` | 1024 | Excellent for RAG, semantic search |

*Nova supports 3072, 1024, 384, 256 dimensions

### Clustering Configuration

Edit `src/example.ts` to tune clustering parameters:

```typescript
const clusteringService = new ClusteringService({
  algorithm: 'dbscan',        // dbscan | optics | kmeans
  epsilon: 0.3,               // Distance threshold (lower = stricter)
  minPoints: 2,               // Minimum points per cluster
  distanceMetric: 'cosine',   // cosine | euclidean
});
```

## Usage

### Option 1: Quick Start with Mock Embeddings (No AWS)

1. Start LocalStack:
```bash
npm run localstack:up
```

2. Run the example:
```bash
npm run dev
```

This uses mock embeddings and LocalStack S3 - no AWS credentials needed!

### Option 2: Using Real AWS Bedrock & S3

1. Configure AWS credentials in `.env`:
```bash
AWS_ACCESS_KEY_ID=your-actual-key
AWS_SECRET_ACCESS_KEY=your-actual-secret
USE_MOCK_EMBEDDINGS=false
USE_LOCALSTACK=false
```

2. Create S3 bucket (if needed):
```bash
aws s3 mb s3://clustering-poc-embeddings --region us-east-1
```

3. Run the example:
```bash
npm run dev
```

### Option 3: Real Bedrock + LocalStack S3

Best of both worlds for testing embeddings without S3 costs:

```bash
USE_MOCK_EMBEDDINGS=false
USE_LOCALSTACK=true
```

## Project Structure

```
clustering-poc/
├── src/
│   ├── types.ts                      # Type definitions
│   ├── services/
│   │   ├── s3-service.ts            # S3 storage operations
│   │   ├── embedding-service.ts     # AWS Bedrock embedding generation
│   │   └── clustering-service.ts    # DBSCAN clustering
│   └── example.ts                   # Complete workflow demo
├── docker-compose.yml               # LocalStack setup
├── package.json
├── tsconfig.json
└── README.md
```

## API Overview

### S3Service

```typescript
const s3Service = new S3Service({
  bucket: 'my-bucket',
  region: 'us-east-1',
  endpoint: 'http://localhost:4566',  // Optional: for localstack
  forcePathStyle: true,
});

// Store embeddings
await s3Service.storeEmbeddings(embeddedDocs);

// Retrieve all embeddings
const docs = await s3Service.getAllEmbeddings();
```

### EmbeddingService

```typescript
const embeddingService = new EmbeddingService(
  EMBEDDING_MODELS.nova,  // or .titan, .cohere
  'us-east-1'
);

// Embed single document
const embeddedDoc = await embeddingService.embedDocument(doc);

// Embed multiple documents
const embeddedDocs = await embeddingService.embedDocuments(docs);
```

### ClusteringService

```typescript
const clusteringService = new ClusteringService({
  algorithm: 'dbscan',
  epsilon: 0.3,
  minPoints: 2,
  distanceMetric: 'cosine',
});

// Cluster documents
const clusters = clusteringService.clusterDocuments(embeddedDocs);

// Find similar documents
const similar = clusteringService.findSimilarDocuments(
  queryDoc,
  allDocs,
  0.8,  // Similarity threshold
  5     // Max results
);

// Get statistics
const stats = clusteringService.getClusteringStats(clusters);
```

## Clustering Algorithms

### DBSCAN (Default - Recommended)

**Pros:**
- Automatically discovers number of clusters
- Handles noise and outliers well
- Finds clusters of arbitrary shapes
- Good for varying density

**Best for:** Documents with natural groupings

**Parameters:**
- `epsilon`: Maximum distance between points (0.1-0.5 recommended for cosine)
- `minPoints`: Minimum cluster size (2-5 recommended)

### OPTICS

**Pros:**
- Hierarchical density-based clustering
- Better than DBSCAN for varying densities
- Can extract cluster hierarchy

**Best for:** Documents with nested topics

### K-Means

**Pros:**
- Fast and simple
- Good for well-separated clusters

**Cons:**
- Must specify number of clusters (k)
- Assumes spherical clusters

## Best Practices

### Cosine Similarity

- **Best for:** High-dimensional embeddings (like text embeddings)
- **Range:** -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
- **Distance:** Use `1 - similarity` for clustering
- **Threshold:** 0.7-0.9 for similar documents

### Epsilon Tuning

| Epsilon | Similarity | Use Case |
|---------|------------|----------|
| 0.1-0.2 | Very strict (0.8-0.9) | Near-duplicates |
| 0.3-0.4 | Moderate (0.6-0.7) | Similar topics |
| 0.5+ | Loose (<0.5) | Broad categories |

### Performance Tips

1. **Normalize vectors** before storage (already done in embedding service)
2. **Batch operations** when storing/retrieving from S3
3. **Use appropriate dimensions** (lower = faster, higher = more accurate)
4. **Consider FAISS/Annoy** for large-scale production (>10k documents)

## Example Output

```
=== Document Clustering with S3 Embeddings ===

Configuration:
  - Embedding Provider: nova
  - Mock Embeddings: true
  - LocalStack: true
  - S3 Bucket: clustering-poc-embeddings
  - Region: us-east-1

Step 1: Generating sample documents...
✓ Generated 12 documents

Step 2: Generating embeddings...
✓ Generated 12 embeddings

Step 3: Storing embeddings in S3...
✓ Stored 12 embeddings in S3

Step 4: Retrieving embeddings from S3...
✓ Retrieved 12 embeddings from S3

Step 5: Clustering documents by cosine similarity...
✓ Found 4 clusters

=== Clustering Results ===

Statistics:
  - Total Documents: 12
  - Number of Clusters: 4
  - Average Cluster Size: 3.00
  - Min Cluster Size: 3
  - Max Cluster Size: 3

Cluster 0 (3 documents):
  - [technology] doc-1: Cloud computing enables scalable infrastructure...
  - [technology] doc-2: AWS provides various cloud services...
  - [technology] doc-3: Kubernetes orchestrates containerized...

Cluster 1 (3 documents):
  - [ml] doc-4: Neural networks are the foundation...
  - [ml] doc-5: Transformers revolutionized natural language...
  - [ml] doc-6: Embedding models convert text...

Cluster 2 (3 documents):
  - [finance] doc-7: Stock markets fluctuate...
  - [finance] doc-8: Investment portfolios should be...
  - [finance] doc-9: Cryptocurrency trading involves...

Cluster 3 (3 documents):
  - [health] doc-10: Regular exercise and balanced nutrition...
  - [health] doc-11: Cardiovascular fitness improves...
  - [health] doc-12: Mental wellness requires adequate...
```

## Troubleshooting

### LocalStack Issues

```bash
# Check if LocalStack is running
docker ps | grep localstack

# View LocalStack logs
docker logs clustering-localstack

# Restart LocalStack
npm run localstack:down
npm run localstack:up
```

### AWS Bedrock Access

Ensure your AWS account has access to Bedrock models:
```bash
aws bedrock list-foundation-models --region us-east-1
```

### TypeScript Build Errors

```bash
# Clean and rebuild
npm run clean
npm run build
```

## Production Considerations

1. **Error Handling**: Add retry logic for S3 and Bedrock API calls
2. **Rate Limiting**: Respect AWS Bedrock quotas (especially for embeddings)
3. **Caching**: Cache embeddings to avoid regeneration
4. **Batch Processing**: Process large document sets in batches
5. **Monitoring**: Add CloudWatch metrics for clustering performance
6. **Vector Database**: Consider using specialized vector DBs (Pinecone, Weaviate, pgvector)

## References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon Nova Embeddings](https://aws.amazon.com/blogs/aws/amazon-nova-multimodal-embeddings-now-available-in-amazon-bedrock/)
- [DBSCAN Algorithm](https://en.wikipedia.org/wiki/DBSCAN)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [LocalStack S3](https://docs.localstack.cloud/user-guide/aws/s3/)

## License

MIT
