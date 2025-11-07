import { S3Service } from './services/s3-service.js';
import { EmbeddingService } from './services/embedding-service.js';
import { ClusteringService } from './services/clustering-service.js';
import {
  Document,
  EmbeddedDocument,
  EMBEDDING_MODELS,
  EmbeddingProvider,
} from './types.js';
import { CreateBucketCommand, S3Client } from '@aws-sdk/client-s3';

/**
 * Configuration
 */
const USE_MOCK_EMBEDDINGS = process.env.USE_MOCK_EMBEDDINGS !== 'false'; // Default to true for testing
const EMBEDDING_PROVIDER = (process.env.EMBEDDING_PROVIDER || 'nova') as EmbeddingProvider;
const USE_LOCALSTACK = process.env.USE_LOCALSTACK !== 'false'; // Default to true
const S3_BUCKET = process.env.S3_BUCKET || 'clustering-poc-embeddings';
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';

/**
 * Generate sample documents for testing
 */
function generateSampleDocuments(): Document[] {
  return [
    // Technology cluster
    {
      id: 'doc-1',
      content: 'Cloud computing enables scalable infrastructure and distributed systems.',
      metadata: { category: 'technology' },
    },
    {
      id: 'doc-2',
      content: 'AWS provides various cloud services including S3, EC2, and Lambda.',
      metadata: { category: 'technology' },
    },
    {
      id: 'doc-3',
      content: 'Kubernetes orchestrates containerized applications in cloud environments.',
      metadata: { category: 'technology' },
    },

    // Machine Learning cluster
    {
      id: 'doc-4',
      content: 'Neural networks are the foundation of modern deep learning systems.',
      metadata: { category: 'ml' },
    },
    {
      id: 'doc-5',
      content: 'Transformers revolutionized natural language processing and generation.',
      metadata: { category: 'ml' },
    },
    {
      id: 'doc-6',
      content: 'Embedding models convert text into high-dimensional vector representations.',
      metadata: { category: 'ml' },
    },

    // Finance cluster
    {
      id: 'doc-7',
      content: 'Stock markets fluctuate based on economic indicators and investor sentiment.',
      metadata: { category: 'finance' },
    },
    {
      id: 'doc-8',
      content: 'Investment portfolios should be diversified across different asset classes.',
      metadata: { category: 'finance' },
    },
    {
      id: 'doc-9',
      content: 'Cryptocurrency trading involves high volatility and market speculation.',
      metadata: { category: 'finance' },
    },

    // Health cluster
    {
      id: 'doc-10',
      content: 'Regular exercise and balanced nutrition are essential for good health.',
      metadata: { category: 'health' },
    },
    {
      id: 'doc-11',
      content: 'Cardiovascular fitness improves through aerobic activities and training.',
      metadata: { category: 'health' },
    },
    {
      id: 'doc-12',
      content: 'Mental wellness requires adequate sleep, stress management, and mindfulness.',
      metadata: { category: 'health' },
    },
  ];
}

/**
 * Generate mock embeddings for testing without AWS Bedrock
 * Creates similar embeddings for documents in the same category
 */
function generateMockEmbeddings(docs: Document[]): EmbeddedDocument[] {
  const dimensions = 1024;

  return docs.map((doc) => {
    // Create a base embedding based on category
    const category = doc.metadata?.category || 'unknown';
    const categorySeeds: Record<string, number[]> = {
      technology: [1, 0.8, 0.6],
      ml: [0.8, 1, 0.7],
      finance: [0.2, 0.3, 1],
      health: [0.5, 0.2, 0.4],
      unknown: [0.5, 0.5, 0.5],
    };

    const seed = categorySeeds[category] || categorySeeds.unknown;
    const embedding: number[] = [];

    // Generate embedding with some randomness but clustering around category seed
    for (let i = 0; i < dimensions; i++) {
      const seedValue = seed[i % seed.length];
      const noise = (Math.random() - 0.5) * 0.2; // Small random noise
      embedding.push(seedValue + noise);
    }

    // Normalize the vector
    const normalized = EmbeddingService.normalizeVector(embedding);

    return {
      ...doc,
      embedding: normalized,
    };
  });
}

/**
 * Ensure S3 bucket exists (for localstack)
 */
async function ensureBucketExists(s3Config: any): Promise<void> {
  const client = new S3Client({
    region: s3Config.region,
    endpoint: s3Config.endpoint,
    forcePathStyle: s3Config.forcePathStyle,
    credentials: s3Config.endpoint
      ? { accessKeyId: 'test', secretAccessKey: 'test' }
      : undefined,
  });

  try {
    await client.send(new CreateBucketCommand({ Bucket: S3_BUCKET }));
    console.log(`✓ Created bucket: ${S3_BUCKET}`);
  } catch (error: any) {
    if (error.name === 'BucketAlreadyOwnedByYou' || error.name === 'BucketAlreadyExists') {
      console.log(`✓ Bucket already exists: ${S3_BUCKET}`);
    } else {
      throw error;
    }
  }
}

/**
 * Main example workflow
 */
async function main() {
  console.log('=== Document Clustering with S3 Embeddings ===\n');

  // Configure S3
  const s3Config = {
    bucket: S3_BUCKET,
    region: AWS_REGION,
    ...(USE_LOCALSTACK && {
      endpoint: 'http://localhost:4566',
      forcePathStyle: true,
    }),
  };

  console.log(`Configuration:
  - Embedding Provider: ${EMBEDDING_PROVIDER}
  - Mock Embeddings: ${USE_MOCK_EMBEDDINGS}
  - LocalStack: ${USE_LOCALSTACK}
  - S3 Bucket: ${S3_BUCKET}
  - Region: ${AWS_REGION}
\n`);

  // Initialize services
  const s3Service = new S3Service(s3Config);
  const embeddingService = new EmbeddingService(
    EMBEDDING_MODELS[EMBEDDING_PROVIDER],
    AWS_REGION
  );
  const clusteringService = new ClusteringService({
    algorithm: 'dbscan',
    epsilon: 0.3, // Cosine distance threshold (lower = more similar required)
    minPoints: 2,
    distanceMetric: 'cosine',
  });

  try {
    // Ensure bucket exists (important for localstack)
    if (USE_LOCALSTACK) {
      await ensureBucketExists(s3Config);
    }

    // Step 1: Generate sample documents
    console.log('Step 1: Generating sample documents...');
    const documents = generateSampleDocuments();
    console.log(`✓ Generated ${documents.length} documents\n`);

    // Step 2: Generate embeddings
    console.log('Step 2: Generating embeddings...');
    let embeddedDocs: EmbeddedDocument[];

    if (USE_MOCK_EMBEDDINGS) {
      console.log('  Using mock embeddings (set USE_MOCK_EMBEDDINGS=false to use real Bedrock)');
      embeddedDocs = generateMockEmbeddings(documents);
    } else {
      console.log(`  Using AWS Bedrock: ${EMBEDDING_MODELS[EMBEDDING_PROVIDER].modelId}`);
      embeddedDocs = await embeddingService.embedDocuments(documents);
    }
    console.log(`✓ Generated ${embeddedDocs.length} embeddings\n`);

    // Step 3: Store embeddings in S3
    console.log('Step 3: Storing embeddings in S3...');
    await s3Service.storeEmbeddings(embeddedDocs);
    console.log(`✓ Stored ${embeddedDocs.length} embeddings in S3\n`);

    // Step 4: Retrieve embeddings from S3
    console.log('Step 4: Retrieving embeddings from S3...');
    const retrievedDocs = await s3Service.getAllEmbeddings();
    console.log(`✓ Retrieved ${retrievedDocs.length} embeddings from S3\n`);

    // Step 5: Cluster documents
    console.log('Step 5: Clustering documents by cosine similarity...');
    const clusters = clusteringService.clusterDocuments(retrievedDocs);
    console.log(`✓ Found ${clusters.length} clusters\n`);

    // Step 6: Display results
    console.log('=== Clustering Results ===\n');

    const stats = clusteringService.getClusteringStats(clusters);
    console.log('Statistics:');
    console.log(`  - Total Documents: ${stats.totalDocuments}`);
    console.log(`  - Number of Clusters: ${stats.numClusters}`);
    console.log(`  - Average Cluster Size: ${stats.avgClusterSize.toFixed(2)}`);
    console.log(`  - Min Cluster Size: ${stats.minClusterSize}`);
    console.log(`  - Max Cluster Size: ${stats.maxClusterSize}\n`);

    clusters.forEach((cluster, index) => {
      console.log(`\nCluster ${cluster.clusterId} (${cluster.documents.length} documents):`);
      cluster.documents.forEach((doc) => {
        const category = doc.metadata?.category || 'unknown';
        console.log(`  - [${category}] ${doc.id}: ${doc.content.substring(0, 60)}...`);
      });
    });

    // Step 7: Calculate similarity matrix for first cluster (if exists)
    if (clusters.length > 0 && clusters[0].documents.length > 1) {
      console.log(`\n=== Similarity Matrix for Cluster 0 ===\n`);
      const matrix = clusteringService.calculateSimilarityMatrix(clusters[0].documents);

      console.log('Document IDs:', clusters[0].documents.map((d) => d.id).join(', '));
      console.log('\nCosine Similarities:');
      matrix.forEach((row, i) => {
        const rowStr = row.map((val) => val.toFixed(3)).join('  ');
        console.log(`${clusters[0].documents[i].id}: ${rowStr}`);
      });
    }

    console.log('\n=== Example Complete ===\n');
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

// Run the example
main();
