import { VectorStoreFactory } from './services/vector-store-factory.js';
import { S3VectorsService } from './services/s3-vectors-service.js';
import { EmbeddingService } from './services/embedding-service.js';
import { ClusteringService } from './services/clustering-service.js';
import {
  Document,
  EmbeddedDocument,
  EMBEDDING_MODELS,
  EmbeddingProvider,
  VectorStoreType,
  VectorStore,
} from './types.js';

/**
 * Configuration
 */
const USE_MOCK_EMBEDDINGS = process.env.USE_MOCK_EMBEDDINGS !== 'false'; // Default to true for testing
const EMBEDDING_PROVIDER = (process.env.EMBEDDING_PROVIDER || 'nova') as EmbeddingProvider;
const VECTOR_STORE_TYPE = (process.env.VECTOR_STORE_TYPE || 'simple-s3') as VectorStoreType;
const S3_BUCKET = process.env.S3_BUCKET || 'clustering-poc-embeddings';
const S3_VECTOR_BUCKET = process.env.S3_VECTOR_BUCKET || 'clustering-poc-vectors';
const S3_VECTOR_INDEX = process.env.S3_VECTOR_INDEX || 'embeddings-index';
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
 * Main example workflow
 */
async function main() {
  console.log('=== Document Clustering with S3 Embeddings ===\n');

  console.log(`Configuration:
  - Embedding Provider: ${EMBEDDING_PROVIDER}
  - Mock Embeddings: ${USE_MOCK_EMBEDDINGS}
  - Vector Store: ${VECTOR_STORE_TYPE}
  - Clustering Algorithm: HDBSCAN
  - Region: ${AWS_REGION}
${VECTOR_STORE_TYPE === 'simple-s3' ? `  - S3 Bucket: ${S3_BUCKET}` : `  - Vector Bucket: ${S3_VECTOR_BUCKET}\n  - Vector Index: ${S3_VECTOR_INDEX}`}
\n`);

  // Initialize services
  const embeddingModelConfig = EMBEDDING_MODELS[EMBEDDING_PROVIDER];

  // Create vector store based on configuration
  let vectorStore: VectorStore;
  if (VECTOR_STORE_TYPE === 'simple-s3') {
    vectorStore = VectorStoreFactory.create({
      type: 'simple-s3',
      s3Config: {
        bucket: S3_BUCKET,
        region: AWS_REGION,
      },
    });
  } else {
    vectorStore = VectorStoreFactory.create({
      type: 's3-vectors',
      s3VectorsConfig: {
        vectorBucket: S3_VECTOR_BUCKET,
        indexName: S3_VECTOR_INDEX,
        region: AWS_REGION,
        dimensions: embeddingModelConfig.dimensions || 1024,
      },
    });
  }

  const embeddingService = new EmbeddingService(
    embeddingModelConfig,
    AWS_REGION
  );
  const clusteringService = new ClusteringService({
    algorithm: 'hdbscan',
    minClusterSize: 2,
    minSamples: 2,
  });

  try {
    // Initialize vector store
    if (VECTOR_STORE_TYPE === 's3-vectors') {
      console.log('Step 0: Initializing S3 Vectors bucket and index...');
      await (vectorStore as S3VectorsService).initialize();
      console.log('✓ Vector store initialized\n');
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

    // Step 3: Store embeddings in vector store
    console.log(`Step 3: Storing embeddings in ${VECTOR_STORE_TYPE}...`);
    await vectorStore.storeEmbeddings(embeddedDocs);
    console.log(`✓ Stored ${embeddedDocs.length} embeddings\n`);

    // Step 4: Retrieve embeddings from vector store
    console.log(`Step 4: Retrieving embeddings from ${VECTOR_STORE_TYPE}...`);
    const retrievedDocs = await vectorStore.getAllEmbeddings();
    console.log(`✓ Retrieved ${retrievedDocs.length} embeddings\n`);

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
