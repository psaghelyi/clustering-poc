#!/usr/bin/env tsx

/**
 * Document Clustering and Merging Workflow with S3 Vectors
 *
 * This script:
 * 1. Loads all documents from JSON files in agent-input-output/outputs/
 * 2. Generates embeddings for each document using AWS Bedrock
 * 3. Stores embeddings in S3 Vectors with metadata
 * 4. Retrieves embeddings from S3 Vectors for clustering
 * 5. Clusters semantically similar documents using HDBSCAN
 * 6. Merges documents in each cluster using Claude Haiku 4.5
 * 7. Outputs a single merged JSON file
 */

import 'dotenv/config';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import { EmbeddingService } from './services/embedding-service.js';
import { S3VectorsService } from './services/s3-vectors-service.js';
import { ClusteringService } from './services/clustering-service.js';
import { ClaudeService } from './services/claude-service.js';
import { DocumentLoader, type SourceDocument } from './services/document-loader.js';
import type { Document, EmbeddedDocument } from './types.js';
import { EMBEDDING_MODELS } from './types.js';

// LLM Model mapping - using cross-region inference profiles
const LLM_MODELS = {
  'claude-haiku': process.env.CLAUDE_MODEL_ID || 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
  'nova': process.env.NOVA_MODEL_ID || 'us.amazon.nova-lite-v1:0',
};

// Configuration
const CONFIG = {
  inputDir: join(process.cwd(), 'agent-input-output', 'outputs'),
  outputFile: join(process.cwd(), 'agent-input-output', 'output.json'),
  awsRegion: process.env.AWS_REGION || 'us-east-1',
  embeddingProvider: (process.env.EMBEDDING_PROVIDER || 'titan') as 'nova' | 'titan' | 'cohere',
  llmModel: (process.env.LLM_MODEL || 'claude-haiku') as keyof typeof LLM_MODELS,
  s3Vectors: {
    vectorBucket: process.env.S3_VECTOR_BUCKET || 'clustering-poc-vectors',
    indexName: process.env.S3_VECTOR_INDEX || 'embeddings-index',
    dimensions: parseInt(process.env.EMBEDDING_DIMENSIONS || '1024'),
  },
  clustering: {
    algorithm: 'hdbscan' as const,
    minClusterSize: parseInt(process.env.MIN_CLUSTER_SIZE || '2'),         // Minimum documents to form a cluster
    minSamples: parseInt(process.env.MIN_SAMPLES || '2'),                  // Minimum samples for core points
    metric: (process.env.DISTANCE_METRIC as 'euclidean' | 'cosine') || 'euclidean', // Distance metric for similarity
  },
};

/**
 * Check if embeddings exist in S3 Vectors
 */
async function checkS3VectorsPopulated(s3Vectors: S3VectorsService): Promise<boolean> {
  try {
    const embeddings = await s3Vectors.getAllEmbeddings();
    return embeddings.length > 0;
  } catch {
    return false;
  }
}

async function main() {
  const forceReindex = process.argv.includes('--reindex');

  console.log('🚀 Starting document clustering and merging workflow with S3 Vectors...\n');

  // Step 1: Initialize S3 Vectors
  console.log('🔧 Initializing S3 Vectors...');
  console.log(`   - Bucket: ${CONFIG.s3Vectors.vectorBucket}`);
  console.log(`   - Index: ${CONFIG.s3Vectors.indexName}`);
  console.log(`   - Dimensions: ${CONFIG.s3Vectors.dimensions}`);
  const s3Vectors = new S3VectorsService({
    vectorBucket: CONFIG.s3Vectors.vectorBucket,
    indexName: CONFIG.s3Vectors.indexName,
    region: CONFIG.awsRegion,
    dimensions: CONFIG.s3Vectors.dimensions,
  });
  await s3Vectors.initialize();
  console.log('✅ S3 Vectors initialized\n');

  // Force reindex if requested
  if (forceReindex) {
    console.log('🗑️  Force reindex requested - deleting all existing embeddings...');
    await s3Vectors.deleteAllEmbeddings();
    console.log('✅ All embeddings deleted\n');
  }

  // Step 2: Load documents
  console.log('📂 Loading documents from:', CONFIG.inputDir);
  const loader = new DocumentLoader();
  const sourceDocs = await loader.loadDocuments(CONFIG.inputDir);
  console.log(`✅ Loaded ${sourceDocs.length} documents from ${new Set(sourceDocs.map(d => d.sourceFile)).size} files\n`);

  if (sourceDocs.length === 0) {
    console.error('❌ No documents found!');
    process.exit(1);
  }

  // Step 3: Generate or load embeddings from S3 Vectors
  let embeddedDocs: EmbeddedDocument[];
  
  const hasExistingEmbeddings = await checkS3VectorsPopulated(s3Vectors);
  
  if (hasExistingEmbeddings) {
    console.log('📦 Loading embeddings from S3 Vectors...');
    embeddedDocs = await s3Vectors.getAllEmbeddings();
    console.log(`✅ Loaded ${embeddedDocs.length} embeddings from S3 Vectors\n`);
  } else {
    console.log('🧠 Generating embeddings using AWS Bedrock...');
    const embeddingModelConfig = EMBEDDING_MODELS[CONFIG.embeddingProvider];
    const embeddingService = new EmbeddingService(embeddingModelConfig, CONFIG.awsRegion);

    const documents: Document[] = sourceDocs.map(doc => ({
      id: loader.generateDocumentId(doc),
      content: doc.description,
      metadata: {
        owner: doc.owner,
        description: doc.description, // Store for S3 Vectors metadata
        sourceFile: doc.sourceFile,
        index: doc.index,
      },
    }));

    // Use batch processing for faster embedding generation
    const totalDocs = documents.length;
    const batchSize = CONFIG.embeddingProvider === 'cohere' ? 96 : 10;

    embeddedDocs = [];
    let processedCount = 0;

    // Process in batches with progress logging
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      const batchResults = await embeddingService.embedDocuments(batch);

      // Validate embeddings before adding them
      for (const doc of batchResults) {
        if (!doc.embedding || !Array.isArray(doc.embedding)) {
          console.error(`⚠️  Warning: Document ${doc.id} has invalid embedding:`, typeof doc.embedding);
        }
      }

      embeddedDocs.push(...batchResults);

      processedCount += batch.length;
      if (processedCount % 10 === 0 || processedCount === totalDocs) {
        console.log(`   Progress: ${processedCount}/${totalDocs} embeddings generated (${Math.round(processedCount/totalDocs*100)}%)`);
      }
    }
    console.log(`✅ Generated ${embeddedDocs.length} embeddings\n`);
    
    // Store embeddings in S3 Vectors with metadata
    console.log('💾 Storing embeddings in S3 Vectors...');
    await s3Vectors.storeEmbeddings(embeddedDocs);
    console.log('✅ Embeddings stored in S3 Vectors\n');
  }

  // Step 4: Cluster similar documents using HDBSCAN
  console.log('🔍 Clustering similar documents...');
  const clusteringService = new ClusteringService(CONFIG.clustering);
  const clusters = clusteringService.clusterDocuments(embeddedDocs);
  
  // Find which documents are in clusters
  const clusteredDocIds = new Set<string>();
  for (const cluster of clusters) {
    for (const doc of cluster.documents) {
      clusteredDocIds.add(doc.id);
    }
  }
  
  // Noise documents are those not in any cluster
  const noiseDocs = embeddedDocs.filter(doc => !clusteredDocIds.has(doc.id));
  
  console.log(`✅ Found ${clusters.length} clusters`);
  console.log(`   - ${noiseDocs.length} documents didn't cluster (will be kept as-is)`);
  
  // Show cluster sizes
  clusters.forEach(cluster => {
    console.log(`   - Cluster ${cluster.clusterId}: ${cluster.documents.length} documents`);
  });
  console.log();

  // Step 5: Merge documents using LLM (only clusters with >2 documents)
  const llmModelId = LLM_MODELS[CONFIG.llmModel];
  const llmName = CONFIG.llmModel === 'nova' ? 'Nova Lite' : 'Claude Haiku 4.5';
  console.log(`🤖 Merging clusters with >2 documents using ${llmName}...`);
  const claudeService = new ClaudeService(CONFIG.awsRegion, llmModelId);

  const mergedDocuments = [];

  // Merge clusters with more than 2 documents
  const clustersToMerge = clusters.filter(c => c.documents.length > 2);
  const smallClusters = clusters.filter(c => c.documents.length <= 2);

  let mergedCount = 0;
  for (const cluster of clustersToMerge) {
    const inputDocs = cluster.documents.map(doc => ({
      description: doc.content,
      owner: doc.metadata?.owner || 'No owner',
    }));

    const merged = await claudeService.mergeDocuments(inputDocs);
    mergedDocuments.push(merged);
    mergedCount++;
    console.log(`   ✓ [${mergedCount}/${clustersToMerge.length}] Merged cluster ${cluster.clusterId} (${cluster.documents.length} docs → 1)`);
  }

  // Add small clusters as-is (no merging needed for <=2 docs)
  for (const cluster of smallClusters) {
    for (const doc of cluster.documents) {
      mergedDocuments.push({
        description: doc.content,
        owner: doc.metadata?.owner || 'No owner',
      });
    }
  }

  // Add noise documents as-is (no merging needed)
  for (const doc of noiseDocs) {
    mergedDocuments.push({
      description: doc.content,
      owner: doc.metadata?.owner || 'No owner',
    });
  }

  console.log(`✅ Merged into ${mergedDocuments.length} final documents\n`);

  // Step 6: Save output
  console.log('💾 Saving merged documents to:', CONFIG.outputFile);
  const output = JSON.stringify(mergedDocuments, null, 2);
  await writeFile(CONFIG.outputFile, output, 'utf-8');
  
  console.log('\n✨ Done! Summary:');
  console.log(`   Input:  ${sourceDocs.length} documents`);
  console.log(`   Output: ${mergedDocuments.length} documents`);
  console.log(`   Reduction: ${((1 - mergedDocuments.length / sourceDocs.length) * 100).toFixed(1)}%`);
  console.log(`\n📄 Output saved to: ${CONFIG.outputFile}`);
}

// Run the workflow
main().catch(error => {
  console.error('❌ Error:', error);
  process.exit(1);
});

