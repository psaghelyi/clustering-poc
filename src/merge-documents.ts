#!/usr/bin/env tsx

/**
 * Document Clustering and Merging Workflow
 * 
 * This script:
 * 1. Loads all documents from JSON files in agent-input-output/outputs/
 * 2. Generates embeddings for each document using AWS Bedrock
 * 3. Clusters semantically similar documents using DBSCAN
 * 4. Merges documents in each cluster using Claude 4.5 Haiku
 * 5. Outputs a single merged JSON file
 */

import { writeFile, readFile, access } from 'fs/promises';
import { join } from 'path';
import { EmbeddingService } from './services/embedding-service.js';
import { ClusteringService } from './services/clustering-service.js';
import { ClaudeService } from './services/claude-service.js';
import { DocumentLoader, type SourceDocument } from './services/document-loader.js';
import type { Document, EmbeddedDocument } from './types.js';
import { EMBEDDING_MODELS } from './types.js';

// Configuration
const CONFIG = {
  inputDir: join(process.cwd(), 'agent-input-output', 'outputs'),
  outputFile: join(process.cwd(), 'agent-input-output', 'output.json'),
  embeddingsCacheFile: join(process.cwd(), 'agent-input-output', 'embeddings-cache.json'),
  awsRegion: process.env.AWS_REGION || 'us-east-1',
  embeddingProvider: (process.env.EMBEDDING_PROVIDER || 'titan') as 'nova' | 'titan' | 'cohere',
  claudeModelId: process.env.CLAUDE_MODEL_ID || 'anthropic.claude-3-5-haiku-20241022-v1:0',
  clustering: {
    algorithm: 'dbscan' as const,
    epsilon: 0.50, // Cosine distance threshold (0.50 = 50% similarity) - lower = stricter clustering
    minPoints: 2,  // Minimum documents to form a cluster
    distanceMetric: 'cosine' as const,
  },
};

/**
 * Check if embeddings cache file exists
 */
async function cacheExists(): Promise<boolean> {
  try {
    await access(CONFIG.embeddingsCacheFile);
    return true;
  } catch {
    return false;
  }
}

/**
 * Load embeddings from cache file
 */
async function loadEmbeddingsCache(): Promise<EmbeddedDocument[]> {
  const content = await readFile(CONFIG.embeddingsCacheFile, 'utf-8');
  return JSON.parse(content);
}

/**
 * Save embeddings to cache file
 */
async function saveEmbeddingsCache(embeddings: EmbeddedDocument[]): Promise<void> {
  const content = JSON.stringify(embeddings, null, 2);
  await writeFile(CONFIG.embeddingsCacheFile, content, 'utf-8');
}

async function main() {
  console.log('🚀 Starting document clustering and merging workflow...\n');

  // Step 1: Load documents
  console.log('📂 Loading documents from:', CONFIG.inputDir);
  const loader = new DocumentLoader();
  const sourceDocs = await loader.loadDocuments(CONFIG.inputDir);
  console.log(`✅ Loaded ${sourceDocs.length} documents from ${new Set(sourceDocs.map(d => d.sourceFile)).size} files\n`);

  if (sourceDocs.length === 0) {
    console.error('❌ No documents found!');
    process.exit(1);
  }

  // Step 2: Generate or load embeddings
  let embeddedDocs: EmbeddedDocument[];
  
  if (await cacheExists()) {
    console.log('📦 Loading embeddings from cache...');
    embeddedDocs = await loadEmbeddingsCache();
    console.log(`✅ Loaded ${embeddedDocs.length} embeddings from cache\n`);
  } else {
    console.log('🧠 Generating embeddings using AWS Bedrock...');
    const embeddingModelConfig = EMBEDDING_MODELS[CONFIG.embeddingProvider];
    const embeddingService = new EmbeddingService(embeddingModelConfig, CONFIG.awsRegion);

    const documents: Document[] = sourceDocs.map(doc => ({
      id: loader.generateDocumentId(doc),
      content: doc.description,
      metadata: {
        owner: doc.owner,
        sourceFile: doc.sourceFile,
        index: doc.index,
      },
    }));

    embeddedDocs = [];
    let processedCount = 0;
    const totalDocs = documents.length;
    
    for (const doc of documents) {
      const embeddedDoc = await embeddingService.embedDocument(doc);
      embeddedDocs.push(embeddedDoc);
      processedCount++;
      
      // Log progress every 10 documents
      if (processedCount % 10 === 0 || processedCount === totalDocs) {
        console.log(`   Progress: ${processedCount}/${totalDocs} embeddings generated (${Math.round(processedCount/totalDocs*100)}%)`);
      }
    }
    console.log(`✅ Generated ${embeddedDocs.length} embeddings\n`);
    
    // Save embeddings to cache
    console.log('💾 Saving embeddings to cache...');
    await saveEmbeddingsCache(embeddedDocs);
    console.log(`✅ Cached embeddings saved to: ${CONFIG.embeddingsCacheFile}\n`);
  }

  // Step 3: Cluster similar documents
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

  // Step 4: Merge documents using Claude
  console.log('🤖 Merging documents with Claude 4.5 Haiku...');
  const claudeService = new ClaudeService(CONFIG.awsRegion, CONFIG.claudeModelId);

  const mergedDocuments = [];

  // Merge clustered documents
  let mergedCount = 0;
  for (const cluster of clusters) {
    const inputDocs = cluster.documents.map(doc => ({
      description: doc.content,
      owner: doc.metadata?.owner || 'No owner',
    }));

    const merged = await claudeService.mergeDocuments(inputDocs);
    mergedDocuments.push(merged);
    mergedCount++;
    console.log(`   ✓ [${mergedCount}/${clusters.length}] Merged cluster ${cluster.clusterId} (${cluster.documents.length} docs → 1)`);
  }

  // Add noise documents as-is (no merging needed)
  for (const doc of noiseDocs) {
    mergedDocuments.push({
      description: doc.content,
      owner: doc.metadata?.owner || 'No owner',
    });
  }

  console.log(`✅ Merged into ${mergedDocuments.length} final documents\n`);

  // Step 5: Save output
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

