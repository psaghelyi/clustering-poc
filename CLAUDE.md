# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document clustering tool that uses AWS Bedrock embeddings, S3 Vectors storage, and HDBSCAN clustering to merge semantically similar documents.

## Workflow (src/merge-documents.ts)

1. Load JSON documents from `agent-input-output/outputs/` (each has `description` and `owner` fields)
2. If S3 Vectors is empty: generate embeddings via AWS Bedrock and store in S3 Vectors
3. If S3 Vectors has embeddings: load them directly
4. Run HDBSCAN clustering to identify similar document groups
5. Merge clusters with >2 documents using Claude/Nova
6. Keep noise documents (outliers) and small clusters as-is
7. Save output to `agent-input-output/output.json`

## Commands

```bash
# Main workflow (requires aws-vault)
npm run merge:aws

# Build
npm run build
```

## Configuration (.env.example)

**Required:**
- `EMBEDDING_PROVIDER`: `titan` | `nova` | `cohere` (for generating embeddings)
- `LLM_MODEL`: `claude-haiku` | `nova` (for merging documents)
- `S3_VECTOR_BUCKET`: S3 bucket name for vector storage
- `S3_VECTOR_INDEX`: Index name in S3 Vectors
- `EMBEDDING_DIMENSIONS`: Vector dimensions (must match provider, typically 1024)

**HDBSCAN Parameters:**
- `DISTANCE_METRIC`: `euclidean` | `cosine` (cosine recommended for text)
- `MIN_CLUSTER_SIZE`: Minimum documents to form a cluster (default: 2)
- `MIN_SAMPLES`: Minimum samples for core points (default: 2)

## Key Implementation Details

**HDBSCAN**: Density-based clustering that automatically finds cluster count. Documents not fitting any cluster are marked as "noise" (label -1) and kept unmerged. The `metric` parameter significantly impacts results - cosine is better for semantic similarity.

**S3 Vectors**: AWS preview service for native vector storage. Only available in specific regions (us-east-1, us-east-2, us-west-2, eu-central-1, ap-southeast-2). Requires bucket initialization on first run.

**Clustering threshold**: Controlled indirectly via `MIN_CLUSTER_SIZE` and `MIN_SAMPLES`. Lower values create more/smaller clusters. HDBSCAN automatically adapts to varying densities.
