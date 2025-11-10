import { readdir, readFile } from 'fs/promises';
import { join } from 'path';
import type { InputDocument } from './claude-service.js';

/**
 * Document with source file information
 */
export interface SourceDocument extends InputDocument {
  sourceFile: string;
  index: number; // Index within the source file
}

/**
 * Service for loading documents from JSON files
 */
export class DocumentLoader {
  /**
   * Load all documents from a directory containing JSON files
   * @param dirPath Path to directory with JSON files
   * @returns Array of documents with source information
   */
  async loadDocuments(dirPath: string): Promise<SourceDocument[]> {
    const files = await readdir(dirPath);
    const jsonFiles = files.filter(f => f.endsWith('.json')).sort();

    const allDocuments: SourceDocument[] = [];

    for (const file of jsonFiles) {
      const filePath = join(dirPath, file);
      const content = await readFile(filePath, 'utf-8');
      
      try {
        const docs = JSON.parse(content) as InputDocument[];
        
        // Add source information to each document
        docs.forEach((doc, idx) => {
          allDocuments.push({
            ...doc,
            sourceFile: file,
            index: idx,
          });
        });
      } catch (error) {
        console.error(`Failed to parse ${file}:`, error);
      }
    }

    return allDocuments;
  }

  /**
   * Generate a unique ID for a document
   */
  generateDocumentId(doc: SourceDocument): string {
    return `${doc.sourceFile.replace('.json', '')}_${doc.index}`;
  }
}

