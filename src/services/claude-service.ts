import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from '@aws-sdk/client-bedrock-runtime';

/**
 * Input document structure from JSON files
 */
export interface InputDocument {
  description: string;
  owner: string;
}

/**
 * Merged document result
 */
export interface MergedDocument {
  description: string;
  owner: string;
  mergedFrom?: string[]; // Original descriptions that were merged
}

/**
 * Service for calling AWS Bedrock Claude models to merge documents
 */
export class ClaudeService {
  private client: BedrockRuntimeClient;
  private modelId: string;

  constructor(region: string = 'us-east-1', modelId: string = 'anthropic.claude-haiku-4-5-20251001-v1:0') {
    this.client = new BedrockRuntimeClient({ 
      region,
      // AWS SDK will automatically use credentials from aws-vault or environment
    });
    this.modelId = modelId;
  }

  /**
   * Merge a cluster of semantically similar documents into a single document
   * @param documents Array of similar documents to merge
   * @returns Merged document
   */
  async mergeDocuments(documents: InputDocument[]): Promise<MergedDocument> {
    if (documents.length === 0) {
      throw new Error('Cannot merge empty document array');
    }

    // If only one document, return it as-is
    if (documents.length === 1) {
      return {
        ...documents[0],
        mergedFrom: [documents[0].description],
      };
    }

    // Prepare the prompt for Claude
    const prompt = this.buildMergePrompt(documents);

    // Call Claude via Bedrock
    const response = await this.invokeModel(prompt);

    // Parse the response
    const mergedDoc = this.parseResponse(response, documents);

    return mergedDoc;
  }

  /**
   * Build the prompt for Claude to merge documents
   */
  private buildMergePrompt(documents: InputDocument[]): string {
    const docList = documents
      .map((doc, idx) => `${idx + 1}. ${doc.description} (Owner: ${doc.owner})`)
      .join('\n');

    return `You are a document analyst tasked with merging semantically similar documents into a single, consolidated document.

Here are ${documents.length} similar documents:

${docList}

Please merge these documents into a single consolidated entry that:
1. Combines all unique information from the descriptions
2. Eliminates redundancy while preserving all distinct requirements or details
3. Maintains clarity and specificity
4. Assigns an appropriate owner (if multiple owners, choose the most relevant or use "Multiple owners")

Respond with a JSON object in this exact format:
{
  "description": "your merged description here",
  "owner": "chosen owner here"
}

Only return the JSON object, no additional text.`;
  }

  /**
   * Invoke the Claude model via Bedrock
   */
  private async invokeModel(prompt: string): Promise<string> {
    const payload = {
      anthropic_version: 'bedrock-2023-05-31',
      max_tokens: 2000,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.3, // Lower temperature for more consistent merging
    };

    const command = new InvokeModelCommand({
      modelId: this.modelId,
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify(payload),
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    // Claude response format: { content: [{ text: "..." }], ... }
    return responseBody.content[0].text;
  }

  /**
   * Parse Claude's response and extract the merged document
   */
  private parseResponse(response: string, originalDocs: InputDocument[]): MergedDocument {
    try {
      // Try to extract JSON from the response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);

      return {
        description: parsed.description,
        owner: parsed.owner,
        mergedFrom: originalDocs.map(d => d.description),
      };
    } catch (error) {
      console.error('Failed to parse Claude response:', response);
      // Fallback: just concatenate descriptions
      return {
        description: originalDocs.map(d => d.description).join('; '),
        owner: originalDocs[0].owner,
        mergedFrom: originalDocs.map(d => d.description),
      };
    }
  }

  /**
   * Batch merge multiple clusters of documents
   * @param clusters Array of document clusters
   * @returns Array of merged documents
   */
  async mergeClusters(clusters: InputDocument[][]): Promise<MergedDocument[]> {
    const results: MergedDocument[] = [];

    for (const cluster of clusters) {
      console.log(`Merging cluster with ${cluster.length} documents...`);
      const merged = await this.mergeDocuments(cluster);
      results.push(merged);
    }

    return results;
  }
}

