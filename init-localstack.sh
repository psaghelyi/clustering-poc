#!/bin/bash

# Script to initialize localstack S3 bucket

echo "Waiting for LocalStack to be ready..."
sleep 5

echo "Creating S3 bucket: clustering-poc-embeddings"
aws --endpoint-url=http://localhost:4566 s3 mb s3://clustering-poc-embeddings --region us-east-1

echo "Verifying bucket creation..."
aws --endpoint-url=http://localhost:4566 s3 ls

echo "LocalStack S3 setup complete!"
