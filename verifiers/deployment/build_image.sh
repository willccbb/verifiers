#!/bin/bash
# Build custom vLLM image with verifiers support

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default image name and tag
IMAGE_NAME="${IMAGE_NAME:-verifiers/vllm-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "Building custom vLLM image..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Build from project root to have access to verifiers code
cd "${PROJECT_ROOT}"

docker build \
    -f verifiers/deployment/Dockerfile.vllm \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

if [ $? -eq 0 ]; then
    echo "Successfully built ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To push to a registry:"
    echo "  docker push ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To use with Kubernetes deployment, update the 'image' field in DeploymentConfig"
else
    echo "Failed to build image"
    exit 1
fi