#!/bin/bash

# Extract version from pyproject.toml
VERSION=$(grep -m1 'version = ' pyproject.toml | cut -d '"' -f 2)

if [ -z "$VERSION" ]; then
  echo "Error: Could not extract version from pyproject.toml"
  exit 1
fi

TAG_NAME="v$VERSION"

echo "Detected version: $VERSION"
echo "Creating git tag: $TAG_NAME"

# Create tag
git tag "$TAG_NAME"

# Push tag
echo "Pushing tag to remote..."
git push origin "$TAG_NAME"

echo "Done."
