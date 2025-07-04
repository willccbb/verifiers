#!/bin/bash

# Run tests with warnings suppressed for cleaner output
# This suppresses external dependency warnings while preserving our own code warnings

export PYTHONWARNINGS="ignore::DeprecationWarning"
exec uv run pytest "$@"