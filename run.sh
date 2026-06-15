#!/usr/bin/env bash
# Builds and runs uYodaBot.
set -euo pipefail

cd "$(dirname "$0")"

dotnet build YodaTransformer.csproj -c Release
dotnet run --project YodaTransformer.csproj -c Release --no-build "$@"
