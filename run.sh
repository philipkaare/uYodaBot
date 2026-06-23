#!/usr/bin/env bash
set -e
# Run the console app, or pass "web" to run the Blazor dev server.
if [ "$1" = "web" ]; then
  dotnet run --project src/YodaTransformer.Web
else
  dotnet run --project src/YodaTransformer.Console
fi
