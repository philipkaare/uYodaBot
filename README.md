uYodaBot is the simplest transformer architecture imaginable.

It turns sentences written in an extremely limited vocabulary into something
Master Yoda might have said, for example:

    i am strong  -->  strong I am

## Run in the browser

A Blazor WebAssembly build runs entirely client-side and is hosted on GitHub Pages:

**https://philip-loventoft.github.io/uYodaBot/**  ← (set to your Pages URL)

It has full parity with the console app: **Chat**, **Verbose** (a visual trace of
every transformer stage), and **Train** (retrain the model in your browser; the
result is saved to local storage).

Run the web app locally:

    ./run.sh web
    # or: dotnet run --project src/YodaTransformer.Web

## Run the console app

    ./run.sh
    # or: dotnet run --project src/YodaTransformer.Console

Requires a recent .NET SDK (the projects target .NET 8).

## Project layout

    src/YodaTransformer.Core      shared transformer + training code
    src/YodaTransformer.Console   terminal app
    src/YodaTransformer.Web       Blazor WebAssembly app
    tests/YodaTransformer.Tests   unit tests
