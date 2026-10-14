# ==============================================================================
# Multi-stage Dockerfile for Kairos103
# Universal container for external benchmarks, Kaggle, PhysioNet, and Hugging Face
# ==============================================================================

# Stage 1: Build & Publish
FROM mcr.microsoft.com/dotnet/sdk:10.0-preview AS builder
WORKDIR /src

# Copy solution and project files first for optimal layer caching
COPY Kairos103.slnx ./
COPY src/Kairos103.Core/Kairos103.Core.csproj src/Kairos103.Core/
COPY src/Kairos103.Benchmarks/Kairos103.Benchmarks.csproj src/Kairos103.Benchmarks/
COPY src/Kairos103.Cli/Kairos103.Cli.csproj src/Kairos103.Cli/
COPY tests/Kairos103.Tests/Kairos103.Tests.csproj tests/Kairos103.Tests/

RUN dotnet restore src/Kairos103.Cli/Kairos103.Cli.csproj

# Copy remaining source code
COPY src/ src/
COPY datasets/ datasets/

# Publish self-contained executable for Linux x64
RUN dotnet publish src/Kairos103.Cli/Kairos103.Cli.csproj \
    -c Release \
    -r linux-x64 \
    --self-contained true \
    -o /app/publish

# Stage 2: Minimal runtime environment for Linux x64
FROM mcr.microsoft.com/dotnet/runtime-deps:10.0-preview
WORKDIR /app

COPY --from=builder /app/publish .
COPY datasets/ /app/datasets/

ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
ENTRYPOINT ["./Kairos103.Cli"]
CMD ["--help"]
