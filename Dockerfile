FROM rust:1.95-trixie AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock README.md LICENSE ./
COPY docs/openapi.json docs/openapi.json
COPY src/ src/
COPY examples/ examples/
RUN cargo build --locked --release --bin loci
RUN mkdir -p /dist/lib && cp target/release/loci /dist/loci && \
    find target/release/ -maxdepth 1 -name 'libonnxruntime*.so*' -exec cp -L '{}' /dist/lib/ ';'

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/* && install -d -o 65532 -g 65532 /data
COPY --from=builder /dist/ /opt/loci/
COPY --from=builder /build/LICENSE /opt/loci/LICENSE
ENV LD_LIBRARY_PATH=/opt/loci/lib
RUN ["/opt/loci/loci", "--version"]
USER 65532:65532
VOLUME ["/data"]
EXPOSE 7707
ENTRYPOINT ["/opt/loci/loci"]
CMD ["serve", "--bind", "0.0.0.0:7707", "--data", "/data", "--credentials", "/run/secrets/loci-credentials.json"]
