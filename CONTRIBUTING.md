# Contributing

Install Rust 1.95 or newer and a C compiler for the bundled SQLite build. On Windows with the MSVC Rust toolchain, install the Visual Studio C++ Build Tools.

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

When changing retrieval, report quality and latency on the same corpus and query split before and after your change. Include the commit, settings, and platform. Use the [benchmark guide](benchmarks/reproduce.md) and keep generated corpora, indexes, and traces under `runs/` locally.

For security reports, follow the [security policy](SECURITY.md).
