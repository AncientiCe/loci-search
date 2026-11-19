use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use loci_search::{
    Embedding, Passage, Query,
    bench::{self, Run},
    elastic, fixture,
    index::{Engine, ingest, ingest_compact},
    profile,
    ranking::{RegionalRerank, SearchConfig, Strategy},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "SQLite-backed semantic region search and retrieval benchmarks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Tune bounded global lexical rescue on development queries only.
    RescueDev {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        dev: PathBuf,
        #[arg(long)]
        frozen_report: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Audit candidate and final ranks on development queries against frozen Elasticsearch.
    RecallDev {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        dev: PathBuf,
        #[arg(long)]
        frozen_report: PathBuf,
        #[arg(long)]
        elastic_url: String,
        #[arg(long)]
        elastic_index: String,
        #[arg(long)]
        out: PathBuf,
        /// Sweep regional reranker weights and write the selected weights here.
        #[arg(long)]
        tune_rerank: Option<PathBuf>,
    },
    /// Profile regional query phases with the settings from a prior comparison.
    Profile {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        frozen_report: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Cap the number of passages scored by global lexical rescue.
        #[arg(long)]
        rescue_max_scored: Option<usize>,
    },
    /// Generate deterministic passages, embeddings, and relevance judgments.
    Fixture {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        chunks: usize,
    },
    /// Atomically replace the index with a complete JSONL snapshot.
    Ingest {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        passages: PathBuf,
        #[arg(long)]
        embedding: PathBuf,
        /// Packed little-endian f32 vectors in passage JSONL order.
        #[arg(long)]
        vectors: Option<PathBuf>,
        /// Store the canonical snapshot without duplicate on-disk regional postings.
        #[arg(long)]
        compact: bool,
        #[arg(long, default_value_t = 1024)]
        leaf_size: usize,
    },
    /// Search with a query JSON file containing a precomputed embedding.
    Search {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        query: PathBuf,
        #[arg(long, value_enum, default_value = "regional")]
        strategy: Strategy,
        #[arg(long)]
        explain: bool,
        /// Use an explicit regional reranker weight file.
        #[arg(long)]
        regional_rerank: Option<PathBuf>,
        /// Cap the number of passages scored by global lexical rescue.
        #[arg(long)]
        rescue_max_scored: Option<usize>,
        #[command(flatten)]
        settings: Settings,
    },
    /// Compare global and regional retrieval on judged queries or a generated fixture.
    Bench {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, conflicts_with_all = ["db", "queries"], required_unless_present = "db")]
        fixture: bool,
        #[arg(long, requires = "queries")]
        db: Option<PathBuf>,
        #[arg(long, requires = "db")]
        queries: Option<PathBuf>,
        #[arg(long, default_value_t = 100_000)]
        chunks: usize,
        #[arg(long, default_value_t = 1024)]
        leaf_size: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 12)]
        exact_queries: usize,
        /// Separate development-query JSONL used to tune the comparison.
        #[arg(long, requires = "elastic_url", conflicts_with = "frozen_report")]
        dev: Option<PathBuf>,
        /// Carry over Loci settings from this report and tune only Elasticsearch on the new dev split.
        #[arg(long, requires = "dev")]
        transfer_report: Option<PathBuf>,
        /// Compare only Loci regional and Elasticsearch, without building the other controls.
        #[arg(long, requires = "transfer_report")]
        head_to_head: bool,
        /// Reuse all settings and verify the original index statistics without tuning.
        #[arg(long, requires = "elastic_url")]
        frozen_report: Option<PathBuf>,
        /// Override only the regional reranker in a frozen comparison.
        #[arg(long, requires = "frozen_report")]
        regional_rerank: Option<PathBuf>,
        /// Change only the regional lexical rescue scoring cap in a frozen comparison.
        #[arg(long, requires = "frozen_report")]
        rescue_max_scored: Option<usize>,
        /// Loopback URL for a prepared single-shard Elasticsearch index.
        #[arg(long, requires_all = ["db", "elastic_index"], conflicts_with = "fixture")]
        elastic_url: Option<String>,
        #[arg(long, requires = "elastic_url")]
        elastic_index: Option<String>,
        #[command(flatten)]
        settings: Settings,
    },
}

#[derive(Args)]
struct Settings {
    #[arg(long, default_value_t = 100)]
    limit: usize,
    #[arg(long, default_value_t = 200)]
    depth: usize,
    #[arg(long, default_value_t = 256)]
    graph_width: usize,
    #[arg(long, default_value_t = 12)]
    max_regions: usize,
    #[arg(long, default_value_t = 6_000)]
    max_chunks: usize,
    #[arg(long, default_value_t = 60_000)]
    max_postings: usize,
    #[arg(long, default_value_t = 40)]
    rescue: usize,
    /// Regional routing time budget in milliseconds. Zero uses work limits alone.
    #[arg(long, default_value_t = 0)]
    routing_ms: u64,
}

impl From<Settings> for SearchConfig {
    fn from(settings: Settings) -> Self {
        Self {
            limit: settings.limit,
            depth: settings.depth,
            graph_width: settings.graph_width,
            max_regions: settings.max_regions,
            max_chunks: settings.max_chunks,
            max_postings: settings.max_postings,
            rescue: settings.rescue,
            rescue_max_scored: None,
            routing_ms: settings.routing_ms,
            regional_rerank: None,
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::RescueDev {
            db,
            dev,
            frozen_report,
            out,
        } => {
            let mut engine = Engine::open(&db)?;
            let development = jsonl::<Query>(&dev)?;
            let frozen = json::<bench::FrozenComparison>(&frozen_report)?;
            let tuning = bench::tune_bounded_rescue(&mut engine, &development, &frozen)?;
            write_json(&out, &tuning)?;
        }
        Command::RecallDev {
            db,
            dev,
            frozen_report,
            elastic_url,
            elastic_index,
            out,
            tune_rerank,
        } => {
            let mut engine = Engine::open(&db)?;
            let queries = jsonl::<Query>(&dev)?;
            let frozen = json::<bench::FrozenComparison>(&frozen_report)?;
            let client = elastic::Client::new(&elastic_url, &elastic_index)?;
            if let Some(weights) = tune_rerank {
                let tuning = bench::tune_regional_rerank(&mut engine, &queries, &frozen, &client)?;
                write_json(&out, &tuning)?;
                write_json(&weights, &tuning.selected)?;
            } else {
                let diagnosis = bench::diagnose_recall(&mut engine, &queries, &frozen, &client)?;
                write_json(&out, &diagnosis)?;
            }
        }
        Command::Profile {
            db,
            queries,
            frozen_report,
            out,
            rescue_max_scored,
        } => {
            let mut engine = Engine::open(&db)?;
            let mut frozen = json::<bench::FrozenComparison>(&frozen_report)?;
            if let Some(cap) = rescue_max_scored {
                frozen = frozen.with_bounded_rescue(cap)?;
            }
            let config = frozen.regional_config(&engine)?;
            let queries = jsonl::<Query>(&queries)?;
            let report = profile::regional(&mut engine, &queries, config)?;
            write_json(&out, &report)?;
        }
        Command::Fixture { out, chunks } => {
            let (embedding, passages, queries) = fixture::generate(chunks)?;
            output_dir(&out)?;
            write_json(&out.join("embedding.json"), &embedding)?;
            write_jsonl(&out.join("passages.jsonl"), &passages)?;
            write_jsonl(&out.join("queries.jsonl"), &queries)?;
            write_json(&out.join("query.json"), &queries[0])?;
        }
        Command::Ingest {
            db,
            passages,
            embedding,
            vectors,
            compact,
            leaf_size,
        } => {
            let mut passages = jsonl::<Passage>(&passages)?;
            let embedding = json::<Embedding>(&embedding)?;
            if let Some(path) = vectors {
                read_packed_vectors(&path, &mut passages, embedding.dimensions)?;
            }
            let manifest = if compact {
                ingest_compact(&db, passages, embedding, leaf_size)?
            } else {
                ingest(&db, passages, embedding, leaf_size)?
            };
            print_json(&manifest)?;
        }
        Command::Search {
            db,
            query,
            strategy,
            explain,
            regional_rerank,
            rescue_max_scored,
            settings,
        } => {
            let query = json::<Query>(&query)?;
            let mut config: SearchConfig = settings.into();
            if let Some(path) = regional_rerank {
                ensure!(
                    strategy == Strategy::Regional,
                    "--regional-rerank requires regional search"
                );
                config.regional_rerank = Some(json(&path)?);
            }
            if rescue_max_scored.is_some() {
                ensure!(
                    strategy == Strategy::Regional,
                    "--rescue-max-scored requires regional search"
                );
                config.rescue_max_scored = rescue_max_scored;
            }
            let result = Engine::open(&db)?.search(&query, strategy, &config)?;
            if explain {
                print_json(&result)?;
            } else {
                let sources = result
                    .hits
                    .iter()
                    .map(|hit| &hit.source)
                    .collect::<Vec<_>>();
                print_json(&sources)?;
            }
        }
        Command::Bench {
            out,
            fixture,
            db,
            queries,
            chunks,
            leaf_size,
            repeats,
            exact_queries,
            dev,
            transfer_report,
            head_to_head,
            frozen_report,
            regional_rerank,
            rescue_max_scored,
            elastic_url,
            elastic_index,
            settings,
        } => {
            let config = settings.into();
            ensure!(repeats > 0, "repeats must be positive");
            output_dir(&out)?;
            let (db, queries, ingest_ms) = if fixture {
                let (embedding, passages, queries) = fixture::generate(chunks)?;
                let db = out.join("fixture.sqlite");
                let started = Instant::now();
                ingest(&db, passages, embedding, leaf_size)?;
                (db, queries, Some(started.elapsed().as_secs_f64() * 1000.0))
            } else {
                (
                    db.context("provide --db")?,
                    jsonl::<Query>(&queries.context("provide --queries")?)?,
                    None,
                )
            };
            let started = Instant::now();
            let mut engine = Engine::open(&db)?;
            let run = Run {
                fixture,
                repeats,
                exact_queries,
                index_bytes: index_bytes(&db)?,
                ingest_ms,
                open_ms: started.elapsed().as_secs_f64() * 1000.0,
            };
            eprintln!(
                "Loaded {} chunks and {} evaluation queries",
                engine.manifest.chunks,
                queries.len()
            );
            let trace_path = out.join("traces.partial");
            let traces = BufWriter::new(File::create(&trace_path)?);
            let report = if let Some(url) = elastic_url {
                let client =
                    elastic::Client::new(&url, &elastic_index.context("provide --elastic-index")?)?;
                if let Some(path) = frozen_report {
                    let mut frozen = json::<bench::FrozenComparison>(&path)?;
                    if let Some(weights) = regional_rerank {
                        frozen = frozen.with_regional_rerank(json::<RegionalRerank>(&weights)?)?;
                    }
                    if let Some(cap) = rescue_max_scored {
                        frozen = frozen.with_bounded_rescue(cap)?;
                    }
                    bench::evaluate_frozen(&mut engine, &queries, &run, &frozen, &client, traces)?
                } else {
                    let dev = jsonl::<Query>(&dev.context("provide --dev or --frozen-report")?)?;
                    let development = bench::Development {
                        queries: &dev,
                        elasticsearch: &client,
                    };
                    if let Some(path) = transfer_report {
                        let source = json::<bench::FrozenComparison>(&path)?;
                        bench::evaluate_transfer(
                            &mut engine,
                            &queries,
                            &run,
                            &source,
                            development,
                            head_to_head,
                            traces,
                        )?
                    } else {
                        bench::evaluate_comparison(
                            &mut engine,
                            &queries,
                            &config,
                            &run,
                            development,
                            traces,
                        )?
                    }
                }
            } else {
                bench::evaluate(&mut engine, &queries, &config, &run, traces)?
            };
            fs::rename(trace_path, out.join("traces.jsonl"))?;
            write_json(&out.join("report.json"), &report)?;
            println!(
                "Benchmark report written to {}",
                out.join("report.json").display()
            );
        }
    }
    Ok(())
}

fn index_bytes(path: &Path) -> Result<u64> {
    let mut wal = path.as_os_str().to_owned();
    wal.push("-wal");
    let wal_bytes = match fs::metadata(wal) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    Ok(fs::metadata(path)?.len() + wal_bytes)
}

fn read_packed_vectors(path: &Path, passages: &mut [Passage], dimensions: usize) -> Result<()> {
    let record_bytes = dimensions
        .checked_mul(size_of::<f32>())
        .context("vector size overflows")?;
    let bytes = record_bytes
        .checked_mul(passages.len())
        .context("packed vector file size overflows")?;
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        record_bytes > 0 && file.metadata()?.len() == bytes as u64,
        "packed vector file does not match passage count and embedding dimensions"
    );
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut buffer = vec![0_u8; record_bytes];
    for passage in passages {
        ensure!(
            passage.vector.is_empty(),
            "passage {} also includes a JSON vector",
            passage.id
        );
        reader.read_exact(&mut buffer)?;
        let (components, _) = buffer.as_chunks::<4>();
        passage.vector = components.iter().copied().map(f32::from_le_bytes).collect();
    }
    ensure!(
        reader.fill_buf()?.is_empty(),
        "packed vector file grew during ingest"
    );
    Ok(())
}

fn output_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path).with_context(|| format!("create new output directory {}", path.display()))
}

fn json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

fn jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut records = Vec::new();
    for (line, contents) in BufReader::new(file).lines().enumerate() {
        let contents = contents?;
        if !contents.trim().is_empty() {
            records.push(
                serde_json::from_str(&contents)
                    .with_context(|| format!("{}:{}", path.display(), line + 1))?,
            );
        }
    }
    Ok(records)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_jsonl(path: &Path, values: &[impl Serialize]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn print_json(value: &impl Serialize) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value)?;
    stdout.write_all(b"\n")?;
    Ok(())
}
