use super::valid_index;
use anyhow::{Context, Result, ensure};
use loci_search::store::{Document, Mutation};
use rusqlite::{Connection, OpenFlags};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

fn validate(db: &Connection) -> Result<()> {
    let integrity: String = db.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "catalog failed SQLite integrity checks");
    db.query_row(
        "SELECT generation, settings FROM loci_indexes LIMIT 0",
        [],
        |_| Ok(()),
    )
    .or_else(|e| {
        if matches!(e, rusqlite::Error::QueryReturnedNoRows) {
            Ok(())
        } else {
            Err(e)
        }
    })?;
    Ok(())
}

pub fn backup(data: &Path, out: &Path) -> Result<()> {
    let db = Connection::open_with_flags(
        data.join("catalog.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    validate(&db)?;
    let file = tempfile::NamedTempFile::new_in(
        out.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    db.execute(
        "VACUUM INTO ?1",
        [file.path().to_str().context("invalid backup path")?],
    )?;
    file.as_file().sync_all()?;
    file.persist_noclobber(out)
        .context("publish backup without replacing an existing file")?;
    Ok(())
}

pub fn restore(input: &Path, data: &Path) -> Result<()> {
    let db = Connection::open_with_flags(input, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    validate(&db)?;
    fs::create_dir_all(data)?;
    ensure!(
        fs::read_dir(data)?.next().is_none(),
        "restore requires an empty destination directory"
    );
    let file = tempfile::NamedTempFile::new_in(data)?;
    db.execute(
        "VACUUM INTO ?1",
        [file.path().to_str().context("invalid restore path")?],
    )?;
    file.as_file().sync_all()?;
    file.persist_noclobber(data.join("catalog.sqlite"))?;
    Ok(())
}

fn mutation(mut obj: Map<String, Value>) -> Result<Mutation> {
    if obj.contains_key("op") {
        return Ok(serde_json::from_value(Value::Object(obj))?);
    }
    let revision = format!("{:x}", Sha256::digest(serde_json::to_vec(&obj)?));
    let id = obj
        .remove("id")
        .and_then(|x| x.as_str().map(str::to_owned))
        .context("record needs a string id")?;
    let revision = obj
        .remove("revision")
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or(revision);
    let readers = obj
        .remove("readers")
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    let mut fields: BTreeMap<String, Value> = obj
        .remove("fields")
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    let text = obj
        .remove("text")
        .map(serde_json::from_value::<String>)
        .transpose()?;
    for (k, v) in obj {
        ensure!(fields.insert(k.clone(), v).is_none(), "duplicate field {k}");
    }
    let text = text.unwrap_or_else(|| {
        fields
            .values()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" ")
    });
    Ok(Mutation::Upsert {
        document: Document {
            id,
            revision,
            text,
            fields,
            readers,
            embed: true,
        },
    })
}

pub fn import(url: &str, index: &str, files: &[PathBuf]) -> Result<()> {
    ensure!(
        valid_index(index) && !files.is_empty(),
        "import requires a valid index and at least one file"
    );
    let endpoint = format!("{}/v1/indexes/{index}/bulk", url.trim_end_matches('/'));
    let token = env::var("LOCI_TOKEN").ok();
    let mut pending = Vec::<Value>::new();
    let mut bytes = 0usize;
    let mut sent = 0usize;
    let mut flush = |batch: &mut Vec<Value>| -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut request = ureq::post(&endpoint);
        if let Some(t) = &token {
            request = request.header("Authorization", &format!("Bearer {t}"));
        }
        request.send_json(json!({"operations":batch}))?;
        sent += batch.len();
        batch.clear();
        Ok(())
    };
    for path in files {
        let mut push = |op: Mutation| -> Result<()> {
            let op = serde_json::to_value(op)?;
            let size = serde_json::to_vec(&op)?.len() + 1;
            ensure!(size < 7 * 1024 * 1024, "record exceeds import limit");
            if pending.len() >= 500 || bytes + size > 7 * 1024 * 1024 {
                flush(&mut pending)?;
                bytes = 0;
            }
            bytes += size;
            pending.push(op);
            Ok(())
        };
        match path
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("jsonl" | "ndjson") => {
                for (n, line) in BufReader::new(fs::File::open(path)?).lines().enumerate() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    push(mutation(serde_json::from_str(&line).with_context(
                        || format!("{} line {}", path.display(), n + 1),
                    )?)?)?;
                }
            }
            Some("txt" | "md" | "markdown") => {
                let text = fs::read_to_string(path)?;
                let revision = format!("{:x}", Sha256::digest(text.as_bytes()));
                let id = path
                    .to_str()
                    .context("invalid source path")?
                    .replace('\\', "/");
                push(Mutation::Upsert {
                    document: Document {
                        id,
                        revision,
                        text,
                        fields: BTreeMap::new(),
                        readers: vec![],
                        embed: true,
                    },
                })?;
            }
            _ => anyhow::bail!("unsupported input {}", path.display()),
        }
    }
    flush(&mut pending)?;
    writeln!(std::io::stdout(), "Imported {sent} operations into {index}")?;
    Ok(())
}
