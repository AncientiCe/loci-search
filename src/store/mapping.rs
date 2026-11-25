use super::{Document, StreamPolicy, stream};
use anyhow::{Result, bail, ensure};
use serde_json::Value;

fn kind(declaration: &Value) -> Result<(&str, bool)> {
    let (raw, array) = if let Some(name) = declaration.as_str() {
        (name, false)
    } else {
        let object = declaration
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid field mapping"))?;
        ensure!(
            object.keys().all(|k| k == "type" || k == "array"),
            "unknown field mapping option"
        );
        let array = match object.get("array") {
            Some(v) => v
                .as_bool()
                .ok_or_else(|| anyhow::anyhow!("array flag must be boolean"))?,
            None => false,
        };
        (
            object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("mapping needs type"))?,
            array,
        )
    };
    let (name, array) = raw
        .strip_suffix("[]")
        .map_or((raw, array), |base| (base, true));
    ensure!(
        matches!(
            name,
            "keyword" | "text" | "number" | "integer" | "boolean" | "date"
        ),
        "unknown field mapping {name}"
    );
    Ok((name, array))
}

pub(super) fn validate_settings(settings: &Value) -> Result<()> {
    if let Some(visibility) = settings.get("default_visibility") {
        ensure!(
            matches!(visibility.as_str(), Some("private" | "public")),
            "invalid default_visibility"
        );
    }
    if let Some(mappings) = settings.get("mappings") {
        let fields = mappings
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("mappings must be an object"))?;
        for (field, declaration) in fields {
            ensure!(!field.is_empty(), "empty mapped field");
            kind(declaration)?;
        }
    }
    if let Some(profile) = settings.get("retrieval_profile") {
        ensure!(
            profile.as_str() == Some("palace"),
            "retrieval_profile must be palace"
        );
    }
    StreamPolicy::from_settings(settings)?;
    Ok(())
}

pub(super) fn validate_document(settings: &Value, doc: &Document) -> Result<()> {
    let Some(fields) = settings.get("mappings").and_then(Value::as_object) else {
        return Ok(());
    };
    for (field, declaration) in fields {
        let Some(value) = doc.fields.get(field) else {
            continue;
        };
        let (name, array) = kind(declaration)?;
        let values = if array {
            value
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("field {field} must be an array"))?
                .as_slice()
        } else {
            std::slice::from_ref(value)
        };
        for v in values {
            let valid = match name {
                "keyword" | "text" => v.is_string(),
                "number" => v.is_number(),
                "integer" => v.is_i64() || v.is_u64(),
                "boolean" => v.is_boolean(),
                "date" => stream::timestamp(v).is_ok(),
                _ => false,
            };
            if !valid {
                bail!("field {field} does not match declared type {name}");
            }
        }
    }
    Ok(())
}
