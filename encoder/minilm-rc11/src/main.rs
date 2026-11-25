//! One-text MiniLM inference on ort 2.0.0-rc.11.
//!
//! Protocol: little-endian u32 byte length, then UTF-8. A zero length exits.
//! Each response is 384 little-endian f32 values, L2-normalized.

use anyhow::{Context, Result, anyhow};
use ndarray::Array;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

const MAX_LEN: usize = 512;
const DIMENSIONS: usize = 384;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("model path")?;
    let vocab = args.next().context("vocab path")?;
    let tokenizer = BertTokenizer::from_vocab_file(Path::new(&vocab))?;
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(&model)
        .context("loading ONNX session")?;
    let need_types = session
        .inputs()
        .iter()
        .any(|input| input.name() == "token_type_ids");
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    loop {
        let mut header = [0u8; 4];
        if input.read_exact(&mut header).is_err() {
            break;
        }
        let length = u32::from_le_bytes(header);
        if length == 0 {
            break;
        }
        let mut bytes = vec![0u8; length as usize];
        input.read_exact(&mut bytes)?;
        let text = String::from_utf8(bytes).context("request is not utf-8")?;
        let vector = embed_one(&mut session, &tokenizer, need_types, &text)?;
        anyhow::ensure!(vector.len() == DIMENSIONS, "unexpected width");
        for value in vector {
            output.write_all(&value.to_le_bytes())?;
        }
        output.flush()?;
    }
    Ok(())
}

fn embed_one(
    session: &mut Session,
    tokenizer: &BertTokenizer,
    need_token_type_ids: bool,
    text: &str,
) -> Result<Vec<f32>> {
    let encoded = tokenizer.encode(text, MAX_LEN);
    let length = encoded.input_ids.len();
    let ids = Array::from_shape_vec((1, length), encoded.input_ids).context("input ids")?;
    let mask_owned = encoded.attention_mask.clone();
    let mask = Array::from_shape_vec((1, length), encoded.attention_mask).context("mask")?;
    let mut inputs = ort::inputs![
        "input_ids" => Value::from_array(ids)?,
        "attention_mask" => Value::from_array(mask)?,
    ];
    if need_token_type_ids {
        let types = Array::from_shape_vec((1, length), encoded.token_type_ids)?;
        inputs.push((
            "token_type_ids".into(),
            Value::from_array(types)?.into(),
        ));
    }
    let outputs = session.run(inputs).context("onnx inference")?;
    let output = outputs
        .values()
        .next()
        .context("model returned no outputs")?;
    let values = output.try_extract_array::<f32>()?;
    let shape = values.shape();
    let pooled = match shape.len() {
        2 => values.iter().copied().collect::<Vec<_>>(),
        3 => mean_pool(values.view(), &mask_owned, length),
        _ => return Err(anyhow!("unexpected output shape {shape:?}")),
    };
    Ok(l2_normalize(&pooled))
}

fn mean_pool(output: ndarray::ArrayViewD<'_, f32>, mask: &[i64], length: usize) -> Vec<f32> {
    let dim = *output.shape().last().unwrap_or(&DIMENSIONS);
    let mut result = vec![0.0f32; dim];
    let mut mask_sum = 0.0f32;
    for token in 0..length {
        let weight = mask[token] as f32;
        if weight == 0.0 {
            continue;
        }
        mask_sum += weight;
        for (lane, value) in result.iter_mut().enumerate() {
            *value += output[[0, token, lane]] * weight;
        }
    }
    if mask_sum > 0.0 {
        for value in &mut result {
            *value /= mask_sum;
        }
    }
    result
}

fn l2_normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    let scale = norm + 1e-12;
    vector.iter().map(|value| value / scale).collect()
}

struct BertTokenizer {
    token_to_id: HashMap<String, i64>,
    cls_id: i64,
    sep_id: i64,
    unk_id: i64,
}

struct EncodedInput {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

impl BertTokenizer {
    fn from_vocab_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let token_to_id: HashMap<String, i64> = content
            .lines()
            .enumerate()
            .map(|(index, token)| (token.to_string(), index as i64))
            .collect();
        let id = |token: &str| {
            token_to_id
                .get(token)
                .copied()
                .ok_or_else(|| anyhow!("vocab missing {token}"))
        };
        Ok(Self {
            cls_id: id("[CLS]")?,
            sep_id: id("[SEP]")?,
            unk_id: id("[UNK]")?,
            token_to_id,
        })
    }

    fn encode(&self, text: &str, max_len: usize) -> EncodedInput {
        let cleaned = clean_and_lower(text);
        let mut wordpieces = Vec::new();
        for token in basic_tokenize(&cleaned) {
            wordpiece_tokenize(&token, &self.token_to_id, self.unk_id, &mut wordpieces);
        }
        let effective_max = max_len.saturating_sub(2);
        if wordpieces.len() > effective_max {
            wordpieces.truncate(effective_max);
        }
        let mut input_ids = Vec::with_capacity(wordpieces.len() + 2);
        input_ids.push(self.cls_id);
        input_ids.extend_from_slice(&wordpieces);
        input_ids.push(self.sep_id);
        let seq_len = input_ids.len();
        EncodedInput {
            input_ids,
            attention_mask: vec![1; seq_len],
            token_type_ids: vec![0; seq_len],
        }
    }
}

fn clean_and_lower(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.nfd() {
        if character.is_control() || character == '\0' || is_combining_mark(character) {
            continue;
        }
        if is_whitespace_char(character) {
            out.push(' ');
        } else {
            for lower in character.to_lowercase() {
                out.push(lower);
            }
        }
    }
    out
}

fn is_combining_mark(character: char) -> bool {
    unicode_normalization::char::canonical_combining_class(character) != 0
}

fn is_whitespace_char(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\n' | '\r')
        || matches!(character as u32, 0x00A0 | 0x1680 | 0x202F | 0x205F | 0x3000)
        || (0x2000..=0x200A).contains(&(character as u32))
}

fn is_punctuation(character: char) -> bool {
    let code = character as u32;
    (0x21..=0x2F).contains(&code)
        || (0x3A..=0x40).contains(&code)
        || (0x5B..=0x60).contains(&code)
        || (0x7B..=0x7E).contains(&code)
        || character.is_ascii_punctuation()
        || matches!(
            code,
            0x2010..=0x2027
                | 0x2030..=0x205E
                | 0x2308..=0x230B
                | 0x2768..=0x2775
                | 0x27E6..=0x27EF
                | 0x2983..=0x2998
                | 0x3001..=0x3003
                | 0x3008..=0x3011
                | 0x3014..=0x301F
                | 0xFE30..=0xFE4F
                | 0xFF01..=0xFF0F
                | 0xFF1A..=0xFF20
                | 0xFF3B..=0xFF3D
                | 0xFF5B..=0xFF65
        )
}

fn is_cjk_character(character: char) -> bool {
    let code = character as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
        || (0x2F800..=0x2FA1F).contains(&code)
}

fn basic_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        let mut current = String::new();
        for character in word.chars() {
            if is_punctuation(character) || is_cjk_character(character) {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(character.to_string());
            } else {
                current.push(character);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
    }
    tokens
}

fn wordpiece_tokenize(token: &str, vocab: &HashMap<String, i64>, unk_id: i64, output: &mut Vec<i64>) {
    if token.is_empty() {
        return;
    }
    if let Some(&id) = vocab.get(token) {
        output.push(id);
        return;
    }
    let chars: Vec<char> = token.chars().collect();
    let mut start = 0;
    let mut is_bad = false;
    while start < chars.len() {
        let mut end = chars.len();
        let mut found = false;
        while start < end {
            let substr: String = if start > 0 {
                let mut piece = String::from("##");
                piece.extend(&chars[start..end]);
                piece
            } else {
                chars[start..end].iter().collect()
            };
            if let Some(&id) = vocab.get(&substr) {
                output.push(id);
                found = true;
                start = end;
                break;
            }
            end -= 1;
        }
        if !found {
            is_bad = true;
            break;
        }
    }
    if is_bad {
        output.truncate(output.len().saturating_sub(chars.len()));
        output.push(unk_id);
    }
}
