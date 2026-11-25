//! Paragraph windows from palace-rs `miner::chunk_text` and exchange chunks
//! from `convo_miner::chunk_exchanges`. Hotpot passages keep their own spans.

const TEXT_CHUNK_SIZE: usize = 800;
const TEXT_CHUNK_OVERLAP: usize = 100;
const TEXT_MIN_CHUNK: usize = 50;
const EXCHANGE_MIN_CHUNK: usize = 30;
const MAX_EXCHANGE_CHARS: usize = 2000;
const MIN_REPLY_CHARS: usize = 500;

pub fn chunk_text(content: &str) -> Vec<(String, usize)> {
    let content = content.trim();
    if content.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let total = content.len();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < total {
        let end = floor_char_boundary(content, (start + TEXT_CHUNK_SIZE).min(total));
        let mut cut = end;

        if cut < total {
            if let Some(pos) = content[start..cut].rfind("\n\n") {
                let abs = start + pos;
                if abs > start + TEXT_CHUNK_SIZE / 2 {
                    cut = abs;
                }
            } else if let Some(pos) = content[start..cut].rfind('\n') {
                let abs = start + pos;
                if abs > start + TEXT_CHUNK_SIZE / 2 {
                    cut = abs;
                }
            }
        }

        let chunk = content[start..cut].trim().to_string();
        if chunk.len() >= TEXT_MIN_CHUNK {
            chunks.push((chunk, chunk_index));
            chunk_index += 1;
        }

        if cut >= total {
            break;
        }
        let next = floor_char_boundary(content, cut.saturating_sub(TEXT_CHUNK_OVERLAP));
        start = if next > start { next } else { cut };
    }

    chunks
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub fn chunk_exchanges(content: &str) -> Vec<(String, usize)> {
    let lines: Vec<&str> = content.lines().collect();
    let quote_count = lines
        .iter()
        .filter(|line| line.trim().starts_with('>'))
        .count();
    if quote_count >= 3 {
        chunk_by_exchange(&lines)
    } else {
        chunk_by_paragraph(content)
    }
}

fn chunk_by_exchange(lines: &[&str]) -> Vec<(String, usize)> {
    let mut chunks = Vec::new();
    let is_text = |line: &&str| !line.is_empty() && !is_horizontal_rule(line);
    let first_turn = lines
        .iter()
        .position(|line| line.trim().starts_with('>'))
        .unwrap_or(lines.len());
    let preamble: Vec<&str> = lines[..first_turn]
        .iter()
        .map(|line| line.trim())
        .filter(is_text)
        .collect();
    for part in split_reply(&preamble, MAX_EXCHANGE_CHARS) {
        if part.len() > EXCHANGE_MIN_CHUNK {
            chunks.push((part, chunks.len()));
        }
    }

    let mut index = first_turn;
    while index < lines.len() {
        let user_turn = lines[index].trim().to_string();
        index += 1;
        let mut reply = Vec::new();
        while index < lines.len() {
            let next = lines[index].trim();
            if next.starts_with('>') {
                break;
            }
            if is_text(&next) {
                reply.push(next);
            }
            index += 1;
        }
        let budget = MAX_EXCHANGE_CHARS
            .saturating_sub(user_turn.len() + 1)
            .max(MIN_REPLY_CHARS);
        let parts = split_reply(&reply, budget);
        if parts.is_empty() {
            if user_turn.len() > EXCHANGE_MIN_CHUNK {
                chunks.push((user_turn, chunks.len()));
            }
        } else {
            for part in parts {
                let content = format!("{user_turn}\n{part}");
                if content.trim().len() > EXCHANGE_MIN_CHUNK {
                    chunks.push((content, chunks.len()));
                }
            }
        }
    }
    chunks
}

fn is_horizontal_rule(line: &str) -> bool {
    line.len() >= 3 && line.chars().all(|character| character == '-')
}

fn split_reply(lines: &[&str], budget: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for word in lines.iter().flat_map(|line| line.split_whitespace()) {
        if !current.is_empty() && current.len() + 1 + word.len() > budget {
            parts.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn chunk_by_paragraph(content: &str) -> Vec<(String, usize)> {
    let lines: Vec<&str> = content.lines().collect();
    let paragraphs: Vec<&str> = content
        .split("\n\n")
        .map(str::trim)
        .filter(|paragraph| !paragraph.is_empty())
        .collect();
    if paragraphs.len() <= 1 && lines.len() > 20 {
        return lines
            .chunks(25)
            .enumerate()
            .filter_map(|(index, chunk)| {
                let group = chunk.join("\n").trim().to_string();
                (group.len() > EXCHANGE_MIN_CHUNK).then_some((group, index))
            })
            .collect();
    }
    paragraphs
        .into_iter()
        .enumerate()
        .filter(|(_, paragraph)| paragraph.len() > EXCHANGE_MIN_CHUNK)
        .map(|(index, paragraph)| (paragraph.to_string(), index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraph_windows_overlap_and_prefer_a_blank_line() {
        let text = format!("{}\n\n{}", "a".repeat(700), "b".repeat(700));
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].0.contains('a'));
        assert!(chunks.iter().any(|(chunk, _)| chunk.contains('b')));
        assert!(chunks.iter().all(|(chunk, _)| chunk.len() >= 50));
    }

    #[test]
    fn exchanges_pair_each_user_turn_with_its_reply() {
        let reply = "assistant reply long enough to pass the minimum size check";
        let text = format!(
            "> one user turn long enough\n{reply}\n> two user turn long enough\n{reply}\n> three user turn long enough\n{reply}\n"
        );
        let chunks = chunk_exchanges(&text);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|(chunk, _)| chunk.starts_with('>')));
    }
}
