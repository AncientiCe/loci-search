const PREFERENCE_PATTERNS: &[&str] = &[
    "i prefer ",
    "i like ",
    "i always ",
    "i never ",
    "i want ",
    "i dislike ",
    "i avoid ",
    "i don't ",
    "i do not ",
    "i hate ",
    "i love ",
    "i tend to ",
    "i usually ",
    "i typically ",
    "i find it ",
    "i'm comfortable",
    "i'm not comfortable",
    "i am comfortable",
    "i am not comfortable",
    "in my experience",
    "my convention",
    "my favorite",
    "my go-to",
    "my preference",
    "my preferred",
    "my style",
    "my approach is",
    "always use ",
    "never use ",
    "prefer to use",
    "prefer to write",
    "prefer to keep",
    "prefer to avoid",
    "don't like ",
    "do not like ",
    "would rather ",
    "feel strongly",
    "strongly prefer",
    "it's important to me",
    "it is important to me",
];

const CONVERSATION_EXTENSIONS: &[&str] = &["txt", "md", "json", "jsonl"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreferenceSource {
    Code,
    Conversation,
    Filed,
}

impl PreferenceSource {
    pub(crate) fn infer(source_file: &str) -> Self {
        let path = std::path::Path::new(source_file);
        let absolute = path.is_absolute() || source_file.starts_with('/');
        if !absolute {
            return Self::Filed;
        }
        let conversation = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                CONVERSATION_EXTENSIONS
                    .iter()
                    .any(|candidate| ext.eq_ignore_ascii_case(candidate))
            });
        if conversation {
            Self::Conversation
        } else {
            Self::Code
        }
    }
}

pub(crate) fn is_preference(text: &str) -> bool {
    let lower = text.to_lowercase();
    PREFERENCE_PATTERNS
        .iter()
        .any(|pattern| lower.contains(pattern))
}

pub(crate) fn preference_span(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    for (original, lowered) in text
        .split_inclusive(['.', '!', '?', '\n'])
        .zip(lower.split_inclusive(['.', '!', '?', '\n']))
    {
        let trimmed = original.trim();
        if !trimmed.is_empty() && has_preference(lowered.trim()) {
            return Some(trimmed.to_string());
        }
    }
    let trimmed = text.trim();
    if !trimmed.is_empty() && has_preference(lower.trim()) {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn has_preference(lowercased: &str) -> bool {
    PREFERENCE_PATTERNS
        .iter()
        .any(|pattern| lowercased.contains(pattern))
}
