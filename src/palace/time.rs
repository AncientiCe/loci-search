//! Relative time in queries, copied from palace-rs `time_query`.

use jiff::civil::{Date, Weekday};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Span, Timestamp};
use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimeTarget {
    pub start: Timestamp,
    pub end: Timestamp,
    pub spread_days: f64,
}

const NUMBER: &str = r"\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:a\s+)?couple(?:\s+of)?|(?:a\s+)?few";
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

static AGO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b({NUMBER})\s+(day|week|month|year)s?\s+(?:ago|back|before)\b"
    ))
    .unwrap()
});
static PAST_SPAN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(?:past|last|previous)\s+(?:({NUMBER})\s+)?(day|week|month|year)s?\b"
    ))
    .unwrap()
});
static THIS_SPAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bthis\s+(week|month|year)\b").unwrap());
static YESTERDAY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(the\s+day\s+before\s+)?yesterday\b").unwrap());
static LAST_WEEKDAY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:last|this\s+past)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    )
    .unwrap()
});
static MONTH_DAY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b({})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        MONTHS.join("|")
    ))
    .unwrap()
});
static IN_MONTH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(in|during|early|mid|late|last)[\s-]+({})\b",
        MONTHS.join("|")
    ))
    .unwrap()
});

pub fn parse_timestamp(text: &str) -> Option<Timestamp> {
    text.parse().ok()
}

/// LongMemEval's "2023/05/20 (Sat) 02:21", RFC 3339, or a bare "2023-05-20".
pub fn parse_dataset_timestamp(text: &str) -> Option<Timestamp> {
    let text = text.trim();
    if let Some(parsed) = parse_timestamp(text) {
        return Some(parsed);
    }
    if let Some(stamp) = parse_longmem_clock(text) {
        return Some(stamp);
    }
    let mut parts = text.split('-');
    let year: i16 = parts.next()?.parse().ok()?;
    let month: i8 = parts.next()?.parse().ok()?;
    let day: i8 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    at_hour(Date::new(year, month, day).ok()?, 12)
}

fn parse_longmem_clock(text: &str) -> Option<Timestamp> {
    let (date, rest) = text.split_once(' ')?;
    if !date.contains('/') {
        return None;
    }
    let time = rest.rsplit_once(' ').map(|(_, time)| time).unwrap_or(rest);
    let mut parts = date.split('/');
    let year: i16 = parts.next()?.parse().ok()?;
    let month: i8 = parts.next()?.parse().ok()?;
    let day: i8 = parts.next()?.parse().ok()?;
    let (hour, minute) = time.split_once(':')?;
    let date = Date::new(year, month, day).ok()?;
    at_clock(date, hour.parse().ok()?, minute.parse().ok()?)
}

pub(crate) fn time_targets(query: &str, now: Timestamp) -> Vec<TimeTarget> {
    let mut targets = Vec::new();
    let days_back = |days: f64| {
        now.checked_sub(SignedDuration::from_secs((days * 86_400.0) as i64))
            .ok()
    };
    let point = |at: Timestamp, spread_days: f64| TimeTarget {
        start: at,
        end: at,
        spread_days,
    };

    for caps in AGO.captures_iter(query) {
        let (Some(count), Some(unit)) = (number(&caps[1]), unit_days(&caps[2])) else {
            continue;
        };
        let days = count * unit;
        let Some(at) = days_back(days) else {
            continue;
        };
        targets.push(point(at, point_spread(days)));
    }
    for caps in PAST_SPAN.captures_iter(query) {
        let count = caps
            .get(1)
            .map_or(Some(1.0), |matched| number(matched.as_str()));
        let (Some(count), Some(unit)) = (count, unit_days(&caps[2])) else {
            continue;
        };
        let Some(start) = days_back(count * unit) else {
            continue;
        };
        targets.push(TimeTarget {
            start,
            end: now,
            spread_days: point_spread(unit),
        });
    }
    for caps in THIS_SPAN.captures_iter(query) {
        let today = date_of(now);
        let start = match caps[1].to_ascii_lowercase().as_str() {
            "week" => {
                let back = i64::from(today.weekday().to_monday_zero_offset());
                today.checked_sub(Span::new().days(back)).unwrap_or(today)
            }
            "month" => today.first_of_month(),
            _ => today.first_of_year(),
        };
        let Some(start) = at_hour(start, 0) else {
            continue;
        };
        targets.push(TimeTarget {
            start,
            end: now,
            spread_days: 1.0,
        });
    }
    for caps in YESTERDAY.captures_iter(query) {
        let days = if caps.get(1).is_some() { 2.0 } else { 1.0 };
        if let Some(at) = days_back(days) {
            targets.push(point(at, 1.0));
        }
    }
    for caps in LAST_WEEKDAY.captures_iter(query) {
        let Some(weekday) = weekday(&caps[1]) else {
            continue;
        };
        let today = i64::from(date_of(now).weekday().to_monday_zero_offset());
        let wanted = i64::from(weekday.to_monday_zero_offset());
        let days = match (today - wanted).rem_euclid(7) {
            0 => 7,
            days => days,
        };
        if let Some(at) = days_back(days as f64) {
            targets.push(point(at, 1.0));
        }
    }
    for caps in MONTH_DAY.captures_iter(query) {
        let Some(month) = month(&caps[1]) else {
            continue;
        };
        let Ok(day) = caps[2].parse::<u32>() else {
            continue;
        };
        if let Some(at) = latest_date(now, month, day) {
            targets.push(point(at, 1.0));
        }
    }
    for caps in IN_MONTH.captures_iter(query) {
        let Some(month) = month(&caps[2]) else {
            continue;
        };
        let Some(first) = latest_date(now, month, 1) else {
            continue;
        };
        let first = date_of(first);
        let last = u32::try_from(first.days_in_month()).unwrap_or(28);
        let (from, to) = match caps[1].to_ascii_lowercase().as_str() {
            "early" => (1, 10),
            "mid" => (11, 20),
            "late" => (21, last),
            _ => (1, last),
        };
        let (Some(start), Some(end)) = (with_day(first, from), with_day(first, to)) else {
            continue;
        };
        let (Some(start), Some(end)) = (at_hour(start, 12), at_hour(end, 12)) else {
            continue;
        };
        let half_day = SignedDuration::from_hours(12);
        targets.push(TimeTarget {
            start: start - half_day,
            end: (end + half_day).min(now),
            spread_days: 3.0,
        });
    }
    targets
}

pub(crate) fn time_match(targets: &[TimeTarget], date: Timestamp) -> f64 {
    targets
        .iter()
        .map(|target| {
            let outside = if date < target.start {
                target.start.duration_since(date)
            } else if date > target.end {
                date.duration_since(target.end)
            } else {
                SignedDuration::ZERO
            };
            let z = outside.as_secs() as f64 / 86_400.0 / target.spread_days;
            (-0.5 * z * z).exp()
        })
        .fold(0.0, f64::max)
}

fn point_spread(days: f64) -> f64 {
    (days / 4.0).max(1.0)
}

fn number(text: &str) -> Option<f64> {
    let text = text.to_ascii_lowercase();
    let word = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let value = match word.as_str() {
        "a" | "an" | "one" => 1.0,
        "two" => 2.0,
        "three" => 3.0,
        "four" => 4.0,
        "five" => 5.0,
        "six" => 6.0,
        "seven" => 7.0,
        "eight" => 8.0,
        "nine" => 9.0,
        "ten" => 10.0,
        "eleven" => 11.0,
        "twelve" => 12.0,
        _ if word.contains("couple") => 2.0,
        _ if word.contains("few") => 3.0,
        _ => word.parse::<u32>().ok()?.into(),
    };
    Some(value)
}

fn unit_days(unit: &str) -> Option<f64> {
    match unit.to_ascii_lowercase().as_str() {
        "day" => Some(1.0),
        "week" => Some(7.0),
        "month" => Some(30.0),
        "year" => Some(365.0),
        _ => None,
    }
}

fn weekday(name: &str) -> Option<Weekday> {
    match name.to_ascii_lowercase().as_str() {
        "monday" => Some(Weekday::Monday),
        "tuesday" => Some(Weekday::Tuesday),
        "wednesday" => Some(Weekday::Wednesday),
        "thursday" => Some(Weekday::Thursday),
        "friday" => Some(Weekday::Friday),
        "saturday" => Some(Weekday::Saturday),
        "sunday" => Some(Weekday::Sunday),
        _ => None,
    }
}

fn month(name: &str) -> Option<u32> {
    let name = name.to_ascii_lowercase();
    MONTHS
        .iter()
        .position(|month| *month == name)
        .map(|index| index as u32 + 1)
}

fn date_of(at: Timestamp) -> Date {
    at.to_zoned_datetime(TimeZone::UTC).date()
}

fn at_hour(date: Date, hour: i8) -> Option<Timestamp> {
    at_clock(date, hour, 0)
}

fn at_clock(date: Date, hour: i8, minute: i8) -> Option<Timestamp> {
    let zoned = date
        .at(hour, minute, 0, 0)
        .to_zoned_datetime(TimeZone::UTC)
        .ok()?;
    Some(zoned.timestamp())
}

fn with_day(date: Date, day: u32) -> Option<Date> {
    date.with().day(i8::try_from(day).ok()?).build().ok()
}

fn latest_date(now: Timestamp, month: u32, day: u32) -> Option<Timestamp> {
    let year = date_of(now).year();
    let at = |year: i16| {
        let date = Date::new(year, i8::try_from(month).ok()?, i8::try_from(day).ok()?).ok()?;
        at_hour(date, 12)
    };
    let this_year = at(year)?;
    if this_year <= now {
        Some(this_year)
    } else {
        at(year - 1)
    }
}
