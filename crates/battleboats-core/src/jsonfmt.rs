//! JSON text that matches `json.dumps(..., sort_keys=True, separators=(",", ":"))`.
//!
//! Only `kill_curve_k` is a float in canonical state. The formatter still follows
//! CPython's shortest-roundtrip rules (`PyOS_double_to_string` format `'r'` with
//! `ADD_DOT_0`) so other magnitudes hash the same way.

use ryu::Buffer;

/// Format a finite or non-finite float the way CPython's json encoder does.
pub fn format_python_json_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x.is_sign_negative() {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let mut buffer = Buffer::new();
    pythonize_ryu(buffer.format_finite(x))
}

/// Ryu's shortest form, adjusted to CPython's exponent spelling.
///
/// CPython always prints a signed exponent of at least two digits (`e-07`,
/// `e+16`). It also switches to scientific notation one decade sooner than
/// Ryu on the small side: `1e-5` is `1e-05` in Python and `0.00001` in Ryu.
/// Ryu already matches Python for exponents <= -6 and >= 16.
fn pythonize_ryu(raw: &str) -> String {
    if let Some(e_idx) = raw.find(['e', 'E']) {
        let mant = &raw[..e_idx];
        let exp: i32 = raw[e_idx + 1..].parse().expect("ryu exponent");
        return format!("{mant}e{exp:+03}");
    }
    if let Some(scientific) = python_small_scientific(raw) {
        return scientific;
    }
    if raw.contains('.') {
        raw.to_string()
    } else {
        format!("{raw}.0")
    }
}

/// Rewrite Ryu's fixed form when the scientific exponent is <= -5.
fn python_small_scientific(raw: &str) -> Option<String> {
    let (sign, body) = if let Some(rest) = raw.strip_prefix('-') {
        ("-", rest)
    } else {
        ("", raw)
    };
    let (int_part, frac) = body.split_once('.')?;
    if int_part != "0" {
        return None;
    }
    let zeros = frac.chars().take_while(|c| *c == '0').count();
    let exp = -((zeros as i32) + 1);
    if exp >= -4 {
        return None;
    }
    let digits = &frac[zeros..];
    if digits.is_empty() {
        return None;
    }
    let mant = if digits.len() == 1 {
        digits.to_string()
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    Some(format!("{sign}{mant}e{exp:+03}"))
}

pub fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn json_string(s: &str) -> String {
    let mut out = String::new();
    push_json_string(&mut out, s);
    out
}

pub fn json_i64(n: i64) -> String {
    n.to_string()
}

pub fn json_bool(b: bool) -> String {
    if b {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

pub fn json_pos(x: i32, y: i32) -> String {
    format!("[{x},{y}]")
}

/// Object with keys already in `sort_keys` order. Values are raw JSON text.
pub fn json_object(fields: &[(&str, String)]) -> String {
    let mut out = String::from("{");
    for (i, (key, value)) in fields.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_json_string(&mut out, key);
        out.push(':');
        out.push_str(value);
    }
    out.push('}');
    out
}

pub fn json_array(items: impl IntoIterator<Item = String>) -> String {
    let mut out = String::from("[");
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&item);
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_json_float_vectors() {
        // Captured from CPython 3.12 `json.dumps`.
        let cases = [
            (0.0, "0.0"),
            (1.0, "1.0"),
            (2.0, "2.0"),
            (4.0, "4.0"),
            (0.5, "0.5"),
            (1.5, "1.5"),
            (2.5, "2.5"),
            (0.1, "0.1"),
            (0.3, "0.3"),
            (1e-7, "1e-07"),
            (1e-6, "1e-06"),
            (1e16, "1e+16"),
            (1e20, "1e+20"),
            (10.0, "10.0"),
            (100.0, "100.0"),
            (2.25, "2.25"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (1.23e-5, "1.23e-05"),
            (2.5e-5, "2.5e-05"),
            (5e-5, "5e-05"),
            (1e15, "1000000000000000.0"),
            (1.5e16, "1.5e+16"),
            (std::f64::consts::PI, "3.141592653589793"),
        ];
        for (value, expected) in cases {
            assert_eq!(format_python_json_float(value), expected, "{value}");
        }
        assert_eq!(format_python_json_float(-0.0), "-0.0");
        assert_eq!(format_python_json_float(-1e-5), "-1e-05");
        assert_eq!(format_python_json_float(f64::INFINITY), "Infinity");
        assert_eq!(format_python_json_float(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(format_python_json_float(f64::NAN), "NaN");
        assert_eq!(
            format_python_json_float(1.0000000000000002),
            "1.0000000000000002"
        );
        assert_eq!(format_python_json_float(2.25_f64.powi(4)), "25.62890625");
    }

    #[test]
    fn string_escape_matches_python_ensure_ascii() {
        assert_eq!(json_string("Destroyer"), "\"Destroyer\"");
        assert_eq!(json_string("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(json_string("é"), "\"\\u00e9\"");
    }
}
