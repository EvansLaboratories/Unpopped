/// HEURISTIC instantiation for the census: the `template <...>` header that precedes `__global__` at `kernel_at` has its type
/// parameters replaced by `float` and its non-type parameters by `128` (bool: `true`) across the whole source, and the header blanked.
/// Not a real instantiation (a real one is the caller's `LiftContext`, a later slice); it shows what stops a templated kernel once its
/// parameters are given.
fn instantiate(src: &str, kernel_at: usize) -> String {
    let Some(t) = src[..kernel_at].rfind("template") else { return src.to_string() };
    let rest = &src[t..];
    let Some(open) = rest.find('<') else { return src.to_string() };
    let (mut depth, mut end) = (0i32, None);
    for (i, c) in rest[open..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(end) = end else { return src.to_string() };
    let header = &rest[open + 1..end];
    let mut params = Vec::new();
    let (mut d, mut cur) = (0i32, String::new());
    for c in header.chars() {
        match c {
            '<' | '(' => d += 1,
            '>' | ')' => d -= 1,
            ',' if d == 0 => {
                params.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    params.push(cur);
    let mut out = src.to_string();
    let hdr_start = t;
    let hdr_end = t + end + 1;
    let blank: String = out[hdr_start..hdr_end].chars().map(|c| if c == '\n' { '\n' } else { ' ' }).collect();
    out.replace_range(hdr_start..hdr_end, &blank);
    for p in params {
        let p = p.split('=').next().unwrap_or("").trim().to_string();
        let words: Vec<&str> = p.split_whitespace().collect();
        let Some(name) = words.last().map(|w| w.trim_start_matches('.').to_string()) else { continue };
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let (kw, repl) = match words.first().copied() {
            Some("typename" | "class") => ("type", "float"),
            Some("bool") => ("value", "true"),
            Some(_) if words.len() >= 2 => ("value", "128"),
            _ => continue,
        };
        let _ = kw;
        let re = regex::Regex::new(&format!(r"\b{}\b", regex::escape(&name))).unwrap();
        out = re.replace_all(&out, repl).to_string();
    }
    out
}
