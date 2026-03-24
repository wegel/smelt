use std::io::{self, Read, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};

static VERBOSE: AtomicBool = AtomicBool::new(false);

macro_rules! vprint {
    ($($arg:tt)*) => {
        if VERBOSE.load(Ordering::Relaxed) {
            eprint!($($arg)*);
        }
    };
}

macro_rules! vprintln {
    ($($arg:tt)*) => {
        if VERBOSE.load(Ordering::Relaxed) {
            eprintln!($($arg)*);
        }
    };
}
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf";
const MODEL_FILE: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";

const MAX_OUTPUT_TOKENS: i32 = 200;
const DEFAULT_CTX_SIZE: u32 = 8192;
// Reserve tokens for the ChatML template wrapping
const TEMPLATE_OVERHEAD: i32 = 100;

/// How to handle input that exceeds the model's context window.
#[derive(Clone, Copy, Debug)]
enum Strategy {
    /// Keep only the tail of the input. One inference pass.
    /// Best for build logs, test output — the verdict is at the end.
    Tail,
    /// Process chunks sequentially, carrying a running summary forward.
    /// Multiple inference passes but nothing is lost.
    Rolling,
}

impl Strategy {
    fn from_args(args: &[String]) -> Self {
        for arg in args {
            match arg.as_str() {
                "--rolling" => return Strategy::Rolling,
                "--last" => return Strategy::Tail,
                _ => {}
            }
        }
        Strategy::Tail
    }
}

fn parse_usize_arg(args: &[String], name: &str) -> Option<usize> {
    for (i, arg) in args.iter().enumerate() {
        if arg == name {
            return args.get(i + 1).and_then(|v| v.parse().ok());
        }
        if let Some(val) = arg.strip_prefix(&format!("{name}=")) {
            return val.parse().ok();
        }
    }
    None
}

fn parse_ctx_size(args: &[String]) -> u32 {
    parse_usize_arg(args, "--ctx-size").unwrap_or(DEFAULT_CTX_SIZE as usize) as u32
}

fn parse_str_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    for (i, arg) in args.iter().enumerate() {
        if arg == name {
            return args.get(i + 1).map(|v| v.as_str());
        }
        if let Some(val) = arg.strip_prefix(&format!("{name}=")) {
            return Some(val);
        }
    }
    None
}

// ── Prompt builders ────────────────────────────────────────────

const SYSTEM_PROMPT: &str = "\
You compress command output for another paid language model.
Rules:
- Keep the answer extremely short (but complete).
- No markdown.
- Prefer one sentence. Never exceed three short lines.
- Never ask for more input.
- If the command output is insufficient, reply only with \"smelt: Insufficient information to output anything.\"
- If the source is already shorter than your answer would be, prefer a minimal answer or reuse the source wording.";

const DEFAULT_PROMPT: &str = "Summarize this command output:";

/// Build a ChatML-formatted prompt (Qwen2.5 chat template).
fn build_prompt(instruction: &str, input: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{instruction}\n{input}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Build a rolling prompt that includes the running summary so far.
fn build_rolling_prompt(instruction: &str, running_summary: &str, chunk: &str) -> String {
    let user_msg = if running_summary.is_empty() {
        format!("{instruction}\n{chunk}")
    } else {
        format!(
            "Here is your summary of the output so far:\n{running_summary}\n\n\
             {instruction}\n{chunk}"
        )
    };

    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user_msg}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

// ── Model download ─────────────────────────────────────────────

fn models_dir() -> Result<PathBuf> {
    let base = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join("smelt").join("models");
    std::fs::create_dir_all(&dir)
        .context("failed to create models directory")?;
    Ok(dir)
}

fn ensure_model() -> Result<PathBuf> {
    let model_path = models_dir()?.join(MODEL_FILE);

    if model_path.exists() {
        return Ok(model_path);
    }

    eprintln!("smelt: downloading model to {}", model_path.display());

    let resp = ureq::get(MODEL_URL)
        .call()
        .context("failed to download model")?;

    let total: u64 = resp.headers().get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Write to a temp file first, then rename (atomic-ish)
    let tmp_path = model_path.with_extension("gguf.part");
    let mut file = std::fs::File::create(&tmp_path)
        .context("failed to create temp file")?;

    let mut reader = resp.into_body().into_reader();
    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 256 * 1024];

    loop {
        let n = reader.read(&mut buf).context("download read failed")?;
        if n == 0 { break; }
        file.write_all(&buf[..n]).context("write failed")?;
        downloaded += n as u64;
        if total > 0 {
            eprint!("\rsmelt: downloading... {:.0}%", downloaded as f64 / total as f64 * 100.0);
        }
    }
    eprintln!();

    std::fs::rename(&tmp_path, &model_path)
        .context("failed to move downloaded model into place")?;

    eprintln!("smelt: model saved to {}", model_path.display());
    Ok(model_path)
}

// ── Inference ──────────────────────────────────────────────────

/// Run a single inference pass: tokenize prompt, prefill, generate.
/// Reuses the provided context (caller must clear KV cache if needed).
fn infer(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    prompt: &str,
) -> Result<String> {
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .context("failed to tokenize")?;

    let n_prompt = tokens.len() as i32;

    // Prefill
    let mut batch = LlamaBatch::new(n_prompt.max(512) as usize, 1);
    let last = n_prompt - 1;
    for (i, &token) in tokens.iter().enumerate() {
        batch.add(token, i as i32, &[0], i as i32 == last)?;
    }
    ctx.decode(&mut batch).context("prefill failed")?;

    // Generate
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.1),
        LlamaSampler::greedy(),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = n_prompt;
    let n_max = n_prompt + MAX_OUTPUT_TOKENS;

    while n_cur < n_max {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .context("token_to_piece failed")?;
        output.push_str(&piece);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch).context("decode failed")?;
        n_cur += 1;
    }

    Ok(output)
}

/// Count tokens for a string without allocating a full prompt.
fn token_count(model: &LlamaModel, text: &str) -> Result<i32> {
    Ok(model
        .str_to_token(text, AddBos::Never)
        .context("tokenize failed")?
        .len() as i32)
}

// ── Strategies ─────────────────────────────────────────────────

/// Max input tokens that can fit alongside template + generation budget.
fn max_input_tokens(ctx_size: u32) -> i32 {
    ctx_size as i32 - MAX_OUTPUT_TOKENS - TEMPLATE_OVERHEAD
}

/// Tail strategy: keep the last lines that fit in context, summarize once.
fn summarize_tail(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    input: &str,
    instruction: &str,
    ctx_size: u32,
) -> Result<String> {
    let budget = max_input_tokens(ctx_size);
    let prompt_text = build_prompt(instruction, input);
    let n_tokens = token_count(model, &prompt_text)?;

    if n_tokens <= budget {
        return infer(model, ctx, &prompt_text);
    }

    // Binary search for the longest tail (by lines) that fits
    let lines: Vec<&str> = input.lines().collect();
    let mut lo = 0usize;
    let mut hi = lines.len();
    let mut best_start = hi.saturating_sub(1);

    while lo < hi {
        let mid = (lo + hi) / 2;
        let candidate: String = lines[mid..].join("\n");
        let candidate_prompt = build_prompt(instruction, &candidate);
        let n = token_count(model, &candidate_prompt)?;
        if n <= budget {
            hi = mid;
            best_start = mid;
        } else {
            lo = mid + 1;
        }
    }

    let truncated: String = lines[best_start..].join("\n");
    let kept = lines.len() - best_start;
    vprintln!(
        "smelt: input truncated to last {kept}/{} lines",
        lines.len()
    );

    ctx.clear_kv_cache();
    infer(model, ctx, &build_prompt(instruction, &truncated))
}

/// Rolling strategy: chunk input, summarize sequentially with carry-forward.
fn summarize_rolling(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    input: &str,
    instruction: &str,
    ctx_size: u32,
) -> Result<String> {
    // Budget per chunk: context minus template, generation, and room for running summary
    let summary_reserve: i32 = MAX_OUTPUT_TOKENS; // previous summary could be this long
    let chunk_budget = max_input_tokens(ctx_size) - summary_reserve;

    if chunk_budget <= 0 {
        // Context too small for rolling, fall back to tail
        return summarize_tail(model, ctx, input, instruction, ctx_size);
    }

    // Split input into chunks that each fit within chunk_budget tokens
    let lines: Vec<&str> = input.lines().collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();

    for line in &lines {
        let candidate = if current_chunk.is_empty() {
            line.to_string()
        } else {
            format!("{current_chunk}\n{line}")
        };

        if token_count(model, &candidate)? > chunk_budget && !current_chunk.is_empty() {
            chunks.push(current_chunk);
            current_chunk = line.to_string();
        } else {
            current_chunk = candidate;
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    // If it fits in one chunk, just do a single pass
    if chunks.len() == 1 {
        return infer(model, ctx, &build_prompt(instruction, &chunks[0]));
    }

    vprintln!(
        "smelt: rolling summary over {} chunks",
        chunks.len()
    );

    // Process chunks sequentially
    let mut running_summary = String::new();

    for (i, chunk) in chunks.iter().enumerate() {
        vprint!(
            "\rsmelt: thinking... chunk {}/{}",
            i + 1,
            chunks.len()
        );

        let prompt = build_rolling_prompt(instruction, &running_summary, chunk);
        ctx.clear_kv_cache();
        running_summary = infer(model, ctx, &prompt)?.trim().to_string();
    }
    vprintln!();

    Ok(running_summary)
}

// ── Main ───────────────────────────────────────────────────────

fn print_help() {
    eprintln!("\
smelt — compress command output using a local LLM

USAGE:
    command | smelt [OPTIONS]

OPTIONS:
    --last            Summarize only the tail of the input (default)
    --rolling         Summarize the full input in sequential chunks
    --head N          Passthrough first N lines (no summarization)
    --tail N          Passthrough last N lines (no summarization)
    --prompt TEXT     Set the instruction prompt (default: \"Summarize this command output:\")
    --ctx-size N      Context window size in tokens (default: 8192)
    -v, --verbose     Show timing and progress on stderr
    --help            Show this help message
    --version         Show version

EXAMPLES:
    cargo build 2>&1 | smelt
    npm test 2>&1 | smelt --rolling
    cargo build 2>&1 | smelt --head 5 --tail 3
    kubectl get pods -A | smelt --ctx-size 16384");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        eprintln!("smelt {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    if args.iter().any(|a| a == "--verbose" || a == "-v") {
        VERBOSE.store(true, Ordering::Relaxed);
    }

    let strategy = Strategy::from_args(&args);
    let ctx_size = parse_ctx_size(&args);
    let head_n = parse_usize_arg(&args, "--head");
    let tail_n = parse_usize_arg(&args, "--tail");
    let instruction = parse_str_arg(&args, "--prompt")
        .unwrap_or(DEFAULT_PROMPT);

    // ── Read stdin ──────────────────────────────────────────────
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read stdin")?;

    if input.trim().is_empty() {
        return Ok(());
    }

    // ── Passthrough head/tail lines ─────────────────────────────
    if head_n.is_some() || tail_n.is_some() {
        let lines: Vec<&str> = input.lines().collect();
        let h = head_n.unwrap_or(0).min(lines.len());
        let t = tail_n.unwrap_or(0).min(lines.len().saturating_sub(h));

        for line in &lines[..h] {
            println!("{line}");
        }
        if h > 0 && t > 0 && h + t < lines.len() {
            println!("... ({} lines omitted) ...", lines.len() - h - t);
        }
        for line in &lines[lines.len() - t..] {
            println!("{line}");
        }
        io::stdout().flush()?;
        return Ok(());
    }

    // ── Load model ─────────────────────────────────────────────
    let t_total = Instant::now();
    let model_path = ensure_model()?;

    let t_load = Instant::now();
    vprint!("smelt: loading model... ");

    let backend = LlamaBackend::init()?;
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model_params = std::pin::pin!(model_params);

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("unable to load model")?;

    let load_time = t_load.elapsed();
    vprintln!("done ({:.1}s)", load_time.as_secs_f32());

    // ── Create context ─────────────────────────────────────────
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()))
        .with_n_batch(ctx_size);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("unable to create context")?;

    // ── Summarize ──────────────────────────────────────────────
    let t_infer = Instant::now();
    vprint!("smelt: thinking... ");

    let result = match strategy {
        Strategy::Tail => summarize_tail(&model, &mut ctx, &input, instruction, ctx_size)?,
        Strategy::Rolling => summarize_rolling(&model, &mut ctx, &input, instruction, ctx_size)?,
    };

    let infer_time = t_infer.elapsed();
    let total_time = t_total.elapsed();

    vprintln!(
        "done ({:.1}s)",
        infer_time.as_secs_f32(),
    );

    // ── Output ─────────────────────────────────────────────────
    let trimmed = result.trim();
    if !trimmed.is_empty() {
        println!("{trimmed}");
    }

    vprintln!(
        "smelt: total {:.1}s (load {:.1}s, inference {:.1}s) [strategy: {strategy:?}]",
        total_time.as_secs_f32(),
        load_time.as_secs_f32(),
        infer_time.as_secs_f32()
    );

    io::stdout().flush()?;
    Ok(())
}
