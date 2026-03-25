use std::collections::VecDeque;
use std::io::{self, BufRead, IsTerminal, Read, Write};
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
use llama_cpp_2::{LlamaBackendDeviceType, list_llama_ggml_backend_devices};
use llama_cpp_2::{LogOptions, send_logs_to_tracing};

const MODEL_URL: &str = "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf";
const MODEL_FILE: &str = "Qwen_Qwen3-1.7B-Q4_K_M.gguf";

const MAX_OUTPUT_TOKENS: i32 = 200;
const DEFAULT_CTX_SIZE: u32 = 8192;
// Reserve tokens for the ChatML template wrapping
const TEMPLATE_OVERHEAD: i32 = 100;
const SAMPLING_SEED: u32 = 42;
const SAMPLING_TOP_K: i32 = 20;
const SAMPLING_TOP_P: f32 = 0.8;
const SAMPLING_MIN_P: f32 = 0.0;
const SAMPLING_TEMP: f32 = 0.7;
const SAMPLING_PRESENCE_PENALTY: f32 = 1.5;
const PENALTY_LAST_N: i32 = 256;
const FINAL_TAIL_LINES: usize = 40;
const ROLLING_EXACT_CHECK_MARGIN: i32 = 128;

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
        Strategy::Rolling
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
- Prefer the final outcome over intermediate progress.
- For builds and tests, say whether they passed or failed when the output reveals that.
- Never ask for more input.
- If the command output is insufficient, reply only with \"smelt: Insufficient information to output anything.\"
- If the source is already shorter than your answer would be, prefer a minimal answer or reuse the source wording.";

const DEFAULT_PROMPT: &str = "Summarize this command output:";

/// Build a ChatML-formatted prompt for Qwen chat models.
fn build_prompt(instruction: &str, input: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{instruction}\n{input}\n/no_think<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// Build a rolling prompt that includes the running summary so far.
fn build_rolling_prompt(instruction: &str, running_summary: &str, chunk: &str) -> String {
    let user_msg = if running_summary.is_empty() {
        format!("{instruction}\n{chunk}\n/no_think")
    } else {
        format!(
            "Here is your summary of the output so far:\n{running_summary}\n\n\
             {instruction}\n{chunk}\n/no_think"
        )
    };

    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user_msg}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// Build a final reconciliation prompt for rolling mode using the running summary and raw tail.
fn build_rolling_finalize_prompt(
    instruction: &str,
    running_summary: &str,
    raw_tail: &str,
) -> String {
    let user_msg = format!(
        "Produce the final summary of the full command output.\n\
         Prioritize the final outcome over intermediate progress.\n\
         For tests and builds, explicitly state pass/fail and counts when present.\n\
         If later lines contradict earlier summary, trust the later lines.\n\n\
         Original instruction:\n{instruction}\n\n\
         Running summary so far:\n{running_summary}\n\n\
         Last raw lines:\n{raw_tail}\n/no_think"
    );

    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user_msg}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

fn finalize_rolling_summary(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    instruction: &str,
    running_summary: &str,
    tail_lines: &VecDeque<String>,
) -> Result<String> {
    let raw_tail = tail_lines
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = build_rolling_finalize_prompt(instruction, running_summary, &raw_tail);
    ctx.clear_kv_cache();
    infer(model, ctx, &prompt).map(|s| s.trim().to_string())
}

struct RollingOutput {
    summary: String,
    total_lines: usize,
    tail_lines: VecDeque<String>,
    streamed_head_lines: usize,
}

#[derive(Clone, Copy)]
struct RollingConfig<'a> {
    instruction: &'a str,
    ctx_size: u32,
    head_n: Option<usize>,
    tail_n: Option<usize>,
}

struct SummaryOutput<'a> {
    summary: &'a str,
    total_lines: usize,
    tail_lines: &'a [&'a str],
    streamed_head_lines: usize,
}

fn print_summary_output(
    output: SummaryOutput<'_>,
    head_n: Option<usize>,
    tail_n: Option<usize>,
) -> Result<()> {
    let trimmed = output.summary.trim();
    if !trimmed.is_empty() {
        println!("{trimmed}");
    }

    let requested_tail = tail_n.unwrap_or(0);
    if requested_tail > 0 && !output.tail_lines.is_empty() {
        let h = output
            .streamed_head_lines
            .max(head_n.unwrap_or(0).min(output.total_lines));
        let tail_start = output.total_lines.saturating_sub(requested_tail).max(h);
        let tail_buffer_start = output.total_lines.saturating_sub(output.tail_lines.len());
        let skip = tail_start.saturating_sub(tail_buffer_start);

        for line in output.tail_lines.iter().skip(skip) {
            println!("{line}");
        }
    }

    io::stdout().flush()?;
    Ok(())
}

fn print_summary(summary: &str) -> Result<()> {
    let trimmed = summary.trim();
    if !trimmed.is_empty() {
        println!("{trimmed}");
    }

    io::stdout().flush()?;
    Ok(())
}

fn detect_gpu_warning(backend: &LlamaBackend) -> Option<String> {
    if backend.supports_gpu_offload() {
        return None;
    }

    let built_with_gpu_backend =
        cfg!(feature = "vulkan") || cfg!(feature = "cuda") || cfg!(feature = "metal");

    if !built_with_gpu_backend {
        return Some(
            "smelt: warning: running without GPU offload; this binary was built without vulkan/cuda/metal support"
                .to_string(),
        );
    }

    let devices = list_llama_ggml_backend_devices();
    let gpu_like_devices = devices
        .iter()
        .filter(|device| {
            matches!(
                device.device_type,
                LlamaBackendDeviceType::Gpu
                    | LlamaBackendDeviceType::IntegratedGpu
                    | LlamaBackendDeviceType::Accelerator
            )
        })
        .map(|device| {
            if device.description.is_empty() {
                device.name.clone()
            } else {
                format!("{} ({})", device.name, device.description)
            }
        })
        .collect::<Vec<_>>();

    if gpu_like_devices.is_empty() {
        Some(
            "smelt: warning: running without GPU offload; no supported GPU backend device was found"
                .to_string(),
        )
    } else {
        Some(format!(
            "smelt: warning: running without GPU offload despite detected devices: {}",
            gpu_like_devices.join(", ")
        ))
    }
}

// ── Model download ─────────────────────────────────────────────

fn models_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join("smelt").join("models");
    std::fs::create_dir_all(&dir).context("failed to create models directory")?;
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

    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Write to a temp file first, then rename (atomic-ish)
    let tmp_path = model_path.with_extension("gguf.part");
    let mut file = std::fs::File::create(&tmp_path).context("failed to create temp file")?;

    let mut reader = resp.into_body().into_reader();
    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 256 * 1024];

    loop {
        let n = reader.read(&mut buf).context("download read failed")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write failed")?;
        downloaded += n as u64;
        if total > 0 {
            eprint!(
                "\rsmelt: downloading... {:.0}%",
                downloaded as f64 / total as f64 * 100.0
            );
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
        LlamaSampler::penalties(PENALTY_LAST_N, 1.0, 0.0, SAMPLING_PRESENCE_PENALTY),
        LlamaSampler::top_k(SAMPLING_TOP_K),
        LlamaSampler::top_p(SAMPLING_TOP_P, 1),
        LlamaSampler::min_p(SAMPLING_MIN_P, 1),
        LlamaSampler::temp(SAMPLING_TEMP),
        LlamaSampler::dist(SAMPLING_SEED),
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

/// Streaming rolling strategy: read stdin incrementally and compact chunks as they fill.
fn summarize_rolling_stream<R: BufRead>(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    reader: &mut R,
    first_line: String,
    config: RollingConfig<'_>,
) -> Result<RollingOutput> {
    // Budget per chunk: context minus template, generation, and room for running summary
    let summary_reserve: i32 = MAX_OUTPUT_TOKENS;
    let chunk_budget = max_input_tokens(config.ctx_size) - summary_reserve;
    let tail_capacity = FINAL_TAIL_LINES.max(config.tail_n.unwrap_or(0));

    if chunk_budget <= 0 {
        let mut input = first_line;
        reader
            .read_to_string(&mut input)
            .context("failed to read stdin")?;
        let summary = summarize_tail(model, ctx, &input, config.instruction, config.ctx_size)?;
        let lines: Vec<String> = input.lines().map(|s| s.to_string()).collect();
        let h = config.head_n.unwrap_or(0).min(lines.len());
        let mut tail_lines = VecDeque::new();
        for line in lines.iter().skip(lines.len().saturating_sub(tail_capacity)) {
            tail_lines.push_back(line.clone());
        }
        return Ok(RollingOutput {
            summary,
            total_lines: lines.len(),
            tail_lines,
            streamed_head_lines: h,
        });
    }

    let stream_compactions = io::stderr().is_terminal();
    let mut running_summary = String::new();
    let mut current_chunk = String::new();
    let newline_tokens = token_count(model, "\n")?;
    let mut current_chunk_tokens = 0i32;
    let mut compacted_chunks = 0usize;
    let mut tail_lines: VecDeque<String> = VecDeque::new();
    let mut total_lines = 0usize;
    let mut streamed_head_lines = 0usize;

    let mut process_line = |line: &str,
                            current_chunk: &mut String,
                            running_summary: &mut String,
                            compacted_chunks: &mut usize|
     -> Result<()> {
        let line_tokens = token_count(model, line)?;
        let estimated_candidate_tokens = if current_chunk.is_empty() {
            line_tokens
        } else {
            current_chunk_tokens + newline_tokens + line_tokens
        };
        let needs_exact_check =
            estimated_candidate_tokens >= chunk_budget - ROLLING_EXACT_CHECK_MARGIN;
        let exact_candidate_tokens = if needs_exact_check {
            let candidate = if current_chunk.is_empty() {
                line.to_string()
            } else {
                format!("{current_chunk}\n{line}")
            };
            let exact_tokens = token_count(model, &candidate)?;
            Some((candidate, exact_tokens))
        } else {
            None
        };
        let candidate_overflows = exact_candidate_tokens
            .as_ref()
            .map(|(_, tokens)| *tokens > chunk_budget)
            .unwrap_or(false);

        if candidate_overflows && !current_chunk.is_empty() {
            *compacted_chunks += 1;
            if !stream_compactions {
                vprint!("\rsmelt: thinking... chunk {}", *compacted_chunks);
            }

            let prompt = build_rolling_prompt(config.instruction, running_summary, current_chunk);
            ctx.clear_kv_cache();
            *running_summary = infer(model, ctx, &prompt)?.trim().to_string();

            if stream_compactions && !running_summary.is_empty() {
                eprintln!("{running_summary}");
            }

            *current_chunk = line.to_string();
            current_chunk_tokens = line_tokens;
        } else if let Some((candidate, exact_tokens)) = exact_candidate_tokens {
            *current_chunk = candidate;
            current_chunk_tokens = exact_tokens;
        } else if current_chunk.is_empty() {
            *current_chunk = line.to_string();
            current_chunk_tokens = line_tokens;
        } else {
            current_chunk.push('\n');
            current_chunk.push_str(line);
            current_chunk_tokens = estimated_candidate_tokens;
        }

        total_lines += 1;
        if streamed_head_lines < config.head_n.unwrap_or(0) {
            println!("{line}");
            io::stdout().flush()?;
            streamed_head_lines += 1;
        }
        tail_lines.push_back(line.to_string());
        if tail_lines.len() > tail_capacity {
            tail_lines.pop_front();
        }

        Ok(())
    };

    process_line(
        first_line.trim_end_matches(['\r', '\n']),
        &mut current_chunk,
        &mut running_summary,
        &mut compacted_chunks,
    )?;

    let mut line = String::new();
    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .context("failed to read stdin")?
            == 0
        {
            break;
        }

        process_line(
            line.trim_end_matches(['\r', '\n']),
            &mut current_chunk,
            &mut running_summary,
            &mut compacted_chunks,
        )?;
    }

    if compacted_chunks == 0 {
        let summary = infer(
            model,
            ctx,
            &build_prompt(config.instruction, &current_chunk),
        )?;
        return Ok(RollingOutput {
            summary,
            total_lines,
            tail_lines,
            streamed_head_lines,
        });
    }

    if !current_chunk.is_empty() {
        let prompt = build_rolling_prompt(config.instruction, &running_summary, &current_chunk);
        ctx.clear_kv_cache();
        running_summary = infer(model, ctx, &prompt)?.trim().to_string();
    }

    let summary = finalize_rolling_summary(
        model,
        ctx,
        config.instruction,
        &running_summary,
        &tail_lines,
    )?;
    Ok(RollingOutput {
        summary,
        total_lines,
        tail_lines,
        streamed_head_lines,
    })
}

fn load_model() -> Result<(LlamaBackend, LlamaModel, std::time::Duration)> {
    let model_path = ensure_model()?;

    let t_load = Instant::now();
    vprint!("smelt: loading model... ");

    let backend = LlamaBackend::init()?;
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    if let Some(warning) = detect_gpu_warning(&backend) {
        eprintln!("{warning}");
    }

    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model_params = std::pin::pin!(model_params);

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("unable to load model")?;

    let load_time = t_load.elapsed();
    vprintln!("done ({:.1}s)", load_time.as_secs_f32());

    Ok((backend, model, load_time))
}

fn create_context<'a>(
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
    ctx_size: u32,
) -> Result<llama_cpp_2::context::LlamaContext<'a>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()))
        .with_n_batch(ctx_size);

    model
        .new_context(backend, ctx_params)
        .context("unable to create context")
}

// ── Main ───────────────────────────────────────────────────────

fn print_help() {
    eprintln!(
        "\
smelt — compress command output using a local LLM

USAGE:
    command | smelt [OPTIONS]

OPTIONS:
    --last            Summarize only the tail of the input
    --rolling         Summarize the full input in sequential chunks (default)
    --head N          Include the first N raw lines before the summary
    --tail N          Include the last N raw lines after the summary
    --prompt TEXT     Set the instruction prompt (default: \"Summarize this command output:\")
    --ctx-size N      Context window size in tokens (default: 8192)
    -v, --verbose     Show timing and progress on stderr
    --help            Show this help message
    --version         Show version

EXAMPLES:
    cargo build 2>&1 | smelt
    npm test 2>&1 | smelt
    cargo build 2>&1 | smelt --head 5 --tail 3
    cargo build 2>&1 | smelt --last
    kubectl get pods -A | smelt --ctx-size 16384"
    );
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
    let instruction = parse_str_arg(&args, "--prompt").unwrap_or(DEFAULT_PROMPT);

    if matches!(strategy, Strategy::Rolling) {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let mut first_line = String::new();

        if reader
            .read_line(&mut first_line)
            .context("failed to read stdin")?
            == 0
        {
            return Ok(());
        }

        let t_total = Instant::now();
        let (backend, model, load_time) = load_model()?;
        let mut ctx = create_context(&backend, &model, ctx_size)?;

        let t_infer = Instant::now();
        vprint!("smelt: thinking... ");

        let output = summarize_rolling_stream(
            &model,
            &mut ctx,
            &mut reader,
            first_line,
            RollingConfig {
                instruction,
                ctx_size,
                head_n,
                tail_n,
            },
        )?;

        let infer_time = t_infer.elapsed();
        let total_time = t_total.elapsed();

        vprintln!("done ({:.1}s)", infer_time.as_secs_f32(),);

        if head_n.is_some() || tail_n.is_some() {
            let tail_lines: Vec<&str> = output.tail_lines.iter().map(String::as_str).collect();
            print_summary_output(
                SummaryOutput {
                    summary: &output.summary,
                    total_lines: output.total_lines,
                    tail_lines: &tail_lines,
                    streamed_head_lines: output.streamed_head_lines,
                },
                head_n,
                tail_n,
            )?;
        } else {
            print_summary(&output.summary)?;
        }

        vprintln!(
            "smelt: total {:.1}s (load {:.1}s, inference {:.1}s) [strategy: {strategy:?}]",
            total_time.as_secs_f32(),
            load_time.as_secs_f32(),
            infer_time.as_secs_f32()
        );

        io::stdout().flush()?;
        return Ok(());
    }

    // ── Read stdin ──────────────────────────────────────────────
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read stdin")?;

    if input.trim().is_empty() {
        return Ok(());
    }

    // ── Load model ─────────────────────────────────────────────
    let t_total = Instant::now();
    let (backend, model, load_time) = load_model()?;
    let mut ctx = create_context(&backend, &model, ctx_size)?;

    // ── Summarize ──────────────────────────────────────────────
    let t_infer = Instant::now();
    vprint!("smelt: thinking... ");

    debug_assert!(matches!(strategy, Strategy::Tail));
    let result = summarize_tail(&model, &mut ctx, &input, instruction, ctx_size)?;

    let infer_time = t_infer.elapsed();
    let total_time = t_total.elapsed();

    vprintln!("done ({:.1}s)", infer_time.as_secs_f32(),);

    // ── Output ─────────────────────────────────────────────────
    if head_n.is_some() || tail_n.is_some() {
        let lines: Vec<&str> = input.lines().collect();
        let h = head_n.unwrap_or(0).min(lines.len());
        for line in &lines[..h] {
            println!("{line}");
        }
        let tail_start = lines.len().saturating_sub(tail_n.unwrap_or(0)).max(h);
        print_summary_output(
            SummaryOutput {
                summary: &result,
                total_lines: lines.len(),
                tail_lines: &lines[tail_start..],
                streamed_head_lines: h,
            },
            head_n,
            tail_n,
        )?;
    } else {
        print_summary(&result)?;
    }

    vprintln!(
        "smelt: total {:.1}s (load {:.1}s, inference {:.1}s) [strategy: {strategy:?}]",
        total_time.as_secs_f32(),
        load_time.as_secs_f32(),
        infer_time.as_secs_f32()
    );
    Ok(())
}
