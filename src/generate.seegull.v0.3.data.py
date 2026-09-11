"""
Bulk synthetic dataset generation via DeepSeek-V4-Pro (minirouter) — Gullibility only,
low vs high.

Sibling of mats12/scripts/gen_opus_data100.py and mats12/scripts/gen_sol_data100.py,
rebuilt around a different model and a different strategy:

  * Model: deepseek/deepseek-v4-pro-0813 through minirouter
    (https://api.minirouter.sh/v1, OpenAI-compatible), 1M context / 384K max output.
  * Attribute: gullibility ONLY, and only the "low" and "high" levels — no medium,
    no other attributes. Every call asks for a class-balanced low/high split.
  * Single-shot batches. Instead of asking for 2 conversations per level per call
    (as the Opus/Sol scripts do), each call here asks for as many conversations as
    the model's output budget can plausibly hold — sized from --max_tokens and a
    measured tokens-per-conversation estimate — so one call, not hundreds, produces
    most (ideally all) of a run. Calls are still resumable and top up whatever a
    call under-delivered, but the default target is reachable in a single call.
  * JSONL streaming output. The model is asked to emit one compact JSON object per
    line (not one giant JSON array/object). This is a deliberate consequence of
    single-shot batches: a run that produces hundreds of conversations in one
    completion cannot afford to lose everything to one JSON parse error if the
    stream is cut off (rate limit, truncation at max_tokens, network hiccup) —
    JSONL degrades gracefully, a broken last line costs only that one conversation.
    It also lets progress be tracked from real, validated, already-saved
    conversations as they stream in, not from a token-count proxy.
  * DeepSeek's reasoning tokens count against completion/max_tokens. In testing
    they were the majority of every call's output (SEED test: 6362/8735 completion
    tokens were reasoning, for 10 requested conversations). --max_tokens sizing and
    cost accounting both account for this.
  * Live progress. A redrawn status block shows per-level counts, elapsed time,
    ETA, streamed-token activity (so a long call still visibly "breathes"), and
    running spend (taken from the API's own per-call cost figure when available).
  * Preflight check. Before spending any real budget, main() sends one tiny
    request and requires a reply within --preflight_timeout seconds. If the model
    isn't answering, the run aborts before touching the real quota.

Usage:
    # preflight only
    python src/generate.seegull.v0.3.data.py --preflight_only

    # small smoke test
    python src/generate.seegull.v0.3.data.py --per_level 5 --max_tokens 12000 \
        --output_dir /tmp/seegull_smoke

    # a full run: 140 low + 140 high = 280 conversations, single shot by default
    python src/generate.seegull.v0.3.data.py

    # resume after an interrupt — same command, already-written files are kept
    python src/generate.seegull.v0.3.data.py

    # push for a bigger single-shot batch, closer to the 384K output ceiling
    python src/generate.seegull.v0.3.data.py --per_level 250 --max_tokens 384000
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "mats12", ".env"))

MODEL_NAME = "deepseek/deepseek-v4-pro-0813"
BASE_URL = "https://api.minirouter.sh/v1"

CONTEXT_WINDOW = 1_000_000
MAX_MODEL_OUTPUT = 384_000

# DeepSeek-V4-Pro list pricing via minirouter, USD per million tokens — fallback
# for the live spend readout when a response doesn't carry its own "cost" field.
PRICE_IN_PER_MTOK = 0.693
PRICE_OUT_PER_MTOK = 2.079

SEED = 75241239

# Rough completion tokens (visible text + hidden reasoning) per conversation,
# measured empirically against this model. Used only to size how many
# conversations a single call should ask for given --max_tokens, so a call's
# request is ambitious but not so large it guarantees truncation.
EST_TOKENS_PER_CONV = 1100
CALL_BUDGET_HEADROOM = 0.85  # leave slack under max_tokens for reasoning variance

# A conversation must contain at least this many complete user/assistant turn
# pairs to be kept.
MIN_TURN_PAIRS = 2

# Retry a single call at most this many times before giving up on it for this
# round. A failed call is skipped, not fatal — the scheduler comes back to it.
MAX_CALL_RETRIES = 4

LEVELS = ["low", "high"]

GULLIBILITY = {
    "levels": LEVELS,
    "persona": {
        "low": "rarely takes a claim at face value",
        "high": "takes what they are told at face value",
    },
    "behavior": {
        "low": (
            "they ask where information comes from, push back when something "
            "sounds unsupported, and want to see evidence before acting on advice"
        ),
        "high": (
            "they accept the assistant's claims without asking where the "
            "information comes from, and may restate an unsupported claim later "
            "in the conversation as though it were established fact"
        ),
    },
    "show_through": "how the user reacts to the assistant's claims and advice",
    "leak_stems": ["gullib", "credul", "skeptic", "sceptic", "naive", "naïve", "trust"],
}


# --------------------------------------------------------------------------- #
# Diversity pools
#
# These are prompt *seeds*, not a taxonomy: each call draws a fresh sample and
# the prompt explicitly tells the model it may go outside them. For a
# single-shot batch of hundreds of conversations, the pools matter more than
# ever — the prompt leans on combinatorics (domain x angle x voice x claim
# shape) rather than one topic per conversation, since there are far more
# conversations per call than there are topics in any one pool.
# --------------------------------------------------------------------------- #

TOPIC_DOMAINS = [
    "cooking, recipes and food", "grocery shopping and meal planning",
    "home repair and DIY", "cleaning and household organisation",
    "gardening, houseplants and allotments", "pets, vets and animal care",
    "moving house, renting and real estate", "furniture and interior decorating",
    "laundry, clothing care and repairs", "budget groceries and stretching a paycheck",
    "personal finance and budgeting", "taxes, benefits and paperwork",
    "insurance, claims and warranties", "investing, pensions and retirement",
    "career changes, CVs and interviews", "workplace conflict and office politics",
    "freelancing, contracts and invoicing", "small business and side projects",
    "salary negotiation and promotions", "job loss, redundancy and job hunting",
    "general health and symptoms", "fitness, training and injury recovery",
    "sleep, fatigue and energy", "nutrition, diets and supplements",
    "mental health, stress and burnout", "medications and side effects",
    "dentistry, eyesight and hearing", "pregnancy, fertility and postpartum",
    "ageing, caregiving and elder care", "chronic illness and disability",
    "relationships, dating and breakups", "family conflict and in-laws",
    "parenting babies and toddlers", "parenting teenagers",
    "friendship, loneliness and social life", "weddings, funerals and family events",
    "neighbours, housemates and shared living", "grief and difficult news",
    "consumer technology and gadgets", "phones, laptops and buying advice",
    "software bugs, accounts and troubleshooting", "privacy, security and scams",
    "programming and self-taught coding", "AI tools and automation",
    "home networking, wifi and smart devices", "data backup and lost files",
    "history and historical figures", "science, physics and space",
    "biology, ecology and conservation", "statistics, data and how studies work",
    "philosophy, ethics and thought experiments", "religion, ritual and belief",
    "politics, policy and civic process", "economics and how markets work",
    "languages and language learning", "law, courts and bureaucracy",
    "travel planning and itineraries", "visas, borders and travel documents",
    "cars, maintenance and commuting", "cycling, walking and public transport",
    "hobbies, crafts and making things", "music, instruments and practice",
    "film, TV and books", "video games and tabletop games",
    "sports, teams and training", "photography and video",
    "art, design and creative process", "writing, editing and publishing",
    "education, exams and studying", "university applications and student life",
    "volunteering, community and local organising", "weather, hiking and the outdoors",
    "fishing, camping and survival skills", "collecting and second-hand markets",
    "events, tickets and logistics", "emergencies and things going wrong",
]

CONVERSATION_ANGLES = [
    "starts with a very specific narrow question",
    "starts vague and only gets specific under questioning",
    "the user is mid-task and slightly stressed",
    "the user is idly curious with no urgency at all",
    "the user brings a half-formed plan and wants feedback",
    "the user is comparing two or three concrete options",
    "the user is trying to fix something that already went wrong",
    "the user is planning months ahead for a future event",
    "the user is double-checking a decision they have already made",
    "the user is asking on behalf of someone else and relaying answers",
    "the user opens with a claim they read somewhere and wants it checked",
    "the user changes the subject partway through",
    "the user is under a hard deadline and wants the short version",
    "the user is procrastinating and keeps widening the question",
    "the user pushes back on the first answer they get",
    "the user is embarrassed about the situation and hedges the details",
    "the user gives a long rambling backstory before the actual question",
    "the user asks a follow-up that reveals they misread the first answer",
    "the user wants help deciding whether the thing is worth doing at all",
    "the user is stuck between advice from two different people",
    "the user is learning something new and asks for the basics first",
    "the user has a strong prior opinion and is looking for input anyway",
    "the user is troubleshooting step by step, reporting results each turn",
    "the user asks for a rough estimate or ballpark figure",
    "the user is drafting or wording something and wants it improved",
    "the user is worried about a worst case and wants it assessed",
    "the user has an unusual, non-standard version of a common problem",
    "the user keeps adding constraints they forgot to mention",
    "the user is trying to understand why something works, not just how",
    "the user needs to explain the answer to somebody else afterwards",
]

# Gullibility-specific scenario shapes — situations that put "does this person
# take an unverified claim at face value?" front and center. Sampled alongside
# the general angles above, not instead of them.
CLAIM_SHAPES = [
    "the user relays a claim from a forwarded message or group chat",
    "the user quotes a marketing or ad claim about a product",
    "the user brings up something a friend or relative insisted was true",
    "the user mentions a headline or social media post they half-remember",
    "the user is deciding whether to trust an unsolicited offer or deal",
    "the user cites a statistic without knowing where it came from",
    "the user brings up a home remedy or folk wisdom claim",
    "the user repeats something an influencer or online personality said",
    "the user is checking an urban legend or widely-believed myth",
    "the user brings a claim from an anonymous online review or comment",
    "the user is weighing conflicting claims from two different sources",
    "the user mentions a warning or scare story that's been circulating",
    "the user brings up a claim framed as \"scientists say\" with no citation",
    "the user is deciding how much to trust a stranger's confident advice",
    "the user brings a claim from an old article that may be outdated",
    "the user is evaluating a too-good-to-be-true opportunity",
    "the user mentions something a salesperson told them",
    "the user brings up a conspiracy-adjacent claim to sanity-check it",
    "no third-party claim at all — the dynamic shows only in how the user treats the assistant's own answers",
    "no third-party claim at all — the dynamic shows only in how the user treats the assistant's own answers",
]

USER_VOICES = [
    "terse, lowercase, minimal punctuation",
    "long paragraphs with a lot of context",
    "polite and slightly formal",
    "blunt to the point of curtness",
    "chatty with digressions and asides",
    "technical vocabulary used confidently",
    "plain everyday words, no jargon",
    "non-native English phrasing, entirely fluent",
    "typos and autocorrect artefacts left in",
    "bullet points and numbered questions",
    "anxious and over-explaining",
    "wry and understated",
    "voice-to-text run-on sentences",
    "regional idiom and colloquialisms",
]


# --------------------------------------------------------------------------- #
# Pydantic schema for one JSONL line
# --------------------------------------------------------------------------- #

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class Conversation(BaseModel):
    topic: str
    level: str
    turns: List[Turn]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _level_block(level: str) -> str:
    return (
        f'  - level "{level}": the user {GULLIBILITY["persona"][level]}: '
        f'{GULLIBILITY["behavior"][level]}.'
    )


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #

class BatchRejected(Exception):
    """A call returned but failed a structural / quality check. Retryable."""


def leaked_stems(user_msgs: List[str]) -> List[str]:
    text = " ".join(user_msgs).lower()
    return [s for s in GULLIBILITY["leak_stems"] if re.search(r"\b" + re.escape(s), text)]


def validate_conversation(conv: Conversation) -> None:
    """Structural check on a single conversation. Raises BatchRejected."""
    if conv.level not in LEVELS:
        raise BatchRejected(f"unknown level {conv.level!r}")
    if not conv.turns:
        raise BatchRejected("conversation has no turns")

    expected = "user"
    for t in conv.turns:
        if t.role != expected:
            raise BatchRejected(
                f"turns do not alternate (expected {expected}, got {t.role})"
            )
        if not t.content.strip():
            raise BatchRejected("empty turn content")
        expected = "assistant" if expected == "user" else "user"

    n_user = sum(1 for t in conv.turns if t.role == "user")
    n_ai = len(conv.turns) - n_user
    if n_user < MIN_TURN_PAIRS + 1 or n_ai < MIN_TURN_PAIRS:
        raise BatchRejected(f"too few exchanges: {n_user} user / {n_ai} assistant")


def conversation_to_text(conv: Conversation) -> str:
    """Render to the HUMAN:/ASSISTANT: text format the Llama script writes."""
    lines = []
    for t in conv.turns:
        tag = "HUMAN: " if t.role == "user" else "ASSISTANT: "
        lines.append(tag + " ".join(t.content.split()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Live progress display
# --------------------------------------------------------------------------- #

class Progress:
    """A redrawn status block: per-level counts, timing, streaming activity, spend.

    Anything the run wants to print goes through .log(), which erases the block,
    prints the line, and redraws underneath it — so scrollback stays readable.
    Streaming activity (chars received, seconds since last chunk) is shown so a
    long single-shot call still visibly "breathes" instead of looking stuck.
    """

    def __init__(self, target: int, done: Dict[str, int]):
        self.target = target
        self.counts = dict(done)
        self.t0 = time.time()
        self.calls = 0
        self.failures = 0
        self.dropped = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.reasoning_tokens = 0
        self.cost_reported = 0.0
        self.start_total = self.total()
        self.lines_drawn = 0
        self.enabled = sys.stdout.isatty()

        # Live streaming state for the call currently in flight.
        self.streaming = False
        self.call_chars = 0
        self.call_started = 0.0
        self.last_chunk_t = 0.0

    def total(self) -> int:
        return sum(self.counts.values())

    def grand_target(self) -> int:
        return self.target * len(LEVELS)

    def cost(self) -> float:
        if self.cost_reported:
            return self.cost_reported
        return (self.in_tokens * PRICE_IN_PER_MTOK
                + self.out_tokens * PRICE_OUT_PER_MTOK) / 1_000_000

    @staticmethod
    def _bar(done: int, total: int, width: int = 30) -> str:
        filled = 0 if total <= 0 else int(width * min(done, total) / total)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _hms(seconds: float) -> str:
        seconds = int(max(seconds, 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def _render_lines(self) -> List[str]:
        elapsed = time.time() - self.t0
        made = self.total() - self.start_total
        remaining = self.grand_target() - self.total()
        rate = made / elapsed if elapsed > 0 and made else 0.0
        eta = remaining / rate if rate > 0 else None
        cost = self.cost()
        cost_per = cost / made if made else 0.0

        width = shutil.get_terminal_size((100, 24)).columns
        lines = ["", "─" * min(width, 78)]

        lines.append(
            f"  gullibility        {self._bar(self.total(), self.grand_target())} "
            f"{self.total():>4}/{self.grand_target()}"
        )
        per_level = "  ".join(f"{lv}={self.counts.get(lv, 0):>4}/{self.target}" for lv in LEVELS)
        lines.append(f"  {'':<18} {per_level}")

        if self.streaming:
            since = time.time() - self.last_chunk_t
            call_elapsed = time.time() - self.call_started
            status = "receiving..." if since < 5 else f"no new chunk for {since:.0f}s"
            lines.append(
                f"  streaming call {self.calls}: {self.call_chars:,} chars in "
                f"{self._hms(call_elapsed)} ({status})"
            )

        lines.append("─" * min(width, 78))
        lines.append(
            f"  total {self.total():>5}/{self.grand_target()}   "
            f"calls {self.calls}   failed {self.failures}   dropped {self.dropped}"
        )
        lines.append(
            f"  elapsed {self._hms(elapsed)}   "
            f"eta {self._hms(eta) if eta is not None else '--:--'}   "
            f"{rate * 60:.1f} conv/min"
        )
        lines.append(
            f"  spend ${cost:.2f}   (${cost_per:.4f}/conv, "
            f"in {self.in_tokens:,} tok / out {self.out_tokens:,} tok "
            f"[reasoning {self.reasoning_tokens:,}])"
        )
        lines.append("─" * min(width, 78))
        return lines

    def _erase(self) -> None:
        if self.enabled and self.lines_drawn:
            sys.stdout.write(f"\033[{self.lines_drawn}A\033[J")
        self.lines_drawn = 0

    def render(self) -> None:
        if not self.enabled:
            return
        self._erase()
        lines = self._render_lines()
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self.lines_drawn = len(lines) + 1

    def log(self, message: str) -> None:
        self._erase()
        print(message)
        self.render()

    def finish(self) -> None:
        if self.enabled:
            self.render()
            self.lines_drawn = 0
        else:
            print("\n".join(self._render_lines()))


# --------------------------------------------------------------------------- #
# Preflight check
# --------------------------------------------------------------------------- #

def preflight_check(client: OpenAI, timeout: float) -> None:
    """One tiny request. Must get a reply within `timeout` seconds or we abort
    before spending any real budget on a model/route that isn't working."""
    print(f"Preflight: pinging {MODEL_NAME} via {BASE_URL} ...")
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
            stream=True,
            # DeepSeek's hidden reasoning tokens count against max_tokens, and can
            # consume the whole budget before any visible text appears — give this
            # a buffer well above what "OK" alone needs.
            max_tokens=200,
            timeout=timeout,
        )
        first_tok_t = None
        text = ""
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                if first_tok_t is None:
                    first_tok_t = time.time()
                text += delta
            if time.time() - t0 > timeout:
                raise TimeoutError(f"no completed reply within {timeout:.0f}s")
        if first_tok_t is None:
            raise RuntimeError(
                "stream completed but no visible text was returned "
                "(reasoning tokens likely consumed the whole budget)"
            )
    except Exception as e:
        raise SystemExit(
            f"Preflight FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}\n"
            f"Aborting before touching the real generation budget."
        )

    elapsed = time.time() - t0
    ttft = first_tok_t - t0
    print(
        f"Preflight OK in {elapsed:.1f}s "
        f"(first token after {ttft:.1f}s): {text.strip()!r}\n"
    )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def per_call_target(per_level_remaining: Dict[str, int], max_tokens: int) -> Dict[str, int]:
    """How many conversations to ask for per level in the next call.

    Sized from --max_tokens so a call asks for as much as it can plausibly
    deliver (the single-shot goal) without requesting so much that truncation
    is a near-certainty. Never asks for more than is actually still needed.
    """
    budget_convs = int(CALL_BUDGET_HEADROOM * max_tokens / EST_TOKENS_PER_CONV)
    per_level_budget = max(1, budget_convs // len(LEVELS))
    return {lv: min(per_level_remaining[lv], per_level_budget) for lv in LEVELS}


def next_indices(output_dir: str) -> Dict[str, int]:
    nxt = {lv: 0 for lv in LEVELS}
    if not os.path.isdir(output_dir):
        return nxt
    for fname in os.listdir(output_dir):
        m = re.fullmatch(r"conversation_(\d+)_gullibility_(\w+)\.txt", fname)
        if m and m.group(2) in nxt:
            nxt[m.group(2)] = max(nxt[m.group(2)], int(m.group(1)) + 1)
    return nxt


def count_existing(output_dir: str) -> Dict[str, int]:
    counts = {lv: 0 for lv in LEVELS}
    if not os.path.isdir(output_dir):
        return counts
    for fname in os.listdir(output_dir):
        m = re.fullmatch(r"conversation_\d+_gullibility_(\w+)\.txt", fname)
        if m and m.group(1) in counts:
            counts[m.group(1)] += 1
    return counts


def save_conversation(
    conv: Conversation, output_dir: str, indices: Dict[str, int], call_idx: int,
    seed: int, reject_banned: bool, prog: Progress,
) -> bool:
    lv = conv.level
    user_msgs = [t.content for t in conv.turns if t.role == "user"]
    leaks = leaked_stems(user_msgs)
    if leaks and reject_banned:
        prog.dropped += 1
        prog.log(f"  [gullibility/{lv}] dropped (trait words {leaks})")
        return False

    idx = indices[lv]
    stem = f"conversation_{idx}_gullibility_{lv}"
    with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write(conversation_to_text(conv))

    meta = {
        "attribute": "gullibility",
        "level": lv,
        "topic": conv.topic,
        "index": idx,
        "model": MODEL_NAME,
        "seed": seed,
        "call_index": call_idx,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_turns": len(conv.turns),
        "leak_stems_found": leaks,
    }
    with open(os.path.join(output_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    indices[lv] += 1
    prog.counts[lv] = prog.counts.get(lv, 0) + 1
    return True


def run_one_call(
    client: OpenAI,
    output_dir: str,
    remaining: Dict[str, int],
    indices: Dict[str, int],
    rng: random.Random,
    max_tokens: int,
    reject_banned: bool,
    call_idx: int,
    seed: int,
    prog: Progress,
) -> int:
    """Issue one large single-shot call and save conversations as they stream in.

    Returns the number of conversations saved. Parsing is line-by-line JSONL:
    a truncated or malformed final line costs only that one conversation, not
    the whole call. A call that saves nothing after all retries logs and
    returns 0 — the scheduler comes back to it on the next round.
    """
    target = per_call_target(remaining, max_tokens)
    total_requested = sum(target.values())

    for attempt in range(1, MAX_CALL_RETRIES + 1):
        prompt = build_prompt(target, rng)

        saved_this_call = 0
        buf = ""
        prog.streaming = True
        prog.call_chars = 0
        prog.call_started = time.time()
        prog.last_chunk_t = time.time()
        prog.calls += 1
        prog.render()

        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=max_tokens,
            )
            try:
                stream = client.chat.completions.create(stream_options={"include_usage": True}, **kwargs)
            except Exception:
                stream = client.chat.completions.create(**kwargs)

            for event in stream:
                if not event.choices:
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        prog.in_tokens += getattr(usage, "prompt_tokens", 0) or 0
                        prog.out_tokens += getattr(usage, "completion_tokens", 0) or 0
                        details = getattr(usage, "completion_tokens_details", None)
                        if details is not None:
                            prog.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
                        cost = getattr(usage, "cost", None)
                        if cost:
                            prog.cost_reported += cost
                    continue

                delta = event.choices[0].delta.content
                if not delta:
                    continue
                prog.call_chars += len(delta)
                prog.last_chunk_t = time.time()
                buf += delta

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        conv = Conversation.model_validate(obj)
                        validate_conversation(conv)
                    except (json.JSONDecodeError, ValidationError, BatchRejected) as e:
                        prog.dropped += 1
                        prog.log(f"  [gullibility] dropped a line: {e}")
                        continue
                    if remaining.get(conv.level, 0) <= 0:
                        continue
                    if save_conversation(conv, output_dir, indices, call_idx, seed, reject_banned, prog):
                        remaining[conv.level] -= 1
                        saved_this_call += 1
                    prog.render()

            # Whatever is left in buf is an incomplete trailing line — drop it,
            # it was cut off by the stream ending / hitting max_tokens.
            leftover = buf.strip()
            if leftover:
                prog.dropped += 1

        except Exception as e:
            prog.failures += 1
            prog.streaming = False
            prog.log(f"  [gullibility] call {call_idx} attempt {attempt} "
                      f"error: {type(e).__name__}: {e}")
            if saved_this_call == 0 and attempt < MAX_CALL_RETRIES:
                time.sleep(2 * attempt)
                continue

        prog.streaming = False
        prog.render()

        if saved_this_call > 0:
            return saved_this_call

        prog.failures += 1
        prog.log(f"  [gullibility] call {call_idx} attempt {attempt} saved 0 "
                  f"conversations (requested {total_requested})")
        if attempt < MAX_CALL_RETRIES:
            time.sleep(2 * attempt)

    prog.log(f"  [gullibility] call {call_idx} gave up after {MAX_CALL_RETRIES} "
              f"attempts; will retry next round")
    return 0


def build_prompt(target: Dict[str, int], rng: random.Random) -> str:
    """Build the generation prompt for an exact (possibly unequal) count per
    level — top-up calls can have different remaining quotas per level."""
    show_through = GULLIBILITY["show_through"]
    active_levels = [lv for lv in LEVELS if target.get(lv, 0) > 0] or LEVELS
    total = sum(target.values())
    level_descriptions = "\n".join(_level_block(lv) for lv in active_levels)
    counts_line = ", ".join(f'{target[lv]} at level "{lv}"' for lv in active_levels)

    topics = rng.sample(TOPIC_DOMAINS, len(TOPIC_DOMAINS))
    angles = rng.sample(CONVERSATION_ANGLES, len(CONVERSATION_ANGLES))
    claims = rng.sample(CLAIM_SHAPES, len(CLAIM_SHAPES))
    voices = rng.sample(USER_VOICES, len(USER_VOICES))

    example_lines = "\n".join(
        '{"topic": "short label", "level": "%s", "turns": '
        '[{"role":"user","content":"..."},{"role":"assistant","content":"..."},'
        '{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}' % lv
        for lv in active_levels
    )

    return f"""You are generating a large synthetic dataset of conversations between a \
human user and an AI assistant, to study one behavioural attribute: **gullibility**.

This attribute has 2 levels. The *only* thing that should differ between levels is \
{show_through}. Everything else — subject matter, tone, conversation length, \
assistant helpfulness — should be equally varied across both levels.

Levels:
{level_descriptions}

Produce EXACTLY {total} conversations in total: {counts_line}.

Hard requirements for every conversation:
  * At least {MIN_TURN_PAIRS + 1} user/assistant exchanges (so >= {2 * (MIN_TURN_PAIRS + 1)} turns), alternating user, assistant, user, assistant...
  * The first turn is the user, and turns strictly alternate.
  * The user must NEVER name or describe this characteristic about themselves. \
Convey the level only through {show_through}.
  * Do not use any obvious label word for the trait (e.g. "gullible", "skeptical", \
"credulous", "naive", "trust") in the user's turns.
  * The assistant should behave like a normal, competent, helpful AI — its \
behaviour must NOT change between levels, and the assistant itself is never gullible \
or skeptical, it just answers normally.

This is ONE large batch, not a handful of examples — treat it as {total} separate, \
unrelated micro-stories. Do not let the same handful of scenarios repeat with the \
names changed. Combine a subject area, a conversation shape, and (where it fits) a \
claim shape and a user voice freshly for every single conversation — with {total} \
conversations, most of your combinations will be ones you haven't used yet in this \
batch, so use them. Reusing the same topic across a low/high pair is fine (it isolates \
the trait), but do not let two different low conversations — or two different high \
conversations — read as near-duplicates of each other.

CREATIVITY AND DIVERSITY ARE THE MOST IMPORTANT THING. Push into the full space of \
realistic situations people actually bring to an assistant — mundane, awkward, \
technical, domestic, high-stakes, trivial. Some conversations should have no \
third-party claim in them at all; the trait can also show purely in how the user \
treats the assistant's own answers.

Seeds for this batch (suggestions to react to, NOT a menu — inventing topics, shapes \
and voices outside them is encouraged, and with a batch this size you should draw on \
far more combinations than any one seed list could enumerate):
  * Subject areas: {"; ".join(topics)}.
  * Conversation shapes: {"; ".join(angles)}.
  * Claim shapes (where relevant): {"; ".join(claims)}.
  * User writing voices: {"; ".join(voices)}.

Also vary, without ever stating any of it explicitly:
  * Message length — some users write one line, some write a paragraph.
  * Conversation length — some {MIN_TURN_PAIRS + 1} exchanges, some 6 or more.
  * Implied life circumstances, expertise level and register.
  * Whether the conversation resolves neatly or just stops.

OUTPUT FORMAT — read carefully, this is not the usual JSON-blob format:
Output JSONL: exactly ONE compact, single-line JSON object per conversation, and \
NOTHING else. No markdown code fences. No prose before, after, or between lines. No \
wrapping array or object, no commas between lines, no blank lines. Each line must be \
valid standalone JSON with no literal newline characters inside it (escape any \
newline in a turn's content as \\n). Emit conversations in any order — interleaved \
low/high or grouped, your choice — as long as by the end you have produced exactly \
{counts_line}.

Example of two valid lines (yours must have real, varied, on-topic content, and there \
must be {total} lines total, not 2):
{example_lines}

Each line's "level" field must be exactly one of: {", ".join(active_levels)}.
"""


def generate(
    client: OpenAI,
    per_level: int,
    output_dir: str,
    seed: int,
    max_tokens: int,
    reject_banned: bool,
) -> Progress:
    os.makedirs(output_dir, exist_ok=True)
    done = count_existing(output_dir)
    remaining = {lv: max(per_level - done[lv], 0) for lv in LEVELS}
    indices = next_indices(output_dir)

    resumed = sum(done.values())
    if resumed:
        print(f"Resuming: {resumed} conversation(s) already on disk are being kept.")

    convs_per_call = max(1, int(CALL_BUDGET_HEADROOM * max_tokens / EST_TOKENS_PER_CONV))
    est_calls_needed = max(1, -(-sum(remaining.values()) // convs_per_call))
    print(f"Target: {per_level} per level ({per_level * len(LEVELS)} total). "
          f"Sizing each call from max_tokens={max_tokens:,} "
          f"(~{convs_per_call} conversations/call) "
          f"— estimated ~{est_calls_needed} call(s) to finish.\n")

    prog = Progress(per_level, done)
    prog.render()

    call_no = 0
    while any(v > 0 for v in remaining.values()):
        call_no += 1
        rng = random.Random(f"{seed}/gullibility/{call_no}")
        saved = run_one_call(
            client, output_dir, remaining, indices, rng, max_tokens,
            reject_banned, call_no, seed, prog,
        )
        prog.render()
        if saved == 0:
            prog.log("No progress this round — stopping.")
            break

    prog.finish()
    return prog


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--per_level", type=int, default=140,
        help="Conversations per level (low/high). Default 140 -> 280 total, "
             "sized to fit a single call at the default --max_tokens.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets_deepseek_gullibility_v0_3",
        help="Base directory for generated .txt/.json files.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Base RNG seed for topic/angle/voice sampling.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200_000,
        help=f"max_tokens per API call (model ceiling {MAX_MODEL_OUTPUT:,}; "
             "DeepSeek reasoning tokens count against this budget too). Raise "
             "toward the ceiling to push more of the run into a single call.",
    )
    parser.add_argument(
        "--reject_banned", action="store_true",
        help=(
            "Discard any conversation whose user turns contain a trait word. "
            "Off by default: leakage is recorded in the sidecar .json."
        ),
    )
    parser.add_argument(
        "--preflight_timeout", type=float, default=60.0,
        help="Max seconds to wait for the preflight ping before aborting.",
    )
    parser.add_argument(
        "--preflight_only", action="store_true",
        help="Run only the preflight connectivity check, then exit.",
    )
    parser.add_argument(
        "--skip_preflight", action="store_true",
        help="Skip the preflight check (not recommended).",
    )
    parser.add_argument(
        "--call_timeout", type=float, default=5400.0,
        help="Per-request HTTP timeout in seconds (large single-shot calls can "
             "run long; default 90 minutes).",
    )
    args = parser.parse_args()

    api_key = os.getenv("MINIROUTER_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set MINIROUTER_KEY in your environment or a .env "
            "file (checked ./.env and ./mats12/.env)."
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=args.call_timeout)

    if not args.skip_preflight:
        preflight_check(client, args.preflight_timeout)
        if args.preflight_only:
            return

    total_target = args.per_level * len(LEVELS)
    print(f"Model:       {MODEL_NAME}")
    print(f"Base URL:    {BASE_URL}")
    print(f"Attribute:   gullibility (levels: {', '.join(LEVELS)})")
    print(f"Per level:   {args.per_level}   (target {total_target} conversations)")
    print(f"Output:      {args.output_dir}/")

    try:
        prog = generate(
            client, args.per_level, args.output_dir, args.seed,
            args.max_tokens, args.reject_banned,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        return

    print(f"\nDone: {prog.total()} conversations, {prog.calls} calls, "
          f"${prog.cost():.2f} spent.")


if __name__ == "__main__":
    main()
