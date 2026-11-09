"""Find 20–60 second promo clips in timestamped microdrama screenplays.

Continuous clips only: one stretch from a single episode. Timestamps are taken
from the screenplay's own beat boundaries. A Gemini model can rewrite the hook,
cliffhanger, and reason, and can reject a weak clip. Without an API key the
same command still ranks clips from the screenplay text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


HEADER_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(
    r"Start: \((\d{1,2}):(\d{2})\.(\d{2})\) \| End: \((\d{1,2}):(\d{2})\.(\d{2})\)"
)
TECHNICAL = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}

COMMAND_RE = re.compile(
    r"^(take|let|don't|dont|stop|look|get|leave|listen|marry|say|tell)\b",
    re.IGNORECASE,
)
SPECTACLE_LINE_RE = re.compile(
    r"\b(marry|my hand|this ring|the ring|be my wife|my wife|i propose)\b",
    re.IGNORECASE,
)
HOOK_EVENT_RE = re.compile(
    r"\b(slap|slaps|slapped|strike|strikes|struck|sword|blade|dagger|knife|"
    r"blood|bleed|kneel|kneels|knee|propos|scream|screams|shriek|grab|grabs|"
    r"seiz|gun|shot|fires|kiss|kisses|betray|corpse|dead|dies|arrest|guard|"
    r"burst|draws|aims|choke|chokes|poison|stab|stabs)\w*\b",
    re.IGNORECASE,
)
STRONG_EMOTION = (
    "shock",
    "horrif",
    "furi",
    "outrag",
    "gasp",
    "betray",
    "scream",
    "command",
    "mock",
    "cruel",
    "threat",
    "desper",
    "accus",
    "cold",
    "vicious",
    "gloat",
    "devastat",
    "pierc",
    "trembl",
    "enrag",
    "menac",
    "savage",
    "panic",
    "lethal",
    "ice",
    "icy",
    "hard",
    "scorn",
    "callous",
    "broken",
    "raw",
    "vindict",
)
RESOLVED_RE = re.compile(
    r"\b(i forgive you|we made it|it is over|it's over|happily ever after|"
    r"the end|i do)\b",
    re.IGNORECASE,
)
EPISODE_FILE_RE = re.compile(r"episode-(\d+)", re.IGNORECASE)


@dataclass
class Beat:
    kind: str
    speaker: str
    emotion: str
    text: str
    start: float
    end: float
    start_label: str
    end_label: str


class DailyQuotaError(Exception):
    pass


@dataclass
class Episode:
    number: int
    label: str
    logline: str
    continuity: str
    source_file: str
    beats: list[Beat] = field(default_factory=list)

    @property
    def title_bits(self) -> str:
        return self.label or f"Episode {self.number}"


def parse_clock(minutes: str, seconds: str, fraction: str) -> tuple[float, str]:
    label = f"{int(minutes):02d}:{seconds}.{fraction}"
    value = int(minutes) * 60 + int(seconds) + int(fraction) / 100
    return value, label


def classify_header(header: str) -> tuple[str, str, str]:
    stem = header.split("(", 1)[0].strip()
    if stem in TECHNICAL:
        return TECHNICAL[stem], "", ""
    emotions = re.findall(r"\(([^)]*)\)", header)
    return "dialogue", stem, ", ".join(emotions)


def parse_episode(path: Path) -> Episode:
    match = EPISODE_FILE_RE.search(path.name)
    number = int(match.group(1)) if match else 0
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    label = ""
    logline = ""
    continuity = ""
    beats: list[Beat] = []
    index = 0
    while index < len(lines):
        raw = lines[index].strip()
        if raw.startswith("EPISODE NUMBER:"):
            label = raw.split(":", 1)[1].strip()
        elif raw.startswith("EPISODE LOGLINE:"):
            logline = raw.split(":", 1)[1].strip()
        elif raw.startswith("CONTINUITY NOTES:"):
            continuity = raw.split(":", 1)[1].strip()

        header_match = HEADER_RE.match(raw)
        if not header_match:
            index += 1
            continue

        time_match = None
        time_index = index + 1
        while time_index < len(lines) and time_index <= index + 4:
            time_match = TIME_RE.search(lines[time_index].strip())
            if time_match:
                break
            time_index += 1
        if not time_match:
            index += 1
            continue

        body: list[str] = []
        cursor = time_index + 1
        while cursor < len(lines):
            nxt = lines[cursor].strip()
            if (
                HEADER_RE.match(nxt)
                or nxt.startswith("SCENE ")
                or nxt.startswith("--- SCRIPT")
                or nxt.startswith("Scene heading")
                or nxt.startswith("Scene start")
                or nxt.startswith("Scene end")
            ):
                break
            if nxt and nxt.lower() != "actions and dialogue:":
                body.append(nxt.strip().strip('"'))
            cursor += 1

        start, start_label = parse_clock(*time_match.group(1, 2, 3))
        end, end_label = parse_clock(*time_match.group(4, 5, 6))
        kind, speaker, emotion = classify_header(header_match.group(1).strip())
        beats.append(
            Beat(
                kind=kind,
                speaker=speaker,
                emotion=emotion,
                text=" ".join(body).strip(),
                start=start,
                end=end,
                start_label=start_label,
                end_label=end_label,
            )
        )
        index = cursor

    beats.sort(key=lambda beat: (beat.start, beat.end))
    return Episode(
        number=number,
        label=label or f"Episode {number}",
        logline=logline,
        continuity=continuity,
        source_file=path.name,
        beats=beats,
    )


def emotion_is_strong(emotion: str) -> bool:
    lowered = emotion.lower()
    return any(token in lowered for token in STRONG_EMOTION)


def hook_strength(beat: Beat) -> float:
    if beat.kind == "dialogue":
        score = 0.34
        text = beat.text.strip()
        if "?" in text:
            score += 0.22
        if "!" in text:
            score += 0.1
        if COMMAND_RE.search(text):
            score += 0.14
        if len(text.split()) <= 12:
            score += 0.14
        if emotion_is_strong(beat.emotion):
            score += 0.24
        if text.startswith("..."):
            score -= 0.28
        if HOOK_EVENT_RE.search(text):
            score += 0.08
        return max(0.0, min(1.0, score))
    if beat.kind in {"action", "sound"} and HOOK_EVENT_RE.search(beat.text):
        return 0.62
    return 0.0


def cliff_strength(beat: Beat | None) -> float:
    if beat is None or beat.kind != "dialogue" or not beat.text.strip():
        return 0.12
    text = beat.text.strip()
    score = 0.18
    if text.endswith("?"):
        score += 0.42
    elif "?" in text:
        score += 0.28
    if text.endswith("...") or text.endswith("—") or text.endswith("-"):
        score += 0.16
    if emotion_is_strong(beat.emotion):
        score += 0.22
    if RESOLVED_RE.search(text):
        score -= 0.45
    if text.endswith(".") and "?" not in text and len(text.split()) > 22:
        score -= 0.08
    return max(0.0, min(1.0, score))


def duration_fit(duration: float) -> float:
    if duration < 20 or duration > 60:
        return 0.0
    if 28 <= duration <= 45:
        return 1.0
    if duration < 28:
        return 0.55 + 0.45 * ((duration - 20) / 8)
    return 0.55 + 0.45 * ((60 - duration) / 15)


def standalone_strength(lines: list[Beat]) -> float:
    if len(lines) < 2:
        return 0.05
    speakers = {beat.speaker for beat in lines if beat.speaker}
    score = 0.2
    if len(lines) >= 3:
        score += 0.22
    if len(lines) >= 6:
        score += 0.08
    if len(speakers) >= 2:
        score += 0.28
    if len(speakers) >= 3:
        score += 0.06
    first = lines[0].text.strip()
    if first.startswith("..."):
        score -= 0.22
    if len(first.split()) >= 4:
        score += 0.08
    return max(0.0, min(1.0, score))


def beats_between(episode: Episode, start: float, end: float, kind: str | None = None) -> list[Beat]:
    chosen = []
    for beat in episode.beats:
        if beat.end < start - 0.05 or beat.start > end + 0.05:
            continue
        if kind and beat.kind != kind:
            continue
        chosen.append(beat)
    return chosen


def trim(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_copy(
    dialogue: list[Beat],
    hook_beat: Beat,
    cliff_beat: Beat | None,
    first_spoken: Beat | None,
) -> tuple[str, str, str]:
    if hook_beat.kind == "dialogue":
        follow = next(
            (
                beat
                for beat in dialogue
                if beat.start > hook_beat.end - 0.05 and beat.start <= hook_beat.start + 3.0
            ),
            None,
        )
        short_open = len(hook_beat.text.split()) <= 3
        if short_open and follow is not None:
            hook = (
                f'{hook_beat.speaker} ({hook_beat.emotion or "spoken"}) '
                f'in the opening second: "{trim(hook_beat.text, 80)}". '
                f'Within three seconds, {follow.speaker} answers: "{trim(follow.text, 120)}".'
            )
        else:
            hook = (
                f'{hook_beat.speaker} ({hook_beat.emotion or "spoken"}) '
                f'in the opening second: "{trim(hook_beat.text, 140)}".'
            )
        spoken_open = hook_beat
    elif first_spoken is not None:
        hook = (
            f"Opens as {trim(hook_beat.text, 110)} "
            f'Within the first three seconds, {first_spoken.speaker} says "{trim(first_spoken.text, 120)}".'
        )
        spoken_open = first_spoken
    else:
        hook = f"Opens on {trim(hook_beat.text, 140)}."
        spoken_open = None

    if cliff_beat and cliff_beat.kind == "dialogue":
        cliff = (
            f'Cuts on {cliff_beat.speaker} ({cliff_beat.emotion or "spoken"}): '
            f'"{trim(cliff_beat.text, 140)}".'
        )
        ending = cliff_beat.text.strip()
    else:
        cliff = "Cuts before the exchange is answered."
        ending = ""

    counts: dict[str, int] = {}
    for beat in dialogue:
        if not beat.speaker or "OFF-SCREEN" in beat.speaker.upper():
            continue
        counts[beat.speaker] = counts.get(beat.speaker, 0) + 1
    names = sorted(counts, key=lambda name: counts[name], reverse=True)
    if len(names) >= 2:
        who = f"{names[0]} against {names[1]}"
    elif names:
        who = names[0]
    else:
        who = "the people on screen"
    if ending.endswith("?"):
        payoff = "the clip ends on a question nobody answers"
    elif cliff_beat and emotion_is_strong(cliff_beat.emotion):
        payoff = "the clip ends on a line that still needs an answer"
    else:
        payoff = "the clip ends before the moment is settled"
    opening_quote = trim(spoken_open.text, 80) if spoken_open else trim(hook_beat.text, 80)
    why = (
        f'A stranger meets {who} already in conflict ("{opening_quote}"), and {payoff}.'
    )
    return hook, cliff, why


def score_window(episode: Episode, start_beat: Beat, end_beat: Beat) -> dict | None:
    start = start_beat.start
    end = end_beat.end
    duration = end - start
    if duration < 19.95 or duration > 60.05:
        return None
    if end_beat.kind != "dialogue":
        return None

    dialogue = [
        beat
        for beat in beats_between(episode, start, end, "dialogue")
        if beat.start >= start - 0.05
    ]
    if len(dialogue) < 2:
        return None

    early_spoken = [beat for beat in dialogue if beat.start <= start + 3.0]
    first_spoken = early_spoken[0] if early_spoken else None
    hook = hook_strength(start_beat)
    if start_beat.kind != "dialogue":
        if first_spoken is None:
            hook = max(0.0, hook - 0.15)
        else:
            hook = max(hook, hook_strength(first_spoken))
    if start_beat.text.strip().startswith("..."):
        hook = max(0.0, hook - 0.2)

    cliff = cliff_strength(end_beat)
    spectacle = 0.0
    if SPECTACLE_LINE_RE.search(end_beat.text) or HOOK_EVENT_RE.search(end_beat.text):
        spectacle = 1.0
    else:
        for beat in beats_between(episode, end - 3.0, end):
            if beat.kind == "camera" or not beat.text.strip():
                continue
            if beat.start >= end - 0.05 or beat.end < end - 3.05:
                continue
            if HOOK_EVENT_RE.search(beat.text) or SPECTACLE_LINE_RE.search(beat.text):
                spectacle = 1.0
                break
    if spectacle:
        cliff = max(cliff, 0.62)
    alone = standalone_strength(dialogue)
    fit = duration_fit(duration)
    score = 0.34 * hook + 0.26 * cliff + 0.16 * alone + 0.08 * fit + 0.16 * spectacle
    hook_text, cliff_text, why = build_copy(dialogue, start_beat, end_beat, first_spoken)
    flags = []
    if hook < 0.5:
        flags.append("Opening is soft for a scrolling viewer.")
    if cliff < 0.4:
        flags.append("The cut sounds finished rather than unresolved.")
    tender = ("gentle", "soft", "tender", "warm", "loving", "protective", "sincere")
    if any(token in end_beat.emotion.lower() for token in tender) and not spectacle:
        flags.append("The last line is emotionally soft, so the cut can feel like a pause.")
    if alone < 0.45:
        flags.append("A new viewer gets too little context.")
    if duration < 22 or duration > 55:
        flags.append("Length sits at the edge of the 20–60 second range.")
    if dialogue[0].text.strip().startswith("..."):
        flags.append("Opens in the middle of a sentence.")
    if len({beat.speaker for beat in dialogue}) < 2:
        flags.append("Only one speaker, so the clip can feel like a fragment.")

    transcript = []
    for beat in beats_between(episode, start, end):
        if beat.kind == "camera" or beat.start < start - 0.05:
            continue
        if beat.kind == "dialogue":
            transcript.append(
                {
                    "kind": "dialogue",
                    "speaker": beat.speaker,
                    "emotion": beat.emotion,
                    "start": beat.start_label,
                    "end": beat.end_label,
                    "line": beat.text,
                }
            )
        elif beat.kind in {"action", "sound"} and beat.text.strip():
            transcript.append(
                {
                    "kind": beat.kind,
                    "speaker": beat.kind.upper(),
                    "emotion": "",
                    "start": beat.start_label,
                    "end": beat.end_label,
                    "line": beat.text,
                }
            )
    return {
        "episode": episode.number,
        "episode_label": episode.label,
        "source_file": episode.source_file,
        "start": start_beat.start_label,
        "end": end_beat.end_label,
        "start_seconds": round(start, 2),
        "end_seconds": round(end, 2),
        "duration_seconds": round(duration, 2),
        "segments": [
            {
                "episode": episode.number,
                "start": start_beat.start_label,
                "end": end_beat.end_label,
            }
        ],
        "hook": hook_text,
        "cliffhanger": cliff_text,
        "why_it_works": why,
        "score": round(score, 4),
        "components": {
            "hook": round(hook, 4),
            "cliffhanger": round(cliff, 4),
            "standalone": round(alone, 4),
            "duration_fit": round(fit, 4),
            "spectacle": spectacle,
        },
        "flags": flags,
        "transcript": transcript,
        "judge": "heuristic",
    }


def propose(episode: Episode) -> list[dict]:
    anchors = []
    for beat in episode.beats:
        if beat.kind == "dialogue" and beat.text.strip():
            anchors.append(beat)
        elif beat.kind in {"action", "sound"} and hook_strength(beat) >= 0.5:
            anchors.append(beat)

    dialogue_ends = [beat for beat in episode.beats if beat.kind == "dialogue" and beat.text.strip()]
    best_by_start: dict[str, dict] = {}
    for anchor in anchors:
        if hook_strength(anchor) < 0.45 and anchor.kind == "dialogue":
            continue
        if anchor.kind != "dialogue" and hook_strength(anchor) < 0.5:
            continue
        for ending in dialogue_ends:
            if ending.end <= anchor.start:
                continue
            window = score_window(episode, anchor, ending)
            if not window:
                continue
            # Keep the best ending for each opening timestamp.
            current = best_by_start.get(window["start"])
            if current is None or window["score"] > current["score"]:
                best_by_start[window["start"]] = window
    ranked = sorted(best_by_start.values(), key=lambda item: item["score"], reverse=True)
    return ranked


def overlap_seconds(left: dict, right: dict) -> float:
    start = max(left["start_seconds"], right["start_seconds"])
    end = min(left["end_seconds"], right["end_seconds"])
    if end <= start:
        return 0.0
    return end - start


def suppress_overlaps(clips: list[dict]) -> list[dict]:
    ordered = sorted(clips, key=lambda clip: clip["score"], reverse=True)
    kept: list[dict] = []
    for clip in ordered:
        if any(
            other["episode"] == clip["episode"] and overlap_seconds(clip, other) >= 0.5
            for other in kept
        ):
            continue
        kept.append(clip)
    return kept


def select_clips(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    chosen: list[dict] = []
    rejected: list[dict] = []
    for candidate in candidates:
        if any(overlap_seconds(candidate, kept) >= 0.5 for kept in chosen):
            candidate = dict(candidate)
            candidate["rejected_because"] = "Overlaps a higher-scoring clip in the same episode."
            rejected.append(candidate)
            continue
        if len(chosen) >= 2:
            candidate = dict(candidate)
            candidate["rejected_because"] = "Episode already has two non-overlapping promos."
            rejected.append(candidate)
            continue
        if chosen and candidate["score"] < 0.64:
            candidate = dict(candidate)
            candidate["rejected_because"] = "Second clip in the episode is below the extra-promo bar."
            rejected.append(candidate)
            continue
        if not chosen and candidate["score"] < 0.5:
            weak = dict(candidate)
            weak["flags"] = list(weak["flags"]) + [
                "Best clip in the episode is still below the posting bar."
            ]
            weak["post_tier"] = "weak"
            chosen.append(weak)
            continue
        picked = dict(candidate)
        picked["post_tier"] = "candidate"
        chosen.append(picked)
    return chosen, rejected[:12]


def format_moment(row: dict) -> str:
    kind = row.get("kind", "dialogue")
    if kind == "dialogue":
        return f'{row["start"]} {row["speaker"]}: {trim(row["line"], 160)}'
    return f'{row["start"]} {kind.upper()}: {trim(row["line"], 160)}'


def transcript_for_model(clip: dict) -> str:
    rows = clip["transcript"]
    if len(rows) <= 16:
        chosen = rows
    else:
        chosen = rows[:10] + rows[-4:]
    return "\n".join(format_moment(row) for row in chosen)


def extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def ask_gemini(api_key: str, model_name: str, prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    generation_config = {
        "temperature": 0.2,
        "maxOutputTokens": 2048,
        "responseMimeType": "application/json",
    }
    if "lite" not in model_name:
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.load(response)
    parts = payload["candidates"][0]["content"].get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    if not texts:
        raise ValueError("Gemini returned no text.")
    return "\n".join(texts)


def judge_episode_with_gemini(
    episode: Episode,
    pool: list[dict],
    model_name: str,
    api_key: str,
) -> tuple[list[dict], str | None]:
    payload = []
    for index, clip in enumerate(pool[:4], start=1):
        payload.append(
            {
                "id": f"c{index}",
                "start": clip["start"],
                "end": clip["end"],
                "duration_seconds": clip["duration_seconds"],
                "heuristic_flags": clip["flags"],
                "transcript": transcript_for_model(clip),
            }
        )
    prompt = f"""You select promo clips for a vertical microdrama.

A promo is one continuous 20–60 second stretch from the episode. It must:
- hook a scrolling viewer in the first 3 seconds
- end on a cliffhanger, cut while something is unresolved
- make sense to someone who has never seen the show

Do not change start or end. Do not invent events, names, or lines that are not in the transcript.
The episode logline is background only. Judge each clip from its transcript.

Episode: {episode.label}
Logline: {episode.logline}

Candidates:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "picks": [
    {{
      "id": "c1",
      "keep": true,
      "hook": "one sentence",
      "cliffhanger": "one sentence",
      "why_it_works": "one sentence for a viewer who has not seen the show",
      "problem": ""
    }}
  ]
}}
Keep at most 2. Put the better promo first. If a clip fails, set keep to false and explain the problem in one sentence.
"""
    parsed = None
    last_error = "no response"
    for attempt in range(2):
        try:
            parsed = extract_json_object(ask_gemini(api_key, model_name, prompt))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and "per day" in detail.lower():
                raise DailyQuotaError(
                    "Free-tier daily request cap reached. This episode kept the screenplay score."
                ) from exc
            last_error = f"HTTP {exc.code}: {detail[:180]}"
            if exc.code not in {429, 500, 503}:
                break
            time.sleep(8 + attempt * 8)
        except Exception as exc:  # noqa: BLE001 - retry timeouts and bad JSON
            last_error = str(exc)
            time.sleep(4)
    if parsed is None:
        return select_clips(pool)[0], f"Gemini failed on {episode.label}: {last_error}"

    by_id = {f"c{index}": clip for index, clip in enumerate(pool[:4], start=1)}
    kept: list[dict] = []
    notes = []
    for pick in parsed.get("picks", []):
        clip = by_id.get(str(pick.get("id", "")))
        if clip is None:
            notes.append(f"Ignored unknown id {pick.get('id')}.")
            continue
        updated = dict(clip)
        if pick.get("keep") is True:
            updated["hook"] = str(pick.get("hook") or clip["hook"]).strip()
            updated["cliffhanger"] = str(pick.get("cliffhanger") or clip["cliffhanger"]).strip()
            updated["why_it_works"] = str(pick.get("why_it_works") or clip["why_it_works"]).strip()
            updated["judge"] = "gemini"
            updated["post_tier"] = "candidate"
            problem = str(pick.get("problem") or "").strip()
            if problem:
                updated["flags"] = list(updated["flags"]) + [problem]
            kept.append(updated)
        else:
            problem = str(pick.get("problem") or "Rejected by the model.").strip()
            notes.append(f'{clip["start"]}-{clip["end"]}: {problem}')
    if not kept:
        fallback, _ = select_clips(pool)
        for clip in fallback[:1]:
            clip = dict(clip)
            clip["flags"] = list(clip["flags"]) + [
                "Gemini rejected every candidate in this episode; the heuristic clip was kept and should be checked by hand."
            ]
            clip["post_tier"] = "weak"
            clip["judge"] = "heuristic-after-gemini-reject"
            kept.append(clip)
        notes.append("Model kept nothing; one heuristic clip was retained.")
    return kept[:2], ("; ".join(notes) or None)


def rank_reason(promo: dict, rank: int) -> str:
    parts = promo["components"]
    if rank == 1:
        lead = "Post this first."
    else:
        lead = f"Rank {rank}."
    return (
        f"{lead} Hook {parts['hook']:.2f}, cliffhanger {parts['cliffhanger']:.2f}, "
        f"standalone {parts['standalone']:.2f}. "
        "Higher means a faster opening, a less settled ending, and enough of a fight "
        "for a new viewer."
    )


def discover_episodes(input_path: Path) -> tuple[list[Path], str]:
    files = sorted(input_path.glob("episode-*.txt"))
    if files:
        return files, input_path.name
    groups = []
    if input_path.is_dir():
        for sub in sorted(path for path in input_path.iterdir() if path.is_dir()):
            found = sorted(sub.glob("episode-*.txt"))
            if found:
                groups.append((sub, found))
    if len(groups) == 1:
        return groups[0][1], groups[0][0].name
    if len(groups) > 1:
        names = ", ".join(sub.name for sub, _ in groups)
        raise SystemExit(f"More than one show folder found ({names}). Pass one show folder.")
    raise SystemExit(f"No episode-*.txt files found in {input_path}.")


def write_markdown(path: Path, show: str, promos: list[dict], judge_mode: str) -> None:
    lines = [
        f"# Promo ranking — {show}",
        "",
        f"Judge: {judge_mode}. Continuous clips only, one episode each, 20–60 seconds.",
        "",
        "Post from the top of this list. Each promo is one segment.",
        "",
    ]
    if not promos:
        lines.append("No promo cleared the posting bar.")
    for promo in promos:
        segment = promo["segments"][0]
        lines.extend(
            [
                f"## {promo['rank']}. {promo['episode_label']} — {segment['start']} to {segment['end']} ({promo['duration_seconds']}s)",
                "",
                f"- Episode: {promo['episode']}",
                f"- Segment: {segment['start']} → {segment['end']}",
                f"- Score: {promo['score']}",
                f"- Tier: {promo.get('post_tier', 'candidate')}",
                f"- Hook: {promo['hook']}",
                f"- Cliffhanger: {promo['cliffhanger']}",
                f"- Why it works: {promo['why_it_works']}",
                f"- Why this rank: {promo['rank_reason']}",
            ]
        )
        if promo["flags"]:
            lines.append("- Check this pick: " + " ".join(promo["flags"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit.strip() or None
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find ranked 20–60 second promo clips in a folder of episode screenplays."
    )
    parser.add_argument("--input", required=True, help="Show folder, or a folder that contains one show folder.")
    parser.add_argument("--output", required=True, help="Folder for unedited JSON and Markdown results.")
    parser.add_argument(
        "--judge",
        choices=("auto", "heuristic", "gemini"),
        default="auto",
        help="auto uses Gemini when an API key is set, otherwise the screenplay scorer.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.6-flash",
        help="Gemini model name. gemini-3.8-flash was refusing traffic when this was written.",
    )
    parser.add_argument("--api-key", default=None, help="Gemini API key. Prefer GEMINI_API_KEY if you omit this.")
    parser.add_argument("--limit", type=int, default=0, help="Read only the first N episodes. 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.4, help="Seconds to wait between Gemini calls.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep episodes already judged. Call Gemini only where the last run recorded a failure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    files, show = discover_episodes(input_path)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    api_key = load_api_key(args.api_key)
    use_gemini = args.judge == "gemini" or (args.judge == "auto" and api_key)
    if args.judge == "gemini" and not api_key:
        raise SystemExit(
            "Gemini was requested, but no API key was found. Set GEMINI_API_KEY or pass --api-key. "
            "A Gemini chat subscription is separate from an API key."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    episode_dir = output_path / "episodes"
    episode_dir.mkdir(exist_ok=True)

    if use_gemini and args.resume:
        judge_mode = f"Gemini (earlier episodes on gemini-3.6-flash, then {args.model})"
    elif use_gemini:
        judge_mode = f"gemini:{args.model}"
    else:
        judge_mode = "heuristic"
    selected_all: list[dict] = []
    weak: list[dict] = []
    model_notes: list[str] = []
    episode_count = 0

    for index, path in enumerate(files, start=1):
        episode = parse_episode(path)
        episode_count += 1
        episode_file = episode_dir / f"episode-{episode.number:03d}.json"
        prior = None
        if args.resume and episode_file.exists():
            prior = json.loads(episode_file.read_text(encoding="utf-8"))
        already_judged = prior is not None and "Gemini failed" not in (prior.get("model_note") or "")
        if already_judged:
            chosen = prior["selected"]
            rejected = prior.get("rejected_preview") or []
            note = prior.get("model_note")
            print(f"[{index}/{len(files)}] kept {episode.label}", flush=True)
        else:
            candidates = propose(episode)
            chosen, rejected = select_clips(candidates)
            note = None
        if use_gemini and not already_judged and candidates:
            print(f"[{index}/{len(files)}] Gemini {episode.label}", flush=True)
            try:
                chosen, note = judge_episode_with_gemini(
                    episode, candidates[:4], args.model, api_key
                )
            except DailyQuotaError as exc:
                note = str(exc)
                use_gemini = False
                print(f"Stopped calling Gemini: {note}", flush=True)
            if args.sleep and use_gemini:
                time.sleep(args.sleep)
        elif not already_judged:
            print(
                f"[{index}/{len(files)}] {episode.label}: {len(chosen)} promo(s) from {len(candidates)} openings",
                flush=True,
            )
        if note:
            model_notes.append(f"{episode.label}: {note}")

        for clip in chosen:
            if clip.get("post_tier") == "weak" or clip["score"] < 0.5:
                clip["post_tier"] = "weak"
                weak.append(clip)
            else:
                clip["post_tier"] = "candidate"
                selected_all.append(clip)

        if not already_judged:
            episode_payload = {
                "show": show,
                "episode": episode.number,
                "episode_label": episode.label,
                "source_file": episode.source_file,
                "logline": episode.logline,
                "judge": judge_mode,
                "beat_count": len(episode.beats),
                "selected": chosen,
                "rejected_preview": rejected,
                "model_note": note,
            }
            episode_file.write_text(
                json.dumps(episode_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    selected_all = suppress_overlaps(selected_all)
    selected_all.sort(key=lambda clip: clip["score"], reverse=True)
    for rank, clip in enumerate(selected_all, start=1):
        clip["rank"] = rank
        clip["rank_reason"] = rank_reason(clip, rank)
        clip["show"] = show
        if rank <= 10:
            clip["post_tier"] = "post_first"
        else:
            clip["post_tier"] = "reserve"

    for clip in weak:
        clip["rank"] = None
        clip["rank_reason"] = "Held out of the posting list because the score or the judge flagged it."
        clip["show"] = show

    series = {
        "show": show,
        "judge": judge_mode,
        "episode_count": episode_count,
        "promo_count": len(selected_all),
        "assumptions": [
            "A promo is one continuous stretch from a single episode.",
            "Start and end are screenplay beat boundaries, not frame-accurate video cuts.",
            "The clip must run 20 to 60 seconds.",
            "The opening beat should land inside the first 3 seconds.",
            "Camera direction alone is not treated as a hook.",
            "At most two non-overlapping promos are kept from one episode.",
        ],
        "promos": selected_all,
    }
    (output_path / "series_ranking.json").write_text(
        json.dumps(series, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(output_path / "series_ranking.md", show, selected_all, judge_mode)

    quality = {
        "show": show,
        "judge": judge_mode,
        "where_the_tool_picks_badly": [
            {
                "episode": clip["episode"],
                "start": clip["start"],
                "end": clip["end"],
                "score": clip["score"],
                "flags": clip["flags"],
                "hook": clip["hook"],
                "cliffhanger": clip["cliffhanger"],
            }
            for clip in selected_all
            if clip["flags"]
        ],
        "held_out_weak_clips": [
            {
                "episode": clip["episode"],
                "start": clip["start"],
                "end": clip["end"],
                "score": clip["score"],
                "flags": clip["flags"],
            }
            for clip in weak
        ],
        "model_notes": model_notes,
        "systematic_limits": [
            "The tool reads the screenplay, so it cannot tell whether a described action is actually visible in the first three seconds of picture.",
            "A finished insult can score as a cliffhanger when the emotion tag is strong, even if the moment feels closed on screen.",
            "Clips late in an episode are missed when fewer than 20 seconds remain after the hook.",
            "Joined moments and cross-episode promos are out of scope for this run.",
            "Standalone sense is estimated from speaker count and the opening line, not from a viewer test.",
        ],
    }
    (output_path / "quality_notes.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {len(selected_all)} ranked promos and {len(weak)} weak clips to {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
