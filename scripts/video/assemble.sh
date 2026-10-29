#!/usr/bin/env bash
# Assemble the Effector Lab video from the clips of scripts/video/capture.mjs.
#
#   scripts/video/assemble.sh [--clips DIR] [--out DIR] [--shots FILE] [--voiceover FILE] [--vo-start S]
#                             [--flag NAME]...
#
#   --clips DIR      the capture folder (clips.json and <id>.webm); default $RTT_VIDEO_DIR/clips
#   --out DIR        where the cut, the thumbnail and the check stills go; default $RTT_VIDEO_DIR
#                    ($RTT_VIDEO_DIR defaults to ~/rtt-video; both must be outside the repository)
#   --shots FILE     the shot list; default scripts/video/shots.json
#   --voiceover F    voiceover.wav, .m4a or .mp3. Without it: a silent timing preview with the
#                    whole voice-over text as lower-third captions. With it: the final cut, the
#                    voice-over laid over the picture (loudness-normalised to -14 LUFS) and only
#                    the short captions for key numbers.
#   --vo-start S     the second of the recording where the video's 0:00 falls (default 0), to drop
#                    a count-in
#   --flag NAME      turn on a flagged shot (scripts/video/shots.mjs); the clips and the voice-over
#                    must have been made with the same flags (clips.json and voiceover.json say so)
#
# Output (H.264 High, 1920x1080, 30 fps, yuv420p, AAC 48 kHz stereo, faststart):
#   effector-lab-timing-preview.mp4 or effector-lab-final.mp4, at most 2:59
#   effector-lab-thumbnail.jpg (1280x720) and stills/ (frames sampled from the cut, with a
#   blank-frame check)
#
# Before any encoding it checks the shot list with scripts/video/shots.mjs: shots back to back from
# 0:00, the total at or under max_duration_s, every voice-over line present in
# docs/VIDEO_SCRIPT.md, no vendor brand token (the repository's lists in
# src/rtt/effectors/types.py and src/rtt/library/brands.py) in any caption, tag, voice-over line or
# the end card, no other company name burned in outside the end card's credit (the sponsor terms
# NVIDIA, Nemotron, Nebius and Token Factory are allowed on screen), and, for the final cut, no
# {{pending: ...}} marker anywhere.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
video_dir="${RTT_VIDEO_DIR:-$HOME/rtt-video}"
clips="$video_dir/clips"
out="$video_dir"
shots="$here/shots.json"
voiceover=""
vo_start="0"
flags=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clips) clips="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --shots) shots="$2"; shift 2 ;;
    --voiceover) voiceover="$2"; shift 2 ;;
    --vo-start) vo_start="$2"; shift 2 ;;
    --flag) flags+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "assemble: unknown argument $1" >&2; exit 2 ;;
  esac
done

for tool in ffmpeg ffprobe node fc-match; do
  command -v "$tool" >/dev/null || { echo "assemble: $tool is required" >&2; exit 2; }
done
mkdir -p "$out"
out="$(cd "$out" && pwd)"
case "$out/" in "$repo"/*) echo "assemble: --out must be outside the repository" >&2; exit 2 ;; esac
[[ -f "$clips/clips.json" ]] || { echo "assemble: no $clips/clips.json (run scripts/video/capture.mjs first)" >&2; exit 2; }
if [[ -n "$voiceover" ]]; then
  [[ -f "$voiceover" ]] || { echo "assemble: no voice-over file $voiceover" >&2; exit 2; }
  case "${voiceover,,}" in *.wav|*.m4a|*.mp3) ;; *) echo "assemble: the voice-over must be .wav, .m4a or .mp3" >&2; exit 2 ;; esac
  mode="final"
else
  mode="preview"
fi

work="$out/.work-$mode"
rm -rf "$work"
mkdir -p "$work/seg"
font_file="$(fc-match -f '%{file}' 'DejaVu Sans:bold')"
font_name="$(fc-match -f '%{family[0]}' 'DejaVu Sans')"

# ---------------------------------------------------------------- plan: checks, segments, captions
node --input-type=module - "$repo" "$shots" "$clips/clips.json" "$work" "$mode" "$font_name" "$voiceover" "${flags[@]}" <<'NODE'
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [repo, shotsFile, clipsFile, work, mode, fontName, voiceover, ...on] = process.argv.slice(2);
const { resolveShots, checkShots } = await import(pathToFileURL(path.join(repo, "scripts", "video", "shots.mjs")).href);
const raw = JSON.parse(readFileSync(shotsFile, "utf8"));
const spec = resolveShots(raw, on);
const clipsDoc = JSON.parse(readFileSync(clipsFile, "utf8"));
const clips = Object.fromEntries(clipsDoc.clips.map((c) => [c.id, c]));
const flagsOn = Object.entries(spec.flags).filter(([, v]) => v).map(([k]) => k).sort();

// 1. the shot list's own rules (timing, script, brands, names on screen; pending markers when final)
const { problems, total_s: t } = checkShots(spec, { repo, final: mode === "final", raw });

// 2. every shot has its clip, recorded with the same flags
for (const s of spec.shots) {
  if (!clips[s.id]) problems.push(`${s.id} has no clip in ${clipsFile}`);
  else if (!existsSync(path.join(path.dirname(clipsFile), clips[s.id].file))) problems.push(`${s.id}: ${clips[s.id].file} is missing`);
}
const same = (a) => JSON.stringify([...(a ?? [])].sort()) === JSON.stringify(flagsOn);
if (clipsDoc.flags !== undefined && !same(clipsDoc.flags)) problems.push(`the clips were recorded with flags [${clipsDoc.flags}], this cut has [${flagsOn}]`);

// 3. the voice-over was made from the same shot list and flags (voiceover.py writes voiceover.json beside it)
if (voiceover) {
  const side = voiceover.replace(/\.(wav|m4a|mp3)$/i, ".json");
  if (existsSync(side)) {
    const vo = JSON.parse(readFileSync(side, "utf8"));
    if (!same(vo.flags)) problems.push(`the voice-over was made with flags [${vo.flags}], this cut has [${flagsOn}]`);
    const ids = spec.shots.map((s) => s.id).join(",");
    if ((vo.lines ?? []).map((l) => l.id).join(",") !== ids) problems.push(`the voice-over's lines are not this shot list's (${side})`);
    if (vo.draft) problems.push(`the voice-over is a draft (pending markers spoken as a stand-in): re-run voiceover.py once every marker is filled`);
  }
}

if (problems.length) {
  console.error("assemble: the shot list does not pass:\n  " + problems.join("\n  "));
  process.exit(1);
}

// 4. segments: clip, trim, duration
writeFileSync(
  path.join(work, "segments.tsv"),
  spec.shots.map((s) => [s.id, path.join(path.dirname(clipsFile), clips[s.id].file), clips[s.id].trim_s, s.duration].join("\t")).join("\n") + "\n",
);

// 5. captions as ASS: a tag per shot (top left), the caption (lower third), a preview mark
const ts = (sec) => {
  const cs = Math.max(0, Math.round(sec * 100));
  const h = Math.floor(cs / 360000), m = Math.floor((cs % 360000) / 6000), s = Math.floor((cs % 6000) / 100), c = cs % 100;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(c).padStart(2, "0")}`;
};
const esc = (x) => x.replace(/\\/g, "\\\\").replace(/[{}]/g, "").replace(/\n/g, "\\N");
// Box alpha in hex (00 opaque): the tag and the key captions sit on an opaque box, so the page
// under them (the site's logo, body text) never shows through.
const style = (name, size, align, mL, mR, mV, bold, alpha = "30") =>
  `Style: ${name},${fontName},${size},&H00FFFFFF,&H00FFFFFF,&H${alpha}141414,&H${alpha}141414,${bold ? -1 : 0},0,0,0,100,100,0,0,3,${Math.round(size / 3)},0,${align},${mL},${mR},${mV},1`;
const events = [];
for (const s of spec.shots) {
  const a = s.start, b = s.start + s.duration;
  if (s.tag) events.push(`Dialogue: 1,${ts(a)},${ts(b)},Tag,,0,0,0,,${esc(spec.labels[s.tag])}`);
  const text = mode === "preview" ? s.voiceover : s.caption;
  if (text) events.push(`Dialogue: 0,${ts(a + 0.25)},${ts(b - 0.25)},${mode === "preview" ? "Caption" : "Key"},,0,0,0,,${esc(text)}`);
}
if (mode === "preview") events.push(`Dialogue: 2,${ts(0)},${ts(t)},Mark,,0,0,0,,TIMING PREVIEW · no voice-over · captions are the script`);
writeFileSync(
  path.join(work, "captions.ass"),
  [
    "[Script Info]", "ScriptType: v4.00+", `PlayResX: ${spec.width}`, `PlayResY: ${spec.height}`, "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
    "[V4+ Styles]",
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    style("Caption", 40, 2, 200, 200, 56, false),
    style("Key", 46, 2, 200, 200, 64, true, "00"),
    style("Tag", 24, 7, 36, 36, 30, false, "00"),
    style("Mark", 24, 9, 36, 36, 30, true),
    "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text", ...events, "",
  ].join("\n"),
);

// 6. the numbers the shell needs
const thumbShot = spec.shots.find((s) => s.id === spec.thumbnail.shot);
writeFileSync(
  path.join(work, "plan.env"),
  [
    `TOTAL=${t}`, `W=${spec.width}`, `H=${spec.height}`, `FPS=${spec.fps}`, `MAX=${spec.max_duration_s}`,
    `THUMB_AT=${thumbShot.start + spec.thumbnail.at_s}`,
    `THUMB_TITLE=${JSON.stringify(spec.thumbnail.title)}`, `THUMB_SUB=${JSON.stringify(spec.thumbnail.subtitle)}`,
  ].join("\n") + "\n",
);
console.log(`plan: ${spec.shots.length} shots, ${t} s, mode ${mode}, flags [${flagsOn}]; checks passed (timing, script, brands, names${mode === "final" ? ", pending markers" : ""})`);
NODE

# shellcheck disable=SC1091
source "$work/plan.env"

# ---------------------------------------------------------------- segments at 1920x1080, 30 fps
list="$work/concat.txt"
: >"$list"
while IFS=$'\t' read -r id file trim dur; do
  ffmpeg -nostdin -v error -y -ss "$trim" -i "$file" -t "$dur" \
    -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=0xf4f3ee,fps=${FPS},tpad=stop_mode=clone:stop_duration=${dur},format=yuv420p" \
    -t "$dur" -an -c:v libx264 -preset veryfast -crf 16 -r "$FPS" "$work/seg/$id.mp4"
  echo "file 'seg/$id.mp4'" >>"$list"
done <"$work/segments.tsv"
(cd "$work" && ffmpeg -v error -y -f concat -safe 0 -i concat.txt -c copy raw.mp4)

# ---------------------------------------------------------------- captions, audio, encode
name="effector-lab-timing-preview"
[[ "$mode" == "final" ]] && name="effector-lab-final"
target="$out/$name.mp4"
vf="ass=$work/captions.ass"
common=(-c:v libx264 -preset medium -crf 18 -profile:v high -pix_fmt yuv420p -r "$FPS" -g $((FPS * 2))
  -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart -t "$TOTAL")
if [[ "$mode" == "preview" ]]; then
  ffmpeg -v error -y -i "$work/raw.mp4" -f lavfi -i "anullsrc=r=48000:cl=stereo" \
    -vf "$vf" -map 0:v -map 1:a "${common[@]}" "$target"
else
  vo_len="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$voiceover")"
  if awk -v l="$vo_len" -v s="$vo_start" -v t="$TOTAL" 'BEGIN { exit !((l - s) > t + 0.5) }'; then
    echo "assemble: the voice-over runs $(awk -v l="$vo_len" -v s="$vo_start" 'BEGIN{printf "%.1f", l-s}') s after --vo-start; the cut is $TOTAL s. Re-record, or trim it." >&2
    exit 1
  fi
  ffmpeg -v error -y -i "$work/raw.mp4" -ss "$vo_start" -i "$voiceover" \
    -filter_complex "[0:v]$vf[v];[1:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,apad[a]" \
    -map "[v]" -map "[a]" "${common[@]}" "$target"
fi

# ---------------------------------------------------------------- thumbnail (from the uncaptioned cut)
thumb="$out/effector-lab-thumbnail.jpg"
esc_dt() { printf '%s' "$1" | sed -e "s/\\\\/\\\\\\\\/g" -e "s/:/\\\\:/g" -e "s/'/\\\\'/g" -e "s/%/\\\\%/g"; }
ffmpeg -v error -y -ss "$THUMB_AT" -i "$work/raw.mp4" -frames:v 1 \
  -vf "scale=1280:720:flags=lanczos,drawbox=x=0:y=ih-200:w=iw:h=200:color=0x141414@0.82:t=fill,drawtext=fontfile=$font_file:text='$(esc_dt "$THUMB_TITLE")':fontcolor=white:fontsize=72:x=48:y=h-178,drawtext=fontfile=$font_file:text='$(esc_dt "$THUMB_SUB")':fontcolor=0xdfe3ff:fontsize=38:x=50:y=h-84" \
  -q:v 2 "$thumb"

# ---------------------------------------------------------------- check: duration, size, frames not blank
rm -rf "$out/stills"
mkdir -p "$out/stills"
dur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$target")"
read -r vw vh vfps vcodec < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,codec_name -of csv=p=0 "$target" | awk -F, '{print $2, $3, $4, $1}')
echo "cut: $target"
echo "     ${dur} s, ${vw}x${vh}, ${vfps} fps, ${vcodec}; thumbnail $thumb ($(ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x "$thumb"))"
awk -v d="$dur" -v m="$MAX" 'BEGIN { exit !(d <= m + 0.05) }' || { echo "assemble: the cut is over ${MAX} s" >&2; exit 1; }
[[ "$vw" == "$W" && "$vh" == "$H" ]] || { echo "assemble: the cut is ${vw}x${vh}, not ${W}x${H}" >&2; exit 1; }
blank=0
for at in 5 18 31 48 64 80 97 111 127 143 152 160 167; do
  awk -v a="$at" -v d="$dur" 'BEGIN { exit !(a < d) }' || continue
  still="$out/stills/t$(printf '%03d' "$at").png"
  ffmpeg -v error -y -ss "$at" -i "$target" -frames:v 1 "$still"
  read -r ymin ymax yavg < <(ffprobe -v error -f lavfi -i "movie=$still,signalstats" \
    -show_entries frame_tags=lavfi.signalstats.YMIN,lavfi.signalstats.YMAX,lavfi.signalstats.YAVG -of default=nw=1 |
    awk -F= '/YMIN/ {mn=$2} /YMAX/ {mx=$2} /YAVG/ {av=$2} END {print mn, mx, av}')
  range=$(( ${ymax%.*} - ${ymin%.*} ))
  flag="ok"
  if (( range < 40 )); then flag="BLANK"; blank=$((blank + 1)); fi
  echo "     still $(basename "$still"): luma ${ymin%.*}-${ymax%.*}, mean ${yavg%.*}  $flag"
done
(( blank == 0 )) || { echo "assemble: $blank sampled frames look blank" >&2; exit 1; }
rm -rf "$work"
echo "done: $mode cut, ${dur} s"
