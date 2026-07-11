#!/usr/bin/env bash
# Narrated slideshow: macOS TTS over slide PNGs -> FrugalRouter_demo.mp4
set -euo pipefail
cd "$(dirname "$0")"
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")

NARR1="This is FrugalRouter — our entry for Track One of the AMD Developer Hackathon: the hybrid token-efficient routing agent."
NARR2="The challenge: complete nineteen varied tasks — math, code, logic, entity extraction and more — using the fewest Fireworks tokens possible, while staying above an accuracy gate. And the grading box is tiny: two CPU cores, four gigabytes of RAM, ten minutes."
NARR3="Most routers try to pick the cheapest cloud model for each task. We inverted the problem: we make cloud calls unnecessary. Two small quantized models are baked into the container and run entirely on the grading hardware — so most answers cost exactly zero tokens."
NARR4="Small models can't be trusted raw, so nothing ships unverified. Math answers come from Python programs we actually execute. Generated code only passes when two independent implementations agree on real, observed behavior. Logic puzzles are brute-forced, and the answer ships only when the solution is provably unique."
NARR5="Every answer carries a confidence score from its verification outcome. Only the weakest escalate to Fireworks — cheapest first, under a hard nine-hundred-token budget. We measured every serverless model live and picked g p t oss one-twenty B at low reasoning effort: about one hundred forty tokens for a full word problem."
NARR6="The clock cannot kill it either. A startup speed probe drives a governor, results are written atomically after every task, and a watchdog guarantees a clean finish. Nineteen tasks complete in under four minutes on the grading hardware."
NARR7="The result: ninety percent accuracy with zero external tokens, ninety-five percent with escalation, in a two point two gigabyte container built on llama C P P, Qwen, and Fireworks AI. FrugalRouter — verified answers, almost free."

rm -f seg*.mp4 narr*.aiff concat.txt
for i in 1 2 3 4 5 6 7; do
  NARR_VAR="NARR$i"
  say -v Samantha -r 185 -o "narr$i.aiff" "${!NARR_VAR}"
  "$FFMPEG" -y -loglevel error -loop 1 -i "slide$i.png" -i "narr$i.aiff" \
    -c:v libx264 -preset veryfast -tune stillimage -pix_fmt yuv420p \
    -c:a aac -b:a 128k -shortest -af "apad=pad_dur=0.6" -shortest "seg$i.mp4"
  echo "file 'seg$i.mp4'" >> concat.txt
done

"$FFMPEG" -y -loglevel error -f concat -safe 0 -i concat.txt -c copy FrugalRouter_demo.mp4
rm -f seg*.mp4 narr*.aiff concat.txt
"$FFMPEG" -i FrugalRouter_demo.mp4 2>&1 | grep -E "Duration" | head -1
echo "DONE: media/FrugalRouter_demo.mp4"
