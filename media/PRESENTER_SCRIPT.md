# Presenter script — cued to FrugalRouter-deck.pptx
~100 seconds at a relaxed pace. Don't read it word-for-word — glance, speak, move on.
Contractions are on purpose. Pause wherever there's a line break.

---

**[Slide 1 — title]**
Hi, I'm Manan, and this is FrugalRouter — my entry for Track 1.
The goal of this track is simple to say and hard to do: answer nineteen varied tasks, and get billed for as few Fireworks tokens as possible.

**[Slide 2 — the problem]**
The catch is the accuracy gate — cheap answers are worthless if they're wrong.
And everything runs on a tiny grading box: two CPU cores, four gigs of RAM, ten minutes. No GPU.

**[Slide 3 — the idea]**
So instead of asking "which cloud model is cheapest for this task?", I asked a different question — what if most tasks never need the cloud at all?
Two small quantized models are baked right into the container and run on those two CPU cores. Local inference costs zero tokens.

**[Slide 4 — verification]**
Now, small models can't just be trusted — so nothing ships unverified.
Math answers come from Python programs that actually run.
Generated code has to survive two independent implementations agreeing on real behavior.
Logic puzzles get brute-forced — the answer only ships when it's provably the unique solution.
Every answer comes out of this with a confidence score.

**[Slide 5 — escalation]**
Only the answers that fail verification escalate to Fireworks — cheapest first, under a hard budget of nine hundred tokens for the whole run.
I benchmarked every serverless model live; gpt-oss-120b at low reasoning effort came out the most frugal — about a hundred and forty tokens for a full word problem.

**[Slide 6 — the clock]**
And the clock can't kill it. A startup probe measures the actual speed of the box and adjusts depth. Results are written after every task. A watchdog guarantees a clean exit.
All nineteen tasks finish in under four minutes — measured on the real constraints.

**[Slide 7 — results]**
Bottom line: ninety percent accuracy with zero external tokens. Ninety-five with escalation. In a two-gigabyte container.
FrugalRouter — verified answers, almost free. Thanks!

---

## Recording (pick one, 5 minutes total)

**Option A — QuickTime (built in):**
1. Open your deck in full-screen presentation mode.
2. QuickTime Player → File → New Screen Recording → Options → choose your mic → Record.
3. Talk through the slides, stop, File → Export As → 1080p.

**Option B — with your face (judges like this):**
1. QuickTime → File → New Movie Recording (webcam window appears) → shrink it, keep it in a corner over the slides.
2. Screen-record as in Option A — your face floats over the deck.

Tips: one take is fine, small stumbles sound human. Don't restart for a flub — just repeat the sentence and keep going.
