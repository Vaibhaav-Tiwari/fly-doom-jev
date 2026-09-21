# Build the complete project — DO NOT STOP EARLY

You are the primary senior engineer/research engineer responsible for implementing the entire project described below.

Your job is to **inspect the existing repository, understand the relevant upstream projects, make the architectural decisions, implement the system end-to-end, test it, document it, and leave a runnable project**.

Do NOT produce a partial scaffold.  
Do NOT stop after implementing the backend.  
Do NOT ask me to say "continue".  
Do NOT give me a plan instead of implementation.  
Do NOT fake functionality that can be implemented for real.  
If something is difficult, implement the smallest scientifically honest working version and document the limitation.

---

# PROJECT

Build an open-source research/demo system that investigates:

**ViZDoom → visual/environment state → Jev → MaleCNS v1.0 → neural activity → motor decoder → ViZDoom**

with a browser dashboard that lets users see:

- DOOM state/gameplay
    
- MaleCNS 3D brain activity
    
- Jev decisions/probabilities
    
- motor output
    
- neural activity over time
    
- action attribution
    
- synchronized timeline
    
- experiment metrics
    
- replayable experiments
    

The research question is:

> Can a probabilistic, high-level decision/state representation produced by Jev be coupled to a fixed biological Drosophila connectome (MaleCNS v1.0) and produce useful, observable closed-loop control behavior in ViZDoom?

This is a **research engineering project**, not a claim that a fly brain literally understands DOOM.

---

# IMPORTANT UPSTREAM PROJECTS

Inspect these before implementing:

- DOOMFLY:  
    [https://github.com/nftechie/doomfly](https://github.com/nftechie/doomfly)
    
- Fly Connectome Template:  
    [https://github.com/cobanov/fly-connectome-template](https://github.com/cobanov/fly-connectome-template)
    
- FlyBrain:  
    [https://github.com/Jhongdlp/FlyBrain](https://github.com/Jhongdlp/FlyBrain)
    
- ViZDoom:  
    [https://github.com/Farama-Foundation/ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
    
- MaleCNS:  
    [https://male-cns.janelia.org](https://male-cns.janelia.org/)
    

Also inspect related MaleCNS implementations when useful.

Before copying code from any repository:

1. inspect its license;
    
2. respect attribution requirements;
    
3. do not vendor incompatible licensed code;
    
4. preserve third-party notices;
    
5. document provenance in THIRD_PARTY.md.
    

The `fly-connectome-template` license/attribution requirements must be respected if anything from it is reused.

---

# NON-NEGOTIABLE ARCHITECTURE

The system MUST follow this conceptual loop:

```text
ViZDoom
   ↓
Visual observation / environment state
   ↓
Sensory / retinal representation
   ↓
MaleCNS sensory pathway
   ↓
MaleCNS v1.0 neural dynamics
   ↑
Jev high-level modulation
   ↓
Motor populations / descending neurons
   ↓
Motor decoder
   ↓
ViZDoom action
```

Jev MUST NOT directly press DOOM keys.

Jev MUST NOT directly become the final action selector in the default experimental mode.

The default causal pathway must be:

**Jev → MaleCNS modulation → neural dynamics → motor decoder → DOOM**

The final action must emerge from the MaleCNS-driven motor pathway.

---

# 1. REAL MaleCNS

Use the real **MaleCNS v1.0** connectome/data.

Do NOT replace the connectome with:

- a random graph
    
- a toy neural network
    
- a synthetic graph
    
- a tiny fake brain
    

Synthetic networks are permitted ONLY for unit tests and development fixtures.

Implement proper:

- MaleCNS download/setup
    
- version verification
    
- metadata
    
- checksum verification
    
- connectome loading
    
- sparse representation
    
- neuron/population indexing
    
- cell-type metadata
    
- anatomical metadata where available
    

Do not silently crop the connectome.

If a reduced graph is required for a browser/demo mode, make the reduction:

- explicit
    
- configurable
    
- reproducible
    
- clearly labeled as reduced
    

The research/full mode must retain the real MaleCNS dataset.

Never invent neural activity.

---

# 2. NEURAL ENGINE

Implement a clean interface approximately like:

```python
reset()
step(...)
inject_input(...)
get_activity(...)
get_population_activity(...)
get_motor_output(...)
```

Use a biologically appropriate/simple documented neuron model such as LIF where appropriate.

Optimize for a Mac development environment:

- CPU-first
    
- Apple Silicon friendly
    
- optional Metal/MPS acceleration where actually useful
    
- sparse matrices
    
- vectorized operations
    
- avoid millions of Python neuron objects
    
- avoid unnecessary dense matrices
    
- use float32 where scientifically acceptable
    
- memory-map large data where useful
    
- profile before optimizing
    

Do NOT assume CUDA.

If a Rust engine is useful for the hot neural loop, implement it cleanly and expose it to Python/browser as appropriate.

Do not prematurely rewrite everything in Rust if Python/vectorized execution is sufficient.

---

# 3. VISUAL INPUT

The visual pathway must be explicit.

Implement:

```text
ViZDoom frame
→ preprocessing
→ retinal/visual representation
→ sensory neurons/populations
→ MaleCNS
```

Do not simply inject arbitrary values into motor neurons.

Use a deterministic visual representation appropriate to the MaleCNS interface.

If the upstream DOOMFLY visual pathway can be reused conceptually, inspect it carefully and improve/adapt it rather than blindly copying it.

Document exactly where biological data ends and engineering assumptions begin.

---

# 4. ENVIRONMENT STATE

Jev MUST NOT receive raw DOOM pixels.

Create a structured state encoder containing useful information such as:

```text
health
ammo
velocity
heading
available actions
visible enemies
enemy relative position
enemy distance
enemy threat
recent damage
recent kills
recent shots
pickups
recent actions
short-term history
episode state
```

Make the schema typed and versioned.

Example:

```json
{
  "health": 82,
  "ammo": 17,
  "velocity": 0.42,
  "heading": -0.18,
  "enemy_visible": true,
  "enemy_distance": 0.37,
  "enemy_angle": 0.21,
  "threat_level": 0.71,
  "recent_damage": 0.0,
  "recent_kill": 0,
  "available_actions": ["forward", "left", "right", "attack"]
}
```

Do not hard-code this exact schema if a better design is justified; keep it extensible.

---

# 5. JEV INTEGRATION

Treat Jev as a **structured probabilistic decision/context model**, NOT as a generic embedding model.

Implement:

```text
environment state
→ typed Jev questions
→ Jev probabilities / scores / confidence
→ modulation signals
→ MaleCNS
```

Create a question bank covering concepts such as:

- ATTACK
    
- RETREAT
    
- EXPLORE
    
- REPOSITION
    
- SEEK_AMMO
    
- THREAT_LEVEL
    
- ENEMY_PRESENT
    
- MOVEMENT_INTENT
    
- TARGET_PRIORITY
    
- ENGAGEMENT_CONFIDENCE
    

Use structured outputs rather than free-form prose.

Jev should run asynchronously at a slower cadence than the neural/game loop.

Start around **2–10 decisions/sec**, configurable.

Do NOT call Jev once per game frame.

Implement:

- timeout
    
- retry
    
- request ID
    
- latency tracking
    
- cost/token tracking if available
    
- caching
    
- deterministic mock mode
    
- API-key protection
    
- backend-only API access
    

The API key must NEVER reach the browser.

The system must remain usable without Jev through replay/mock mode.

---

# 6. JEV → MaleCNS BRIDGE

This is one of the core research contributions.

Jev outputs must modulate **upstream/intermediate neural populations**, NOT directly inject the final motor/readout population.

For example, signals may be mapped into biologically/experimentally relevant populations such as optic-lobe/intermediate populations where justified.

Do NOT create:

```text
Jev ATTACK → motor neuron → fire weapon
```

because that bypasses the brain.

Instead implement:

```text
Jev ATTACK probability
→ modulation/input population
→ MaleCNS propagation
→ downstream neural activity
→ motor decoder
```

Every Jev→MaleCNS mapping must be documented as one of:

1. supported by biological evidence;
    
2. derived from known MaleCNS structure;
    
3. an explicit engineering hypothesis.
    

Never present engineering hypotheses as biological facts.

Add validation preventing accidental Jev injection directly into motor/readout populations.

---

# 7. MOTOR DECODER

Implement a configurable motor decoder using neural activity.

Prefer population activity / descending-neuron activity rather than arbitrary single-neuron mappings.

The decoder should produce:

```text
action scores
confidence
selected action
```

Possible DOOM actions:

```text
forward
backward
left
right
turn_left
turn_right
attack
strafe
noop
```

Use only actions supported by the selected ViZDoom scenario.

Make mappings configurable.

The decoder should be able to expose the neural populations contributing to each action.

---

# 8. EXPERIMENT MODES

Implement at least:

```text
baseline
jev-only
fly-only
jev-fly
```

Optionally:

```text
jev-fly-arbitrated
```

But the primary research mode must remain:

```text
Jev → MaleCNS → motor decoder
```

The comparison modes must use the same environment and metrics where possible.

---

# 9. GAME LOOP / SCHEDULING

Separate the system into different rates:

```text
ViZDoom loop       = fast
MaleCNS loop       = fast
Jev loop           = slower/asynchronous
telemetry          = configurable
browser rendering  = independent
```

Do not block the game/neural loop waiting synchronously for every Jev response.

Implement a scheduler that safely handles stale Jev decisions.

A temporary previous decision may remain active until a new valid decision arrives.

Document this behavior.

---

# 10. TELEMETRY

Create structured telemetry containing:

```text
timestamp
episode_id
game_frame
health
ammo
position/velocity where available
observation metadata
Jev decision
Jev probabilities
Jev latency
neural timestamp
neural population activity
motor scores
selected action
action confidence
controller latency
fallback events
```

Support telemetry levels:

```text
minimal
standard
research
full
```

Do NOT stream every neuron's complete activity every browser tick by default.

Use:

- population summaries
    
- sparse activity
    
- sampled neurons
    
- event windows
    
- configurable recording
    

Full neural recordings should be stored/replayed offline.

---

# 11. ACTION ATTRIBUTION

Implement a causal/temporal attribution view.

For every selected DOOM action, allow the user to inspect:

```text
game state
→ Jev decision
→ Jev probability
→ modulation
→ relevant neural populations
→ neural activity
→ motor population activity
→ decoded action
```

Support temporal windows around actions.

Implement useful research metrics such as:

- action-triggered neural averages
    
- population activity around actions
    
- Jev-to-action lag
    
- neural-to-action lag
    
- action confidence
    
- Jev entropy
    
- action entropy
    
- fallback frequency
    

Do NOT claim causality merely because two events correlate.

Call these attribution/temporal association unless causal intervention has actually been performed.

---

# 12. BRAIN VISUALIZATION

Build a React + TypeScript + Three.js browser dashboard.

The 3D brain viewer must support:

- MaleCNS neuron positions
    
- activity intensity
    
- population activity
    
- region filtering
    
- cell-type filtering
    
- neuron search
    
- selected neuron details
    
- descending/motor neuron highlighting
    
- temporal playback
    
- camera controls
    
- activity timeline
    

Use instancing/LOD/efficient GPU rendering.

Do NOT render all ~25M synaptic edges by default.

Support filtered edges/populations when useful.

Clearly distinguish:

- observed activity
    
- decoded output
    
- inferred function
    
- engineering annotations
    

Do not label a neuron "fire neuron", "attack neuron", etc. merely because it contributes to a decoder.

---

# 13. DASHBOARD

Build a polished research dashboard containing:

### DOOM

- live/replay viewport
    
- health
    
- ammo
    
- kills
    
- episode state
    
- current action
    

### JEV

- current questions
    
- probabilities
    
- confidence
    
- decision history
    
- latency
    

### BRAIN

- 3D MaleCNS
    
- active populations
    
- selected neurons
    
- temporal playback
    

### MOTOR

- action scores
    
- selected action
    
- confidence
    
- contributing populations
    

### TIMELINE

Synchronize:

```text
DOOM frame
Jev decision
neural event
motor output
game action
```

Clicking a timeline event should move the other views to the corresponding timestamp.

---

# 14. REPLAY MODE — MANDATORY

Replay must be a first-class feature, not an afterthought.

A recorded experiment must be replayable in the browser without running:

- ViZDoom
    
- MaleCNS simulation
    
- Jev API
    

Replay should reproduce:

- DOOM observations/video where recorded
    
- game state
    
- Jev decisions
    
- neural telemetry
    
- motor outputs
    
- timeline
    
- metrics
    

This is critical for a public demo because visitors must not consume API credits.

Public deployment should default to replay mode.

Live mode should be optional.

---

# 15. LIVE BACKEND API

Use FastAPI or an equally appropriate Python API framework.

Implement endpoints for:

- health
    
- experiments
    
- episodes
    
- current state
    
- telemetry
    
- replay
    
- configuration
    
- metrics
    

Use WebSocket or SSE for live telemetry.

Keep Jev API credentials server-side only.

---

# 16. PROJECT STRUCTURE

Use a clean structure approximately like:

```text
jev-doom-fly/
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── THIRD_PARTY.md
├── PROVENANCE.md
├── SCIENCE.md
├── REPRODUCIBILITY.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   └── flydoom/
│       ├── doom/
│       ├── malecns/
│       ├── vision/
│       ├── jev/
│       ├── motor/
│       ├── integration/
│       ├── telemetry/
│       ├── experiments/
│       └── api/
│
├── engine/
│   └── rust/
│
├── web/
│   └── src/
│       ├── components/
│       ├── three/
│       ├── workers/
│       ├── api/
│       └── types/
│
├── configs/
├── data/
│   ├── manifests/
│   ├── schemas/
│   └── processed/
│
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
└── outputs/
    ├── runs/
    ├── recordings/
    ├── telemetry/
    ├── metrics/
    └── figures/
```

Do not commit the raw MaleCNS dataset.

---

# 17. TESTING

Implement real tests.

At minimum:

### Unit

- state encoder
    
- Jev question generation
    
- Jev response parsing
    
- modulation mapping
    
- neural dynamics
    
- sparse graph loading
    
- motor decoder
    
- telemetry schema
    
- replay serialization
    

### Integration

- ViZDoom → state
    
- state → Jev mock
    
- Jev → MaleCNS
    
- MaleCNS → decoder
    
- decoder → ViZDoom
    
- full closed-loop episode
    
- telemetry recording/replay
    

### Safety/integrity tests

Explicitly test that:

- Jev cannot directly inject into motor populations;
    
- missing Jev responses do not crash the simulation;
    
- API keys never appear in frontend telemetry;
    
- synthetic graphs are not silently used in research mode;
    
- replay works without external services.
    

---

# 18. BENCHMARKING

Create reproducible benchmark scripts.

Measure:

- neural simulation throughput
    
- environment throughput
    
- Jev latency
    
- end-to-end controller latency
    
- telemetry overhead
    
- memory usage
    
- browser rendering performance where practical
    

Do NOT invent benchmark numbers.

Only report numbers actually measured.

---

# 19. SCIENTIFIC INTEGRITY

Create explicit documentation separating:

```text
DATA
MODEL
ENGINEERING ASSUMPTIONS
OBSERVATIONS
INFERENCES
LIMITATIONS
```

The documentation must explicitly state that:

- MaleCNS is a biological connectome model/data source;
    
- Jev is an artificial probabilistic decision model;
    
- the Jev→MaleCNS bridge contains engineering hypotheses unless biologically supported;
    
- useful DOOM behavior does not prove biological equivalence;
    
- correlation does not establish causation;
    
- a successful controller does not demonstrate that a real fly could understand or play DOOM;
    
- any learning/plasticity claims require actual implemented experiments and measured evidence.
    

Never write marketing claims such as:

> "We made a fly understand DOOM."

unless experimentally demonstrated.

---

# 20. CONFIGURATION

Everything important must be configurable:

- ViZDoom scenario
    
- timestep
    
- neural timestep
    
- Jev cadence
    
- Jev timeout
    
- telemetry level
    
- connectome path
    
- graph reduction mode
    
- motor decoder
    
- experiment mode
    
- random seed
    
- recording
    
- replay
    
- browser visualization settings
    

Provide:

```text
configs/default.yaml
configs/local.yaml
configs/demo.yaml
configs/experiments/
```

---

# 21. LOCAL DEVELOPMENT

The primary development target is:

**Apple Silicon MacBook**

Do not assume NVIDIA/CUDA.

Provide simple commands such as:

```bash
make setup
make download-data
make verify-data
make test
make run
make run-demo
make run-experiment
make replay
make benchmark
```

The README must explain exactly how to get from a fresh clone to:

1. installing dependencies;
    
2. obtaining MaleCNS data;
    
3. verifying the data;
    
4. launching ViZDoom;
    
5. running a mock Jev experiment;
    
6. running real Jev;
    
7. running MaleCNS;
    
8. launching the dashboard;
    
9. recording an experiment;
    
10. replaying it.
    

---

# 22. PUBLIC DEPLOYMENT

Design for eventual public deployment.

The public demo must NOT require visitors to provide a Jev API key.

Default architecture:

```text
Browser
   ↓
Replay server / static recording
   ↓
Dashboard
```

Optional live architecture:

```text
Browser
   ↓
Backend
   ├── ViZDoom
   ├── MaleCNS
   └── Jev API
```

API keys remain backend-only.

Provide Docker configuration.

---

# 23. README

Write a genuinely useful README containing:

- project overview
    
- research question
    
- architecture diagram
    
- screenshots/placeholders if available
    
- quick start
    
- local development
    
- dataset setup
    
- Jev configuration
    
- experiment modes
    
- replay
    
- dashboard
    
- benchmarks
    
- scientific limitations
    
- provenance
    
- third-party licenses
    
- citation
    
- contribution instructions
    

Do not claim benchmark results that were not measured.

---

# 24. IMPLEMENTATION PRIORITY

Follow this order:

### Phase 1 — REAL CORE

Get this working first:

```text
ViZDoom
→ state/vision
→ Jev/mock Jev
→ MaleCNS
→ motor decoder
→ ViZDoom
```

### Phase 2 — REAL TELEMETRY

Record every important event with timestamps.

### Phase 3 — EXPERIMENTS

Implement baseline/fly-only/Jev-only/Jev-fly comparisons.

### Phase 4 — REPLAY

Make recorded experiments independently replayable.

### Phase 5 — DASHBOARD

Build the React/Three.js visualization.

### Phase 6 — PERFORMANCE

Profile and optimize the actual bottlenecks.

### Phase 7 — DOCUMENTATION

Finish scientific/provenance/reproducibility/open-source documentation.

### Phase 8 — PUBLIC DEMO

Dockerize and ensure replay mode works without secrets or external APIs.

Do NOT stop after Phase 1.

---

# 25. FAILURE HANDLING

If the full MaleCNS simulation cannot run at full scale on the development machine:

- keep the real MaleCNS dataset;
    
- implement a documented reduced/ROI mode;
    
- make the reduction explicit;
    
- preserve the same interfaces;
    
- ensure research mode refuses to silently substitute a toy graph.
    

If Jev is unavailable:

- use a deterministic mock Jev;
    
- continue testing the rest of the system.
    

If ViZDoom cannot run in the current environment:

- implement a deterministic environment fixture for tests;
    
- but do NOT claim that the real closed loop was validated.
    

Every fallback must be visible in logs/configuration/UI.

---

# 26. IMPORTANT DESIGN RULES

NEVER:

- fake MaleCNS activity;
    
- fabricate benchmark numbers;
    
- silently replace MaleCNS with a toy network;
    
- let Jev directly choose the final action in the default mode;
    
- inject Jev directly into motor/readout neurons;
    
- send raw pixels to Jev;
    
- expose Jev API keys;
    
- block the neural/game loop waiting on Jev;
    
- claim biological meaning unsupported by evidence;
    
- claim causality from correlation;
    
- commit large raw datasets;
    
- copy third-party code without checking its license;
    
- leave major TODOs while claiming the project is complete.
    

---

# 27. DEFINITION OF DONE

The project is DONE only when all of the following exist and work as far as the available environment permits:

-  clean repository structure
    
-  real ViZDoom integration
    
-  real MaleCNS v1.0 data path
    
-  sparse neural simulation
    
-  explicit visual/sensory pathway
    
-  structured Jev integration
    
-  asynchronous Jev scheduler
    
-  Jev→MaleCNS modulation
    
-  protection against direct Jev→motor bypass
    
-  configurable motor decoder
    
-  closed-loop controller
    
-  experiment modes
    
-  telemetry
    
-  recording
    
-  replay
    
-  action attribution
    
-  React dashboard
    
-  Three.js brain visualization
    
-  synchronized timeline
    
-  benchmark tooling
    
-  unit tests
    
-  integration tests
    
-  reproducibility scripts
    
-  scientific documentation
    
-  provenance/third-party documentation
    
-  Docker support
    
-  Mac-friendly setup
    
-  README with complete setup instructions
    
-  no exposed secrets
    
-  no fake results
    
-  no unexplained toy substitutions
    

---

# FINAL INSTRUCTION

Act as the **lead engineer**, not a consultant.

Inspect the repository and upstream projects first.

Then implement the entire system.

Make reasonable engineering decisions without repeatedly asking me for permission.

When a decision is ambiguous, choose the option that maximizes:

1. scientific honesty,
    
2. reproducibility,
    
3. real end-to-end functionality,
    
4. clean architecture,
    
5. open-source usability.
    

Use the available context aggressively, but do not create unnecessary abstractions.

**The goal is a working, testable, scientifically honest open-source project — not a beautiful incomplete scaffold.**

Do not stop until you have completed the implementation to the maximum extent possible in the available environment, run the relevant tests/checks, and documented anything that genuinely cannot be executed locally.

# PUBLIC DEMO ARCHITECTURE — IMPORTANT

The public website is primarily an **interactive showcase of pre-recorded experiments**, not a public Jev inference service.

Visitors MUST NOT be asked to:

- provide a Jev API key;
    
- provide a Moonshot/Kimi API key;
    
- create an account merely to watch an experiment;
    
- run MaleCNS locally;
    
- run ViZDoom locally;
    
- wait for a live Jev inference request.
    

The public website should work immediately after loading.

## PUBLIC MODE

Pre-record experiments locally using the research/live system.

Store recordings containing, where available:

- DOOM frames/video;
    
- game state;
    
- timestamps;
    
- Jev questions;
    
- Jev probabilities/scores;
    
- Jev decisions;
    
- Jev latency;
    
- Jev request metadata excluding secrets;
    
- MaleCNS population activity;
    
- selected neuron activity;
    
- motor population activity;
    
- motor decoder scores;
    
- selected actions;
    
- action confidence;
    
- episode metrics;
    
- experiment configuration;
    
- software/data version information.
    

The browser then reconstructs the experiment from these recordings.

Public replay must require:

```text
Browser
  ↓
Static/recorded experiment data
  ↓
Interactive visualization
```

and NOT:

```text
Browser
  ↓
Jev API
```

or:

```text
Browser
  ↓
Live MaleCNS simulation
```

or:

```text
Browser
  ↓
Live ViZDoom server
```

unless an explicitly separate live-demo deployment is later configured.

## PUBLIC WEBSITE EXPERIENCE

The homepage should make the concept understandable visually within seconds.

A visitor should be able to:

1. select a recorded experiment;
    
2. press Play;
    
3. watch the DOOM run;
    
4. see the Jev decision appear at the correct timestamp;
    
5. see the corresponding MaleCNS activity;
    
6. see the motor decoder;
    
7. see the resulting action;
    
8. pause/scrub the timeline;
    
9. click an action;
    
10. inspect the associated neural activity;
    
11. rotate/zoom the 3D brain;
    
12. inspect the relevant populations;
    
13. compare experiments.
    

The website should clearly label recordings as:

**Recorded Experiment**

rather than implying that the visitor is currently running a live biological simulation.

## OPTIONAL LIVE MODE

A separate live mode may exist for the researcher/developer.

Architecture:

```text
Researcher's Browser
        ↓
Authenticated/controlled backend
        ↓
ViZDoom
        ↓
MaleCNS
        ↑
Jev API
        ↓
Motor Decoder
        ↓
ViZDoom
```

The Jev API key MUST remain server-side.

The live system must never send the raw API key to the browser.

Live mode should be disabled by default in the public deployment unless explicitly configured.

## RESEARCH RECORDING WORKFLOW

Provide commands similar to:

```bash
make run-live
make record
make replay
make build-demo
```

A researcher should be able to:

```text
run experiment
      ↓
record experiment
      ↓
validate recording
      ↓
publish recording
      ↓
deploy public website
```

The published website should then work without Jev credits.

## PUBLIC DEMO DATA

Include at least one small, validated demonstration recording in the repository or downloadable demo assets where licensing permits.

Do not commit large raw datasets or unnecessarily large neural recordings.

For large recordings, provide a documented download mechanism.

The public demo should remain functional even when the Jev API is unavailable.

## IMPORTANT

The purpose of the public website is to **show and explain the behavior of the system**, not to provide free public access to the Jev API.

Visitors are observing an actual previously recorded experiment and its synchronized neural/decision telemetry.

Do not fabricate recordings.

Do not generate fake neural activity merely to make the visualization look impressive.

Every demo recording must identify:

- experiment mode;
    
- date/version;
    
- MaleCNS version;
    
- Jev configuration/model;
    
- ViZDoom scenario;
    
- relevant random seed;
    
- recording configuration;
    
- whether the run was live, replayed, or simulated;
    
- any reduced-connectome/demo mode used.