# EuroDLLM — Plan (v0.2)

**Working title.** A community-trained, EU-jurisdiction, contributor-owned open language model. Compute donated by a four-tier contributor network — EU AI Factories, EU sovereign cloud providers with off-peak surplus, EuroHPC centres, and EU-resident volunteers. Weights released under a community license restricted to EU/EEA residents and EU-domiciled entities. Inference free for contributors and EU residents up to a fair-use quota.

Changes from v0.1: ambition raised across the board (70B Y1 → 200B MoE Y2 → 405B-class Y3), individual volunteers reframed as the *long tail* of a primarily institutional contributor network, full EU regulatory stack now mapped (AI Act, GDPR, DSM Art. 4, **Data Governance Act / Data Altruism**, **Cyber Resilience Act**, **Data Act**, eIDAS 2.0, NIS2), Phase 0 prototype confirmed working on RTX 3060 GPU at ~38k tok/s per worker.

---

## 0. TL;DR

1. **Contributor network is four tiers, not one.**  
   (a) **Anchor partners** — EU AI Factories, sovereign cloud providers committing multi-month dedicated capacity.  
   (b) **Cloud sponsors** — EU cloud providers donating off-peak surplus (OVHcloud, Scaleway, Hetzner, IONOS, Stackit, Exoscale, UpCloud, 3DS Outscale).  
   (c) **HPC partners** — scheduled allocations from EuroHPC JU sites (LUMI, LEONARDO, JUPITER, MareNostrum 5, MeluXina, Karolina, Discoverer, Vega, Deucalion).  
   (d) **Individual volunteers** — long tail of EU-resident enthusiasts with home/office GPUs. Same DiLoCo workers as v0.1, but now the long tail rather than the centre of gravity.
2. **Aim higher.**  
   **Y1: 70B dense** (was 30B). **Y2: 200B MoE (active ~30B)** or 120B dense. **Y3: 405B-class** dense or 400B MoE, intentionally engaging the EU AI Act systemic-risk regime.
3. **Algorithm unchanged:** [DiLoCo](https://arxiv.org/abs/2311.08105) inner/outer split, ~500 inner steps between syncs, ~500× less communication than synchronous data-parallel. [INTELLECT-2](https://arxiv.org/abs/2505.07291) is the existence proof at 32B; we go bigger by adding institutional pods.
4. **Full EU regulatory stack baked in:**  
   - **EU AI Act** — GPAI obligations + voluntary [Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers) adoption + deliberate FLOPs management (stay below 10²⁵ for Y1–Y2).  
   - **GDPR** — minimised personal data, DPIA, DPO, EU-only data residency.  
   - **DSM Directive Art. 4** — TDM exception with machine-readable opt-out (robots.txt + ai.txt + TDM-Rep).  
   - **Data Governance Act (DGA)** — register Stichting as a **Data Altruism Organisation Recognised in the Union** ([EU register](https://digital-strategy.ec.europa.eu/en/policies/data-altruism-organisations)). Free legal status + official label + QR code. Substantial credibility/legitimacy uplift.  
   - **Cyber Resilience Act (CRA)** — worker client is software with digital elements; **vulnerability reporting obligations from 11 Sept 2026**, full obligations from 11 Dec 2027. We register as an **Open Source Steward** under the CRA's open-source carve-out — lighter obligations than commercial manufacturers but still real.  
   - **Data Act** — affects how we mediate data between contributors and the foundation; reviewed by counsel.  
   - **eIDAS 2.0** — use the EU Digital Identity Wallet for residency verification on Tier 1 model access.  
   - **NIS2** — coordinator infrastructure analysed; likely out of scope but counsel to confirm.  
5. **Phasing (v0.2):**  
   - **Phase 0** ✓ done — DiLoCo loop, 1 coord + 2 workers verified on a single RTX 3060 (~38k tok/s/worker).  
   - **Phase 1** — WAN deploy + bf16/8-bit transport + first cloud-sponsor pool + 1B model trained.  
   - **Phase 2** — 7B model + first AI Factory partnership + DGA registration + CRA open-source steward declaration.  
   - **Phase 3** — **70B production run** (was 30B), 9–12 months wall-clock with anchor partners.  
   - **Phase 4** — 200B MoE.  
   - **Phase 5** — 405B-class, intentional systemic-risk engagement.  
6. **Y1 cash budget: ~€2.5M** (up from €1M). Y1 in-kind compute target: **€15–30M** equivalent from anchor partners + cloud sponsors + HPC allocations. Y1 individual-volunteer contribution: ~€2M-equivalent (down from "primary" to "supplementary").
7. **A working repo exists.** Phase 0 code in `D:\Projects\dllm`; verified end-to-end. Loss curve descends cleanly through 10 outer rounds. GPU path validated.

---

## 1. Mission & contributor classes

### 1.1 Mission

Train and release an EU-jurisdiction open language model whose compute, data, governance, and beneficiaries are all anchored in the EU. Compute donated by EU institutions and individuals. Weights released to EU residents and EU-domiciled organisations under the EU Residents Community License.

### 1.2 Success criteria (revised)

**12-month (Y1):**
- ≥3 anchor partners signed (≥2 AI Factories or sovereign cloud providers, ≥1 EuroHPC centre).
- ≥10 cloud sponsors active in the off-peak pool.
- ≥3,000 active individual EU-resident worker nodes sustained ≥3 months.
- One **70B-parameter dense** (or 120B MoE) base model trained end-to-end.
- Eval performance ≥90% of best contemporary open model (Llama-equivalent) on EU-multilingual evals; ≥80% on English-dominant evals.
- Stichting formed, **registered as Data Altruism Organisation Recognised in the Union**.
- Released under EU Residents Community License with eIDAS 2.0 verification on full-fat tier.
- Inference endpoint with ≥50k EU-resident users.

**24-month (Y2):**
- **200B MoE** (active ~30B per token) or 120B dense next-gen.
- Sustaining funding pipeline (Horizon, national grants, AI Factory shared-cost, corporate sponsorships).
- Open Source Steward status under CRA, security audit completed.
- Domain variants released (legal, medical, scientific) as continued pre-training + SFT.
- ≥5 anchor partners, ≥20 cloud sponsors, ≥10k individual workers.

**36-month (Y3):**
- **405B-class dense** or 400B MoE run engaging EU AI Act systemic-risk regime intentionally.
- AI Office formal engagement; we are a reference implementation of a transparent, compliant systemic-risk GPAI provider.
- Foundation revenue from above-threshold commercial licensees funds ongoing work.

### 1.3 Non-goals

- Beating closed frontier (GPT-5/Claude/Gemini) on US-English benchmarks. We will lose; that's fine.
- Becoming a commercial AI lab. Foundation, not company.
- Issuing financial tokens or selling securities.
- Replacing centralised training entirely — post-training (SFT, DPO, RLHF) uses rented H100s on EU CSPs.

### 1.4 Contributor classes

| Tier | Examples | Commitment | Tech requirement | Reward |
|---|---|---|---|---|
| **A: Anchor** | EuroHPC AI Factory, sovereign CSP, national lab | Multi-month dedicated capacity (≥100 GPUs) | Pipeline-parallel pod host; low-latency intra-pod (≥25 Gbit); operator on-call | Co-listing on model card; Foundation Council seat; influence over training-run schedule; CSR credit |
| **B: Cloud sponsor** | OVHcloud, Scaleway, Hetzner, IONOS, Stackit | Off-peak surplus (≤1000 GPU-hours/week) via API | Containerised worker; auto-shed on tenant load; reverse-billed | Logo on contributors page; tax-deductible donation receipts (per MS rules); priority API quota |
| **C: HPC partner** | EuroHPC centre, university cluster | Scheduled batch slots (Slurm/PBS) | Slurm-job worker; checkpointable; runs during low-priority windows | Acknowledgement on model card; access to internal research credits; institutional inference quota |
| **D: Individual** | EU-resident with home/office GPU | Intermittent; configurable schedule | Cross-platform worker; pause-aware; ≥8 GB VRAM | Compute Credits → inference quota, governance vote weight, early weight access, leaderboard recognition |

Anchor partners and HPC partners get the *pipeline-parallel* pod hosting for the heaviest stages of training (model-parallel slices of the 70B/200B model). Cloud sponsors and individuals carry data-parallel DiLoCo pods at smaller per-pod capacity.

---

## 2. Feasibility math (revised)

### 2.1 Compute budget — Y1 70B target

Reference: Llama 3.1 70B ~6.4M H100-hours for 15T tokens. Chinchilla-optimal 70B = ~1.4T tokens, ~3M H100-hours. We over-train for quality: target **3T tokens, ~6M H100-hours**.

Anchor + cloud + HPC + individual contribution model:

| Contributor class | Active GPUs (Y1 target) | Effective hrs/mo per GPU | Class-share H100-equivalent per month | Months | H100-equiv hours |
|---|---:|---:|---:|---:|---:|
| Anchor partners (H100/H200/MI300) | 800 | 600 | 480,000 | 9 | 4,320,000 |
| Cloud sponsors (mix; effective 0.6× H100) | 1,500 | 200 | 180,000 | 9 | 1,620,000 |
| HPC partners (H100/MI250 mix; 0.7× H100) | 1,000 | 300 | 210,000 | 9 | 1,890,000 |
| Individuals (4090/5090/3090; 0.22× H100) | 3,000 | 192 | 127,000 | 9 | 1,143,000 |
| **Total** | | | | | **~9M H100-hours** |

→ ~9M H100-hours over 9 months. After ~25% DiLoCo+comms overhead → **~7M effective**, which clears the 6M-hour target.

**Stay below systemic-risk threshold.** 70B × 3T tokens ≈ 1.3×10²⁴ FLOPs. Comfortably under the 10²⁵ systemic-risk line. (We track FLOPs as a first-class metric in the coordinator.)

### 2.2 Compute budget — Y2 200B MoE

200B total, ~30B active per token. Compute scales with *active* params for MoE: ~30B × 4T tokens ≈ 7.2×10²³ FLOPs, still under 10²⁵.

Wall-clock estimate at the same contributor mix expanded ~50%: ~12 months for the 200B MoE run.

### 2.3 Compute budget — Y3 405B

405B × 15T tokens ≈ 3.6×10²⁵ FLOPs → **systemic risk** under EU AI Act. Intentional. By Y3 we have the compliance machinery (CoP signatory, AI Office relationship, ISO/IEC 42001 audit, red-team programme).

### 2.4 Bandwidth budget

Unchanged from v0.1. DiLoCo + 8-bit compression keeps inter-pod traffic at ~15 GB per outer cycle even for 70B bf16 (~140 GB raw → 17 GB at 8-bit). Outer cycle every ~30 min at residential 30 Mbps upload is plausible; cloud-sponsor and anchor uplinks (10+ Gbit) make outer cycles seconds, not minutes.

### 2.5 Memory budget per worker tier

| Worker tier | VRAM per GPU | Native dense ceiling (bf16 + grad ckpt + 8-bit AdamW) | With ZeRO-3 + pipeline within pod |
|---|---|---|---|
| Individual (12–32 GB) | 12–32 | 1.5B–5B | up to 30B in a 4-GPU pod |
| Cloud sponsor (24–80 GB H100/L40S) | 24–80 | 5B–40B | 70B in a 4-GPU pod |
| HPC (H100/MI250, 40–96 GB) | 40–96 | 40B–60B | 200B in an 8-GPU pod |
| Anchor (H100/H200/MI300, 80–192 GB) | 80–192 | 70B | 405B in a 16-GPU pod |

The heavy pipeline-parallel pods live on anchor + HPC tiers. Cloud sponsors and individuals run the lighter data-parallel ring.

---

## 3. Technical architecture

### 3.1 Stack overview (revised — institutional tiers)

```
                           ┌────────────────────────────────┐
                           │     Coordinator (EU cloud)     │
                           │  - tier-aware shard assignment │
                           │  - outer optimizer + sched     │
                           │  - reg + checkpoint store      │
                           │  - trust scoring + FLOPs meter │
                           └──────────────┬─────────────────┘
                                          │
              ┌──────────────────────┬────┴────┬──────────────────┐
              │                      │         │                  │
        ┌─────▼─────┐         ┌──────▼─────┐  ┌▼───────────┐  ┌──▼─────────┐
        │ Tier A    │         │ Tier B     │  │ Tier C     │  │ Tier D     │
        │ Anchor    │         │ Cloud      │  │ HPC partner│  │ Individual │
        │ pod (16G) │         │ sponsor    │  │ Slurm pod  │  │ workers    │
        │ PP+TP+DP  │         │ pool (DP)  │  │ (DP)       │  │ (DP)       │
        └───────────┘         └────────────┘  └────────────┘  └────────────┘
            ▲                       ▲                ▲                ▲
            │                       │                │                │
       100 Gbit fabric         10 Gbit DC      Slurm batch          residential
       within pod              between sites   queues               broadband
```

### 3.2–3.6 (unchanged from v0.1 except as noted below)

Algorithm: DiLoCo with inner AdamW + outer Nesterov; ~500 inner steps; SHARDCAST-style weight broadcast; 8-bit comm; trimmed-mean Byzantine aggregation.

Coordinator stack: FastAPI + Postgres + Redis + EU S3-compatible object storage. **NEW: tier-aware scheduling** — each anchor pod is its own logical "super-worker" submitting many deltas; cloud-sponsor pool is auto-scaled by queue depth; individual workers are async with timeout-based barrier.

### 3.7 Byzantine tolerance & verification

Unchanged from v0.1. Worth noting: anchor partners are presumed trusted (verified counterparties, legal agreements). Verification spend is concentrated on individual and cloud-sponsor deltas.

### 3.8 Weight broadcast (SHARDCAST)

Unchanged. Anchor partners with high bandwidth function as natural seed nodes for individual workers downloading new state.

### 3.9 Tooling

Unchanged. Add **Slurm worker wrapper** for HPC partners; **Terraform/OpenTofu modules** for each Tier-B cloud sponsor (OVH, Scaleway, Hetzner, IONOS, Stackit) to auto-provision worker pools from their respective APIs.

### 3.10 Institutional integration (NEW)

**Tier A — Anchor partner onboarding:**
1. Legal: signed Compute Donation Agreement (CDA) — multi-month commitment, SLA, IP terms, attribution.
2. Tech: anchor deploys our worker container on N GPUs in a pod (typically 16). We provide a Terraform/OpenTofu module per provider for one-shot deploy.
3. Pod runs persistent pipeline-parallel pod state. Worker container exposes its pod-internal coordinator address.
4. Anchor pod has its own pod-local coordinator that handles tensor-/pipeline-parallel; submits aggregated pseudo-grad to global coordinator.
5. Operator on-call rotation for our team during anchor commitment windows.

**Tier B — Cloud sponsor onboarding:**
1. Sponsor signs lighter Sponsor Agreement (Apache-style, foundation-style).
2. They expose an API key for their cloud (OVH/Scaleway/Hetzner) — limited to "create/destroy worker VM" scope.
3. Our **Sponsor Orchestrator** service polls queue depth + sponsor-defined budget cap (e.g. "donate up to 200 GPU-hours/week"), spins up worker VMs accordingly.
4. Workers run a container image; tear down at end of window.
5. Reverse-billing: at month end, foundation issues a tax-deductible donation receipt under the sponsor's MS rules (per counsel).

**Tier C — HPC partner onboarding:**
1. Mediated via a research collaboration agreement with the hosting entity.
2. We provide Slurm-job templates; partner runs them at their discretion within scheduled allocation windows.
3. Checkpointable: worker can be SIGTERM'd by Slurm scheduler at slot end, resumes next window.
4. Useful especially for the heaviest pipeline-parallel pods.

**Tier D — Individual onboarding:**
1. Visit project site, attest EU residency, download signed worker binary.
2. Per [worker.py](src/dllm/client/worker.py).
3. Long-tail compute; trust-scored.

---

## 4. Data

(largely unchanged from v0.1; tightening below)

### 4.1 Sources

CulturaX (EU subset), OSCAR (EU langs), Wikipedia (24 EU langs), Europeana (public-domain books), EuroParl + EUR-Lex + ECHR + national gazettes, OpenLegalData, The Stack v2 (permissive only), PubMed OA, arXiv, Project Gutenberg EU, Common Crawl filtered through DSM Art. 4 + machine-readable opt-out.

### 4.2 Volume

**Target raised: 3T tokens** for the 70B Y1 run (was 3T for 30B — now keeping same volume, more over-trained). Y2 200B MoE: 4T tokens. Y3 405B: 8–15T tokens.

Language mix: ≥40% non-English EU content.

### 4.3 Pipeline

Per v0.1. **NEW: provenance hash chain** — every shard is content-addressed and the chain of {source → fetch → filter → tokenize} steps is recorded so any datum can be traced back to its source URL + license + fetch date.

### 4.4 Provenance & opt-out

- Public manifest of all sources with licenses, fetch dates, sample counts.
- Respect machine-readable opt-out: `robots.txt`, `ai.txt`, [TDMRep](https://www.w3.org/community/reports/tdmrep/CG-FINAL-tdmrep-20240202/).
- **Quarterly shard refresh** incorporating new opt-outs.
- Public **opt-out registry**: anyone can submit a URL or domain to be excluded from current and future shards.
- Public **training data summary** per [EU AI Act Article 53(1)(d) template](https://digital-strategy.ec.europa.eu/en/library/template-public-summary-training-content-general-purpose-ai-models).

### 4.5 Evaluation

Per v0.1. We publish the EuroLaw, EuroMultilingualReason, EuroCivics benchmarks as a contribution.

---

## 5. Legal & governance

### 5.1 Entity

**Stichting (Dutch foundation)**, seat in Amsterdam. Same as v0.1. Y2 introduces a Contributors' Council; Y2/Y3 may evolve to SCE (European Cooperative Society) if member-control desired.

### 5.2 Licensing

**Code**: Apache 2.0.

**Weights**: **EU Residents Community License (EU-RCL)** — to be drafted with EU IP counsel. Provisions per v0.1.

**Data**: training data manifest published; transformative use under DSM Art. 4 with opt-out respected.

### 5.3 EU AI Act (expanded)

We are a **GPAI provider** as defined under [AI Act Article 51](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai). Obligations effective 2 Aug 2025:

- ✅ **Technical documentation** (Article 53(1)(a)): model card with architecture, training data summary, evals, FLOPs estimate.
- ✅ **Public training-data summary** (Article 53(1)(d)): manifest from §4.4 using AI Office template.
- ✅ **Copyright compliance policy** (Article 53(1)(c)): written policy, opt-out registry, takedown SLA.
- ✅ **Cooperate with AI Office** (Article 53(1)(b)): register, respond to queries.
- ✅ **Voluntary adoption of [GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers)** (Transparency + Copyright chapters at minimum). Adopting CoP gives presumption of compliance. Strong yes.
- 🚫 **Stay below 10²⁵ FLOPs for Y1–Y2** — engineering target tracked in coord; alarms at 5×10²⁴.
- ⚠ **Y3 systemic-risk engagement** — intentional. Triggers Safety & Security chapter of CoP, AI Office notification within 2 weeks of foreseeing threshold, model evaluation, adversarial testing, incident reporting, cybersecurity protections.

### 5.4 GDPR

Unchanged from v0.1. DPO appointed before public beta. DPIA before Phase 2. EU-only data residency. Right to erasure honoured for worker accounts.

### 5.5 EU residency verification

Unchanged: 3-tier scheme. **Tier 1 (download / free inference) uses EU Digital Identity Wallet (eIDAS 2.0)** as primary method, with VAT-ID / utility-bill fallback. **Tier 2 (high quota / API)** uses SumSub/Veriff EU-operated KYC.

### 5.6 Open questions for counsel

Per v0.1, plus:
- DGA Data Altruism Organisation status: scope clarity (is compute-contribution-for-AI-model-training within scope of "data" altruism, or must we declare the training data shards themselves as the donated artifact?). Likely the latter is cleaner.
- Tier B reverse-billing: tax-deductible donation receipts vary by MS; we'll need a per-country reading.
- EU-RCL: residency restriction as contract term — enforceability in all 27 MS.
- Cross-border transfers to AI Factory hosting entities outside our Stichting MS.

### 5.7 Cyber Resilience Act (NEW)

[CRA](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act) entered into force 10 Dec 2024. Vulnerability **reporting obligations apply from 11 Sept 2026** (4 months from now). Main obligations from 11 Dec 2027.

Worker client = "product with digital elements" → in scope **if** made available on the market in commercial activity. As a not-for-profit Foundation distributing free open-source software, we benefit from the [open-source carve-out](https://digital-strategy.ec.europa.eu/en/policies/cra-open-source), but with the catch: a legal person providing "systematic support on a sustained basis" qualifies as an **Open Source Steward** — lighter obligations than full manufacturers, but real:

- ✅ Establish and publish a written **cybersecurity policy** for the project.
- ✅ **Cooperate with market surveillance authorities** on request.
- ✅ **Report actively exploited vulnerabilities** to ENISA and affected users within stipulated timeframes.
- ✅ Coordinated vulnerability disclosure process (security@ inbox + PGP, advisory channel).
- ✅ Software Bill of Materials (SBOM) — CycloneDX or SPDX format — for each release of the worker client.
- ✅ Reproducible builds (or signed builds with verifiable supply chain) for the worker binary.

Action: declare Open Source Steward status; publish security policy + SBOM by **11 Sept 2026** (the reporting-obligations start date).

### 5.8 Data Governance Act — Data Altruism (NEW)

[DGA](https://digital-strategy.ec.europa.eu/en/policies/data-governance-act) Articles 16–25. Voluntary registration as **Data Altruism Organisation Recognised in the Union** ([EU register](https://digital-strategy.ec.europa.eu/en/policies/data-altruism-organisations)).

Requirements:
- Not-for-profit (Stichting qualifies).
- General-interest purpose (yes — sovereign EU AI for the public).
- Transparency: annual activity report; record of data uses; safeguards for data subjects.
- Operate independently of commercial entities.

Benefits:
- Official **"Data Altruism Organisation Recognised in the Union" label + common logo + QR code** linking to EU public register. Strong credibility signal for both data donors (publisher opt-outs become opt-INs; private corpora become donatable) and grant funders.
- Standardised consent forms make it easier for EU citizens/orgs to donate corpora to us.
- Eligibility for EU R&D programmes that require/prefer recognised orgs.

Action: file notification with the competent authority in NL (per Stichting seat) immediately after Stichting registration. Reference: [DGA notification form](https://digital-strategy.ec.europa.eu/en/library/notification-form-member-states-recognised-data-altruism-organisations).

### 5.9 Data Act (NEW)

[Data Act](https://digital-strategy.ec.europa.eu/en/policies/data-act) effective 12 Sept 2025. Mainly about IoT data and B2B data sharing — not centrally about LLM training. Relevance:

- We are not in scope as "data holder" of connected products.
- We may *use* data covered by Data Act sharing obligations (e.g. if a future partner provides industrial data to us under DA-mandated sharing) — would be specific to each such agreement.

Action: counsel review of any future DA-implicated partnerships.

### 5.10 NIS2 (analysis)

[NIS2 Directive](https://digital-strategy.ec.europa.eu/en/policies/nis2-directive). Applies to "essential" and "important" entities in 18 sectors. AI training infrastructure is not an enumerated sector. Cloud services providers ARE in scope, but we're a consumer of CSPs, not a CSP ourselves. Coordinator infrastructure does not appear to trigger NIS2 unless we cross size thresholds.

Action: counsel review at Y2 scale; document analysis in the entity governance docs.

### 5.11 Standards & certifications

Voluntary but signal-valuable:

- **ISO/IEC 42001** (AI Management System) — pursue certification by Y2.
- **ISO/IEC 23894** (AI Risk Management) — adopt the framework.
- **ISO/IEC 5259** series (AI Data Quality) — apply to our data pipeline.
- **SOC 2 Type II** — for inference hosting in Y2+.

### 5.12 Training-data compliance checklist (operational)

For every source we ingest:
1. License recorded (CC-BY-SA, public-domain, EU government, etc.).
2. Fetch date + content hash recorded.
3. DSM Art. 4 opt-out signals checked (robots.txt + ai.txt + TDM-Rep).
4. PII screened (regex + spaCy + Presidio + EU-specific patterns: IBAN, BSN, NIE, NHS, passport, etc.).
5. Special-category data (GDPR Art. 9) screened and excluded by default.
6. Quality-classified and deduped (MinHash + exact-suffix).
7. Tokenised with our 128k EU-balanced BPE.
8. Sharded with provenance manifest.
9. Published in the public manifest under §4.4.
10. Re-checked quarterly against the live opt-out registry.

---

## 6. Economics (revised)

### 6.1 Y1 cash budget (estimate, €)

| Category | v0.1 | v0.2 | Notes |
|---|---:|---:|---|
| Core team (FTE × 12 mo) | 450,000 | 900,000 | 6 FTE: 2 systems/ML, 1 distributed-systems lead, 1 community/ops, 1 legal/policy, 1 partnerships |
| Legal: entity, license, DGA, CRA prep | 60,000 | 150,000 | More regulatory work in v0.2 |
| Coordinator + storage | 60,000 | 120,000 | HA across two EU regions |
| Data prep + CDN | 80,000 | 200,000 | Larger corpus + provenance tooling |
| Eval + post-training H100 time | 180,000 | 400,000 | More rigorous evals; bigger post-train run on 70B |
| Inference hosting (free tier) | 180,000 | 350,000 | More users at 70B scale |
| Community, comms, events | 30,000 | 100,000 | Anchor-partner outreach + grant comms |
| Audits (security + AI Act + CRA) | 40,000 | 150,000 | Adds SBOM tooling, CRA conformity prep |
| Contingency (15%) | 165,000 | 280,000 | |
| **Total** | **~€1.25M** | **~€2.65M** | |

### 6.2 Funding paths (revised priority)

1. **EU AI Factories shared-cost partnership** — most direct route to anchor-tier compute + co-funding. Each AI Factory has SME/research access mandates we fit naturally.
2. **EU Horizon Europe Cluster 4** — open-source AI calls.
3. **National grants**: France 2030, Germany SPRIND/BMBF, NL AINed/NWO, Spain AESIA, Italy PNRR-AI.
4. **Foundations**: Mozilla, Open Society, Bertelsmann Stiftung, Robert Bosch Stiftung, NLnet (NL).
5. **Corporate in-kind**: Tier-B cloud sponsors.
6. **Crowdfunding / membership**: for legitimacy and long-tail funding.

### 6.3 In-kind contribution valuation (for grant applications)

| Class | Y1 GPUs | H100-eq hrs/yr | €-equiv @ €2/H100-hr |
|---|---:|---:|---:|
| Anchor | 800 | 4.3M | €8.6M |
| Cloud sponsor | 1,500 | 1.6M | €3.2M |
| HPC partner | 1,000 | 1.9M | €3.8M |
| Individual | 3,000 | 1.1M | €2.2M |
| **Total in-kind** | | **~9M** | **~€17.8M** |

→ Headline number for grant proposals: **~€18M in-kind Y1**.

### 6.4 Sustainability beyond Y1

- Above-SME-threshold commercial licensees under EU-RCL — revenue-share back to Foundation.
- Continued grants.
- Foundation membership for institutions.
- Training-data-as-a-service for research institutions (the cleaned EU corpus is independently valuable).
- Inference API for compliant enterprise users.

---

## 7. Contributor model (restructured — 4 tiers)

### 7.1 Tier A — Anchor partners

**Who:** EuroHPC AI Factories, sovereign cloud providers (e.g. T-Systems / OTC, Orange Business / Cloud Avenue, Deutsche Telekom / Stackit, Aruba / Outscale).

**What they give:** dedicated GPU pods, multi-month commitment, signed CDA.

**What they get:** seat on Foundation Council, listing on model card and all publications, CSR credit, AI-leadership narrative for their PR, co-publication rights on technical papers.

**Recognition:** "Anchor Partner" tier on the Foundation site; logos on /thanks page; first paragraph of model card.

### 7.2 Tier B — Cloud sponsors

**Who:** EU CSPs (OVHcloud, Scaleway, Hetzner, IONOS, Stackit, Exoscale, UpCloud, 3DS Outscale, Aruba, Open Telekom Cloud).

**What they give:** off-peak GPU surplus via API, up to sponsor-defined ceiling per period.

**What they get:** "Cloud Sponsor" listing; donation receipts (per-MS rules); foundation membership eligibility; logo on contributors page; visibility in foundation comms.

**Recognition:** "Cloud Sponsor" tier badge; logo on /sponsors page; mentioned in monthly transparency report.

### 7.3 Tier C — HPC partners

**Who:** EuroHPC JU sites (LUMI, LEONARDO, JUPITER, MareNostrum 5, MeluXina, Karolina, Discoverer, Vega, Deucalion, etc.), national HPC, university clusters.

**What they give:** scheduled batch allocations via Slurm/PBS.

**What they get:** acknowledgement on model card, internal research credits, foundation institutional inference quota, co-authorship on technical papers.

**Recognition:** "HPC Partner" tier on Foundation site.

### 7.4 Tier D — Individual contributors

**Who:** EU residents with home/office/gaming GPUs.

**What they give:** intermittent worker time on the DiLoCo network.

**What they get:** **Compute Credits** (CC) — non-transferable, non-financial ledger entry. Benefits scale with CC:
- Inference quota (X tokens/day per CC).
- Early access to checkpoints (1 cycle ahead of public).
- Governance vote weight (log-scale, capped).
- Foundation membership eligibility (above thresholds).
- Public recognition: leaderboard, contributor badges.

**Recognition:** opt-in leaderboard; cohort badges per training run; top contributors listed on model card.

**Explicitly NOT given:** cash, transferable tokens, equity. (Avoids MiCA, avoids 27-jurisdiction tax/labour mess.)

### 7.5 Onboarding flows (per-tier)

- **A**: relationship-driven, hand-shake outreach by Foundation lead, signed CDA, technical deployment with our help.
- **B**: self-serve via a Foundation portal; sign sponsor agreement online; provide API key; auto-orchestrator handles the rest.
- **C**: research collaboration agreement; Slurm-job template provided; institution runs at their discretion.
- **D**: download client → attest EU residency → connect.

### 7.6 Anti-abuse

Trust scoring per worker/pod; redundant assignment for unverified workers; spot-check honeypots; FLOPs accounting reconciled monthly. Anchors and HPC partners presumed trusted; cloud sponsors high-trust; individuals start at low trust and earn up.

---

## 8. Phased roadmap (revised)

### Phase 0 — Prototype ✓ DONE

DiLoCo loop verified end-to-end on one RTX 3060 with 2 simulated workers. ~38k tok/s/worker. Loss curve clean across 10 outer rounds.

### Phase 1 — WAN + first sponsor (months 1–3)

- Worker client: signed binary, bf16 transport, 8-bit delta compression.
- Coordinator: HA Postgres, EU S3 checkpoints, FLOPs meter.
- Deploy coord on Netcup VPS (`dllm.planetbass.de`).
- Stichting filed (Amsterdam).
- DGA notification filed.
- First Tier-B cloud sponsor pool live (target: one of Hetzner / OVH / Scaleway).
- Train a **1B model** end-to-end on real WAN: 1 anchor pod + 10 individual workers + Tier-B sponsor pool.
- Public design doc + technical blog post.
- IP counsel engaged.

### Phase 2 — Open beta + first anchor (months 4–7)

- Worker client v1 release with EU residency Tier 0/1 verification.
- Public site launch.
- Stichting active; DGA recognition received.
- CRA open-source steward declaration published; security policy + SBOM in place.
- EU-RCL v1 published.
- DPIA complete; GDPR notices live.
- Train a **7B model** with first anchor partnership (AI Factory or sovereign CSP).
- Inference endpoint live for verified EU residents.
- AI Office registration filed.

### Phase 3 — 70B production run (months 8–18)

- Pod formation logic deployed; pipeline parallelism within anchor pods.
- ≥3 anchor partners, ≥10 cloud sponsors, ≥3k individuals.
- SHARDCAST broadcast; TOPLOC verification for RL post-training.
- Apply for and (hopefully) secure Horizon grant.
- Post-training on rented H100s (SFT, DPO, targeted RLHF for safety + EU compliance).
- Red-team rounds.
- Release **70B base + chat** under EU-RCL to EU residents.
- ≥50k active inference users.
- ISO/IEC 42001 audit started.

### Phase 4 — 200B MoE (months 19–30)

- Architecture decision: dense vs MoE locked.
- ≥5 anchor partners, ≥20 cloud sponsors.
- Train 200B MoE.
- Foundation Contributors' Council elected.
- AI Office formal relationship; voluntary advance dialogue on Y3 systemic-risk run.

### Phase 5 — 405B-class (months 31–48)

- Engage EU AI Act systemic-risk regime intentionally.
- Notify AI Office 2 weeks pre-threshold.
- CoP Safety & Security chapter adoption; model evaluations; adversarial testing; incident reporting; cybersecurity protections.
- Release 405B (or 400B MoE) under EU-RCL.

---

## 9. Risks & mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Anchor partner does not materialise | M | H | Phase plan tolerates 1B/7B without anchors; only 70B requires anchor compute |
| R2 | Cloud-sponsor pool unstable (preemption) | M | M | Async DiLoCo; sponsors are bonus, not critical-path |
| R3 | DiLoCo convergence fails at 70B | M | H | Validate at 1B, 7B; have centralised fallback for final ~10% of training |
| R4 | Byzantine attack | M | H | Trimmed mean + trust score + honeypots + anchor verification |
| R5 | Legal challenge to training data | M | M | DSM Art. 4 + opt-out registry + DGA-DAO status + transparent manifest |
| R6 | Funding shortfall | M | H | Phased — 7B is the minimum viable release |
| R7 | EU AI Act systemic-risk crossed unintentionally | L | H | FLOPs meter; hard alarm at 5×10²⁴; Y3 crossing is deliberate |
| R8 | CRA non-compliance | L | H | Open-source steward declaration + security policy + SBOM by Sept 2026 |
| R9 | Worker client supply-chain attack | L | C | Reproducible builds, code signing, security audit pre Phase 2 |
| R10 | DAO status denied | L | M | Re-apply after addressing feedback; benefit is signalling, not load-bearing |
| R11 | Cloud-sponsor reverse-billing rejected in some MS | M | M | Per-country guidance; if blocked in MS X, sponsors there get acknowledgement only |
| R12 | Geo-gating bypass | H | L | Contractual not technical; accept some leakage |
| R13 | Power-cost defection of individuals | H | L | Individuals now long-tail; institutional tiers dominate |
| R14 | Internal disagreement / fork | M | M | Clear governance from Y1; fork-friendly licensing |
| R15 | Coordinator outage during long run | L | M | HA; S3 checkpoints; resumable |
| R16 | Model quality below frontier | M | M | Lean on EU-multilingual differentiation; manage expectations |
| R17 | Bus factor | M | M | Doc-first; pair on critical systems |

---

## 10. Open decisions

Carried from v0.1, plus new:

1. **Project name** (still placeholder "EuroDLLM"). Defer until founding board picks.
2. **Dense 70B vs MoE 100B for Phase 3.** Decision at month 7 based on Phase 2 learnings.
3. **Stichting seat**: Amsterdam recommended.
4. **First anchor partner candidate**: outreach to LUMI AI Factory (FI) + JUPITER AI Factory (DE) + BSC (ES) first; they have the highest single-site GPU density and a research-friendly culture.
5. **First Tier-B cloud sponsor**: Hetzner is most likely to engage early (smaller, founder-led, EU-sovereign positioning); Scaleway and OVH have more PR upside; Stackit (Schwarz Gruppe) has interesting strategic alignment.
6. **Worker client sandboxing**: Docker image hash-pinned or seccomp-wrapped subprocess. Security audit recommends.
7. **Non-EU contributors**: accept compute, restrict model access to EU. Confirmed.
8. **Inference hosting**: self-host on Tier-B sponsor donated capacity + partial OVH/Scaleway rental.
9. **Post-training data**: mix synthesized (via partner labs) + translated open datasets + EU-specific instruction data.
10. **Tokenizer**: train our own 128k EU-balanced BPE.
11. **CoP voluntary adoption**: strong yes; sign Transparency + Copyright chapters in Phase 2, Safety & Security in Phase 5.
12. **DGA-DAO scope** (NEW): file scoped to "donated training-data shards and contributor metadata". Counsel to confirm.
13. **CRA Open Source Steward declaration** (NEW): publish by July 2026, ahead of Sept reporting-obligations start.

---

## 11. Immediate next actions (this week)

1. **Phase 0 verified** ✓ (smoke test passes on GPU, 22× CPU speedup).
2. **bf16 transport + 8-bit delta compression** — Phase 0.5 patch to coord + serialize, ~1 day.
3. **Coordinator on Netcup VPS** — deploy current code to `dllm.planetbass.de` behind nginx + Let's Encrypt; test one remote worker.
4. **Sketch the wire protocol v2** (gRPC + Protobuf for Phase 1; current Phase-0 HTTP is fine for prototyping).
5. **Outreach drafts**: Anchor (LUMI / BSC / JUPITER), Tier-B (Hetzner, Scaleway, OVH).
6. **IP counsel engagement** — Bird & Bird (UK/EU), Osborne Clarke (DE), Fieldfisher, iRights.Law (DE). Brief them on EU-RCL + DGA-DAO + CRA Open Source Steward.
7. **Stichting paperwork**: notary, KvK registration, draft governance.
8. **Public manifesto** — separate from this plan; ≤1 page.
9. **EU Horizon call identification** — find a 2026/2027 open call we can apply to.
10. **Reserve trademark candidates** at EUIPO.

---

## 12. References

### Algorithms & systems
- DiLoCo — [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)
- INTELLECT-2 — [arXiv:2505.07291](https://arxiv.org/abs/2505.07291)
- OpenDiLoCo — [Prime Intellect](https://www.primeintellect.ai/blog/opendiloco)
- SWARM Parallelism — Ryabinin et al., ICML 2023
- Petals — [petals.dev](https://petals.dev)
- Hivemind — `github.com/learning-at-home/hivemind`
- PowerSGD — Vogels et al., NeurIPS 2019
- nanoGPT — `github.com/karpathy/nanoGPT`
- "What happens when nanochat meets DiLoCo?" — [arXiv:2511.13761](https://arxiv.org/abs/2511.13761)

### EU regulation
- [EU AI Act consolidated text](https://artificialintelligenceact.eu)
- [GPAI Code of Practice (July 2025)](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers)
- [DSM Directive Art. 4 (TDM)](https://eur-lex.europa.eu/eli/dir/2019/790/oj)
- [GDPR](https://gdpr.eu/)
- [Data Governance Act](https://digital-strategy.ec.europa.eu/en/policies/data-governance-act)
- [Data Altruism Organisations register](https://digital-strategy.ec.europa.eu/en/policies/data-altruism-organisations)
- [Cyber Resilience Act](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act)
- [CRA Open Source provisions](https://digital-strategy.ec.europa.eu/en/policies/cra-open-source)
- [Data Act](https://digital-strategy.ec.europa.eu/en/policies/data-act)
- [eIDAS 2.0 Digital Identity Wallet](https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET)
- [NIS2 Directive](https://digital-strategy.ec.europa.eu/en/policies/nis2-directive)

### EU AI infrastructure
- [EuroHPC AI Factories](https://www.eurohpc-ju.europa.eu/ai-factories_en)
- [AI Factory selections (Dec 2024, Mar 2025, Oct 2025)](https://digital-strategy.ec.europa.eu/en/policies/ai-factories)
- [EuroHPC supercomputing systems](https://www.eurohpc-ju.europa.eu/supercomputers/our-supercomputers_en)

### Comparable efforts
- BigScience BLOOM; Mistral; Aleph Alpha; Kyutai; EuroLLM (Unbabel); TRELLIS/Silo AI

---

## 13. Glossary

- **AI Factory** — EuroHPC JU initiative; AI-optimised supercomputing platforms + ecosystem support, hosted at existing EuroHPC sites.
- **DiLoCo** — Distributed Low-Communication training; inner AdamW + outer Nesterov.
- **DGA** — Data Governance Act.
- **DAO** (here) — Data Altruism Organisation Recognised in the Union (DGA Art. 16+). Not blockchain.
- **CRA** — Cyber Resilience Act.
- **Open Source Steward** — CRA category: legal person providing systematic support for FOSS, lighter obligations than commercial manufacturers.
- **GPAI** — General-Purpose AI model (EU AI Act).
- **CoP** — Code of Practice (EU AI Act voluntary compliance instrument).
- **MoE** — Mixture of Experts.
- **Pod** — small group of workers running pipeline-parallel within, data-parallel between.
- **Pseudo-gradient (Δθ)** — local−snapshot diff; unit of outer-loop communication.
- **TOPLOC** — locality-sensitive hashing for verifying untrusted inference rollouts.
- **SHARDCAST** — chunked peer-assisted broadcast of weights.
- **EU-RCL** — EU Residents Community License.
- **CC** — Compute Credit, non-transferable contribution ledger entry.
- **eIDAS** — EU electronic identification regulation; 2.0 introduces the EU Digital Identity Wallet.

---

*v0.2 — supersedes v0.1. Drafted 2026-05-25 after Phase 0 GPU verification. Next revision after counsel review of §5 and confirmed outreach with first anchor / Tier-B candidates.*
