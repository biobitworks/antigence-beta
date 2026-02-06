# IMMUNOS Lexicon

## Antigents
**Definition**: A blended term for *antigen + agents*. Antigents are task‑specialized AI agents that detect, classify, and respond to inputs as if they were immune antigens.

**Usage**:
- Antigents are tagged with an immune cell role (e.g., NKCell, BCell, Dendritic).
- Antigents can be grouped into domain packs (e.g., Research Verification, Code Security).
- Antigents produce “self / non‑self” judgments to support trust and provenance.

**Notes**:
- Antigents are not datasets; they are active agents that consume datasets and tools.
- SciFact is a dataset and pipeline feature; it can be wrapped by a Research Verification antigent.

## Antigence™ (TM pending)
**Definition**: Branding term for *antigen + intelligence*. Intended to describe the overall platform or operating system that coordinates antigents.

**Usage**:
- Antigence™ can refer to the full platform (core, domain packs, and secure self boundary).
- If used publicly, always clarify that immunOS is the technical architecture behind Antigence™.

## Antigentic
**Definition**: Adjective for systems or workflows that are designed around antigen detection and immune-style response logic.

**Usage**:
- Antigentic workflows emphasize self/non-self boundaries and auditable provenance.
- Antigentic agents are specialized antigents aligned to a domain pack.

## Antigent (singular)
**Definition**: A single antigen-specialized agent instance within a domain pack.

**Usage**:
- Use singular to describe one agent (e.g., a Network NK antigent).
- Use plural “antigents” for the full set in a domain pack.

## Antigen Receptors (immunOS)
**Definition**: In immunOS, antigen receptors are computational detectors that operate on feature representations of inputs rather than genetic sequences.

**Mapping**:
- B Cell receptors: trained classifiers that map claim/evidence feature vectors to labels (support/contradict/NEI).
- NK receptors: negative-selection detector vectors in Dendritic feature space used to flag non-self/anomalous claims.
- Dendritic presentation: feature extraction layer that defines the antigen representation exposed to B/NK cells.
- Memory: stored validated patterns that act like reusable receptors over time.

## GANs (Generative Adversarial Networks)
**Definition**: A training setup with two neural networks in competition: a generator creates samples, a discriminator detects fakes. The objective is to produce realistic synthetic data.

**How immunOS is similar**
- Both use adversarial logic: detection vs generation.
- Both improve via iterative feedback.

**How immunOS is different**
| Dimension | GANs | immunOS |
| --- | --- | --- |
| Primary goal | Generate realistic data | Verify outputs/actions as self vs non-self |
| Architecture | 2 neural nets (G/D) | Multi-agent immune stack (NK/B/Dendritic/Memory/T) |
| Training signal | Adversarial loss | Domain features + evidence + policy/provenance |
| Deployment | Centralized model training | Local-first, air-gapped capable |
| Safety model | No provenance or policy gates | Signed policies + tamper-evident logs |

**Use in immunOS (safe pattern)**
- GANs can generate candidate antigen receptors in a sandboxed training environment.
- Candidates must be validated on held-out data and signed before promotion.
- The cryptographic core (self) should never be modified by GAN output directly.

**Lessons immunOS can adopt**
- Use adversarial “red team” generators to stress-test detectors.
- Emphasize hard-negative generation for more robust NK detectors.

## Trademark + First Use Tracking

**Note**: First-use dates below are based on document dates and internal publication logs. Verify actual
public first-use dates when a term is posted publicly (e.g., Zenodo or website).

| Term | Type | First documented use | Source | Notes |
| --- | --- | --- | --- | --- |
| Antigence™ | Brand | 2025-12-24 | immunos-preprint/publication/manuscript/immunos-preprint.md | Document date; update with public first-use. |
| antigent | Term | 2025-12-24 | immunos-preprint/publication/manuscript/immunos-preprint.md | Document date; update with public first-use. |
| antigents | Term | 2025-12-24 | immunos-preprint/publication/manuscript/immunos-preprint.md | Document date; update with public first-use. |
| antigentic | Term | 2025-12-24 | immunos-preprint/publication/manuscript/immunos-preprint.md | Document date; update with public first-use. |
