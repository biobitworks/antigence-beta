# Antigence Alpha Essentials

This checklist enumerates the minimal, security-conscious release set for Antigence Alpha and the
canonical sources inside `<workspace>` (e.g., your local projects directory).

## Must-Have (Alpha Release)

| Item | Alpha Location | Canonical Source | Purpose |
| --- | --- | --- | --- |
| README | antigence-alpha/README.md | (local) | Release scope + version label |
| SOURCE_MAP | antigence-alpha/SOURCE_MAP.md | (local) | Traceability to canonical sources |
| .gitignore | antigence-alpha/.gitignore | (local) | Prevent accidental secrets/logs |
| Tier 0 manuscript | antigence-alpha/manuscript/immunos-preprint-tier0.md | immunos-preprint/publication/manuscript/immunos-preprint-tier0.md | Zenodo Tier 0 content |
| Tier 0 short | antigence-alpha/manuscript/immunos-preprint-tier0-short.md | immunos-preprint/publication/manuscript/immunos-preprint-tier0-short.md | Zenodo short summary |
| Tier 1 outline | antigence-alpha/manuscript/immunos-preprint-tier1-outline.md | immunos-preprint/publication/manuscript/immunos-preprint-tier1-outline.md | Evaluation plan |
| Branding | antigence-alpha/docs/branding.md | docs/kb/branding.md | Naming + usage |
| Lexicon | antigence-alpha/docs/lexicon.md | docs/kb/lexicon.md | Antigence/antigent definitions |
| Pitch | antigence-alpha/docs/pitch.md | docs/kb/pitch.md | Grant/partner summary |
| Substack draft | antigence-alpha/blog/antigence-tier0-zenodo-substack.md | immunos-preprint/publication/blog/antigence-tier0-zenodo-substack.md | Announcement copy |

## Should-Have (Security + Governance)

| Item | Suggested Alpha Location | Canonical Source | Purpose |
| --- | --- | --- | --- |
| LICENSE | antigence-alpha/LICENSE | (choose) | Open-source terms for alpha |
| SECURITY policy | antigence-alpha/SECURITY.md | docs/policies/retention-redaction.md (adapt) | Disclosure channel + data handling |
| IR playbook | antigence-alpha/IR-PLAYBOOK.md | docs/ir-playbook.md (adapt) | Incident response summary |
| Release notes | antigence-alpha/CHANGELOG.md | (local) | Version history |

## Nice-to-Have (Operational)

| Item | Suggested Alpha Location | Canonical Source | Purpose |
| --- | --- | --- | --- |
| Citation | antigence-alpha/CITATION.cff | (local) | DOI/citation metadata |
| Contribution | antigence-alpha/CONTRIBUTING.md | (local) | Contributor workflow |
| Code of Conduct | antigence-alpha/CODE_OF_CONDUCT.md | (local) | Community norms |

## Security Guardrails for Alpha

- Do not copy `.immunos/`, `.claude`, or local logs into alpha.
- Avoid raw datasets or large assets; keep the alpha repo text-only.
- Use the SOURCE_MAP to link to canonical paths instead of duplicating data.
- Keep “TM pending” in Antigence naming until filing status changes.

## Next Steps

1. Decide license (CC-BY-4.0 vs Apache-2.0 vs MIT) for alpha artifacts.
2. Add SECURITY + IR stubs if publishing publicly.
3. Add CHANGELOG and CITATION if Zenodo release is imminent.
