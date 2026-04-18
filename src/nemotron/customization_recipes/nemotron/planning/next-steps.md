# Finding Your Data

Most organizations beginning a sovereign AI program already have the data they need. It is not labeled as training data. It lives in filing systems, staff portals, help desks, policy repositories, and archived communications. The first step is learning to see it.

This page walks through five real use cases and shows what data sources each organization is likely sitting on, what it takes to surface them, and what to do with them first. The goal is not to hand you a checklist — it is to help you recognize what you already have before you start looking for something you don't.

One principle applies across all of them: **build your evaluation set before your training set.** A small collection of expert-written question-and-answer examples, drawn from your own materials, tells you what good looks like. Without it, you have no way to know whether tuning helped.

## Ministry of Public Health, Sri Lanka
*Sinhala and English assistant for rural clinic guidance and patient-intake support*

| Source | How to surface it | Main constraint | First use |
|--------|------------------|-----------------|-----------|
| Internal ministry guidance | Clinic protocols, intake forms, triage guides, and public health circulars | Internal review and de-identification | Build eval examples, then SFT |
| Public health websites | Ministry pages, WHO guidance, and public patient advisories | Cleanup and deduplication | Curate for baseline domain data |
| Hospital FAQ material | Recurring staff or patient questions from help desks and support channels | Privacy review | Create instruction-style examples |
| Translated English health content | Trusted English health guidance adapted into Sinhala | Terminology review | Run Stage 0 translation |
| Synthetic clinic dialogues | Staff-patient and staff-assistant conversations generated from approved protocols | Grounding in real policy | Bootstrap conversational SFT data |

---

## Department of Agriculture, Kenya
*Swahili and English support for crop advisory and extension workers*

| Source | How to surface it | Main constraint | First use |
|--------|------------------|-----------------|-----------|
| Extension manuals | Bulletins, field guides, and agronomy documents used by extension staff | Formal tone may need adaptation | Domain eval and SFT data |
| Open agricultural publications | FAO, CGIAR, ministry publications, and pest guidance sources | Mixed formats | Curate and cluster by task |
| Farmer-facing messages | SMS advisories, radio scripts, and plain-language notices | Permissions and normalization | Tone and response style reference |
| Swahili-translated agronomy content | High-value English crop and pest guidance translated into Swahili | Terminology consistency | Stage 0 translation and review |
| Synthetic crop Q&A | "What should I do if…" examples generated from official protocols | Needs verification against trusted sources | Starter instruction set |

---

## National Archives Authority, Jordan
*Arabic document search, summarization, and triage for internal staff*

| Source | How to surface it | Main constraint | First use |
|--------|------------------|-----------------|-----------|
| Filing manuals and records policies | Internal rules governing classification and routing | Access approvals | Triage benchmarks |
| Public archival collections | Open records, metadata, and archive descriptions | May not reflect internal workflows | Search and summarization evals |
| Archivist annotations | A small sample of records labeled by type, topic, or routing path by practicing archivists | Expert time | Gold benchmark set |
| OCR'd scanned records | Text extracted from scanned archival material | Cleanup burden | Test noisy-text handling |
| Synthetic metadata-to-summary pairs | First-pass summaries generated from archive metadata and document descriptors | Can become generic without grounding | Bootstrap summarization data |

---

## Provincial Department of Justice, Quebec
*French and English legal research and policy memo drafting*

| Source | How to surface it | Main constraint | First use |
|--------|------------------|-----------------|-----------|
| Public legal texts | Statutes, regulations, tribunal decisions, and policy pages | Weak on internal memo style | Baseline legal eval |
| Internal templates and guidance | Memo templates, drafting guides, and internal policy standards | Privilege and confidentiality review | Style-aware SFT set |
| Bilingual legal material | Aligned French-English legal and policy text | Alignment and formatting work | Terminology consistency eval |
| Expert-written gold examples | Strong answers and memo outlines written by legal staff | Expensive expert time | Benchmark before tuning |
| Synthetic legal drafting prompts | Memo and policy questions generated from public legal materials | Needs expert spot-checking | Bootstrap instruction tuning |

---

## Ministry of Education, Morocco
*Darija, Arabic, and French assistant for curriculum lookup and teacher support*

| Source | How to surface it | Main constraint | First use |
|--------|------------------|-----------------|-----------|
| Curriculum and policy documents | Official standards, circulars, and teacher guidance | Restructuring and cleanup | Curriculum QA benchmark |
| Public ministry portals | School policy pages, curriculum portals, and teacher-facing content | Duplication and uneven formatting | Curate for eval and retrieval |
| Teacher support communities | Recurring teacher questions, training notes, and classroom support material | Permissions and consistency | Practical instruction data |
| Adapted Darija content | Arabic and French guidance rewritten into teacher-friendly Darija | Naturalness review | Language adaptation set |
| Synthetic teacher dialogues | Classroom-support and policy-explainer exchanges generated from official documents | Careful grounding required | Bootstrap conversational SFT |

---

## The Common Pattern

Every example above follows the same logic, regardless of language, domain, or organization type.

**Tier 1 — What you already own.** Internal guidance, protocols, manuals, templates, and annotated records. This material is specific to your workflows and terminology. It is hard to replicate from public sources. Start here.

**Tier 2 — What is public and relevant.** Ministry websites, open publications, translated guidance, and public portals. Useful for domain coverage and baseline evaluation, but less distinctive than your internal material.

**Tier 3 — What you can generate.** Synthetic dialogues, Q&A pairs, and summary examples grounded in approved material. Useful for bootstrapping instruction data when real examples are sparse, but only as reliable as the source documents they are derived from.

**Build your benchmark first.** In each case, a small set of expert-written or expert-annotated examples comes before any large-scale data collection. That set defines what you are trying to achieve. Everything else is in service of it.

**Name your constraints early.** Privacy review, access approvals, terminology consistency, and expert time are the real blockers — not data volume. Surfacing these constraints in the first week is more valuable than collecting data you cannot use.
