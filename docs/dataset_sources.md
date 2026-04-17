# Dataset Sources

This document summarizes the external source families used to assemble the current canonical dataset snapshot.

The repository dataset is curated, compact, and static by design.
It is not an automated live sync of the biomedical literature.

Official source references below were checked against their current official pages on April 13, 2026.

## Literature Search And Full-Text Sources

- `PubMed`
  Main biomedical literature search interface used for discovery and article screening.
  Official page: <https://pubmed.ncbi.nlm.nih.gov/>

- `PubMed Central (PMC)`
  Full-text biomedical archive used to access open articles and inspect evidence directly.
  Official page: <https://pmc.ncbi.nlm.nih.gov/>

- `PMC Open Access Subset`
  Reusable PMC subset relevant for corpus construction and text-mining-oriented workflows.
  Official page: <https://pmc.ncbi.nlm.nih.gov/tools/openftlist/>

- `Cochrane Library`
  High-value source for systematic reviews and evidence synthesis.
  Official site: <https://www.cochranelibrary.com/>

## Institutional And Applied Reference Sources

- `NIH Office of Dietary Supplements (ODS)`
  Institutional reference source for supplement background, mechanisms, and evidence framing.
  Official site: <https://ods.od.nih.gov/>

- `NIH ODS Fact Sheets`
  Ingredient-level fact sheets and topic pages used to frame claims and scope boundaries.
  Official site: <https://ods.od.nih.gov/factsheets/list-all/>

- `Australian Institute of Sport (AIS) Supplements`
  Applied athlete-oriented framework used for practical supplement categorization and evidence framing.
  Official site: <https://www.ais.gov.au/nutrition/supplements>

- `AIS Sports Supplement Framework`
  Practical framework used as a secondary reference for supplement categories and support context.
  Official page: <https://www.ais.gov.au/__data/assets/pdf_file/0005/1085711/36837_AIS-sports-supplements-framework-position-statement-contextual-information.pdf>

## Domain-Specific Position Statements

- `Journal of the International Society of Sports Nutrition (JISSN)`
  Position stands and review articles relevant to creatine, caffeine, protein, and related sports-nutrition claims.
  Official journal site: <https://jissn.biomedcentral.com/>

- `ISSN Position Stands`
  Official position papers published through the society journal were used as high-value domain references.
  Example official position stand page: <https://jissn.biomedcentral.com/articles/10.1186/s12970-017-0177-8>

## ML Dataset Sources

The auxiliary classifier dataset in `data/ml/claim_type_dataset.csv` was assembled from:

- `matrix_scope.csv` example claims
- `reasoning_eval_cases.csv` claims with known labels
- PubMed-derived paraphrases from sports nutrition position-stand literature
- controlled domain augmentations to diversify phrasings across claim categories

Each row records its provenance in the `source` column.

## ML Dataset Sources

The claim-type classification dataset (`data/ml/claim_type_dataset.csv`) was derived from:

- `matrix_scope.csv` example claims (direct extraction)
- `reasoning_eval_cases.csv` claim texts with known labels (direct extraction)
- ISSN position stand abstracts retrieved from PubMed (claim-level paraphrases)
- Domain-vocabulary augmentations (systematic combination of known ingredients and outcome terms)

Each row in the dataset records its provenance in the `source` column.

Key PubMed references used for deriving training examples:

- ISSN position stand on caffeine and exercise performance (PMID 33388079)
- ISSN position stand on beta-alanine (PMID 26175657)
- ISSN position stand on creatine supplementation (PMID 28615996)
- Common questions and misconceptions about creatine supplementation (PMID 33557850)
- Protein distribution and muscle-building (PMID 29497353)
- Dietary protein and muscle hypertrophy with resistance exercise (PMID 29414855)
- BCAAs and muscle protein synthesis (PMID 28852372)
- BCAA supplement timing and exercise-induced muscle soreness (PMID 28944645)

## Notes

- The current corpus is not derived from one single source.
- The evidence snapshot was assembled from a mix of literature databases, institutional references, and domain-specific reviews.
- Repository updates may refine keywords, annotations, and benchmark expectations without implying a full external literature refresh.
