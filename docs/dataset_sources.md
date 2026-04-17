# Dataset Sources

This document summarizes the main information sources used to build the dataset.

These sources were used to support:

- document discovery
- source screening
- corpus construction
- evidence extraction
- claim framing
- benchmark design

## Core Literature Sources

- `PubMed`
  Biomedical literature search platform used to identify relevant papers and filter by article type, date, and availability.

- `PubMed Central (PMC)`
  Free full-text repository of biomedical and life sciences articles.

- `PMC Open Access Subset`
  Reusable subset of PMC suitable for text mining and corpus construction.

- `Cochrane Library`
  High-value source for systematic reviews and health evidence synthesis.

## Institutional Sources

- `NIH Office of Dietary Supplements (ODS)`
  Institutional portal with scientific information on dietary supplements.

- `NIH ODS Fact Sheets`
  Technical and outreach fact sheets on supplement ingredients and related evidence.

- `Australian Institute of Sport (AIS) - Supplements`
  Applied framework for sports supplements and athlete-oriented evidence review.

- `AIS Group A Supplements`
  Reference subset of supplements with stronger practical support within the AIS framework.

## Domain-Specific Reference Sources

- `International Society of Sports Nutrition (ISSN)`
  Reference society in sports nutrition.

- `ISSN Position Stands Collection`
  Collection of position stands useful for topics such as creatine, caffeine, protein, and nutrient timing.

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

The dataset is not derived from a single source.
It was assembled from a combination of literature databases, institutional references, and domain-specific review material.
