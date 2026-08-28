# Credit-Card Data Pipeline

Turns Indian bank credit-card documentation into a structured, scoreable card database.

## Flow

```
bank sitemap.xml
   → card URL list                          (CreditCardScraper.ipynb, stage 1)
   → PDF crawl            → per_card_data/<BANK>/<CARD>/{main.pdf,TERMS/,OFFERS/,LOUNGE/}
   → text extraction      (pdfplumber / pypdf)
   → LLM merge-extraction (Cerebras gpt-oss-120b, resumable via state.json)
                          → card_attributes/<BANK>/<CARD>.json      [148 cards]
   → manual curation      → final_decision/<BANK>/<CARD>.json       [4 cards]
   → scoring + LLM report (credit_card_recommendation_engine.ipynb)
```

## Directories

| Path | Role | Regenerable |
|---|---|---|
| `per_card_data/` | Source PDF corpus. **4-card AXIS sample** — the full ~8.7 GB scrape is not in the repo. | yes, by re-scraping |
| `card_attributes/` | LLM extraction output, 148 cards. | yes, but costs ~2,100 LLM calls |
| `final_decision/` | Curated cards the recommendation engine actually reads. | no — hand-curated |
| `state.json` | Resume ledger of processed files. **Relative paths.** | yes (deleting forces full re-extraction) |

## Notes

- Run Jupyter **from this directory** — notebook paths are relative to it.
- `large_pdfs/` is an overflow directory the splitter creates on demand; it is gitignored
  and absent until you run the pipeline on oversized PDFs.
- Requires `CEREBRAS_API_KEY` (extraction) and `GROQ_API_KEY` (recommendation report).
