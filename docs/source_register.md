# Source Register

| Source | Type | Refresh | Implementation |
|---|---|---|---|
| Synthetic transactions | Synthetic | Daily | `synthetic/generate_transactions.py` |
| Synthetic inventory | Synthetic | Daily | `synthetic/generate_inventory.py` |
| Synthetic customers | Synthetic | Daily | `synthetic/generate_crm.py` |
| Synthetic campaigns | Synthetic | Daily | `synthetic/generate_crm.py` |
| Synthetic costs | Synthetic | Weekly | `synthetic/generate_costs.py` |
| Google Trends | External API | Weekly | `ingestion/google_trends.py` |
| Open-Meteo weather | External API | Daily | `ingestion/open_meteo_weather.py` |
| UK bank holidays | External API | Annual | `ingestion/uk_bank_holidays.py` |
| ONS retail sales | External API | Monthly | `ingestion/ons_retail_sales.py` |
| ONS internet sales | External API | Monthly | `ingestion/ons_internet_sales.py` |
| UCI online retail | External dataset | Ad hoc | `ingestion/uci_online_retail.py` |

## Notes

- Demo mode is fully supported through the synthetic generators under `synthetic/`.
- External source wrappers remain available for future production integration.
- Daily and weekly pipelines currently use synthetic inputs unless real extracts are staged into `data/synthetic`.