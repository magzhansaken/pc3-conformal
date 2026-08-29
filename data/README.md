# Data folder

The paper uses eight public datasets. Most are obtained automatically:

| Dataset | How it is obtained |
|---------|--------------------|
| **UCI Concrete** | Downloaded automatically by `pc3.py`. |
| **DFT elastic moduli** (`elastic_tensor_2015`, de Jong et al., 2015) | Loaded programmatically by `elastic_experiment.py` via `matminer` (`load_dataset("elastic_tensor_2015")`). |
| **Textile-polymer composite** (Malashin et al., 2024) | Downloaded automatically by `composite_real_experiments.py` from `github.com/catauggie/TPCM` and cached here as `Polymer_TPCM.xlsx`. |
| **Steel-fibre-reinforced concrete** (Shafighfard et al., 2022) | **Manual:** download from Mendeley Data, `doi:10.17632/hjrfgys29n.1`, and place `SFRC_Data_v1.xlsx` in this folder. |
| **UHPC compilation** (Chen494820) | **Manual:** from `github.com/Chen494820/code-and-dataset`, save `Raw material data.csv` here as `uhpc.csv`. |
| **SCC after elevated temperature** (Quanchaochao) | **Manual:** from `github.com/Quanchaochao/Explainable-prediction-model-for-high-temperature-compressive-strength-of-self-compacting-concrete`, save `real_data.csv` here as `scc_ht.csv`. |
| **Metakaolin geopolymer** (223 records) and **hybrid alkali-activated concrete** (262 records) | **Manual:** from `github.com/tkjafla`, save as `mk_geopolymer.csv` and `hybrid_aac.csv`. |

For the grouped validation (`grouped_families.py`) the UCI Concrete file auto-caches here as
`concrete.csv`; the Eurocode 2 decay envelope (EN 1992-1-2, siliceous) is coded in the script.

Only the SFRC, UHPC, SCC, MK-geopolymer and AAC files must be placed here by hand (Mendeley does not allow direct
programmatic download). Everything else is fetched on first run.
