# Demand Forecasting Benchmark

An end-to-end benchmarking framework comparing statistical and ML-based forecasting models across different demand patterns. Designed for supply chain planning processes.

## Motivation

This project answers the question "Which model should you use, and under what condition?"

The benchmarking pipeline:
1. Demand classification
2. 8 statistical models (including three intermittent-specific methods)
3. 2 ML models with supply-chain-relevant feature engineering
4. Walk-forward (time-series cross-validation) — no data leakage
5. Supply chain metrics: WMAPE, Forecast Accuracy, Bias%, Coverage Rate
6. Interactive Streamlit dashboard for exploration

## Demand classification

Demand classification is the process of sorting our SKU portfolio into groups based on the nature of their demand patterns, so you we apply the right forecasting method, inventory policy, and planning attention to each group rather than treating everything the same way.

For classifying the demand, we will use the **SBC (Syntectos-Boylan-Croston)** framework. This framework classifies the demand based on two parameters - the **Average Demand Volume** and the **Demand Variability**. To be specific, it uses two metrics. 

1. ADI (Average Demand Interval): the average number of periods between non-zero demand. This captures the irregularity of the demand.  
2. CV<sup>2</sup> (Coefficient of Variation): this captures how *consistent* the demand is when it does occur.

Based on the above metrics, the demand can fall into one of 4 quadrants. 

|                         | Low CV² (stable quantity) | High CV² (lumpy quantity) |
| ----------------------- | ------------------------- | ------------------------- |
| Low ADI (frequent)      | Smooth                    | Erratic                   |
| High ADI (intermittent) | Intermittent              | Lumpy                     |

High-volume SKUs have rich demand histories which means the model has enough data to learn from. Low volume SKUs are statistically sparse, and small fluctuations can look like large percentage swings. Similarly, a low CV means demand is predictable and regular, and a high CV means demand is lumpy and hard to forecast. 

## Statistical forecasting

### Demand types covered

| Type         | ADI (Avg Demand Interval) | (CV) squared | Examples                               |
| ------------ | ------------------------ | ------------ | --------------------------------------- |
| Smooth       | < 1.32                   | < 0.49       | Fast-moving FMCG, high-volume generics  |
| Erratic      | < 1.32                   | ≥ 0.49       | Promotional items, hospital consumables |
| Intermittent | ≥ 1.32                   | < 0.49       | Slow-moving pharma, MRO spares          |
| Lumpy        | ≥ 1.32                   | ≥ 0.49       | Rare APIs, project-based demand         |

## Project structure

```
demand-forecast-benchmark/
├── data/
│   └── generate_data.py        # Synthetic demand generation (SBC-classified)
├── src/
│   ├── models/
│   │   ├── statistical.py      # Croston, SBA, TSB, ETS, ARIMA, baselines
│   │   └── ml_models.py        # XGBoost, LightGBM + feature engineering
│   ├── evaluation.py           # WMAPE, FA%, Bias%, Coverage, Diebold-Mariano
│   └── visualizations.py       # Matplotlib/Seaborn charts
├── notebooks/
│   ├── 01_eda.ipynb            # Demand pattern exploration
│   └── 02_benchmarking.ipynb  # Full benchmark walkthrough (narrative)
├── outputs/                    # Generated results (gitignored)
├── app.py                      # Streamlit interactive dashboard
├── run_benchmark.py            # CLI entry point
└── requirements.txt
```

