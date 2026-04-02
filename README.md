# Demand Forecasting Benchmark

An end-to-end benchmarking framework comparing statistical and ML-based forecasting models across different demand patterns. Designed for supply chain planning processes.

## Motivation

This project answers the question "Which model should you use, and under what condition?"

The benchmarking pipeline:
1. Demand classification module
2. Statistical baseline module (8 statistical models + 2 ML models)
4. Walk-forward (time-series cross-validation) — no data leakage
5. Supply chain metrics: WMAPE, Forecast Accuracy, Bias%, Coverage Rate
6. Interactive Streamlit dashboard for exploration

### Datasets considered
1. M5 Forecasting dataset: sales_train_evaluation (https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

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

### How to use
If running the M5 dataset, run the command below while in the root directory.
```
python src/classifier/run_classification.py --source m5
```
You'll get the result in `data/results/classifications.csv` with SKU wise classification of the demand type. This output can be considered as the updated segmentation master, a table that maps every SKU to a forecasting method, a review frequency and a replenishment policy. 

## Statistical forecasting

### Demand types covered

| Type         | ADI (Avg Demand Interval) | (CV) squared | Examples                               |
| ------------ | ------------------------ | ------------ | --------------------------------------- |
| Smooth       | < 1.32                   | < 0.49       | Fast-moving FMCG, high-volume generics  |
| Erratic      | < 1.32                   | ≥ 0.49       | Promotional items, hospital consumables |
| Intermittent | ≥ 1.32                   | < 0.49       | Slow-moving pharma, MRO spares          |
| Lumpy        | ≥ 1.32                   | ≥ 0.49       | Rare APIs, project-based demand         |

As a thumb rule, we associate different demand types to different forecasting methods.

- Smooth SKUs → statistical forecasting (ETS or ARIMA), reviewed monthly in consensus
- Intermittent SKUs → Croston or SBA, higher safety stock multiplier, less frequent review
- Lumpy SKUs → often managed manually or on a make-to-order basis
- Erratic SKUs → ML models or causal forecasting if drivers can be identified (promotions, weather)

## Models
 
### Statistical Models
 
| Model | Best For | Notes |
|---|---|---|
| **Simple Moving Average** | Smooth | Baseline reference |
| **Exponential Smoothing (SES)** | Smooth, Erratic | Holt-Winters variant available |
| **Croston's Method** | Intermittent | Separate smoothing for intervals and sizes |

### Machine Learning Models
 
| Model | Best For | Notes |
|---|---|---|
| **XGBoost** | All classes | Lag features, calendar features, rolling stats |
| **LightGBM** | All classes | Faster training; handles high-cardinality categoricals |

## Evaluation