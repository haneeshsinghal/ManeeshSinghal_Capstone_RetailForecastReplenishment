# Retail Demand Forecasting & Inventory Replenishment Planner

## Project Overview

This project implements an **end-to-end analytics and inventory planning solution** for a multi-store retail business facing two structurally opposing challenges:

- **Stockouts**, resulting in lost sales and poor customer experience  
- **Overstock**, leading to blocked working capital, higher holding costs, and potential waste  

The solution integrates **demand forecasting**, **inventory risk monitoring**, and a **reorder point–based replenishment policy** into an interactive **Tableau dashboard**. It enables consistent, data-driven purchase and replenishment decisions at a granular **store–SKU level**.

---

## North Star KPI

### Fill Rate (Service Level)

> Measures the proportion of true customer demand that is fulfilled from available inventory.

Improving Fill Rate—while maintaining disciplined inventory levels—is the primary objective of this project. It directly reflects customer experience and is operationally controllable through forecasting accuracy and replenishment policy levers.

---

## Forecasting Methods Used

Demand forecasting is performed at the **Store–SKU–Day** level with a **28-day (4-week) planning horizon**.

### Demand Signal / Forecast Proxy

- **Baseline demand proxy:** `avg_daily_demand`, calculated from the most recent 4–8 weeks of historical sales
- This proxy is used consistently across:
  - Lost sales estimation
  - Fill rate computation
  - Replenishment and safety stock calculations

This ensures alignment between forecasting, performance measurement, and ordering logic.

---

### Forecast Accuracy Metrics

The following metrics are computed to assess forecast quality and build decision confidence:

- **MAPE (Mean Absolute Percentage Error)**  
  Average daily percentage error (excluding zero-sales days)

- **WAPE (Weighted Absolute Percentage Error)**  
  Volume-weighted error metric that remains stable across both high- and low-volume SKUs

- **Forecast Bias**  
  Signed error indicating systematic over-forecasting or under-forecasting

These metrics are used for **diagnostic and validation purposes only**; forecasting models are not tuned dynamically within Tableau.

---

## End-to-End ETL Flow (How to Run)

### Step 1: Raw Data Inputs

Copy the following raw data files into the `/etl` directory alongside `etl_pipeline.py`:

- `calendar.csv`
- `inventory_daily.csv`
- `products.json`
- `purchase_orders.csv`
- `sales_daily.csv`
- `stores.csv`

### Step 2: Execute the ETL Pipeline

From the project root, run:

```bash
cd etl
python etl_pipeline.py
```
## Curated Output Files Generated

The ETL pipeline generates the following curated datasets in the `/data/` directory:

### 1. `fact_sales_store_sku_daily.csv`

- `date`, `store_id`, `sku_id`
- `units_sold`, `revenue`
- `promo_flag`, `holiday_flag`, `day_of_week`

---

### 2. `fact_inventory_store_sku_daily.csv`

- `date`, `store_id`, `sku_id`
- `on_hand_units`
- `stockout_flag`
- `days_of_cover`

---

### 3. `replenishment_inputs_store_sku.csv`

- `avg_daily_demand`
- `demand_std_dev`
- `lead_time_days`
- `service_level_target`
- `safety_stock`
- `reorder_point`
- `recommended_order_qty`

These datasets collectively form the **analytical backbone** for forecast evaluation, inventory risk monitoring, and replenishment decision-making.

---

## Dashboard Tool Used

### Tableau Public

#### How to Open the Dashboard

1. Open **Tableau Public**
2. Load the packaged workbook:  
   `RetailForecastReplenishment.twbx` from the `/dashboard/` directory
3. Navigate across the following dashboard views:
   - Executive Summary
   - Forecast Explorer
   - Inventory Risk Monitor
   - Replenishment Planner

---

## Data Modeling Notes

- All calculations and business logic are embedded directly within the Tableau workbook
- Data is modeled using **Tableau Relationships (Logical Layer)** only
- No physical joins are used

### Relationships Defined

- Sales ↔ Inventory on `(store_id, sku_id, date)`
- Sales ↔ Replenishment Inputs on `(store_id, sku_id)`

This modeling approach avoids grain mismatch, prevents double counting, and ensures accurate aggregations.

---

## PART A – Business Framing

For detailed problem framing, metric rationale, and stakeholder alignment, refer to:

**`final_story/PartA_Framing.pdf`**