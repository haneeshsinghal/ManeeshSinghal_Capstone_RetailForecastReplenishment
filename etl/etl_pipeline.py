import os
import sys
import subprocess
import logging
from pathlib import Path

REQUIRED_FILES = {
    "stores": "stores.csv",
    "products": "products.json",
    "inventory": "inventory_daily.csv",
    "sales": "sales_daily.csv",
    "purchase_orders": "purchase_orders.csv",
    "calendar": "calendar.csv",
}

LOG_LEVEL = "INFO"

# Set up logging to file and console with a consistent format
def setup_logging(log_file, level):
    logger = logging.getLogger("retail_elt")
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger

# Ensure directories exist for outputs and logs
def ensure_dirs_exist(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

# Install required libraries from a requirements.txt file using pip, with error handling
def install_requirements(package_file, logger):
    try:
        if not os.path.exists(package_file):
            logger.warning("requirements.txt not found at %s — skipping install", package_file)
            return

        logger.info("Installing dependencies from %s", package_file)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(package_file)
        ])
        logger.info("Dependency installation completed successfully")

    except subprocess.CalledProcessError as exc:
        logger.exception("pip install failed | Error: %s", exc)
        raise

# Resolve required input files and ensure they exist, returning a dict of absolute paths
def resolve_inputs_files(base_dir, required_files, logger):
    base_dir = Path(base_dir).resolve()
    missing = []
    found = {}

    for key, fname in required_files.items():
        file_path_name = base_dir / fname
        if not file_path_name.exists():
            missing.append(fname)
        else:
            found[key] = file_path_name.resolve()

    if missing:
        raise FileNotFoundError(
            "Missing required input files next to script: "
            + ", ".join(missing)
            + "\nExpected directory: "
            + str(base_dir)
        )

    logger.info("All required inputs found next to script in: %s", base_dir)
    return found


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# Utility functions to read CSV and JSON safely with error handling and logging
def read_csv_files(path, logger, **kwargs):
    try:
        path = Path(path)
        logger.info("Reading CSV: %s", path)
        df = pd.read_csv(path, **kwargs)

        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if unnamed:
            logger.warning("Dropping unnamed columns from %s: %s", path.name, unnamed)
            df = df.drop(columns=unnamed)

        return df
    except FileNotFoundError:
        logger.error("CSV file not found: %s", path)
        raise
    except Exception as exc:
        logger.exception("Failed to read CSV: %s | Error: %s", path, exc)
        raise


# Utility function to read JSON safely
def read_json_files(path, logger):
    try:
        path = Path(path)
        logger.info("Reading JSON: %s", path)
        return pd.read_json(path)
    except FileNotFoundError:
        logger.error("JSON file not found: %s", path)
        raise
    except Exception as exc:
        logger.exception("Failed to read JSON: %s | Error: %s", path, exc)
        raise

# Utility function to standardize ID columns by stripping whitespace and converting to uppercase, with error handling
def standardize_ids(df, cols, logger=None):
    try:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        return df
    except Exception as exc:
        if logger:
            logger.exception("Failed to standardize ID columns %s | Error: %s", cols, exc)
        raise

# Utility function to normalize category columns to lowercase snake_case and fill missing values with 'unknown'
def normalize_category(series):
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

# Utility function to coerce datetime columns safely with error handling
def coerce_datetime(df, col, logger=None):
    try:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as exc:
        if logger:
            logger.exception("Failed to convert datetime for column '%s' | Error: %s", col, exc)
        raise

# Utility function to create a complete daily grid 
def create_complete_daily_grid(df, date_col, key_cols, logger, min_date=None, max_date=None):
    try:
        out = df.copy()
        out = coerce_datetime(out, date_col,logger)

        if min_date is None:
            min_date = out[date_col].min()
        if max_date is None:
            max_date = out[date_col].max()

        if pd.isna(min_date) or pd.isna(max_date):
            raise ValueError("%s contains no valid dates after parsing." % date_col)

        all_dates = pd.date_range(min_date, max_date, freq="D")
        logger.info(
            "Building complete daily grid %s..%s for keys=%s",
            str(min_date.date()), str(max_date.date()), key_cols
        )

        parts = []
        grouped = out.groupby(key_cols, dropna=False)

        for keys, g in grouped:
            g = g.sort_values(date_col).set_index(date_col)
            g = g.reindex(all_dates)

            if not isinstance(keys, tuple):
                keys = (keys,)
            for i, kc in enumerate(key_cols):
                g[kc] = keys[i]

            g = g.reset_index().rename(columns={"index": date_col})
            parts.append(g)

        return pd.concat(parts, ignore_index=True)

    except Exception as exc:
        if logger:
            logger.exception("Failed to create complete daily grid | Error: %s", exc)
        raise

# This is a simple implementation of outlier detection using the IQR method.
def add_outlier_flag_iqr_by_group(df, group_cols, value_col, flag_col, logger):
    try:
        out = df.copy()
        if value_col not in out.columns:
            logger.warning("Value column %s not present; cannot compute outliers.", value_col)
            out[flag_col] = False
            return out

        q1 = out.groupby(group_cols)[value_col].transform(lambda s: s.quantile(0.25))
        q3 = out.groupby(group_cols)[value_col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        out[flag_col] = (out[value_col] < lower) | (out[value_col] > upper)
        logger.info("Outlier flag added: %s (IQR by %s on %s)", flag_col, group_cols, value_col)
        return out

    except Exception as exc:
        if logger:
            logger.exception("Failed outlier detection | Error: %s", exc)
        raise

# This function saves a boxplot of the specified value column to the given path.
def save_boxplot_image(df, value_col, title, out_path, logger):
    try:        
        if value_col not in df.columns:
            logger.warning("Cannot save boxplot: %s not present.", value_col)
            return

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving boxplot -> %s", out_path)

        plt.figure(figsize=(12, 5))
        sns.boxplot(x=df[value_col].dropna())
        plt.title(title)
        plt.xlabel(value_col)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        logger.info("Boxplot saved successfully: %s", out_path)

    except Exception as exc:
        logger.exception("Failed to save boxplot | Error: %s", exc)
        raise

# Function to clean and standardize the calendar dataset by coercing dates, normalizing flags, and deduplicating
def clean_calendar_dataset(calendar, logger):
    try:
        logger.info("Cleaning calendar")
        cal = calendar.copy()
        cal = coerce_datetime(cal, "date", logger)

        for col in ["day_of_week", "is_weekend", "promo_flag", "holiday_flag"]:
            if col in cal.columns:
                cal[col] = pd.to_numeric(cal[col], errors="coerce").fillna(0).astype(int)

        cal = cal.drop_duplicates(subset=["date"]).sort_values("date")
        return cal

    except Exception as exc:
        logger.exception("Failed to clean calendar | Error: %s", exc)
        raise

# Function to clean and standardize the stores dataset by standardizing IDs, normalizing region and store size, and ensuring city tier is numeric
def clean_stores_dataset(stores, logger):
    try:
        logger.info("Cleaning stores")
        st = stores.copy()
        st = standardize_ids(st, ["store_id"], logger)

        if "region" in st.columns:
            st["region"] = st["region"].astype(str).str.strip().str.upper()
        if "city_tier" in st.columns:
            st["city_tier"] = pd.to_numeric(st["city_tier"], errors="coerce")
        if "store_size" in st.columns:
            st["store_size"] = st["store_size"].astype(str).str.strip().str.upper()

        st = st.drop_duplicates(subset=["store_id"])
        return st

    except Exception as exc:
        logger.exception("Failed to clean stores | Error: %s", exc)
        raise

# Function to clean and standardize the products dataset by standardizing IDs, normalizing category, and ensuring price, cost, shelf life, and MOQ are numeric
def clean_products_dataset(products, logger):
    try:
        logger.info("Cleaning products")
        pr = products.copy()
        pr = standardize_ids(pr, ["sku_id"], logger)

        if "category" in pr.columns:
            pr["category"] = normalize_category(pr["category"])

        for col in ["price", "cost", "shelf_life_days", "moq_units"]:
            if col in pr.columns:
                pr[col] = pd.to_numeric(pr[col], errors="coerce")

        return pr

    except Exception as exc:
        logger.exception("Failed to clean products | Error: %s", exc)
        raise

# Function to clean and standardize the purchase orders dataset by standardizing IDs, coercing dates, calculating lead time, and deduplicating
def clean_purchase_orders_dataset(purchase_orders, logger):
    try:
        logger.info("Cleaning purchase_orders")
        po = purchase_orders.copy()
        po = standardize_ids(po, ["po_id", "store_id", "sku_id"], logger)

        po = coerce_datetime(po, "order_date", logger)
        po = coerce_datetime(po, "expected_receipt_date", logger)

        for col in ["order_qty", "lead_time_days"]:
            if col in po.columns:
                po[col] = pd.to_numeric(po[col], errors="coerce")

        if "po_id" in po.columns:
            po = po.drop_duplicates(subset=["po_id"], keep="first")
        else:
            subset = []
            for c in ["store_id", "sku_id", "order_date", "expected_receipt_date", "order_qty"]:
                if c in po.columns:
                    subset.append(c)
            if subset:
                po = po.drop_duplicates(subset=subset, keep="first")

        if "lead_time_days" in po.columns and "order_date" in po.columns and "expected_receipt_date" in po.columns:
            missing = po["lead_time_days"].isna()
            if missing.any():
                inferred = (po.loc[missing, "expected_receipt_date"] - po.loc[missing, "order_date"]).dt.days
                po.loc[missing, "lead_time_days"] = inferred

        return po

    except Exception as exc:
        logger.exception("Failed to clean purchase orders | Error: %s", exc)
        raise
    
# Function to build the fact_sales_store_sku_daily dataset by merging sales with product info and calculating revenue and margin proxy
def build_fact_sales(sales, calendar, products, logger):
    try:
        logger.info("Building fact_sales_store_sku_daily")

        out = sales.copy()
        # Standardize identifiers
        out = standardize_ids(out, ["store_id", "sku_id"], logger)

        # Date handling
        out = coerce_datetime(out, "date", logger)

        # Units sold
        if "units_sold" in out.columns:
            out["units_sold"] = pd.to_numeric(out["units_sold"], errors="coerce").fillna(0.0)
        else:
            logger.warning("sales missing units_sold; defaulting to 0")
            out["units_sold"] = 0.0

        # Deduplicate: sum units per day-store-sku
        out = (
            out.groupby(["date", "store_id", "sku_id"], as_index=False)
               .agg({"units_sold": "sum"})
        )

        # Ensure continuous daily series
        out = create_complete_daily_grid(
            out,
            date_col="date",
            key_cols=["store_id", "sku_id"],
            logger=logger
        )

        # Calendar enrichment
        cal_cols = ["date", "day_of_week", "promo_flag", "holiday_flag"]
        cal_cols = [c for c in cal_cols if c in calendar.columns]

        calendar_clean = calendar[cal_cols].drop_duplicates(subset=["date"])
        out = out.merge(calendar_clean, on="date", how="left")

        for c in ["day_of_week", "promo_flag", "holiday_flag"]:
            if c not in out.columns:
                out[c] = 0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

        # Product enrichment (price, cost)
        prod_cols = ["sku_id"]
        if "price" in products.columns:
            prod_cols.append("price")
        if "cost" in products.columns:
            prod_cols.append("cost")

        products_clean = (
            products[prod_cols]
            .drop_duplicates(subset=["sku_id"])
            .copy()
        )

        out = out.merge(products_clean, on="sku_id", how="left")

        # Revenue
        if "price" in out.columns:
            out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
            out["revenue"] = out["units_sold"] * out["price"]
        else:
            logger.warning("price not available; revenue set to 0")
            out["revenue"] = 0.0

        # Margin proxy
        if "price" in out.columns and "cost" in out.columns:
            out["cost"] = pd.to_numeric(out["cost"], errors="coerce").fillna(0.0)
            out["margin_proxy"] = out["units_sold"] * (out["price"] - out["cost"])
        else:
            logger.warning("cost/price not fully available; margin_proxy set to 0")
            out["margin_proxy"] = 0.0

        out["revenue"] = out["revenue"].round(3)
        out["margin_proxy"] = out["margin_proxy"].round(3)

        # Final column order (explicit contract)
        out = out[
            [
                "date",
                "store_id",
                "sku_id",
                "units_sold",
                "revenue",
                "margin_proxy",
                "promo_flag",
                "holiday_flag",
                "day_of_week",
            ]
        ]

        logger.info("Sales fact table created successfully")
        return out

    except Exception as exc:
        logger.exception("Failed to build fact sales | Error: %s", exc)
        raise

# Function to build the fact_inventory_store_sku_daily dataset by merging inventory with sales to calculate stockout flags and days of cover
def build_fact_inventory(inventory, sales, logger):
    try:
        cover_window_days = 28
        logger.info("Building fact_inventory_store_sku_daily")

        inv = inventory.copy()

        # Standardize identifiers
        inv = standardize_ids(inv, ["store_id", "sku_id"], logger)

        # Date handling
        inv = coerce_datetime(inv, "date", logger)

        # Inventory units
        if "on_hand_close" not in inv.columns:
            logger.warning("inventory missing on_hand_close; defaulting to NaN")
            inv["on_hand_close"] = pd.NA

        inv["on_hand_close"] = pd.to_numeric(inv["on_hand_close"], errors="coerce")

        # Deduplicate: keep last snapshot per day-store-sku
        inv = (
            inv.sort_values("date")
               .drop_duplicates(subset=["date", "store_id", "sku_id"], keep="last")
        )

        # Ensure continuous daily grid
        inv = create_complete_daily_grid(
            inv,
            date_col="date",
            key_cols=["store_id", "sku_id"],
            logger=logger
        )

        # Forward fill inventory within store-sku
        inv = inv.sort_values(["store_id", "sku_id", "date"])
        inv["on_hand_units"] = (
            inv.groupby(["store_id", "sku_id"])["on_hand_close"]
               .ffill()
               .fillna(0.0)
        )

        # Stockout flag (strict requirement: on_hand == 0)
        inv["stockout_flag"] = (inv["on_hand_units"] == 0).astype(int)

        # Demand calculation (avg daily demand over 4 weeks)
        sales_demand = sales.copy()
        sales_demand = standardize_ids(sales_demand, ["store_id", "sku_id"], logger)
        sales_demand = coerce_datetime(sales_demand, "date", logger)

        if "units_sold" not in sales_demand.columns:
            raise ValueError("sales data missing units_sold; cannot compute days_of_cover")

        sales_demand["units_sold"] = pd.to_numeric(
            sales_demand["units_sold"], errors="coerce"
        ).fillna(0.0)

        max_date = sales_demand["date"].max()
        window_start = max_date - pd.Timedelta(days=cover_window_days - 1) # Use cover_window_days for demand stats calculation

        recent_sales = sales_demand[
            sales_demand["date"].between(window_start, max_date)
        ]

        avg_demand = (
            recent_sales.groupby(["store_id", "sku_id"], as_index=False)["units_sold"]
                        .mean()
                        .rename(columns={"units_sold": "avg_daily_demand_4w"})
        )

        inv = inv.merge(avg_demand, on=["store_id", "sku_id"], how="left")

        inv["avg_daily_demand_4w"] = (
            pd.to_numeric(inv["avg_daily_demand_4w"], errors="coerce")
              .fillna(0.0)
        )

        # Days of cover (safe divide)
        inv["days_of_cover"] = 0.0
        non_zero_demand = inv["avg_daily_demand_4w"] > 0

        inv.loc[non_zero_demand, "days_of_cover"] = (
            inv.loc[non_zero_demand, "on_hand_units"]
            / inv.loc[non_zero_demand, "avg_daily_demand_4w"]
        )

        inv["days_of_cover"] = inv["days_of_cover"].round(3)

        # Final output schema (explicit contract)
        inv = inv[
            [
                "date",
                "store_id",
                "sku_id",
                "on_hand_units",
                "stockout_flag",
                "days_of_cover",
            ]
        ]

        logger.info("Inventory fact table created successfully")
        return inv

    except Exception as exc:
        logger.exception("Failed to build fact_inventory_store_sku_daily | Error: %s", exc)
        raise

# Function to build the replenishment_inputs_store_sku dataset by calculating demand statistics, lead time, service level, safety stock, reorder point, and recommended order quantity for each store-sku combination
def build_replenishment_inputs(sales, purchase_orders, products, logger):
    try:
        logger.info("Building replenishment_inputs_store_sku")

        demand_window_days = 56      # 8 weeks (meets 4–8 week requirement)
        cover_window_days = 28       # next cycle (4 weeks)
        default_lead_time_days = 3.0
        default_service_level = 0.95

        # Validate and prepare sales
        required_sales_cols = ["date", "store_id", "sku_id", "units_sold"]
        missing = [c for c in required_sales_cols if c not in sales.columns]
        if missing:
            raise ValueError(f"sales missing required columns: {missing}")

        df_sales = sales.copy()
        df_sales["date"] = pd.to_datetime(df_sales["date"], errors="coerce")
        df_sales["units_sold"] = pd.to_numeric(df_sales["units_sold"], errors="coerce").fillna(0.0)

        max_date = df_sales["date"].max()
        if pd.isna(max_date):
            raise ValueError("sales contains no valid dates")

        start_date = max_date - pd.Timedelta(days=demand_window_days - 1)
        recent = df_sales.loc[
            (df_sales["date"] >= start_date) & (df_sales["date"] <= max_date)
        ].copy()

        # Demand statistics
        demand_stats = (
            recent.groupby(["store_id", "sku_id"], as_index=False)["units_sold"]
                  .agg(avg_daily_demand="mean", demand_std_dev="std")
        )

        demand_stats["avg_daily_demand"] = (
            pd.to_numeric(demand_stats["avg_daily_demand"], errors="coerce")
              .fillna(0.0)
        )
        demand_stats["demand_std_dev"] = (
            pd.to_numeric(demand_stats["demand_std_dev"], errors="coerce")
              .fillna(0.0)
        )

        # Lead time from purchase orders
        po = purchase_orders.copy() if purchase_orders is not None else pd.DataFrame()

        if (
            len(po) > 0
            and {"store_id", "sku_id", "lead_time_days"}.issubset(po.columns)
        ):
            po["lead_time_days"] = pd.to_numeric(po["lead_time_days"], errors="coerce")
            lead_time = (
                po.groupby(["store_id", "sku_id"], as_index=False)["lead_time_days"]
                  .median()
            )

            global_median_lt = po["lead_time_days"].median()
            if pd.isna(global_median_lt):
                global_median_lt = default_lead_time_days
        else:
            lead_time = pd.DataFrame(columns=["store_id", "sku_id", "lead_time_days"])
            global_median_lt = default_lead_time_days

        # Merge demand + lead time
        out = demand_stats.merge(
            lead_time, on=["store_id", "sku_id"], how="left"
        )

        out["lead_time_days"] = (
            pd.to_numeric(out["lead_time_days"], errors="coerce")
              .fillna(global_median_lt)
              .clip(lower=0.0)
        )

        # Product category & service level
        prod = products.copy() if products is not None else pd.DataFrame()

        if "sku_id" not in prod.columns:
            prod["sku_id"] = pd.NA
        if "category" not in prod.columns:
            prod["category"] = "unknown"

        prod = prod[["sku_id", "category"]].drop_duplicates(subset=["sku_id"])
        prod["category"] = normalize_category(prod["category"])

        out = out.merge(prod, on="sku_id", how="left")
        out["category"] = out["category"].fillna("unknown")

        service_level_map = {
            "dairy": 0.95,
            "grocery": 0.97,
            "snacks": 0.96,
            "beverages": 0.96,
            "home_care": 0.94,
            "personal_care": 0.94,
        }

        out["service_level_target"] = (
            out["category"].map(service_level_map).fillna(default_service_level)
        )

        out["service_level_target"] = (
            pd.to_numeric(out["service_level_target"], errors="coerce")
              .fillna(default_service_level)
        )

        # Z-score using SciPy
        out["z_score"] = out["service_level_target"].apply(
            lambda p: float(norm.ppf(p))
        )

        # Safety stock & ROP
        out["safety_stock"] = (
            out["z_score"]
            * out["demand_std_dev"]
            * np.sqrt(out["lead_time_days"])
        )

        out["reorder_point"] = (
            out["avg_daily_demand"] * out["lead_time_days"]
            + out["safety_stock"]
        )

        out["recommended_order_qty"] = np.ceil(
            out["avg_daily_demand"] * cover_window_days
        ).clip(lower=0.0)

        # Rounding for output
        out["safety_stock"] = out["safety_stock"].round(2)
        out["reorder_point"] = out["reorder_point"].round(2)
        out["recommended_order_qty"] = out["recommended_order_qty"].round(0)

        
        out["avg_daily_demand"] = out["avg_daily_demand"].round(3)
        out["demand_std_dev"] = out["demand_std_dev"].round(3)

        # Final output schema
        out = out[
            [
                "store_id",
                "sku_id",
                "avg_daily_demand",
                "demand_std_dev",
                "lead_time_days",
                "service_level_target",
                "reorder_point",
                "safety_stock",
                "recommended_order_qty",
            ]
        ].copy()

        logger.info(
            "Replenishment inputs built successfully | rows=%d", len(out)
        )
        return out

    except Exception as exc:
        logger.exception(
            "Failed to build replenishment inputs | Error: %s", exc
        )
        raise

# Validation function to ensure required columns are present        
def validate_minimum_column(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("%s missing required columns: %s" % (name, missing))
    if len(df) == 0:
        raise ValueError("%s has 0 rows after processing." % name)

# ============================================================
# MAIN
# ============================================================
def run_pipeline():
    if "__file__" in globals():
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path.cwd().resolve()

    output_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    plots_dir = output_dir / "plots"
    required_library_path = base_dir / "requirements.txt"

    ensure_dirs_exist([output_dir, logs_dir, plots_dir])

    logger = setup_logging(logs_dir / "retail_demand_forecasting_etl.log", LOG_LEVEL)

    try:
        logger.info("Script/base directory: %s", base_dir)

        # Install required packages
        install_requirements(required_library_path, logger)

        base_dir = Path(__file__).resolve().parent

        # Resolve input files
        inputs = resolve_inputs_files(base_dir, REQUIRED_FILES, logger)

        sales_raw = read_csv_files(inputs["sales"], logger)
        inventory_raw = read_csv_files(inputs["inventory"], logger)
        calendar_raw = read_csv_files(inputs["calendar"], logger)
        purchase_orders_raw = read_csv_files(inputs["purchase_orders"], logger)
        stores_raw = read_csv_files(inputs["stores"], logger)
        products_raw = read_json_files(inputs["products"], logger)

        calendar = clean_calendar_dataset(calendar_raw, logger)
        stores = clean_stores_dataset(stores_raw, logger)
        products = clean_products_dataset(products_raw, logger)
        purchase_orders = clean_purchase_orders_dataset(purchase_orders_raw, logger)

        fact_sales = build_fact_sales(sales_raw, calendar, products, logger)
        fact_inventory = build_fact_inventory(inventory_raw, sales_raw, logger)

        if "units_sold" in fact_sales.columns:
            fact_sales = add_outlier_flag_iqr_by_group(
                fact_sales,
                group_cols=["store_id", "sku_id"],
                value_col="units_sold",
                flag_col="outlier_flag",
                logger=logger,
            )
            save_boxplot_image(
                fact_sales,
                value_col="units_sold",
                title="Boxplot – units_sold (all store-sku)",
                out_path=plots_dir / "units_sold_boxplot.png",
                logger=logger,
            )

        # Build curated outputs (your existing build_fact_* functions can be reused)
        repl_inputs = build_replenishment_inputs(fact_sales, purchase_orders, products, logger)

        validate_minimum_column(fact_sales, ["date", "store_id", "sku_id", "units_sold"], "fact_sales")
        validate_minimum_column(fact_inventory, ["date", "store_id", "sku_id", "on_hand_units", "stockout_flag", "days_of_cover"], "fact_inventory")
        validate_minimum_column(repl_inputs, ["store_id", "sku_id"], "repl_inputs")

        out_sales = output_dir / "fact_sales_store_sku_daily.csv"
        out_inv = output_dir / "fact_inventory_store_sku_daily.csv"
        out_repl = output_dir / "replenishment_inputs_store_sku.csv"

        logger.info("Writing outputs to %s", output_dir)
        fact_sales.to_csv(out_sales, index=False)
        fact_inventory.to_csv(out_inv, index=False)
        repl_inputs.to_csv(out_repl, index=False)

        logger.info("Quick Quality Checks")
        logger.info("Rows – Sales fact: %s", "{:,}".format(len(fact_sales)))
        logger.info("Rows – Inventory fact: %s", "{:,}".format(len(fact_inventory)))
        logger.info("Rows – Replenishment inputs: %s", "{:,}".format(len(repl_inputs)))

        if "outlier_flag" in fact_sales.columns:
            logger.info("Outliers flagged (units_sold): %s", "{:,}".format(int(fact_sales["outlier_flag"].sum())))

        stockouts = (
            fact_inventory.groupby(["store_id", "sku_id"], as_index=False)["stockout_flag"]
                          .sum()
                          .sort_values("stockout_flag", ascending=False)
                          .head(10)
        )
        logger.info("Top 10 store-sku by stockout days:\n%s", stockouts.to_string(index=False))

        _ = stores

        logger.info("ETL completed successfully.")
        return 0

    except Exception as exc:
        logger.exception("ETL pipeline failed | Error: %s", exc)
        sys.exit(1)	


# Entry point for the script
if __name__ == "__main__":
    run_pipeline()