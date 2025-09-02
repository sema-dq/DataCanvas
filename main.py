# backend/main.py
from typing import List, Dict, Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ConfigDict, SecretStr
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Database connection libraries
import psycopg2
import mysql.connector
import pymssql


# --- Enhanced Application Setup ---
tags_metadata = [
    {
        "name": "Data Sources",
        "description": "Endpoints for connecting to and loading data from various sources.",
    },
    {
        "name": "Data Cleaning & Transformation",
        "description": "Endpoints for preparing and cleaning datasets.",
    },
    {
        "name": "Statistical Analysis",
        "description": "Endpoints for running statistical tests and models.",
    },
    {
        "name": "Machine Learning",
        "description": "Endpoints for predictive modeling and pattern recognition.",
    },
    {
        "name": "Chart Generation",
        "description": "Endpoints for preparing data specifically for visualizations.",
    },
]

app = FastAPI(
    title="DataCanvas Apex API",
    description="A massively enhanced, enterprise-grade API for advanced data analysis, machine learning, and visualization.",
    version="5.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

# NEW: Models for Database Connections
class DBConnectionPayload(BaseModel):
    db_type: Literal['postgresql', 'mysql', 'sqlserver']
    host: str
    port: int
    user: str
    password: SecretStr
    dbname: str

class DBTablePayload(DBConnectionPayload):
    table_name: str

class BaseDataPayload(BaseModel):
    records: List[Dict[str, Any]]

    @field_validator('records')
    def check_records_not_empty(cls, v):
        if not v:
            raise ValueError("The 'records' list cannot be empty.")
        return v

class ChartPayload(BaseDataPayload):
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color: Optional[str] = None
    filters: List[Dict] = []
    chart_type: Literal['bar', 'line', 'scatter', 'pie', 'heatmap', 'treemap', 'combo', 'boxplot', 'sankey', 'wordCloud', 'gantt']
    aggregation: Literal['sum', 'mean', 'count', 'median', 'min', 'max', 'stdev', 'var'] = 'sum'
    analytics: Optional[Dict] = None
    dimensions: Optional[List[str]] = None
    measures: Optional[List[str]] = None
    
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "records": [{"Category": "A", "Sales": 100, "Region": "North"}, {"Category": "B", "Sales": 150, "Region": "North"}],
            "x_axis": "Category",
            "y_axis": "Sales",
            "chart_type": "bar",
        }]
    })

class AnovaPayload(BaseDataPayload):
    measure: str
    dimension: str

class ChiSquarePayload(BaseDataPayload):
    dimension1: str
    dimension2: str

class KMeansPayload(BaseDataPayload):
    fields: List[str]
    n_clusters: int = 3

class ImputationPayload(BaseDataPayload):
    field: str
    strategy: Literal['mean', 'median', 'mode', 'constant']
    constant_value: Optional[Any] = 0

class ForecastPayload(BaseDataPayload):
    date_field: str
    value_field: str
    forecast_periods: int
    model: Literal['linear', 'smoothing'] = 'linear'

class GroupValuesPayload(BaseDataPayload):
    field_name: str
    new_group_name: str
    values_to_group: List[Any]

class UniqueValuesPayload(BaseDataPayload):
    field_name: str

class CorrelationPayload(BaseDataPayload):
    measure1: str
    measure2: str

class TTestPayload(BaseDataPayload):
    measure: str
    dimension: str

class ZScorePayload(BaseDataPayload):
    measure: str

class BinningPayload(BaseDataPayload):
    measure: str
    bin_size: int
    bin_name: str

# --- Database Connector ---
class DatabaseConnector:
    def __init__(self, params: DBConnectionPayload):
        self.db_type = params.db_type
        self.host = params.host
        self.port = params.port
        self.user = params.user
        self.password = params.password.get_secret_value()
        self.dbname = params.dbname
        self.connection = None

    def __enter__(self):
        try:
            if self.db_type == 'postgresql':
                self.connection = psycopg2.connect(
                    dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port
                )
            elif self.db_type == 'mysql':
                self.connection = mysql.connector.connect(
                    host=self.host, port=self.port, user=self.user, password=self.password, database=self.dbname
                )
            elif self.db_type == 'sqlserver':
                self.connection = pymssql.connect(
                    server=self.host, port=self.port, user=self.user, password=self.password, database=self.dbname
                )
            return self.connection
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

# --- Data Processing Logic ---
class DataProcessor:
    def __init__(self, records: List[Dict[str, Any]]):
        self.df = pd.DataFrame(records).copy()
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            # Attempt to convert object columns to datetime if they look like dates
            elif self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except (ValueError, TypeError):
                    pass # Ignore columns that can't be converted

    def _validate_fields(self, *fields: str):
        missing_fields = [f for f in fields if f and f not in self.df.columns]
        if missing_fields:
            raise HTTPException(status_code=404, detail=f"Field(s) not found: {', '.join(missing_fields)}")

    @staticmethod
    def _clean_for_json(data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.replace({np.nan: None, pd.NaT: None})
        if isinstance(data, list):
            return [DataProcessor._clean_for_json(item) for item in data]
        if isinstance(data, dict):
            return {k: DataProcessor._clean_for_json(v) for k, v in data.items()}
        if isinstance(data, (np.integer, np.floating)):
            return data.item()
        return data

    def to_records(self) -> List[Dict[str, Any]]:
        cleaned_df = self._clean_for_json(self.df)
        return cleaned_df.to_dict('records')

    # --- Data Cleaning & Transformation ---
    def impute_missing_values(self, field: str, strategy: str, constant_value: Any = 0):
        self._validate_fields(field)
        if strategy == 'mean':
            fill_value = self.df[field].mean()
        elif strategy == 'median':
            fill_value = self.df[field].median()
        elif strategy == 'mode':
            fill_value = self.df[field].mode()[0]
        elif strategy == 'constant':
            fill_value = constant_value
        else:
            raise HTTPException(status_code=400, detail="Invalid imputation strategy.")
        self.df[field].fillna(fill_value, inplace=True)

    def group_values(self, field: str, new_name: str, values: List[Any]):
        self._validate_fields(field)
        new_col_name = f"{field}_grouped"
        self.df[new_col_name] = np.where(self.df[field].isin(values), new_name, self.df[field])

    def get_unique_values(self, field: str) -> List[Any]:
        self._validate_fields(field)
        return sorted(self.df[field].dropna().unique().tolist())

    def bin_data(self, measure: str, bin_size: int, bin_name: str):
        self._validate_fields(measure)
        bins = np.arange(self.df[measure].min(), self.df[measure].max() + bin_size, bin_size)
        labels = [f'{i}-{i+bin_size-1}' for i in bins[:-1]]
        self.df[bin_name] = pd.cut(self.df[measure], bins=bins, labels=labels, right=False)

    # --- Statistical & ML Methods ---
    def run_correlation_analysis(self, measure1: str, measure2: str):
        self._validate_fields(measure1, measure2)
        corr_matrix = self.df[[measure1, measure2]].corr()
        return {"correlation_coefficient": corr_matrix.loc[measure1, measure2]}

    def run_t_test(self, measure: str, dimension: str):
        self._validate_fields(measure, dimension)
        unique_groups = self.df[dimension].dropna().unique()
        if len(unique_groups) < 2:
            raise HTTPException(status_code=400, detail="T-test requires at least two groups.")
        
        group1_data = self.df[self.df[dimension] == unique_groups[0]][measure].dropna()
        group2_data = self.df[self.df[dimension] == unique_groups[1]][measure].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            raise HTTPException(status_code=400, detail="Each group must have at least two data points.")
        
        t_statistic, p_value = stats.ttest_ind(group1_data, group2_data)
        return {"t_statistic": t_statistic, "p_value": p_value}

    def run_z_score_outliers(self, measure: str):
        self._validate_fields(measure)
        std_dev = self.df[measure].std()
        if std_dev == 0:
            return {"outliers": [], "message": "Standard deviation is zero; no outliers detected."}
        
        self.df['z_score'] = stats.zscore(self.df[measure].dropna())
        outliers_df = self.df[self.df['z_score'].abs() > 3]
        return {"outliers": self._clean_for_json(outliers_df.to_dict('records'))}

    def run_time_series_forecast(self, date_field: str, value_field: str, forecast_periods: int, model_type: str):
        self._validate_fields(date_field, value_field)
        temp_df = self.df[[date_field, value_field]].copy()
        temp_df[date_field] = pd.to_datetime(temp_df[date_field])
        temp_df.set_index(date_field, inplace=True)
        temp_df = temp_df.asfreq(pd.infer_freq(temp_df.index), method='ffill').dropna()

        if model_type == 'smoothing' and len(temp_df) >= 24: # Holt-Winters needs sufficient data
            model = ExponentialSmoothing(temp_df[value_field], trend='add', seasonal='add', seasonal_periods=12).fit()
            predictions = model.forecast(forecast_periods)
        else: # Default to linear regression
            temp_df['time_index'] = np.arange(len(temp_df))
            model = LinearRegression()
            model.fit(temp_df[['time_index']], temp_df[value_field])
            future_indices = np.arange(len(temp_df), len(temp_df) + forecast_periods).reshape(-1, 1)
            predictions = model.predict(future_indices)
        
        return {"predictions": self._clean_for_json(predictions.tolist())}

    def run_anova(self, measure: str, dimension: str):
        self._validate_fields(measure, dimension)
        formula = f"`{measure}` ~ C(`{dimension}`)"
        model = ols(formula, data=self.df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return {"f_statistic": anova_table['F'][0], "p_value": anova_table['PR(>F)'][0]}

    def run_chi_square_test(self, dimension1: str, dimension2: str):
        self._validate_fields(dimension1, dimension2)
        contingency_table = pd.crosstab(self.df[dimension1], self.df[dimension2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        return {"chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof}

    def run_kmeans_clustering(self, fields: List[str], n_clusters: int):
        self._validate_fields(*fields)
        data = self.df[fields].dropna()
        if len(data) < n_clusters:
            raise HTTPException(status_code=400, detail="Not enough data points for the specified number of clusters.")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data)
        
        self.df.loc[data.index, 'Cluster'] = [f'Cluster {c+1}' for c in clusters]

# --- API Endpoints ---

@app.post("/data/connect", tags=["Data Sources"])
async def api_connect_to_db(payload: DBConnectionPayload):
    query = ""
    if payload.db_type == 'postgresql':
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    elif payload.db_type == 'mysql':
        query = "SHOW TABLES"
    elif payload.db_type == 'sqlserver':
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"

    try:
        with DatabaseConnector(payload) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            tables = [row[0] for row in cursor.fetchall()]
            return {"status": "success", "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/load-table", tags=["Data Sources"])
async def api_load_db_table(payload: DBTablePayload):
    if not payload.table_name.isalnum() and "_" not in payload.table_name:
        raise HTTPException(status_code=400, detail="Invalid table name.")
        
    query = f'SELECT * FROM "{payload.table_name}"'
    if payload.db_type == 'mysql':
        query = f"SELECT * FROM `{payload.table_name}`"
    elif payload.db_type == 'sqlserver':
        query = f'SELECT * FROM [{payload.table_name}]'

    try:
        with DatabaseConnector(payload) as conn:
            df = pd.read_sql(query, conn)
            processor = DataProcessor(df.to_dict('records'))
            return {"status": "success", "records": processor.to_records()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/actions/impute", tags=["Data Cleaning & Transformation"])
async def api_impute(payload: ImputationPayload):
    processor = DataProcessor(payload.records)
    processor.impute_missing_values(payload.field, payload.strategy, payload.constant_value)
    return {"status": "success", "records": processor.to_records()}

@app.post("/actions/group-values", tags=["Data Cleaning & Transformation"])
async def api_group_values(payload: GroupValuesPayload):
    processor = DataProcessor(payload.records)
    processor.group_values(payload.field_name, payload.new_group_name, payload.values_to_group)
    return {"status": "success", "records": processor.to_records()}

@app.post("/actions/bin-data", tags=["Data Cleaning & Transformation"])
async def api_bin_data(payload: BinningPayload):
    processor = DataProcessor(payload.records)
    processor.bin_data(payload.measure, payload.bin_size, payload.bin_name)
    return {"status": "success", "records": processor.to_records()}

@app.post("/analysis/correlation", tags=["Statistical Analysis"])
async def api_run_correlation(payload: CorrelationPayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "result": processor.run_correlation_analysis(payload.measure1, payload.measure2)}

@app.post("/analysis/t-test", tags=["Statistical Analysis"])
async def api_run_t_test(payload: TTestPayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "result": processor.run_t_test(payload.measure, payload.dimension)}

@app.post("/analysis/anova", tags=["Statistical Analysis"])
async def api_run_anova(payload: AnovaPayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "result": processor.run_anova(payload.measure, payload.dimension)}

@app.post("/analysis/chi-square", tags=["Statistical Analysis"])
async def api_run_chi_square(payload: ChiSquarePayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "result": processor.run_chi_square_test(payload.dimension1, payload.dimension2)}

@app.post("/analysis/forecast", tags=["Statistical Analysis"])
async def api_run_forecast(payload: ForecastPayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "result": processor.run_time_series_forecast(payload.date_field, payload.value_field, payload.forecast_periods, payload.model)}

@app.post("/ml/z-score-outliers", tags=["Machine Learning"])
async def api_run_z_score(payload: ZScorePayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "result": processor.run_z_score_outliers(payload.measure)}

@app.post("/ml/kmeans-clustering", tags=["Machine Learning"])
async def api_run_kmeans(payload: KMeansPayload):
    processor = DataProcessor(payload.records)
    processor.run_kmeans_clustering(payload.fields, payload.n_clusters)
    return {"status": "success", "records": processor.to_records()}

@app.post("/charts/generate", tags=["Chart Generation"])
async def api_generate_chart(payload: ChartPayload):
    # This endpoint is currently a passthrough as the frontend handles chart data processing.
    # It can be expanded later for complex server-side rendering if needed.
    return {"status": "success", "chartData": payload.records}

@app.post("/data/unique-values", tags=["Data Cleaning & Transformation"])
async def api_get_unique_values(payload: UniqueValuesPayload):
    processor = DataProcessor(payload.records)
    return {"status": "success", "values": processor.get_unique_values(payload.field_name)}

@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the DataCanvas Pro API!"}

