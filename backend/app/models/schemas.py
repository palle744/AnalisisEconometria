from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import date

class OptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: date = Field(..., alias="from")
    end_date: date = Field(..., alias="to")
    risk_free_rate: float = 0.035
    allow_short: bool = False
    constraints: Optional[Dict[str, List[float]]] = None
    weights: Optional[List[float]] = None

    class Config:
        populate_by_name = True

class PortfolioMetrics(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]

class FrontierPoint(BaseModel):
    return_: float = Field(..., alias="return")
    std_dev: float
    sharpe: float

class OptimizationResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    portfolio_std: float
    sharpe: float
    frontier: List[FrontierPoint]
    latest_prices: Optional[Dict[str, float]] = None
    start_prices: Optional[Dict[str, float]] = None
    historical_dates: Optional[List[str]] = None
    historical_prices: Optional[Dict[str, List[float]]] = None
    news: Optional[Dict[str, List[Dict]]] = None
    cumulative_performance: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None

class PortfolioAnalysisRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    start_date: date = Field(..., alias="from")
    end_date: date = Field(..., alias="to")
    risk_free_rate: float = 0.035
