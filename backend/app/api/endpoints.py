from fastapi import APIRouter, HTTPException
from app.models.schemas import OptimizationRequest, OptimizationResponse
from app.services import optimization, fundamental, technical, reporting
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi.responses import Response
import traceback

router = APIRouter()

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    try:
        result = optimization.run_optimization(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            risk_free_rate=request.risk_free_rate,
            allow_short=request.allow_short,
            constraints=request.constraints
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

from app.models.schemas import PortfolioAnalysisRequest

@router.post("/analyze", response_model=OptimizationResponse)
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    try:
        result = optimization.analyze_portfolio(
            tickers=request.tickers,
            weights=request.weights,
            start_date=request.start_date,
            end_date=request.end_date,
            risk_free_rate=request.risk_free_rate
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class FundamentalRequest(BaseModel):
    ticker: str
    years: List[int]

@router.post("/analysis/fundamental")
async def fundamental_analysis(request: FundamentalRequest):
    try:
        return fundamental.analyze_fundamental(request.ticker, request.years)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TechnicalRequest(BaseModel):
    ticker: str
    from_date: str
    to_date: str
    indicators: List[str]

@router.post("/analysis/technical")
async def technical_analysis(request: TechnicalRequest):
    try:
        return technical.analyze_technical(request.ticker, request.from_date, request.to_date, request.indicators)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report/pdf")
async def get_pdf_report(request: OptimizationRequest):
    try:
        # Run optimization or analysis to get data
        if request.weights:
             opt_result = optimization.analyze_portfolio(
                tickers=request.tickers,
                weights=request.weights,
                start_date=request.start_date,
                end_date=request.end_date,
                risk_free_rate=request.risk_free_rate
            )
        else:
            opt_result = optimization.run_optimization(
                request.tickers, request.start_date, request.end_date, 
                request.risk_free_rate, request.allow_short, request.constraints
            )
        
        # Fetch fundamental data
        tickers = list(opt_result['weights'].keys())
        fundamental_data = {}
        import datetime
        current_year = datetime.datetime.now().year
        years = [current_year - 1, current_year - 2, current_year - 3]
        
        for ticker in tickers:
            try:
                # Run in threadpool since it might be blocking (yfinance)
                # But for now direct call is fine as it's not heavily async optimized app
                fundamental_data[ticker] = fundamental.analyze_fundamental(ticker, years)
            except Exception as e:
                print(f"Error fetching fundamentals for {ticker}: {e}")

        # Generate PDF
        pdf_content = reporting.generate_pdf_report({
            "portfolio": opt_result, 
            "fundamental": fundamental_data
        })
        
        return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report/excel")
async def get_excel_report(request: OptimizationRequest):
    try:
        # Run optimization or analysis to get data
        if request.weights:
             opt_result = optimization.analyze_portfolio(
                tickers=request.tickers,
                weights=request.weights,
                start_date=request.start_date,
                end_date=request.end_date,
                risk_free_rate=request.risk_free_rate
            )
        else:
            opt_result = optimization.run_optimization(
                request.tickers, request.start_date, request.end_date, 
                request.risk_free_rate, request.allow_short, request.constraints
            )
        
        # Fetch fundamental data
        tickers = list(opt_result['weights'].keys())
        fundamental_data = {}
        import datetime
        current_year = datetime.datetime.now().year
        years = [current_year - 1, current_year - 2, current_year - 3]
        
        for ticker in tickers:
            try:
                # Run in threadpool since it might be blocking (yfinance)
                fundamental_data[ticker] = fundamental.analyze_fundamental(ticker, years)
            except Exception as e:
                print(f"Error fetching fundamentals for {ticker}: {e}")

        # Generate Excel
        excel_content = reporting.generate_excel_report({
            "portfolio": opt_result, 
            "fundamental": fundamental_data
        })
        
        return Response(content=excel_content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=report.xlsx"})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
