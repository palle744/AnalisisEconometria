from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def generate_pdf_report(data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=8))
    
    elements = []

    # Title
    elements.append(Paragraph("Reporte de Análisis Econométrico", styles['Title']))
    elements.append(Paragraph(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    portfolio = data.get('portfolio', {})
    
    # --- 1. Portfolio Metrics & Weights ---
    elements.append(Paragraph("Resumen del Portafolio", styles['Heading2']))
    
    # Metrics
    metrics_data = [
        f"Retorno Esperado (Anual): {portfolio.get('expected_return', 0):.2%}",
        f"Volatilidad (Anual): {portfolio.get('portfolio_std', 0):.2%}",
        f"Sharpe Ratio: {portfolio.get('sharpe', 0):.2f}"
    ]
    
    for m in metrics_data:
        elements.append(Paragraph(m, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Weights Table
    elements.append(Paragraph("Asignación de Activos", styles['Heading3']))
    weights_data = [["Activo", "Peso"]]
    for ticker, weight in portfolio.get('weights', {}).items():
        weights_data.append([ticker, f"{weight:.2%}"])
    
    t_weights = Table(weights_data, colWidths=[2*inch, 2*inch])
    t_weights.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t_weights)
    elements.append(Spacer(1, 24))

    # --- 2. Efficient Frontier Plot ---
    if 'frontier' in portfolio and portfolio['frontier']:
        elements.append(Paragraph("Frontera Eficiente", styles['Heading2']))
        
        frontier_data = portfolio['frontier']
        vols = [p['std_dev'] for p in frontier_data]
        rets = [p['return'] for p in frontier_data]
        
        plt.figure(figsize=(7, 4))
        plt.plot(vols, rets, 'b-', label='Frontera Eficiente')
        # Plot current portfolio
        plt.plot(portfolio.get('portfolio_std'), portfolio.get('expected_return'), 'r*', markersize=15, label='Portafolio Optimizado')
        
        plt.title("Frontera Eficiente de Markowitz")
        plt.xlabel("Volatilidad (Riesgo)")
        plt.ylabel("Retorno Esperado")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100)
        img_buffer.seek(0)
        plt.close()
        
        elements.append(Image(img_buffer, width=450, height=250))
        elements.append(Spacer(1, 12))

    # --- 3. Cumulative Performance Plot ---
    if 'cumulative_performance' in portfolio and portfolio['cumulative_performance']:
        elements.append(Paragraph("Desempeño Histórico Acumulado", styles['Heading2']))
        
        cum_perf = portfolio['cumulative_performance']
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in cum_perf['dates']]
        port_vals = cum_perf['portfolio']
        bench_vals = cum_perf['benchmark']
        
        plt.figure(figsize=(7, 4))
        plt.plot(dates, port_vals, label='Portafolio')
        if bench_vals:
            plt.plot(dates, bench_vals, label='Benchmark (SPY)', alpha=0.7)
            
        plt.title("Retorno Acumulado (%)")
        plt.xlabel("Fecha")
        plt.ylabel("Retorno (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_buffer_perf = io.BytesIO()
        plt.savefig(img_buffer_perf, format='png', dpi=100)
        img_buffer_perf.seek(0)
        plt.close()
        
        elements.append(Image(img_buffer_perf, width=450, height=250))
        elements.append(Spacer(1, 12))

    elements.append(PageBreak())

    # --- 4. Correlation Matrix ---
    if 'correlation_matrix' in portfolio and portfolio['correlation_matrix']:
        elements.append(Paragraph("Matriz de Correlación", styles['Heading2']))
        
        corr_data = portfolio['correlation_matrix']
        tickers = corr_data['index']
        matrix = np.array(corr_data['data'])
        
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(tickers)), tickers, rotation=45)
        plt.yticks(range(len(tickers)), tickers)
        plt.title("Correlación entre Activos")
        plt.tight_layout()
        
        img_buffer_corr = io.BytesIO()
        plt.savefig(img_buffer_corr, format='png', dpi=100)
        img_buffer_corr.seek(0)
        plt.close()
        
        elements.append(Image(img_buffer_corr, width=400, height=330))
        elements.append(Spacer(1, 24))

    # --- 5. Fundamental Analysis ---
    fundamental_data = data.get('fundamental', {})
    if fundamental_data:
        elements.append(Paragraph("Análisis Fundamental", styles['Heading2']))
        
        for ticker, fdata in fundamental_data.items():
            elements.append(Paragraph(f"Activo: {ticker}", styles['Heading3']))
            
            if "financials" in fdata and fdata['financials']:
                # Create table for this ticker
                # Columns: Metric, Year 1, Year 2, Year 3
                years = sorted([y for y in fdata['financials'].keys() if isinstance(y, int)], reverse=True)
                
                if not years:
                    elements.append(Paragraph("No hay datos fundamentales disponibles.", styles['Normal']))
                    continue
                
                header = ["Métrica"] + [str(y) for y in years]
                table_data = [header]
                
                # Metrics to show
                metrics_map = {
                    "revenue": "Ingresos",
                    "net_income": "Ingreso Neto",
                    "total_assets": "Activos Totales",
                    "total_equity": "Patrimonio",
                }
                ratios_map = {
                    "net_margin": "Margen Neto",
                    "roe": "ROE",
                    "roa": "ROA",
                    "asset_turnover": "Rotación Activos"
                }
                
                # Add rows
                for key, label in metrics_map.items():
                    row = [label]
                    for y in years:
                        val = fdata['financials'][y].get(key, 0)
                        if isinstance(val, (int, float)):
                             row.append(f"{val:,.0f}")
                        else:
                             row.append(str(val))
                    table_data.append(row)
                    
                for key, label in ratios_map.items():
                    row = [label]
                    for y in years:
                        ratios = fdata['financials'][y].get('ratios', {})
                        val = ratios.get(key, 0)
                        if isinstance(val, (int, float)):
                             row.append(f"{val:.4f}")
                        else:
                             row.append(str(val))
                    table_data.append(row)
                
                t_fund = Table(table_data, colWidths=[1.5*inch] + [1.2*inch]*len(years))
                t_fund.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                ]))
                elements.append(t_fund)
                elements.append(Spacer(1, 12))
            else:
                 elements.append(Paragraph("Datos no disponibles o error al obtener.", styles['Normal']))

    # --- 6. News ---
    news_data = portfolio.get('news', {})
    if news_data:
        elements.append(PageBreak())
        elements.append(Paragraph("Noticias Recientes", styles['Heading2']))
        
        for ticker, news_items in news_data.items():
            if not news_items:
                continue
                
            elements.append(Paragraph(f"Noticias para {ticker}", styles['Heading3']))
            
            # Top 3 news
            for item in news_items[:3]:
                title = item.get('title', 'Sin título')
                publisher = item.get('publisher', 'Desconocido')
                
                text = f"<b>{publisher}:</b> {title}"
                elements.append(Paragraph(text, styles['Normal']))
                elements.append(Spacer(1, 4))
            
            elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def generate_excel_report(data: dict) -> bytes:
    buffer = io.BytesIO()
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    # We use openpyxl as it is more standard for reading/writing .xlsx in pandas now
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        portfolio = data.get('portfolio', {})
        
        # Sheet 1: Summary
        summary_data = {
            "Métrica": ["Retorno Esperado", "Volatilidad", "Sharpe Ratio"],
            "Valor": [
                portfolio.get('expected_return', 0),
                portfolio.get('portfolio_std', 0),
                portfolio.get('sharpe', 0)
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Resumen', index=False)
        
        # Sheet 2: Weights
        weights_data = []
        for ticker, weight in portfolio.get('weights', {}).items():
            weights_data.append({"Activo": ticker, "Peso": weight})
        df_weights = pd.DataFrame(weights_data)
        df_weights.to_excel(writer, sheet_name='Pesos', index=False)
        
        # Sheet 3: Historical Prices
        if 'historical_prices' in portfolio and 'historical_dates' in portfolio:
            df_hist = pd.DataFrame(portfolio['historical_prices'])
            df_hist['Fecha'] = portfolio['historical_dates']
            # Move Fecha to first column
            cols = ['Fecha'] + [c for c in df_hist.columns if c != 'Fecha']
            df_hist = df_hist[cols]
            df_hist.to_excel(writer, sheet_name='Precios Históricos', index=False)
            
        # Sheet 4: Fundamental Analysis
        fundamental_data = data.get('fundamental', {})
        if fundamental_data:
            fund_rows = []
            for ticker, fdata in fundamental_data.items():
                if "financials" in fdata and fdata['financials']:
                    for year, metrics in fdata['financials'].items():
                        if isinstance(metrics, dict): # Skip error strings
                            row = {"Activo": ticker, "Año": year}
                            # Add main metrics
                            row.update({k: v for k, v in metrics.items() if k != 'ratios'})
                            # Add ratios
                            if 'ratios' in metrics:
                                row.update(metrics['ratios'])
                            fund_rows.append(row)
            
            if fund_rows:
                df_fund = pd.DataFrame(fund_rows)
                df_fund.to_excel(writer, sheet_name='Fundamental', index=False)

        # Sheet 5: Correlation
        if 'correlation_matrix' in portfolio and portfolio['correlation_matrix']:
            corr_data = portfolio['correlation_matrix']
            df_corr = pd.DataFrame(corr_data['data'], columns=corr_data['index'], index=corr_data['index'])
            df_corr.to_excel(writer, sheet_name='Correlación')

    buffer.seek(0)
    return buffer.getvalue()
