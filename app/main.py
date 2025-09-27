import os
import re
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import httpx
import PyPDF2

app = FastAPI(title="Budget Analyzer PDF")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, "*" allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenRouter API key in your environment variables
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """
    Extracts text from a PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file.file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF: {e}")

async def analyze_transactions(text: str) -> dict:
    """
    Send text to DeepSeek R1 Distill Llama 70B to categorize, summarize, and offer advice in structured JSON.
    """
    prompt = f"""
    You are a financial assistant.
    Analyze the following transactions and return the result in JSON format with the keys:
    - categories: List of {{"category": str, "amount": float}}
    - monthly_income: float
    - total_expenses: float
    - available_savings: float
    - summary: str (summary of where most money is spent)
    - advice: List of strings (financial tips)

    Transactions:
    {text}

    IMPORTANT: Return strictly valid JSON.
    """
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "deepseek/deepseek-r1-distill-llama-70b:free",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Strip Markdown code blocks if present
            cleaned = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
            
            try:
                data = json.loads(cleaned)
                return data
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse JSON from model output: {cleaned}"
                )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            raise HTTPException(status_code=402, detail="API quota exceeded or payment required.")
        raise HTTPException(status_code=500, detail=f"API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing transactions: {e}")

@app.post("/analyze_pdf/")
async def analyze_pdf(pdf_file: UploadFile):
    """
    Upload a PDF of financial statements and get a summarized analysis in structured JSON.
    """
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    text = extract_text_from_pdf(pdf_file)
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF.")
    
    analysis = await analyze_transactions(text)
    return JSONResponse(content=analysis)
