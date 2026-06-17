# main.py
import os
import json
import requests
from fastapi import FastAPI, HTTPException
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from langsmith import traceable, wrappers

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models for Strict Data Validation ---

class CategoryEnum(str, Enum):
    TECHNICAL_ISSUE = "Technical Issue"
    BILLING_INQUIRY = "Billing Inquiry"
    FEATURE_REQUEST = "Feature Request"
    GENERAL_QUESTION = "General Question"
    ACCOUNT_MANAGEMENT = "Account Management"
    BUG_REPORT = "Bug Report"
    COMPLAINT_ESCALATION = "Complaint/Escalation"
    SECURITY_PRIVACY = "Security/Privacy"
    REFUND_REQUEST = "Refund Request"
    INTEGRATION_ISSUE = "Integration Issue"
    PERFORMANCE_ISSUE = "Performance Issue"
    DOCUMENTATION_API = "Documentation/API Question"
    DATA_REQUEST = "Data Request"
    SERVICE_STATUS = "Service Status"

class UrgencyEnum(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class SentimentEnum(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

class TicketRequest(BaseModel):
    ticket: str = Field(..., min_length=1, description="The combined content of the support ticket.")

class RelevanceJudgeResponse(BaseModel):
    is_relevant: bool      # True if ticket is relevant to the service
    confidence: float      # 0-1: How confident is the judge
    feedback: str         # Why it is/isn't relevant

class TicketAnalysis(BaseModel):
    category: CategoryEnum
    urgency: UrgencyEnum
    sentiment: SentimentEnum

class GuidanceRequest(BaseModel):
    ticket: str
    analysis: TicketAnalysis


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Support Ticket Router",
    description="Processes support tickets using an AI workflow via Hugging Face.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allows your React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust for your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root Endpoint for Health Check ---
@app.get("/")
async def read_root():
    return {"message": "AI Support Ticket Router API is running."}

# --- OpenAI Client for Hugging Face Router ---
openai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGING_FACE_API_KEY"),
)

# Wrap with LangSmith tracing
client = wrappers.wrap_openai(
    openai_client,
    tracing_extra={
        "tags": ["openai", "hugging-face-router", "python"],
        "metadata": {
            "integration": "openai",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        },
    },
)

@traceable(name="query_hf_router", run_type="tool")
def query_hf_router(model: str, prompt: str, max_tokens: int = 150):
    """Helper function to query the Hugging Face Router using the OpenAI client."""
    api_key = os.getenv("HUGGING_FACE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Hugging Face API key not configured.")
        
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except OpenAIError as e:
        error_detail = f"AI service unavailable: {e}"
        raise HTTPException(status_code=503, detail=error_detail)


# --- Prompt Engineering Templates ---
def get_relevance_judge_prompt(ticket: str) -> str:
    """Generate a prompt to validate if the ticket is relevant to the support system."""
    return f"""You are a support ticket relevance validator. Determine if this is a genuine support issue for a software/service platform.

CUSTOMER TICKET:
{ticket}

VALID SUPPORT ISSUES include:
- Software bugs, crashes, errors
- Feature requests for the application
- Account and login issues
- Payment/billing problems
- API integration questions
- Performance or technical problems
- Service status/outage reports
- Data access/export requests
- Security or privacy concerns
- Documentation/help questions

INVALID/IRRELEVANT ISSUES include:
- Personal/life problems unrelated to the service
- Lost personal items (pen, wallet, keys, etc.)
- General life advice
- Questions about unrelated topics
- Spam or promotional content
- Completely off-topic messages

Determine if this ticket is relevant to a support system.

Return ONLY this JSON format:
{{"is_relevant": true, "confidence": 0.95, "feedback": "This is a relevant support issue."}}

If NOT relevant, explain why in feedback. Be direct and clear."""

def get_analysis_prompt(ticket: str) -> str:
    return f"""
Analyze the following support ticket and extract the category, urgency, and sentiment.
Provide the output ONLY in a valid JSON format with the keys "category", "urgency", and "sentiment".

Allowed categories: [
  "Technical Issue",          # App crashes, errors, bugs
  "Billing Inquiry",          # Payments, invoices, charges
  "Feature Request",          # Enhancement suggestions
  "General Question",         # How-tos, account info
  "Account Management",       # Password reset, profile updates
  "Bug Report",               # Specific software bug
  "Complaint/Escalation",     # Customer dissatisfaction
  "Security/Privacy",         # Data breach, privacy concerns
  "Refund Request",           # Money back requests
  "Integration Issue",        # Third-party integrations
  "Performance Issue",        # Slow app, latency
  "Documentation/API Question",  # API docs, guides
  "Data Request",             # Export data, access
  "Service Status"            # System down, outage
]
Allowed urgencies: ["Low", "Medium", "High"]
Allowed sentiments: ["Positive", "Neutral", "Negative"]

Ticket: "{ticket}"

JSON Output:
"""

def get_troubleshooting_prompt(ticket: str) -> str:
    return f'A customer has a high-urgency issue. Provide immediate, actionable troubleshooting steps they can take right now. Be concise and clear. Do not add any conversational fluff.\n\nTicket: "{ticket}"\n\nTroubleshooting Steps:'

def get_self_service_prompt(ticket: str) -> str:
    return f'A customer has a low or medium urgency issue. Provide detailed self-service guidance and links to potential knowledge base articles they can use to solve their problem. Be helpful and empowering.\n\nTicket: "{ticket}"\n\nSelf-Service Guidance:'

def get_email_prompt(ticket: str, analysis: dict, guidance: str) -> str:
    return f'Generate a professional and empathetic customer response email based on the provided information. Acknowledge the user\'s problem from the original ticket. Use the provided analysis and guidance to form the body of the email. The tone should match the urgency: reassuring for \'High\' urgency, and empowering for \'Low\'/\'Medium\'.\n\nOriginal Ticket: "{ticket}"\nTicket Analysis: {json.dumps(analysis)}\nGenerated Guidance: "{guidance}"\n\nCustomer Email:'


# --- API Endpoints ---

@app.post("/api/judge-relevance", response_model=RelevanceJudgeResponse)
@traceable(name="judge_relevance", run_type="chain")
async def judge_relevance(ticket_request: TicketRequest):
    """Validates if the ticket is relevant to the support system."""
    ticket = ticket_request.ticket
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    relevance_prompt = get_relevance_judge_prompt(ticket)
    relevance_response_text = query_hf_router(model=model_name, prompt=relevance_prompt, max_tokens=200)

    try:
        json_start_index = relevance_response_text.find('{')
        json_end_index = relevance_response_text.rfind('}')
        if json_start_index == -1 or json_end_index == -1:
            raise ValueError("No JSON object found in relevance judge response")
        json_string = relevance_response_text[json_start_index : json_end_index + 1]
        judge_json = json.loads(json_string)

        return RelevanceJudgeResponse(
            is_relevant=judge_json.get("is_relevant", True),
            confidence=float(judge_json.get("confidence", 0.5)),
            feedback=judge_json.get("feedback", "Relevance validation completed.")
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error parsing relevance judge response: {e}\nResponse was: {relevance_response_text}")
        return RelevanceJudgeResponse(
            is_relevant=True,
            confidence=0.5,
            feedback="Unable to validate relevance. Proceeding with caution."
        )

@app.post("/api/analyze", response_model=TicketAnalysis)
@traceable(name="analyze_ticket", run_type="chain")
async def analyze_ticket(ticket_request: TicketRequest):
    """Receives a ticket, performs analysis, and returns the analysis."""
    ticket = ticket_request.ticket
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    
    analysis_prompt = get_analysis_prompt(ticket)
    analysis_text = query_hf_router(model=model_name, prompt=analysis_prompt, max_tokens=100)

    try:
        json_start_index = analysis_text.find('{')
        json_end_index = analysis_text.rfind('}')
        if json_start_index == -1 or json_end_index == -1:
            raise ValueError("No JSON object found in the AI's response.")
        json_string = analysis_text[json_start_index : json_end_index + 1]
        analysis_json = json.loads(json_string)
        analysis = TicketAnalysis(**analysis_json)
        return analysis
    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        print(f"Error parsing AI analysis: {e}\nResponse was: {analysis_text}")
        raise HTTPException(status_code=500, detail="AI failed to generate a valid analysis.")

@app.post("/api/guidance")
@traceable(name="get_guidance", run_type="chain")
async def get_guidance(guidance_request: GuidanceRequest):
    """Receives a ticket and its analysis, and returns generated guidance."""
    ticket = guidance_request.ticket
    analysis = guidance_request.analysis
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    guidance_prompt = ""
    if analysis.urgency == 'High':
        guidance_prompt = get_troubleshooting_prompt(ticket)
    else:
        guidance_prompt = get_self_service_prompt(ticket)

    guidance_text = query_hf_router(model=model_name, prompt=guidance_prompt, max_tokens=300).strip()
    return {"guidance": guidance_text}

class EmailRequest(BaseModel):
    ticket: str
    analysis: TicketAnalysis
    guidance: str

@app.post("/api/email")
@traceable(name="get_email", run_type="chain")
async def get_email(email_request: EmailRequest):
    """Receives ticket, analysis, and guidance, and returns a final email."""
    ticket = email_request.ticket
    analysis = email_request.analysis
    guidance = email_request.guidance
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    email_prompt = get_email_prompt(ticket, analysis.dict(), guidance)
    final_email_text = query_hf_router(model=model_name, prompt=email_prompt, max_tokens=400).strip()
    return {"finalEmail": final_email_text}


# ==================== LLM JUDGE FEATURE ====================

# --- Analysis Validation Judge ---
class AnalysisJudgeRequest(BaseModel):
    ticket: str
    analysis: TicketAnalysis

class AnalysisJudgeResponse(BaseModel):
    is_correct: bool
    confidence: float  # 0-1
    feedback: str

# --- Final Quality Judge ---
class JudgeRequest(BaseModel):
    ticket: str
    analysis: TicketAnalysis
    guidance: str
    finalEmail: str

class JudgeResponse(BaseModel):
    quality_score: int  # 1-10
    correctness_score: int  # 1-10
    relevance_score: int  # 1-10
    overall_score: int  # 1-10
    feedback: str
    is_approved: bool

def get_analysis_judge_prompt(ticket: str, analysis: dict) -> str:
    """Generate a prompt to validate the ticket analysis."""
    return f"""You are an expert ticket classification validator. Analyze this ticket and verify if the provided analysis is correct.

CUSTOMER TICKET:
{ticket}

PROVIDED ANALYSIS:
- Category: {analysis.get('category')}
- Urgency: {analysis.get('urgency')}
- Sentiment: {analysis.get('sentiment')}

VALID CATEGORIES:
1. Technical Issue - App crashes, errors, bugs
2. Billing Inquiry - Payments, invoices, charges
3. Feature Request - Enhancement suggestions
4. General Question - How-tos, account info questions
5. Account Management - Password reset, profile updates, account access
6. Bug Report - Specific software bug reports
7. Complaint/Escalation - Customer dissatisfaction, escalations
8. Security/Privacy - Data breach, privacy concerns
9. Refund Request - Money back requests
10. Integration Issue - Third-party integrations
11. Performance Issue - Slow app, latency problems
12. Documentation/API Question - API docs, technical guides
13. Data Request - Export data, data access requests
14. Service Status - System down, outage reports

VALID URGENCIES: Low, Medium, High
VALID SENTIMENTS: Positive, Neutral, Negative

Validate:
1. Is the category correct? Choose from the 14 valid categories above.
2. Is the urgency level accurate?
3. Does the sentiment match the customer's tone?

Return ONLY this JSON format:
{{"is_correct": true, "confidence": 0.95, "feedback": "Analysis is accurate."}}

If analysis is incorrect, set is_correct to false and explain why in feedback. Suggest the correct category from the valid list."""

def get_judge_prompt(ticket: str, analysis: dict, guidance: str, email: str) -> str:
    """Generate a prompt for the LLM to judge the ticket response."""
    return f"""You are an expert support ticket quality judge. Evaluate the following ticket handling:

ORIGINAL TICKET:
{ticket}

ANALYSIS:
- Category: {analysis.get('category')}
- Urgency: {analysis.get('urgency')}
- Sentiment: {analysis.get('sentiment')}

GUIDANCE PROVIDED:
{guidance}

CUSTOMER EMAIL:
{email}

Evaluate on the following criteria (1-10 scale):
1. QUALITY: Is the response professional, clear, and well-structured?
2. CORRECTNESS: Is the category and urgency assessment correct?
3. RELEVANCE: Does the response address the customer's issue?

Provide your evaluation in JSON format:
{{
    "quality_score": <1-10>,
    "correctness_score": <1-10>,
    "relevance_score": <1-10>,
    "feedback": "Brief feedback on strengths and weaknesses",
    "is_approved": <true if overall score >= 7, false otherwise>
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

@app.post("/api/judge-analysis", response_model=AnalysisJudgeResponse)
@traceable(name="judge_analysis", run_type="chain")
async def judge_analysis(analysis_judge_request: AnalysisJudgeRequest):
    """Validates the ticket analysis before proceeding to guidance generation."""
    ticket = analysis_judge_request.ticket
    analysis = analysis_judge_request.analysis
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    analysis_prompt = get_analysis_judge_prompt(ticket, analysis.dict())
    analysis_response_text = query_hf_router(model=model_name, prompt=analysis_prompt, max_tokens=200)

    try:
        json_start_index = analysis_response_text.find('{')
        json_end_index = analysis_response_text.rfind('}')
        if json_start_index == -1 or json_end_index == -1:
            raise ValueError("No JSON object found in analysis judge response")
        json_string = analysis_response_text[json_start_index : json_end_index + 1]
        judge_json = json.loads(json_string)

        return AnalysisJudgeResponse(
            is_correct=judge_json.get("is_correct", True),
            confidence=float(judge_json.get("confidence", 0.5)),
            feedback=judge_json.get("feedback", "Analysis validation completed.")
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error parsing analysis judge response: {e}\nResponse was: {analysis_response_text}")
        return AnalysisJudgeResponse(
            is_correct=True,
            confidence=0.5,
            feedback="Unable to validate analysis. Proceeding with caution."
        )

@app.post("/api/judge", response_model=JudgeResponse)
@traceable(name="judge_response", run_type="chain")
async def judge_response(judge_request: JudgeRequest):
    """LLM Judge evaluates the quality of the ticket response."""
    ticket = judge_request.ticket
    analysis = judge_request.analysis
    guidance = judge_request.guidance
    email = judge_request.finalEmail

    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    judge_prompt = get_judge_prompt(ticket, analysis.dict(), guidance, email)
    judge_response_text = query_hf_router(model=model_name, prompt=judge_prompt, max_tokens=300)

    try:
        # Extract JSON from response
        json_start_index = judge_response_text.find('{')
        json_end_index = judge_response_text.rfind('}')
        if json_start_index == -1 or json_end_index == -1:
            raise ValueError("No JSON object found in judge response")
        json_string = judge_response_text[json_start_index : json_end_index + 1]
        judge_json = json.loads(json_string)

        # Calculate overall score as average
        overall_score = round((
            judge_json.get("quality_score", 5) +
            judge_json.get("correctness_score", 5) +
            judge_json.get("relevance_score", 5)
        ) / 3)

        return JudgeResponse(
            quality_score=judge_json.get("quality_score", 5),
            correctness_score=judge_json.get("correctness_score", 5),
            relevance_score=judge_json.get("relevance_score", 5),
            overall_score=overall_score,
            feedback=judge_json.get("feedback", "No feedback provided"),
            is_approved=judge_json.get("is_approved", overall_score >= 7)
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error parsing judge response: {e}\nResponse was: {judge_response_text}")
        # Return default scores if judge fails
        return JudgeResponse(
            quality_score=5,
            correctness_score=5,
            relevance_score=5,
            overall_score=5,
            feedback="Judge evaluation failed. Please review manually.",
            is_approved=False
        )