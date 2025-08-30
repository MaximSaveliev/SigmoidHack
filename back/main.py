from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
import json, re, os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
import logging

# Load .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Travel Advisor (Conversational)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Models
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    history: List[ChatTurn] = Field(default_factory=list)
    message: str

class ChatResponse(BaseModel):
    status: Literal["incomplete", "complete", "prompt_next"]
    question: str | None = None
    preferences: Dict[str, Any] | None = None
    recommendations: List[Dict[str, Any]] | None = None
    flights: List[Dict[str, Any]] | None = None
    hotels: List[Dict[str, Any]] | None = None

# Initialize LLM
llm = ChatOpenAI(model="gpt-5-nano", temperature=0.6)

# Setup memory (window of 10 interactions)
memory = ConversationBufferWindowMemory(
    memory_key="memory_history",
    input_key="message",
    k=10
)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_advisor")

# Web search tools
search_tool = DuckDuckGoSearchRun()
flight_tool = DuckDuckGoSearchRun(name="FlightSearch", description="Searches for flight information based on user preferences (e.g., origin, destination, dates).")
hotel_tool = DuckDuckGoSearchRun(name="HotelSearch", description="Searches for hotel information based on user preferences (e.g., destination, budget, dates).")
tools = [search_tool, flight_tool, hotel_tool]

# Initialize LangGraph ReAct agents
plan_agent = create_react_agent(llm, tools=[search_tool])
flight_agent = create_react_agent(llm, tools=[flight_tool])
hotel_agent = create_react_agent(llm, tools=[hotel_tool])

# City-to-region mapping
CITY_TO_REGION = {
    "florence": "Italy",
    "rome": "Italy",
    "venice": "Italy",
    "amalfi": "Italy",
    "sicily": "Italy",
    # Add more mappings as needed
}

# Helper: Robust JSON extraction
def try_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.DOTALL)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed] if parsed else []
    except:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                pass
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return [parsed] if parsed else []
            except:
                pass
        # Fallback: extract flights or hotels manually
        if "airline" in text.lower() or "price" in text.lower():
            items = re.findall(r"(\w+[\w\s]*)\s*:\s*\$?(\d+[\d\w\s-]*)\s*\((.*?)\)", text)
            return [{"airline": item[0].strip(), "price": item[1].strip(), "details": item[2].strip()} for item in items]
        elif "hotel" in text.lower() or "price" in text.lower():
            items = re.findall(r"(\w+[\w\s]*)\s*:\s*\$?(\d+[\d\w\s/-]*)\s*\((.*?)\)", text)
            return [{"name": item[0].strip(), "price": item[1].strip(), "details": item[2].strip()} for item in items]
        return []

# Intent Classifier
intent_prompt = PromptTemplate.from_template("""
You are a travel assistant. Based on the conversation and latest message, classify the user's intent into one of:
- preferences: Asking about travel plans or destinations (e.g., "places in Rome", without mentioning flights or hotels).
- flights: Requesting flight information (e.g., "flights from Chisinau" or "places and flights").
- hotels: Requesting hotel information (only if explicitly mentioned without flights).
- refine: Modifying existing preferences or recommendations (e.g., "change to Spain", "focus on Rome places").
- other: Unrelated or unclear intent.

Conversation (oldest first):
{memory_history}

Latest user message:
{message}

If the message mentions both destinations (e.g., "places in Rome") and flights, classify as "flights" but flag to include recommendations.
If the message specifies a city (e.g., Rome), treat it as a destination within its region (e.g., Italy).
If the message implies modifying prior recommendations (e.g., "change them a little bit", "focus on Rome"), classify as "refine".
Return ONLY JSON:
{{
  "intent": "preferences|flights|hotels|refine|other",
  "include_recommendations": true|false
}}
""")
intent_chain = intent_prompt | llm

# Preference Interpreter
pref_prompt = PromptTemplate.from_template("""
You are a travel assistant. From the conversation below, extract or update structured preferences.

Required fields (ask if missing):
- region (destination, e.g., "Italy" or "Europe"; infer from city names like Rome -> Italy)
- when (season or dates, e.g., "October" or "2023-09-07..2023-09-14"; handle flexible dates like "+-2 days")
- origin (city for flights, e.g., "Chisinau")

Optional fields (use if provided, else leave unset):
- destination (specific city, e.g., "Rome")
- duration (e.g., "7 days")
- budget (one of: low, medium, high)
- interests (array of tags like "culture","food","beach")

Conversation (oldest first):
{memory_history}

Latest user message:
{message}

If a city (e.g., Rome) is mentioned, map it to its region (e.g., Italy) and set as destination.
If flexible dates are mentioned (e.g., "7 September or +-2 days"), include the range in "when" (e.g., "2023-09-05..2023-09-09").
Return ONLY valid JSON:
{{
  "preferences": {{
    "region": "...",
    "when": "...",
    "origin": "...",
    "destination": "...",
    "duration": "...",
    "budget": "...",
    "interests": ["..."]
  }},
  "missing": ["field1","field2"]  // required fields only
}}
""")
pref_chain = pref_prompt | llm

# Helper: Load preferences and recommendations
def load_preferences_and_recommendations(history: List[ChatTurn]):
    preferences = {}
    recommendations = []
    # First, try to get from history
    for turn in reversed(history):
        if turn.role == "assistant":
            try:
                data = try_json(turn.content)
                data = data[0] if data else {}
                if isinstance(data, dict):
                    if "preferences" in data:
                        preferences = data.get("preferences", {})
                    if "recommendations" in data and data["recommendations"]:
                        recommendations = data["recommendations"]
            except:
                continue
    # Fallback to memory
    saved_prefs = memory.load_memory_variables({}).get("memory_history", "")
    if saved_prefs:
        for line in reversed(saved_prefs.split("\n")):
            if "Assistant: " in line:
                try:
                    data = try_json(line.split("Assistant: ")[-1])
                    data = data[0] if data else {}
                    if isinstance(data, dict):
                        if "preferences" in data:
                            preferences = data.get("preferences", {})
                        if "recommendations" in data and data["recommendations"]:
                            recommendations = data["recommendations"]
                except:
                    continue
    missing = [f for f in ["region", "when", "origin"] if not preferences.get(f)]
    return preferences, missing, recommendations

# Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # Build history for memory
        hist_lines = [f"{turn.role.capitalize()}: {turn.content}" for turn in req.history]
        history_text = "\n".join(hist_lines)

        # Save user message to memory
        memory.save_context({"message": req.message}, {"output": ""})

        # Step 1: Classify intent
        intent_raw = await intent_chain.ainvoke({"memory_history": memory.load_memory_variables({})["memory_history"], "message": req.message})
        logger.info(f"LLM (intent) raw output: {getattr(intent_raw, 'content', intent_raw)}")
        intent_data = try_json(intent_raw.content if hasattr(intent_raw, "content") else intent_raw)
        intent_data = intent_data[0] if intent_data else {}
        intent = intent_data.get("intent", "other")
        include_recommendations = intent_data.get("include_recommendations", False)

        # Load existing preferences and recommendations
        current_prefs, current_missing, current_recommendations = load_preferences_and_recommendations(req.history)

        # Step 2: Handle intent
        if intent in ["preferences", "refine"]:
            # Interpret preferences
            pref_raw = await pref_chain.ainvoke({"memory_history": memory.load_memory_variables({})["memory_history"], "message": req.message})
            logger.info(f"LLM (preferences) raw output: {getattr(pref_raw, 'content', pref_raw)}")
            pref_data = try_json(pref_raw.content if hasattr(pref_raw, "content") else pref_raw)
            pref_data = pref_data[0] if pref_data else {}
            
            new_prefs = pref_data.get("preferences", {}) if isinstance(pref_data, dict) else {}
            preferences = {**current_prefs, **{k: v for k, v in new_prefs.items() if v and v != "..."}}
            missing = [f for f in ["region", "when", "origin"] if not preferences.get(f)]

            # Save merged preferences
            memory.save_context({"message": req.message}, {"output": json.dumps({"preferences": preferences, "missing": missing})})

            # Ask one question at a time for required fields
            if missing:
                questions = {
                    "region": f"I see you mentioned {preferences.get('region', 'a destination')}. Please confirm or specify where you want to travel (e.g., Italy, Europe). For better results, provide your travel dates and departure city.",
                    "when": f"I see you're planning to travel to {preferences.get('region', 'a destination')}. When do you want to travel (e.g., October, or specific dates like 2023-09-10)? For better results, provide your departure city.",
                    "origin": f"I see you're planning to travel to {preferences.get('region', 'a destination')} around {preferences.get('when', 'some time')}. Which city are you traveling from (e.g., Chisinau)? For better results, provide any preferences like budget or interests.",
                }
                priority_missing = next((f for f in ["region", "when", "origin"] if f in missing), None)
                ask = questions.get(priority_missing, "Please clarify your travel plans (e.g., where are you going, when, and from where?).")
                return ChatResponse(status="incomplete", question=ask, preferences=preferences)

            # Plan destinations if preferences are complete
            destination = preferences.get("destination", preferences["region"])
            plan_query = f"""
Given these user travel preferences (JSON):
{json.dumps(preferences, ensure_ascii=False)}

Suggest 1-2 destinations focusing on {destination}. If a specific city (e.g., Rome) is mentioned, provide attractions or activities specific to that city.
For each, provide a brief reason why it matches the preferences and list 3-5 top attractions or activities.
Return ONLY JSON array like:
[
  {{
    "destination": "...",
    "why": "...",
    "attractions": ["...","...","..."]
  }}
]
"""
            plan_raw = await plan_agent.ainvoke({"messages": [{"role": "user", "content": plan_query}]})
            logger.info(f"LLM (plan) raw output: {plan_raw['messages'][-1].content}")
            recommendations = try_json(plan_raw['messages'][-1].content)

            # If refine intent, filter or adjust recommendations
            if intent == "refine" and current_recommendations:
                recommendations = [r for r in current_recommendations if r["destination"].lower() == destination.lower()] or recommendations

            # Save recommendations
            memory.save_context({"message": req.message}, {"output": json.dumps({"preferences": preferences, "recommendations": recommendations})})

            return ChatResponse(
                status="prompt_next",
                preferences=preferences,
                recommendations=recommendations or [],
                question="Would you like me to find flights or hotels for your trip?"
            )

        elif intent == "flights":
            # Interpret preferences from the message
            pref_raw = await pref_chain.ainvoke({"memory_history": memory.load_memory_variables({})["memory_history"], "message": req.message})
            logger.info(f"LLM (preferences) raw output: {getattr(pref_raw, 'content', pref_raw)}")
            pref_data = try_json(pref_raw.content if hasattr(pref_raw, "content") else pref_raw)
            pref_data = pref_data[0] if pref_data else {}
            
            new_prefs = pref_data.get("preferences", {}) if isinstance(pref_data, dict) else {}
            preferences = {**current_prefs, **{k: v for k, v in new_prefs.items() if v and v != "..."}}
            missing = [f for f in ["region", "when", "origin"] if not preferences.get(f)]

            if missing:
                questions = {
                    "region": f"Please specify where you want to travel (e.g., Italy, Europe). For better results, provide your travel dates and departure city.",
                    "when": f"I see you're planning to travel to {preferences.get('region', 'a destination')}. When do you want to travel (e.g., October, or specific dates like 2023-09-10)? For better results, provide your departure city.",
                    "origin": f"I see you're planning to travel to {preferences.get('region', 'a destination')} around {preferences.get('when', 'some time')}. Which city are you traveling from (e.g., Chisinau)?",
                }
                priority_missing = next((f for f in ["region", "when", "origin"] if f in missing), None)
                ask = questions.get(priority_missing, "Please provide your destination, travel dates, and departure city for flight search.")
                return ChatResponse(status="incomplete", question=ask, preferences=preferences)

            # Generate recommendations if requested
            recommendations = current_recommendations
            if include_recommendations:
                destination = preferences.get("destination", preferences["region"])
                plan_query = f"""
Given these user travel preferences (JSON):
{json.dumps(preferences, ensure_ascii=False)}

Suggest 1-2 destinations focusing on {destination}. If a specific city (e.g., Rome) is mentioned, provide attractions or activities specific to that city.
For each, provide a brief reason why it matches the preferences and list 3-5 top attractions or activities.
Return ONLY JSON array like:
[
  {{
    "destination": "...",
    "why": "...",
    "attractions": ["...","...","..."]
  }}
]
"""
                plan_raw = await plan_agent.ainvoke({"messages": [{"role": "user", "content": plan_query}]})
                logger.info(f"LLM (plan) raw output: {plan_raw['messages'][-1].content}")
                recommendations = try_json(plan_raw['messages'][-1].content)

            # Save preferences and recommendations
            memory.save_context({"message": req.message}, {"output": json.dumps({"preferences": preferences, "recommendations": recommendations})})

            flight_query = f"""
Search for flights from {preferences['origin']} to {preferences.get('destination', preferences['region'])} for {preferences['when']}.
If dates include a range (e.g., +-2 days), search within that range.
Include budget preference ({preferences.get('budget', 'any')}) if specified, and ensure at least 2-3 flight options.
Return ONLY JSON array like:
[
  {{
    "airline": "...",
    "price": "...",
    "details": "..."
  }}
]
"""
            flight_raw = await flight_agent.ainvoke({"messages": [{"role": "user", "content": flight_query}]})
            logger.info(f"LLM (flights) raw output: {flight_raw['messages'][-1].content}")
            flights = try_json(flight_raw['messages'][-1].content)

            # Fallback if no flights found
            if not flights:
                flights = [
                    {"airline": "Unknown", "price": "N/A", "details": f"No flights found for {preferences['origin']} to {preferences.get('destination', preferences['region'])} in {preferences['when']}"}
                ]

            memory.save_context({"message": req.message}, {"output": json.dumps({"flights": flights})})

            return ChatResponse(
                status="complete",
                preferences=preferences,
                recommendations=recommendations or [],
                flights=flights
            )

        elif intent == "hotels":
            # Interpret preferences from the message
            pref_raw = await pref_chain.ainvoke({"memory_history": memory.load_memory_variables({})["memory_history"], "message": req.message})
            logger.info(f"LLM (preferences) raw output: {getattr(pref_raw, 'content', pref_raw)}")
            pref_data = try_json(pref_raw.content if hasattr(pref_raw, "content") else pref_raw)
            pref_data = pref_data[0] if pref_data else {}
            
            new_prefs = pref_data.get("preferences", {}) if isinstance(pref_data, dict) else {}
            preferences = {**current_prefs, **{k: v for k, v in new_prefs.items() if v and v != "..."}}
            missing = [f for f in ["region", "when"] if not preferences.get(f)]

            if missing:
                questions = {
                    "region": f"Please specify where you want to travel (e.g., Italy, Europe). For better results, provide your travel dates.",
                    "when": f"I see you're planning to travel to {preferences.get('region', 'a destination')}. When do you want to travel (e.g., October, or specific dates like 2023-09-10)?",
                }
                priority_missing = next((f for f in ["region", "when"] if f in missing), None)
                ask = questions.get(priority_missing, "Please provide your destination and travel dates for hotel search.")
                return ChatResponse(status="incomplete", question=ask, preferences=preferences)

            # Generate recommendations if requested
            recommendations = current_recommendations
            if include_recommendations:
                destination = preferences.get("destination", preferences["region"])
                plan_query = f"""
Given these user travel preferences (JSON):
{json.dumps(preferences, ensure_ascii=False)}

Suggest 1-2 destinations focusing on {destination}. If a specific city (e.g., Rome) is mentioned, provide attractions or activities specific to that city.
For each, provide a brief reason why it matches the preferences and list 3-5 top attractions or activities.
Return ONLY JSON array like:
[
  {{
    "destination": "...",
    "why": "...",
    "attractions": ["...","...","..."]
  }}
]
"""
                plan_raw = await plan_agent.ainvoke({"messages": [{"role": "user", "content": plan_query}]})
                logger.info(f"LLM (plan) raw output: {plan_raw['messages'][-1].content}")
                recommendations = try_json(plan_raw['messages'][-1].content)

            # Save preferences and recommendations
            memory.save_context({"message": req.message}, {"output": json.dumps({"preferences": preferences, "recommendations": recommendations})})

            hotel_query = f"""
Search for hotels in {preferences.get('destination', preferences['region'])} for {preferences['when']}.
Include budget preference ({preferences.get('budget', 'any')}) if specified, and ensure at least 2-3 hotel options.
Return ONLY JSON array like:
[
  {{
    "name": "...",
    "price": "...",
    "details": "..."
  }}
]
"""
            hotel_raw = await hotel_agent.ainvoke({"messages": [{"role": "user", "content": hotel_query}]})
            logger.info(f"LLM (hotels) raw output: {hotel_raw['messages'][-1].content}")
            hotels = try_json(hotel_raw['messages'][-1].content)

            # Fallback if no hotels found
            if not hotels:
                hotels = [
                    {"name": "Unknown", "price": "N/A", "details": f"No hotels found in {preferences.get('destination', preferences['region'])} for {preferences['when']}"}
                ]

            memory.save_context({"message": req.message}, {"output": json.dumps({"hotels": hotels})})

            return ChatResponse(
                status="complete",
                preferences=preferences,
                recommendations=recommendations or [],
                hotels=hotels
            )

        else:
            preferences, _ = load_preferences_and_recommendations(req.history)
            memory.save_context({"message": req.message}, {"output": "Please clarify your request (e.g., destinations, flights, hotels, or changes)."})
            return ChatResponse(
                status="incomplete",
                question="Please clarify your request (e.g., where are you going, or do you need flights or hotels?).",
                preferences=preferences
            )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Clear Memory Endpoint
@app.post("/clear-memory")
async def clear_memory():
    memory.clear()
    return {"ok": True}

# Health Endpoint
@app.get("/health")
async def health():
    return {"ok": True}