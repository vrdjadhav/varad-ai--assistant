# src/chatbot.py
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# 1. Load environment variables safely
load_dotenv()

# 2. Initialize the modern Gemini client
client = genai.Client()

# 3. Define a strict schema for Type-Safe Routing Outputs
class IntentRouterSchema(BaseModel):
    reasoning: str = Field(description="A brief sentence explaining why this command is global or local.")
    routing_mode: Literal["GLOBAL", "LOCAL"] = Field(description="Must be strictly 'GLOBAL' or 'LOCAL'")

def determine_routing_intent(user_message):
    """
    Type-safe, schema-enforced background evaluation to determine whether 
    the command requires a global full-text sweep or localized FAISS fragments.
    Includes a resilient keyword fallback guard.
    """
    # Defensive Fallback List in case the API drops connection mid-flight
    global_keywords = ["index", "chapters", "summary", "summarize", "overview", "syllabus", "notes", "quiz"]
    cleaned_msg = user_message.lower()

    try:
        router_prompt = (
            "Analyze the user's command and classify its scope into either 'GLOBAL' or 'LOCAL'.\n\n"
            "CRITERIA:\n"
            "- GLOBAL: The user wants an index, table of contents, full chapter list, a holistic summary of the "
            "whole book, a comprehensive study note guide of everything, or a complete document quiz.\n"
            "- LOCAL: The user is asking about a specific term, definition, single concept, snippet of code, "
            "or particular detail found on a specific page."
        )
        
        # Requesting a structurally guaranteed JSON response matching our Pydantic schema
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{router_prompt}\n\nUser Command: {user_message}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=IntentRouterSchema,
                max_output_tokens=100
            )
        )
        
        # Safely parse the structural object response
        structured_data = IntentRouterSchema.model_validate_json(response.text)
        return structured_data.routing_mode

    except Exception as e:
        print(f"\n[ROUTER API ERROR]: {e}. Activating Resilient Keyword Fallback Guard...")
        # Step 3 Optimization: Fallback search if the API stumbles
        for word in global_keywords:
            if word in cleaned_msg:
                print(f"[FALLBACK SUCCESS]: Found keyword '{word}'. Routing to GLOBAL.")
                return "GLOBAL"
        
        print("[FALLBACK DEFAULT]: No global keys found. Routing to LOCAL.")
        return "LOCAL"

def start_new_chat():
    """
    Initializes a fresh Gemini chat session with an approachable, knowledgeable,
    and encouraging Graduate Teaching Assistant (TA) persona.
    """
    try:
        ta_instructions = (
            "You are an approachable, knowledgeable, and highly encouraging Graduate Teaching Assistant (TA). "
            "Your goal is to help the student break down their research documents and study materials clearly. "
            "Use a warm, friendly, collaborative, and peer-to-peer tone. Keep explanations structurally organized, "
            "easy to digest, and simplify complex academic logic without losing technical accuracy. "
            "Always base your responses strictly on the provided document context. If a concept or question "
            "isn't supported by the context text, let the student know honestly and guide them back to what is available."
        )

        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=ta_instructions,
                temperature=0.4
            )
        )
        return chat
    except Exception as e:
        print(f"\n[CRITICAL] Failed to initialize Chat Session: {e}\n")
        return None

def send_chat_message(chat_session, user_message, context=None, routing_mode="LOCAL"):
    """
    Forwards commands along with context. Implements a Context Truncation Guard
    to handle massive text injections smoothly.
    """
    if chat_session is None:
        return "⚠️ Hey there! It looks like our chat session couldn't be initialized. Let's try refreshing the page."
        
    # Step 2 Optimization: Context Truncation Guard (approx. 800k character threshold)
    MAX_CHARACTER_THRESHOLD = 800000
    if context and len(context) > MAX_CHARACTER_THRESHOLD:
        print(f"[GUARDRAIL]: Context string length ({len(context)}) exceeds safety threshold. Truncating.")
        context = context[:MAX_CHARACTER_THRESHOLD] + "\n\n[SYSTEM NOTICE: Context safely truncated due to capacity guardrails.]"
        
    try:
        if context:
            if routing_mode == "GLOBAL":
                full_prompt = (
                    f"Hey! I've bypassed the local chunk filters and executed a full global scan "
                    f"of the entire document package to give you a broad overview.\n\n"
                    f"=== FULL SYSTEM DOCUMENT MATRIX ===\n{context}\n===================================\n\n"
                    f"Student Query: {user_message}"
                )
            else:
                full_prompt = (
                    f"Hey! I ran a targeted scan on our vector database coordinates and found "
                    f"these specific source text fragments to help answer your question.\n\n"
                    f"=== SYSTEM DOCUMENT SCAN ===\n{context}\n============================\n\n"
                    f"Student Query: {user_message}"
                )
        else:
            full_prompt = user_message

        response = chat_session.send_message(full_prompt)
        return response.text
        
    except Exception as e:
        print(f"\n[CORE ERROR]: {e}\n")
        return f"⚠️ Oh no, it looks like an error popped up while getting the response: {e}"