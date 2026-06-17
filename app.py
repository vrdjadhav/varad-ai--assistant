# app.py
import streamlit as st
import time  # For giving visual step-by-step assurance delays

# 1. Page Configuration (Clean, Unbranded Layout)
st.set_page_config(
    page_title="Intelligent Research Companion",
    page_icon="🤖",
    layout="centered"
)

# 2. Imports
from src.chatbot import start_new_chat, send_chat_message, determine_routing_intent
from src.pdf_handler import extract_text_from_pdf
from src.text_splitter import split_text_into_chunks
from src.vector_store import create_vector_db, search_vector_db, load_vector_db, clear_local_cache
from src.agent_engine import generate_evaluation_quiz, evaluate_quiz_submission

# 3. Initialize Multi-Document Session States 
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = start_new_chat()

if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  

if "global_text_pool" not in st.session_state:
    st.session_state.global_text_pool = ""     

if "faiss_index" not in st.session_state or "mapped_chunks" not in st.session_state:
    cached_index, cached_chunks = load_vector_db()
    
    st.session_state.faiss_index = cached_index
    st.session_state.mapped_chunks = cached_chunks
    
    if cached_chunks:
        extracted_names = set()
        for chunk in cached_chunks:
            if chunk.startswith("[") and "]:" in chunk:
                filename = chunk.split("]:")[0].lstrip("[")
                extracted_names.add(filename)
        st.session_state.processed_files = extracted_names
        
        # Build back the global text pool layout for global summaries
        st.session_state.global_text_pool = "\n\n".join(cached_chunks)

if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = None          # Stores the generated quiz JSON layout

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}           # Tracks real-time inputs tied to question IDs

if "quiz_evaluation" not in st.session_state:
    st.session_state.quiz_evaluation = None      # Stores the final graded MasterReport JSON


# App Headers (Production Configuration)
st.title("🤖 Intelligent Research Companion")
st.caption("AI-powered document intelligence, contextual retrieval, and adaptive evaluation.")

# 4. Sidebar UI: Project Info, PDF Uploader, and Actions
with st.sidebar:
    st.header("System Properties")
    st.write("**Core Architecture:** Contextual RAG & Agentic Workflows")
    st.write("**Data Infrastructure:** Persistent Local Disk Cache")
    st.write("**System Status:** Fully Operational")
    
    st.markdown("---")
    
    # Configured to accept multiple files simultaneously
    uploaded_files = st.file_uploader(
        "Upload Study Documents (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True, 
        key="pdf_uploader_widget"
    )
    
# 5. High-Performance Incremental Processing Pipeline Engine
    if uploaded_files:
        for file in uploaded_files:
            # Only run processing if the file has not yet been processed in this session
            if file.name not in st.session_state.processed_files:
                
                # Wrap the entire pipeline into a single, clean production status indicator
                with st.sidebar.status(f"📥 Ingesting: {file.name}...", expanded=False) as status:
                    
                    # 1. Document Text Extraction
                    text = extract_text_from_pdf(file)
                    
                    if text:
                        # Append file contents to global full text pool for global analytics
                        st.session_state.global_text_pool += f"\n\n--- DOCUMENT START: {file.name} ---\n{text}\n--- DOCUMENT END: {file.name} ---"
                        
                        # 2. Semantic Text Chunking & Document Tagging
                        raw_chunks = split_text_into_chunks(text)
                        tagged_chunks = [f"[{file.name}]: {chunk}" for chunk in raw_chunks]
                            
                        # 3. Mathematical Vector Indexing 
                        index, valid_chunks = create_vector_db(
                            new_chunks=tagged_chunks,
                            existing_index=st.session_state.faiss_index,
                            existing_chunks=st.session_state.mapped_chunks
                        )
                            
                        if index is not None:
                            # Update system states with new knowledge vectors
                            st.session_state.faiss_index = index
                            st.session_state.mapped_chunks = valid_chunks
                            st.session_state.processed_files.add(file.name)
                            
                            # Mark the status tracker as a clean success
                            status.update(label=f"✅ {file.name} fully indexed!", state="complete", expanded=False)
                        else:
                            status.update(label=f"❌ Ingestion failed for {file.name}", state="error", expanded=True)
                            st.sidebar.error("Vector storage indexing dropped. Please re-upload.")
                    else:
                        status.update(label=f"❌ Failed to read {file.name}", state="error", expanded=True)
                        st.sidebar.error("Could not parse text layer. Verify the PDF is not encrypted.")
                st.markdown("---")
                
    st.markdown("---")
    
    # Reset Controls (Upgraded to handle hard drive and quiz files clear)
    if st.button("🗑️ Clear Chat & Document Matrix", use_container_width=True):
        st.session_state.gemini_chat = start_new_chat()
        st.session_state.ui_history = []  
        st.session_state.processed_files = set()
        st.session_state.global_text_pool = ""
        st.session_state.faiss_index = None
        st.session_state.mapped_chunks = []
        
        # Reset Quiz Trackers too
        st.session_state.active_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_evaluation = None
        
        # Hard purge local binary files from storage directory
        clear_local_cache()
        
        if "pdf_uploader_widget" in st.session_state:
            del st.session_state["pdf_uploader_widget"]
            
        st.rerun()


# =====================================================================
# MAIN SCREEN LAYOUT (TAB ISOLATED INTERFACE DESIGN)
# =====================================================================

chat_tab, quiz_tab = st.tabs(["💬 Interactive Research Chat", "🧠 Agentic Performance Quiz"])

# --- TAB 1: INTERACTIVE RESEARCH CHAT ---
with chat_tab:
    chat_container = st.container()

    # Display Existing Chat History with Recalled Citations
    with chat_container:
        for message in st.session_state.ui_history:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])
                
                # Re-render citation expander blocks if they belong to past responses
                if message.get("role") == "assistant" and message.get("citations"):
                    with st.expander("🔍 System Scan: View Retrieved Source Fragments"):
                        for idx, chunk in enumerate(message["citations"], 1):
                            st.caption(f"**Fragment Vector Match #{idx}:**")
                            st.write(chunk)
                            st.markdown("---")

    # Handle New User Input (Updated placeholder prompt to match TA persona)
    if user_input := st.chat_input("Ask a question about the study materials...", key="chat_input_widget"):
        
        # Store user command in UI list immediately
        st.session_state.ui_history.append({"role": "user", "text": user_input, "citations": None})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
                
            with st.chat_message("assistant"):
                context_payload = None
                matched_fragments = []
                
                # Step A: Evaluate Traffic Intent Router (Updated loader spinner context)
                routing_mode = "LOCAL"
                if st.session_state.faiss_index is not None:
                    with st.spinner("Analyzing your question's focus route..."):
                        routing_mode = determine_routing_intent(user_input)
                
                # Step B: Assemble Context Based On Intended Route (Updated loader spinner context)
                if st.session_state.faiss_index is not None:
                    if routing_mode == "GLOBAL":
                        context_payload = st.session_state.global_text_pool
                    else:
                        with st.spinner("Scanning relevant document segments..."):
                            matched_fragments = search_vector_db(
                                index=st.session_state.faiss_index,
                                valid_chunks=st.session_state.mapped_chunks,
                                query=user_input,
                                top_k=3
                            )
                            if matched_fragments:
                                context_payload = "\n\n---\n\n".join(matched_fragments)

                # Step C: Dispatch Request to Backend Engine (Updated loader spinner context)
                with st.spinner("Formulating explanation..."):
                    response_text = send_chat_message(
                        chat_session=st.session_state.gemini_chat, 
                        user_message=user_input, 
                        context=context_payload,
                        routing_mode=routing_mode
                    )
                    st.markdown(response_text)
                    
                    # Step D: Present Source Citations Expander (Only during Local Search route matches)
                    if routing_mode == "LOCAL" and matched_fragments:
                        with st.expander("🔍 System Scan: View Retrieved Source Fragments"):
                            for idx, chunk in enumerate(matched_fragments, 1):
                                st.caption(f"**Fragment Vector Match #{idx}:**")
                                st.write(chunk)
                                st.markdown("---")
                    
                    # Cache response and citations into history tracking list
                    st.session_state.ui_history.append({
                        "role": "assistant", 
                        "text": response_text,
                        "citations": matched_fragments if (routing_mode == "LOCAL" and matched_fragments) else None
                    })
                    
        st.rerun()


# --- TAB 2: AGENTIC PERFORMANCE QUIZ WORKSPACE ---
with quiz_tab:
    st.subheader("🧠 Weakness Mapping & Evaluation Portal")
    st.caption("This engine scans your active document pools to evaluate your conceptual comprehension.")
    
    if st.session_state.faiss_index is None:
        st.info("ℹ️ Please upload study documents in the sidebar to activate the Quiz Engine.")
    else:
        # Action trigger to invoke the Quiz Generation Agent
        if st.session_state.active_quiz is None:
            if st.button("⚡ Generate Customized Evaluation Quiz", use_container_width=True):
                with st.spinner("Analyzing study text blocks to formulate your quiz questions..."):
                    quiz_payload = generate_evaluation_quiz(st.session_state.mapped_chunks)
                    if quiz_payload:
                        st.session_state.active_quiz = quiz_payload
                        st.rerun()
                    else:
                        st.error("❌ Failed to compile quiz assets. Please try again.")
        else:
            # Render Active Quiz Layout
            st.markdown(f"### 📋 Evaluation: **{st.session_state.active_quiz['quiz_title']}**")
            st.write("Answer the conceptual questions below based on your readings:")
            st.markdown("---")
            
            # Dynamically loop through generated questions
            for q in st.session_state.active_quiz["questions"]:
                st.markdown(f"**Question {q['question_id']}:** {q['question_text']}")
                st.caption(f"📚 *Focus Area:* {q['topic']}")
                
                # Render unique text areas for responses with isolated dynamic keys (Updated placeholder prompt)
                st.session_state.quiz_answers[str(q['question_id'])] = st.text_area(
                    "Your Answer Core Input:",
                    key=f"q_input_{q['question_id']}",
                    placeholder="Type your explanation here..."
                )
                st.markdown("---")
                
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Submit Completed Assessment", use_container_width=True, type="primary"):
                    with st.spinner("Reviewing response accuracy and compiling feedback report..."):
                        report = evaluate_quiz_submission(
                            quiz_data=st.session_state.active_quiz,
                            user_answers=st.session_state.quiz_answers,
                            mapped_chunks=st.session_state.mapped_chunks
                        )
                        if report:
                            st.session_state.quiz_evaluation = report
                            st.rerun()
            with col2:
                if st.button("🔄 Reset Quiz Matrix", use_container_width=True):
                    st.session_state.active_quiz = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_evaluation = None
                    st.rerun()

            # --- RENDER EVALUATOR CRITIC REPORT CARD ---
            if st.session_state.quiz_evaluation:
                st.markdown("## 📊 Evaluation Report Card")
                score = st.session_state.quiz_evaluation["total_score_percentage"]
                
                if score >= 75:
                    st.success(f"🏆 **Great job! Cumulative Score: {score}%**")
                elif score >= 45:
                    st.warning(f"⚠️ **Conceptual Gaps Found. Cumulative Score: {score}%**")
                else:
                    st.error(f"🚨 **Remedial Focus Recommended. Cumulative Score: {score}%**")
                    
                st.markdown(f"### 🎯 Primary Knowledge Gap Identified:\n> *{st.session_state.quiz_evaluation['primary_knowledge_gap']}*")
                
                st.markdown("### 🔍 Itemized Diagnostic Breakdown")
                for eval_item in st.session_state.quiz_evaluation["evaluations"]:
                    q_id = eval_item["question_id"]
                    st.markdown(f"#### ❓ Question Reference Matrix #{q_id} (Score: **{eval_item['score_out_of_10']}/10**)")
                    st.write(f"🍏 **Strengths:** {eval_item['strengths']}")
                    st.write(f"🍎 **Weaknesses:** {eval_item['weaknesses']}")
                    
                    with st.expander("📖 View Remedial Learning Guide for this Concept"):
                        st.info(eval_item["remedial_guidance"])
                
                st.markdown("### 📋 Recommended Focus Topics for Next Session")
                for topic in st.session_state.quiz_evaluation["recommended_focus_topics"]:
                    st.markdown(f"* 📚 {topic}")