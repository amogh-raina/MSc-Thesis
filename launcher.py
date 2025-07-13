"""
MSc Thesis - Legal Knowledge System Launcher
============================================

This launcher allows you to run any of the available evaluation systems asynchronously.
Each system will run on its own port, allowing you to work on multiple systems
simultaneously.

Usage: python launcher.py
"""

import sys
import subprocess
import streamlit as st
from pathlib import Path
import os
import socket
from contextlib import closing

# Add Main folder to path for shared components
project_root = Path(__file__).resolve().parent
main_folder = project_root / "Main"
if str(main_folder) not in sys.path:
    sys.path.insert(0, str(main_folder))

# --- Utility Functions ---

def find_free_port():
    """Find and return an available port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

# --- Main Application ---

def main():
    st.set_page_config(
        page_title="Legal Knowledge System Launcher",
        page_icon="⚖️",
        layout="centered"
    )
    
    # Initialize session state for managing launched systems
    if "launched_systems" not in st.session_state:
        st.session_state.launched_systems = {}
    
    st.title("⚖️ Legal Knowledge System Launcher")
    
    # --- Active Systems Management ---
    active_systems_management()
    
    st.markdown("### 🚀 Launch a System")
    st.markdown("Choose which evaluation system to run. Each will open on a new port.")
    
    # --- System Definitions ---
    system_definitions()
    
    # --- Additional Information ---
    additional_information()

def active_systems_management():
    """UI section to manage currently running systems."""
    if st.session_state.launched_systems:
        st.markdown("---")
        st.subheader("🟢 Active Systems")
        st.caption("These systems are currently running. You can stop them here to free up resources.")
        
        for app_file, info in list(st.session_state.launched_systems.items()):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"**{info['name']}**")
            with col2:
                st.info(f"🌐 Running on: `http://localhost:{info['port']}`")
            with col3:
                if st.button("🛑 Stop", key=f"stop_{info['port']}", type="secondary"):
                    stop_system(app_file)
                    st.rerun()

def system_definitions():
    """UI section to display and launch available systems."""
    systems = {
        "🧠 LLM-Only System": {
            "description": "Pure LLM evaluation without external knowledge sources.",
            "file": "LLM-Only/ui/llm_only_ui.py",
            "features": ["Direct LLM evaluation", "Question bank integration", "Batch processing"],
            "status": "✅ Available"
        },
        "🔍 RAG System": {
            "description": "Retrieval-Augmented Generation with document search.",
            "file": "RAG_Pipeline/ui/RAG_ui.py", 
            "features": ["Document retrieval", "Vector search", "Context-aware answers"],
            "status": "✅ Available"
        },
        "🕸️ GraphRAG System": {
            "description": "Graph-enhanced RAG with multiple retrieval variants.",
            "file": "GraphRAG_Pipeline/ui/GraphRAG_ui.py",
            "features": ["Knowledge graph integration", "3 retrieval variants", "Authority scoring", "LangChain reranking"],
            "status": "✅ Available"
        }
    }
    
    st.markdown("---")
    
    for system_name, system_info in systems.items():
        with st.container():
            col1, col2 = st.columns([3, 1])
            file_path = Path(system_info['file'])
            
            with col1:
                st.subheader(system_name)
                st.markdown(f"**{system_info['description']}**")
                features_text = " • ".join(system_info['features'])
                st.markdown(f"*{features_text}*")
                
                if not file_path.exists():
                    st.error(f"File not found: {system_info['file']}")

            with col2:
                system_key = system_name.split()[0].lower().replace("🧠", "llm").replace("🔍", "rag").replace("🕸️", "graphrag").replace("🔬", "demo")
                
                if file_path.exists():
                    if st.button(f"🚀 Launch", key=f"launch_{system_key}", type="primary"):
                        launch_system(system_info['file'], system_name)
                else:
                    st.button("❌ Missing", key=f"launch_{system_key}", disabled=True)

            st.markdown("---")

def additional_information():
    """Display system requirements and configuration info."""
    with st.expander("📋 System Requirements & Configuration"):
        st.info("""
        **Before launching any system:**
        - Ensure all dependencies are installed: `pip install -r requirements.txt`
        - Set up your API keys in a `.env` file.
        - Prepare your question bank data (JSON files in `LLM-Only/ui/JSON Trial 1/`)
        - For RAG: Prepare your document datasets (CSV format).
        """)
        st.info("""
        **Shared components are located in the `Main/` folder:**
        - `core/`: Model management, evaluation logic, question bank
        - `config/`: Global configuration settings  
        - `utils/`: Custom LLM implementations and utilities
        """)

def launch_system(app_file: str, system_name: str):
    """Launch the selected system automatically on a free port."""
    if not Path(app_file).exists():
        st.error(f"❌ File not found: {app_file}")
        return

    port = find_free_port()
    
    try:
        command = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.headless", "false" # Automatically open in a new browser tab
        ]
        
        # For non-UI scripts, just run them with python
        is_ui_app = app_file.endswith(('_ui.py', '_app.py'))
        if not is_ui_app:
             command = [sys.executable, app_file]

        process = subprocess.Popen(command, cwd=str(Path(__file__).parent))
        
        st.session_state.launched_systems[app_file] = {
            "process": process,
            "port": port,
            "name": system_name,
            "is_ui": is_ui_app
        }
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to launch {system_name}: {e}")

def stop_system(app_file: str):
    """Stop a running system and remove it from the session state."""
    if app_file in st.session_state.launched_systems:
        info = st.session_state.launched_systems[app_file]
        info['process'].terminate()  # Send termination signal
        info['process'].wait()       # Wait for process to exit
        del st.session_state.launched_systems[app_file]
        st.success(f"🛑 Stopped {info['name']} on port {info['port']}. You can now safely close its browser tab.")

if __name__ == "__main__":
    main() 