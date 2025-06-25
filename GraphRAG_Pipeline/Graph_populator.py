def render_llm_selection(key_prefix=""):
    """Reusable LLM selection component"""
    providers = ModelManager.get_available_llm_providers()
    # ... selection logic
    return selected_provider, selected_model

def render_embedding_selection(key_prefix=""):
    """Reusable embedding selection component"""
    # ... selection logic
    return selected_provider, selected_model
