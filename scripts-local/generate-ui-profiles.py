#!/usr/bin/env python3
import json
import os
import re
import sys
import subprocess

def parse_bash_conf(file_path):
    """Source the bash config and return the variables as a dict."""
    try:
        # Use set -a to export all variables defined in the sourced file
        cmd = f"set -a; source {file_path}; env"
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
        env_dict = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env_dict[key] = value
        return env_dict
    except Exception as e:
        print(f"Error parsing bash config: {e}")
        return {}

def get_float(val, default):
    try:
        return float(val)
    except:
        return default

def generate_personas(conf, config_name):
    # IDs should be short, lowercase, hyphenated (Standard #7)
    base_name = config_name.replace(".gguf.conf", "").lower()
    
    # Simplify common prefixes/suffixes for shortness
    base_id = base_name.replace("google-", "").replace("google_", "")
    base_id = re.sub(r'[^a-z0-9]', '-', base_id)
    base_id = re.sub(r'-+', '-', base_id).strip('-')
    
    # Heuristic for descriptive yet short base_id
    parts = base_id.split("-")
    if len(parts) > 2:
        family = parts[0]
        size = ""
        tag = ""
        
        # Better heuristic for family and size
        if "qwen" in family:
            family = "qw"
            # Find version and size
            for p in parts:
                if p == "3" or p == "4":
                    family += p
                if "b" in p and any(c.isdigit() for c in p):
                    size = p
                    break
        elif "gemma" in family:
            family = "gemma"
            for p in parts:
                if "b" in p and any(c.isdigit() for c in p):
                    size = p
                    break
        elif "gemma4" in family:
            family = "gemma4"
            for p in parts:
                if "b" in p and any(c.isdigit() for c in p):
                    size = p
                    break
                    
        # Find a significant tag (apex, prism, q6, etc.)
        for p in parts:
            if p in ("apex", "prism", "dq", "q6", "k", "pro", "distill"):
                tag = p
                break
        
        if not size and len(parts) > 1:
            size = parts[1]
        if not tag and len(parts) > 2 and parts[2] not in (size, "3", "4", "6"):
            tag = parts[2]
            
        base_id = "-".join(filter(None, [family, size, tag]))
    
    is_qwen = "qwen" in config_name.lower()
    is_gemma4 = "gemma-4" in config_name.lower() or "gemma4" in config_name.lower()
    is_agentic = is_qwen or is_gemma4
    
    # Critical for OpenWeb-UI mapping: Must match the model name/id from llama-server
    # Use MODEL_ALIAS if available as it's what llama-server reports if set
    base_model_id = conf.get("MODEL_ALIAS")
    if not base_model_id:
        model_path = conf.get("MODEL_PATH", "")
        base_model_id = os.path.basename(model_path) if model_path else "unknown"
    
    # Global sampler defaults from .conf
    temp = get_float(conf.get("TEMP"), 0.4)
    min_p = get_float(conf.get("MIN_P"), 0.02)
    top_p = get_float(conf.get("TOP_P"), 0.95)
    top_k = get_float(conf.get("TOP_K"), 50.0)
    repeat_penalty = get_float(conf.get("REPEAT_PENALTY"), 1.1)
    
    # XTC and DRY defaults from .conf
    xtc_prob = get_float(conf.get("XTC_PROBABILITY"), 0.1)
    xtc_threshold = get_float(conf.get("XTC_THRESHOLD"), 0.1)
    dry_multiplier = get_float(conf.get("DRY_MULTIPLIER"), 0.8)
    dry_base = get_float(conf.get("DRY_BASE"), 1.75)
    dry_allowed = get_float(conf.get("DRY_ALLOWED_LENGTH"), 2.0)
    dry_last_n = get_float(conf.get("DRY_PENALTY_LAST_N"), 4096.0)

    # Architectural Differentiation
    if is_qwen:
        identity = "Qwen 3.6 APEX-I Optimized Engine"
        chat_kwargs = { "preserve_thinking": True }
        continuity = "Maintain logical continuity by recursively validating your current plan against previous reasoning traces in this conversation history.\n\n"
        thinking_budget = 2048.0
        samplers = "dry;top_k;top_p;xtc;min_p;temperature"
    elif is_gemma4:
        identity = "Gemma-4 PRISM-Optimized Assistant"
        chat_kwargs = {} # No preserve_thinking for Gemma-4
        continuity = ""  # No recursive logic for Gemma-4 (memoryless thoughts)
        thinking_budget = 1024.0
        samplers = "dry;top_k;top_p;xtc;min_p;temperature"
    else:
        identity = "Standard LLM Engine"
        chat_kwargs = {}
        continuity = ""
        thinking_budget = 1024.0
        samplers = "dry;top_k;top_p;xtc;min_p;temperature"

    # Strict reasoning extraction directive
    reasoning_protocol = "IMPORTANT: You MUST wrap all internal thinking and logical scratchpads inside <think> and </think> tags. This is mandatory for the server to correctly extract and hide your reasoning.\n\n"

    persona_templates = [
        {
            "suffix": "fim",
            "name": "🚀 Ultra Fast | Instant Predictor",
            "params": {
                "system": f"{reasoning_protocol}Predict the next most likely tokens. No conversational filler.",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40.0,
                "min_p": 0.05,
                "repeat_penalty": 1.0,
                "samplers": "top_k;top_p",
                "chat_template_kwargs": { "enable_thinking": False }
            },
            "meta": { "description": "Predictive autocomplete for text and code.", "capabilities": { "file_context": True, "code_interpreter": True } }
        },
        {
            "suffix": "fast",
            "name": "⚡ Fast | Direct Assistant",
            "params": {
                "system": f"You are a {identity}. Provide immediate, high-density facts.\n\n{reasoning_protocol}Keep internal reasoning extremely brief and skip drafting the final response. If your thinking is interrupted, immediately transition to the final answer format.\n\n### ✅ Response Format\n- Use **bold** for key terms.\n- Use double-newlines and keep paragraphs under 3 lines.\n- Use `---` for dividers.",
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "xtc_probability": xtc_prob,
                "xtc_threshold": xtc_threshold,
                "repeat_penalty": repeat_penalty,
                "repeat_last_n": 16.0,
                "samplers": "top_k;top_p",
                "thinking_budget_tokens": 64.0,
                "dry_multiplier": dry_multiplier,
                "dry_base": dry_base,
                "dry_allowed_length": dry_allowed,
                "dry_penalty_last_n": dry_last_n,
                "chat_template_kwargs": { "enable_thinking": True }
            },
            "meta": { "description": "High-speed direct answers.", "capabilities": { "vision": True, "web_search": True, "file_upload": True, "status_updates": True, "builtin_tools": True } }
        },
        {
            "suffix": "coder",
            "name": "💻 Coder | Architect",
            "params": {
                "system": f"You are a Lead Software Engineer using {identity}. " + 
                          f"{continuity}" + 
                          f"{reasoning_protocol}" + 
                          ("Use XML tags (<function=name>) for all tool calls.\n\n" if is_qwen else "") + 
                          "### 🏗️ Engineering\n- Use 🏗️ for architecture, 🛡️ for security, and ⚙️ for logic.\n- Provide production-ready code blocks.\n- Use double-newlines and keep paragraphs under 3 lines.\n- Focus on complexity (🚀) and edge cases (⚠️).",
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": top_k,
                "min_p": min_p,
                "xtc_probability": xtc_prob,
                "xtc_threshold": xtc_threshold,
                "repeat_penalty": 1.1,
                "samplers": "dry;top_k;min_p",
                "thinking_budget_tokens": 750.0,
                "dry_multiplier": dry_multiplier,
                "dry_base": dry_base,
                "dry_allowed_length": dry_allowed,
                "dry_penalty_last_n": dry_last_n,
                "chat_template_kwargs": chat_kwargs
            },
            "meta": { "description": "High-precision code and design.", "capabilities": { "code_interpreter": True, "file_context": True, "builtin_tools": True } }
        },
        {
            "suffix": "pro",
            "name": "🧠 Pro | Deep Thinker",
            "params": {
                "system": f"You are an Elite Analytical Agent using {identity}. " + 
                          f"{continuity}" + 
                          f"{reasoning_protocol}" + 
                          "### 🔍 Analysis\n- Always start sections with contextual emojis.\n- Use **LaTeX** ($inline$) for technical notation.\n- Use 🆚 to contrast and ⚖️ for trade-offs.\n- Use double-newlines and keep paragraphs under 3 lines.",
                "temperature": 0.5,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "xtc_probability": xtc_prob,
                "xtc_threshold": xtc_threshold,
                "repeat_penalty": 1.05,
                "samplers": "dry;top_p;temperature",
                "thinking_budget_tokens": 1536.0,
                "dry_multiplier": dry_multiplier,
                "dry_base": dry_base,
                "dry_allowed_length": dry_allowed,
                "dry_penalty_last_n": dry_last_n,
                "chat_template_kwargs": chat_kwargs
            },
            "meta": { "description": "Multi-perspective reasoning.", "capabilities": { "vision": True, "web_search": True, "builtin_tools": True } }
        },
        {
            "suffix": "research",
            "name": "🔬 Research | Analyst",
            "params": {
                "system": f"You are a Senior Research Analyst using {identity}. {continuity}{reasoning_protocol}Perform exhaustive root-cause stress-testing.\n\n### 📊 Research Report\n- Use Markdown **tables** (📊) for data.\n- Label: ✅ Verified · ⚠️ Speculative · ❓ Uncertain.\n- Use double-newlines and keep paragraphs under 3 lines.\n- Use 📈 for trends.",
                "temperature": 0.35,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "xtc_probability": xtc_prob,
                "xtc_threshold": xtc_threshold,
                "repeat_penalty": 1.08,
                "samplers": samplers,
                "thinking_budget_tokens": 2048.0,
                "dry_multiplier": dry_multiplier,
                "dry_base": dry_base,
                "dry_allowed_length": dry_allowed,
                "dry_penalty_last_n": dry_last_n,
                "chat_template_kwargs": chat_kwargs
            },
            "meta": { "description": "Maximum logical depth.", "capabilities": { "web_search": True, "code_interpreter": True, "builtin_tools": True } }
        },
        {
            "suffix": "creative",
            "name": "🎨 Creative | Stylist",
            "params": {
                "system": f"You are a Master Storyteller using {identity}. " + 
                          ("Perform frame-precise temporal analysis for video and image inputs.\n\n" if is_agentic else "") + 
                          f"{reasoning_protocol}" + 
                          "### ✨ Narrative Style\n- Use evocative, descriptive language.\n- Use 🎭 for character shifts and ✨ for key moments.\n- Maintain rhythm with short paragraphs and double-newlines.",
                "temperature": 0.85,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "xtc_probability": xtc_prob,
                "xtc_threshold": xtc_threshold,
                "repeat_penalty": 1.02,
                "samplers": "dry;top_p",
                "thinking_budget_tokens": 1024.0,
                "dry_multiplier": dry_multiplier,
                "dry_base": dry_base,
                "dry_allowed_length": dry_allowed,
                "dry_penalty_last_n": dry_last_n,
                "chat_template_kwargs": chat_kwargs
            },
            "meta": { "description": "High-temperature creativity.", "capabilities": { "vision": True } }
        },
        {
            "suffix": "math",
            "name": "🔢 Math | Logic Master",
            "params": {
                "system": f"You are a Pure Mathematician using {identity}. {reasoning_protocol}Use extreme precision and step-by-step derivation.\n\n### 📐 Derivation\n- Always use **LaTeX** for all equations.\n- Use $\\therefore$ for 'therefore' and $\\implies$ for 'implies'.\n- Use double-newlines and keep paragraphs under 3 lines.",
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20.0,
                "min_p": 0.01,
                "repeat_penalty": 1.0,
                "samplers": "top_k;temperature",
                "thinking_budget_tokens": 1024.0,
                "chat_template_kwargs": chat_kwargs
            },
            "meta": { "description": "Ultra-precise math.", "capabilities": { "code_interpreter": True } }
        },
        {
            "suffix": "extractor",
            "name": "📋 Extractor | Data Miner",
            "params": {
                "system": f"You are a Data Extraction Engine. {reasoning_protocol}Pull structured info from messy text.\n\n### 📦 Data Output\n- Return ONLY **Markdown Tables**, **JSON**, or **Lists**.\n- No conversational filler.\n- Use double-newlines and keep paragraphs under 3 lines.",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40.0,
                "min_p": 0.05,
                "repeat_penalty": 1.0,
                "samplers": "top_k;top_p",
                "thinking_budget_tokens": 256.0,
                "chat_template_kwargs": chat_kwargs
            },
            "meta": { "description": "Fast data extraction.", "capabilities": { "file_context": True } }
        }
    ]

    personas = []
    for t in persona_templates:
        p = {
            "id": f"{base_id}-{t['suffix']}",
            "user_id": "99e3b1f7-6bbb-45ce-8781-867f9414ce84",
            "base_model_id": base_model_id,
            "name": t["name"],
            "params": t["params"],
            "meta": t["meta"],
            "is_active": True,
            "write_access": True
        }
        p["meta"]["profile_image_url"] = "/static/favicon.png"
        
        # Strict Float Casting (Standard #5)
        for k, v in p["params"].items():
            if isinstance(v, (int, float)) and k != "samplers" and k != "chat_template_kwargs":
                p["params"][k] = float(v)
        
        personas.append(p)

    return personas

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs = [f for f in os.listdir(script_dir) if f.endswith(".conf")]

    if not configs:
        print("No .conf files found.")
        return

    for conf_file in configs:
        print(f"Generating UI profiles for {conf_file}...")
        conf_path = os.path.join(script_dir, conf_file)
        conf_vars = parse_bash_conf(conf_path)
        
        output_file = f"openweb-ui_models_{conf_file.replace('.conf', '')}.json"
        output_path = os.path.join(script_dir, output_file)
        
        personas = generate_personas(conf_vars, conf_file)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(personas, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Created {output_file}")

if __name__ == "__main__":
    main()
