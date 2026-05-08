#!/bin/bash
# Comprehensive tool call test for Qwen3.6-35B-A3B
# Usage: ./test_tool_calls.sh [port] 
# Default port: 8085

PORT="${1:-8085}"
BASE="http://localhost:$PORT/v1/chat/completions"
PASS=0
FAIL=0

check() {
    local name="$1" expected="$2" actual="$3"
    if echo "$actual" | grep -q "$expected"; then
        echo "  ✅ $name"
        ((PASS++))
    else
        echo "  ❌ $name (expected '$expected', got: $actual)"
        ((FAIL++))
    fi
}

echo "=== Tool Call Tests ==="
echo "Server: http://localhost:$PORT"
curl -s "http://localhost:$PORT/health" | grep -q ok || { echo "❌ Server not running"; exit 1; }
echo ""

# Test 1: Basic tool call
echo "Test 1: Basic tool call trigger"
R=$(curl -s --max-time 60 "$BASE" -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is the weather in London?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
  "max_tokens": 256, "temperature": 0
}')
TCALLS=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls');print(tc[0]['function']['name'] if tc else 'none')" 2>/dev/null)
FINISH=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['finish_reason'])" 2>/dev/null)
check "triggers tool call" "get_weather" "$TCALLS"
check "finish_reason=tool_calls" "tool_calls" "$FINISH"

# Test 2: Tool response → final answer (no loop)
echo ""
echo "Test 2: Tool response produces final answer (no loop)"
R=$(curl -s --max-time 60 "$BASE" -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "user", "content": "What is the weather in London?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"London\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 15, \"condition\": \"cloudy\"}"}
  ],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
  "max_tokens": 256, "temperature": 0
}')
CONTENT=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message'].get('content',''))" 2>/dev/null)
TCALLS=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls');print('has_calls' if tc else 'no_calls')" 2>/dev/null)
FINISH=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['finish_reason'])" 2>/dev/null)
check "has content" "London\|15\|cloudy" "$CONTENT"
check "no further tool calls" "no_calls" "$TCALLS"
check "finish_reason=stop" "stop" "$FINISH"

# Test 3: Correct tool selection
echo ""
echo "Test 3: Selects correct tool from multiple"
R=$(curl -s --max-time 60 "$BASE" -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Search for latest AI news"}],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
  ],
  "max_tokens": 256, "temperature": 0
}')
TCALLS=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls');print(tc[0]['function']['name'] if tc else 'none')" 2>/dev/null)
check "picks search_web not get_weather" "search_web" "$TCALLS"

# Test 4: No tool call when not needed
echo ""
echo "Test 4: No tool call for simple question"
R=$(curl -s --max-time 60 "$BASE" -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
  "max_tokens": 256, "temperature": 0
}')
CONTENT=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message'].get('content',''))" 2>/dev/null)
TCALLS=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls');print('has_calls' if tc else 'no_calls')" 2>/dev/null)
check "answers directly with 4" "4" "$CONTENT"
check "no tool call" "no_calls" "$TCALLS"

# Test 5: Multi-step tool use
echo ""
echo "Test 5: Multi-step (2 cities)"
R=$(curl -s --max-time 60 "$BASE" -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Compare weather in London and Paris"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
  "max_tokens": 256, "temperature": 0
}')
TCALLS=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls',[]);print(len(tc))" 2>/dev/null)
check "calls tool (1 or 2 calls)" "[12]" "$TCALLS"

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

# Test 6: Nested quote escaping (stress test)
echo ""
echo "Test 6: Nested bash quote escaping (3 rounds)"
TOOLS_T='[{"type":"function","function":{"name":"terminal","description":"Execute a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'

R1=$(curl -s --max-time 120 "$BASE" -H "Content-Type: application/json" -d "{
  \"messages\":[{\"role\":\"user\",\"content\":\"Run: bash ~/script/proxy.sh \\\"web read --url \\\\\\\"https://example.com/\\\\\\\"\\\"\"}],
  \"tools\":$TOOLS_T, \"max_tokens\":512, \"temperature\":1.0, \"top_p\":0.95, \"top_k\":20, \"presence_penalty\":1.5
}")
CMD1=$(echo "$R1" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls',[]);print(tc[0]['function']['arguments'] if tc else 'no_call')" 2>/dev/null)

R2=$(curl -s --max-time 120 "$BASE" -H "Content-Type: application/json" -d "{
  \"messages\":[
    {\"role\":\"user\",\"content\":\"Run: bash ~/script/proxy.sh \\\"web read --url \\\\\\\"https://example.com/\\\\\\\"\\\"\"},
    {\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"terminal\",\"arguments\":$CMD1}}]},
    {\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"{\\\"output\\\":\\\"bash: unexpected EOF\\\\nSTATUS:FAILURE\\\",\\\"exit_code\\\":1}\"}
  ],
  \"tools\":$TOOLS_T, \"max_tokens\":512, \"temperature\":1.0, \"top_p\":0.95, \"top_k\":20, \"presence_penalty\":1.5
}")
CMD2=$(echo "$R2" | python3 -c "import sys,json;d=json.load(sys.stdin);tc=d['choices'][0]['message'].get('tool_calls',[]);print(tc[0]['function']['arguments'] if tc else 'gave_up')" 2>/dev/null)

if [ "$CMD1" = "$CMD2" ]; then
    echo "  ⚠️  identical commands (potential loop)"
    ((FAIL++))
else
    echo "  ✅ commands differ across retries (no loop)"
    ((PASS++))
fi
echo "    R1: $CMD1"
echo "    R2: $CMD2"

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
