#!/usr/bin/env python3
# debug_ollama.py — chạy file này trước để kiểm tra Ollama output
import requests, json

BASE_URL = "http://localhost:11434"
MODEL    = "qwen3.5:4b"

def test(label, payload):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
    print(f"Status: {r.status_code}")
    data = r.json()
    content = data.get("message", {}).get("content", "")
    thinking = data.get("message", {}).get("thinking", "")
    print(f"Content length : {len(content)}")
    print(f"Thinking length: {len(thinking)}")
    print(f"Content preview:\n{content[:400]}")

# Test 1: không có options gì
test("No options", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with this exact JSON: {\"hello\": \"world\"}"}],
    "stream": False,
})

# Test 2: think=False trong options
test("think=False in options", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with this exact JSON: {\"hello\": \"world\"}"}],
    "stream": False,
    "options": {"temperature": 0.1, "think": False},
})

# Test 3: think=False ở top-level (Ollama >= 0.6.4 syntax)
test("think=False top-level", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with this exact JSON: {\"hello\": \"world\"}"}],
    "stream": False,
    "think": False,
})

# Test 4: think=False top-level + /no-think suffix trong system
test("system /no_think suffix", {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. /no_think"},
        {"role": "user",   "content": "Reply with this exact JSON: {\"hello\": \"world\"}"}
    ],
    "stream": False,
})