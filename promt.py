import json
import os
import urllib.request
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

COUNTRIES: List[Dict[str, Any]] = [
    {"country": "Brazil", "wins": 5, "hosts": 2, "goals": 237},
    {"country": "Germany", "wins": 4, "hosts": 1, "goals": 226},
    {"country": "Italy", "wins": 4, "hosts": 2, "goals": 178},
    {"country": "Argentina", "wins": 3, "hosts": 2, "goals": 161},
    {"country": "France", "wins": 2, "hosts": 2, "goals": 145},
    {"country": "Uruguay", "wins": 2, "hosts": 1, "goals": 110},
    {"country": "England", "wins": 1, "hosts": 1, "goals": 115},
    {"country": "Spain", "wins": 1, "hosts": 1, "goals": 123},
    {"country": "Netherlands", "wins": 0, "hosts": 0, "goals": 120},
    {"country": "Portugal", "wins": 0, "hosts": 0, "goals": 98},
]

MEMORY_FILE = Path(__file__).with_name("agent_memory.json")
# STATE stores session-state keyed by session id, and a top-level 'episodes' list
STATE: Dict[str, Any] = {}


def load_state() -> None:
    global STATE
    if MEMORY_FILE.exists():
        try:
            with MEMORY_FILE.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            STATE = loaded if isinstance(loaded, dict) else {}
        except Exception:
            STATE = {}
    else:
        STATE = {}

    # ensure memory containers
    if isinstance(STATE, dict):
        STATE.setdefault("episodes", [])
        STATE.setdefault("semantic_memories", [])


def save_state() -> None:
    with MEMORY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(STATE, handle, indent=2)


load_state()


# --- Dataset tools ---
def get_country_data() -> List[Dict[str, Any]]:
    return COUNTRIES


def get_country_stats(country_name: str) -> Dict[str, Any]:
    for country in COUNTRIES:
        if country["country"].lower() == country_name.lower():
            return country
    return {"error": f"Country '{country_name}' not found"}


def find_top_country_by_wins() -> Dict[str, Any]:
    return max(COUNTRIES, key=lambda item: item["wins"])


def find_top_country_by_goals() -> Dict[str, Any]:
    return max(COUNTRIES, key=lambda item: item["goals"])


def find_top_country_by_hosts() -> Dict[str, Any]:
    return max(COUNTRIES, key=lambda item: item["hosts"])


def get_average_goals() -> Dict[str, Any]:
    total_goals = sum(item["goals"] for item in COUNTRIES)
    average_goals = total_goals / len(COUNTRIES) if COUNTRIES else 0
    return {"average_goals": average_goals}


def get_countries_with_min_wins(min_wins: int = 1) -> List[Dict[str, Any]]:
    return [item for item in COUNTRIES if item["wins"] >= min_wins]


def compare_countries(country_a: str, country_b: str) -> Dict[str, Any]:
    a = get_country_stats(country_a)
    b = get_country_stats(country_b)
    if "error" in a or "error" in b:
        return {"error": "One or both country names were not found"}
    return {
        "country_a": a,
        "country_b": b,
        "winner_by_wins": a if a["wins"] >= b["wins"] else b,
        "winner_by_goals": a if a["goals"] >= b["goals"] else b,
    }


def get_top_n_countries(n: int) -> List[Dict[str, Any]]:
    return sorted(COUNTRIES, key=lambda item: (-item["wins"], -item["goals"]))[:n]


def summarize_dataset() -> Dict[str, Any]:
    return {
        "total_countries": len(COUNTRIES),
        "top_wins_country": find_top_country_by_wins(),
        "top_goals_country": find_top_country_by_goals(),
        "countries": COUNTRIES,
    }


# --- Episodic memory helpers ---
def add_episodic_memory(content: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    entry = {
        "id": str(uuid4()),
        "session_id": session_id,
        "time": datetime.utcnow().isoformat() + "Z",
        "content": content,
    }
    STATE.setdefault("episodes", []).append(entry)
    save_state()
    return entry


def get_episodic_memory(n: int = 5) -> List[Dict[str, Any]]:
    eps = STATE.get("episodes", [])
    return eps[-n:]


def search_episodic_memory(query: str) -> List[Dict[str, Any]]:
    q = query.lower()
    return [ep for ep in STATE.get("episodes", []) if q in ep.get("content", "").lower()]


# --- Semantic memory helpers ---
def add_semantic_memory(content: str, topic: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    entry = {
        "id": str(uuid4()),
        "session_id": session_id,
        "topic": topic or "general",
        "time": datetime.utcnow().isoformat() + "Z",
        "content": content,
    }
    STATE.setdefault("semantic_memories", []).append(entry)
    save_state()
    return entry


def get_semantic_memory(n: int = 5) -> List[Dict[str, Any]]:
    memories = STATE.get("semantic_memories", [])
    return memories[-n:]


def search_semantic_memory(query: str) -> List[Dict[str, Any]]:
    q = query.lower()
    return [memory for memory in STATE.get("semantic_memories", []) if q in memory.get("content", "").lower() or q in memory.get("topic", "").lower()]


# --- Tool registry ---
TOOLS = {
    "get_country_data": get_country_data,
    "get_country_stats": get_country_stats,
    "find_top_country_by_wins": find_top_country_by_wins,
    "find_top_country_by_goals": find_top_country_by_goals,
    "find_top_country_by_hosts": find_top_country_by_hosts,
    "get_average_goals": get_average_goals,
    "get_countries_with_min_wins": get_countries_with_min_wins,
    "compare_countries": compare_countries,
    "get_top_n_countries": get_top_n_countries,
    "summarize_dataset": summarize_dataset,
    "add_episodic_memory": add_episodic_memory,
    "get_episodic_memory": get_episodic_memory,
    "search_episodic_memory": search_episodic_memory,
    "add_semantic_memory": add_semantic_memory,
    "get_semantic_memory": get_semantic_memory,
    "search_semantic_memory": search_semantic_memory,
}


def get_tool_catalog() -> List[Dict[str, Any]]:
    return [
        {"name": name, "description": tool.__name__.replace("_", " ")}
        for name, tool in TOOLS.items()
    ]


# --- Session and memory management ---
def ensure_session(session_id: str) -> Dict[str, Any]:
    if session_id not in STATE:
        STATE[session_id] = {"history": [], "facts": []}
    return STATE[session_id]


def update_memory(session_state: Dict[str, Any], user_prompt: str, session_id: Optional[str] = None) -> None:
    prompt = user_prompt.strip()
    if not prompt:
        return

    lower_prompt = prompt.lower()
    fact = None
    if "remember" in lower_prompt:
        fact = prompt.split("remember", 1)[1].strip().strip(".?!")
    elif "i prefer" in lower_prompt:
        fact = prompt.strip().rstrip(".?!")
    elif "my name is" in lower_prompt:
        fact = prompt.strip().rstrip(".?!")

    if fact:
        if fact not in session_state.get("facts", []):
            session_state.setdefault("facts", []).append(fact)
        try:
            add_semantic_memory(fact, topic="user_preference", session_id=session_id)
        except Exception:
            pass

    history = session_state.setdefault("history", [])
    if len(history) > 12:
        session_state["history"] = history[-12:]


def build_memory_context(session_state: Dict[str, Any]) -> str:
    facts = session_state.get("facts", [])
    history = session_state.get("history", [])
    memory_lines: List[str] = []

    if facts:
        memory_lines.append("User memory: " + "; ".join(facts[-3:]))

    if history:
        recent_turns = [f"{entry['role']}: {entry['content']}" for entry in history[-4:]]
        memory_lines.append("Recent context: " + " | ".join(recent_turns))

    # Include recent episodic memory (global)
    episodes = STATE.get("episodes", [])
    if episodes:
        last_eps = episodes[-3:]
        ep_lines = [f"{ep.get('time','')}: {ep.get('content','')}" for ep in last_eps]
        memory_lines.append("Recent episodes: " + " | ".join(ep_lines))

    # Include recent semantic memory (global)
    semantic_memories = STATE.get("semantic_memories", [])
    if semantic_memories:
        last_semantics = semantic_memories[-3:]
        sem_lines = [f"{memory.get('topic','general')}: {memory.get('content','')}" for memory in last_semantics]
        memory_lines.append("Semantic memory: " + " | ".join(sem_lines))

    return " ".join(memory_lines)


def trim_history(session_state: Dict[str, Any], max_entries: int = 12) -> None:
    history = session_state.setdefault("history", [])
    if len(history) > max_entries:
        session_state["history"] = history[-max_entries:]


def make_system_prompt(session_state: Optional[Dict[str, Any]] = None) -> str:
    tool_catalog = get_tool_catalog()
    memory_context = ""
    if session_state:
        memory_context = build_memory_context(session_state)
        if memory_context:
            memory_context = f" Session memory: {memory_context}."

    if tool_catalog:
        tool_names = ", ".join(tool["name"] for tool in tool_catalog)
        descriptions = "; ".join(f"{tool['name']}: {tool['description']}" for tool in tool_catalog)
        tools_part = f"Available tools: {tool_names}. Tool descriptions: {descriptions}."
        use_tools_sentence = "Use the available tools whenever the user asks about countries, winners, goals, or comparisons. "
    else:
        tools_part = "No external tools are available."
        use_tools_sentence = "Answer directly using only internal logic; do not attempt to call external tools. "

    return (
        "You are an agentic FIFA World Cup assistant. "
        f"{use_tools_sentence}"
        "Keep short-term state across this chat and remember simple user preferences when they are stated. "
        f"{tools_part}"
        f"{memory_context}"
        " Return valid JSON only with one of these shapes: "
        '{"tool": "tool_name", "arguments": {...}} or {"final_answer": "..."}.'
    )


def call_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception:
        return {}


def fallback_plan(user_prompt: str) -> Dict[str, Any]:
    prompt = user_prompt.lower()

    if "compare" in prompt:
        parts = [p.strip() for p in prompt.replace("compare", "").split("and") if p.strip()]
        if len(parts) >= 2:
            return {"tool": "compare_countries", "arguments": {"country_a": parts[0], "country_b": parts[1]}}

    if ("top" in prompt or "most" in prompt) and "wins" in prompt:
        return {"tool": "find_top_country_by_wins", "arguments": {}}

    if ("top" in prompt or "most" in prompt) and "goals" in prompt:
        return {"tool": "find_top_country_by_goals", "arguments": {}}

    if ("top" in prompt or "most" in prompt) and "hosts" in prompt:
        return {"tool": "find_top_country_by_hosts", "arguments": {}}

    if "average" in prompt and "goals" in prompt:
        return {"tool": "get_average_goals", "arguments": {}}

    if "countries" in prompt and "wins" in prompt and "at least" in prompt:
        try:
            min_wins = int([token for token in prompt.split() if token.isdigit()][0])
        except (IndexError, ValueError):
            min_wins = 1
        return {"tool": "get_countries_with_min_wins", "arguments": {"min_wins": min_wins}}

    if "top" in prompt and "countries" in prompt:
        try:
            n = int(prompt.split()[-1])
        except ValueError:
            n = 3
        return {"tool": "get_top_n_countries", "arguments": {"n": n}}

    if "summary" in prompt or "summarize" in prompt:
        return {"tool": "summarize_dataset", "arguments": {}}

    if "stats" in prompt or "data" in prompt:
        country = prompt.replace("stats", "").replace("data", "").strip()
        if country:
            return {"tool": "get_country_stats", "arguments": {"country_name": country}}

    return {"final_answer": "I can help you compare teams, find the leader in wins or goals, or summarize the dataset. Try asking: 'Which country has the most wins?'"}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    tool = TOOLS.get(tool_name)
    if not tool:
        return {"error": f"Unknown tool: {tool_name}"}

    if tool_name == "get_country_stats":
        return tool(arguments.get("country_name", ""))
    if tool_name == "compare_countries":
        return tool(arguments.get("country_a", ""), arguments.get("country_b", ""))
    if tool_name == "get_top_n_countries":
        return tool(arguments.get("n", 3))
    if tool_name == "add_episodic_memory":
        return tool(arguments.get("content", ""), arguments.get("session_id"))
    if tool_name == "get_episodic_memory":
        return tool(arguments.get("n", 5))
    if tool_name == "search_episodic_memory":
        return tool(arguments.get("query", ""))
    if tool_name == "add_semantic_memory":
        return tool(arguments.get("content", ""), arguments.get("topic"), arguments.get("session_id"))
    if tool_name == "get_semantic_memory":
        return tool(arguments.get("n", 5))
    if tool_name == "search_semantic_memory":
        return tool(arguments.get("query", ""))
    return tool()


def build_agent_state(user_prompt: str, plan: Dict[str, Any], tool_name: Optional[str], session_state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (user_prompt or "").strip()
    lower_prompt = prompt.lower()

    if "remember" in lower_prompt or "i prefer" in lower_prompt or "my name is" in lower_prompt:
        plan_summary = "Store the user preference or fact in memory."
    elif tool_name:
        plan_summary = f"Use the {tool_name} tool to answer the request."
    else:
        plan_summary = "Answer directly without tool use."

    facts = session_state.get("facts", [])
    return {
        "plan": plan_summary,
        "tool_used": tool_name,
        "memory_facts": facts[-3:],
        "episodic_count": len(STATE.get("episodes", [])),
        "semantic_count": len(STATE.get("semantic_memories", [])),
    }


def format_result(tool_name: str, result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        return result["error"]
    if tool_name == "find_top_country_by_wins":
        return f"{result['country']} has the most wins with {result['wins']} wins."
    if tool_name == "find_top_country_by_goals":
        return f"{result['country']} has the most goals with {result['goals']} goals."
    if tool_name == "get_country_stats":
        return f"{result['country']} has {result['wins']} wins, {result['hosts']} hosts, and {result['goals']} goals."
    if tool_name == "compare_countries":
        return (
            f"{result['country_a']['country']} vs {result['country_b']['country']}: "
            f"{result['country_a']['country']} has {result['country_a']['wins']} wins and {result['country_a']['goals']} goals; "
            f"{result['country_b']['country']} has {result['country_b']['wins']} wins and {result['country_b']['goals']} goals. "
            f"Winner by wins: {result['winner_by_wins']['country']}. "
            f"Winner by goals: {result['winner_by_goals']['country']}."
        )
    if tool_name == "find_top_country_by_hosts":
        return f"{result['country']} has hosted the World Cup the most times with {result['hosts']} hosts."
    if tool_name == "get_average_goals":
        return f"The average goal total in the dataset is {result['average_goals']:.1f}."
    if tool_name == "get_countries_with_min_wins":
        if not result:
            return "No countries in the dataset meet that minimum win threshold."
        lines = [f"{item['country']} ({item['wins']} wins)" for item in result]
        return "Countries with at least that many wins: " + ", ".join(lines)
    if tool_name == "get_top_n_countries":
        lines = [f"{idx + 1}. {item['country']} ({item['wins']} wins, {item['goals']} goals)" for idx, item in enumerate(result)]
        return "\n".join(lines)
    if tool_name == "summarize_dataset":
        return (
            f"There are {result['total_countries']} countries in the dataset. "
            f"The top wins country is {result['top_wins_country']['country']} and the top goals country is {result['top_goals_country']['country']}."
        )
    return json.dumps(result, indent=2)


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            with open("index.html", "r", encoding="utf-8") as handle:
                content = handle.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (ConnectionAbortedError, BrokenPipeError):
                return
        elif parsed.path == "/episodes":
            # optional ?n= to limit most-recent episodes
            qs = parse_qs(parsed.query)
            try:
                n = int(qs.get("n", ["10"])[0])
            except Exception:
                n = 10
            eps = get_episodic_memory(n)
            payload = json.dumps({"episodes": eps}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (ConnectionAbortedError, BrokenPipeError):
                return
        elif parsed.path == "/episodes/search":
            qs = parse_qs(parsed.query)
            q = qs.get("q", [""])[0]
            results = search_episodic_memory(q) if q else []
            payload = json.dumps({"results": results}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (ConnectionAbortedError, BrokenPipeError):
                return
        elif parsed.path == "/memory/timeline":
            timeline = []
            for memory in STATE.get("semantic_memories", [])[-8:]:
                timeline.append({"type": "semantic", "content": memory.get("content", ""), "topic": memory.get("topic", "general"), "time": memory.get("time", "")})
            for episode in STATE.get("episodes", [])[-8:]:
                timeline.append({"type": "episodic", "content": episode.get("content", ""), "topic": "episode", "time": episode.get("time", "")})
            timeline.sort(key=lambda item: item.get("time", ""), reverse=True)
            payload = json.dumps({"timeline": timeline[:8]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (ConnectionAbortedError, BrokenPipeError):
                return
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/ask":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            prompt = payload.get("prompt", "")
            session_id = payload.get("session_id")

            result = handle_prompt(prompt, session_id)
            reply = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            try:
                self.wfile.write(reply)
            except (ConnectionAbortedError, BrokenPipeError):
                return
            return

        if self.path == "/episodes":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            content = payload.get("content", "")
            session_id = payload.get("session_id")
            if not content:
                self.send_response(400)
                self.end_headers()
                return
            entry = add_episodic_memory(content, session_id)
            reply = json.dumps(entry).encode("utf-8")
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            try:
                self.wfile.write(reply)
            except (ConnectionAbortedError, BrokenPipeError):
                return
            return

        # unknown POST path
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


# ---------------------------------------------------------------------------
# State graph engine
#
# A minimal, dependency-free state graph: a set of named nodes (plain
# functions that take the shared state dict and return it, mutated), wired
# together with static edges (node -> node) and conditional edges
# (node -> router function -> node name). `compile()` returns a Graph whose
# `invoke(state)` walks nodes until it reaches END, so the whole request
# pipeline is an explicit, inspectable graph instead of one long function.
#
#                        ┌─────────────┐
#                        │    START    │
#                        └──────┬──────┘
#                               ↓
#                        ┌─────────────┐
#                        │   INTAKE    │
#                        │update_memory│
#                        │extract facts│
#                        └──────┬──────┘
#                               ↓
#                        ┌─────────────┐
#                        │BUILD_CONTEXT│
#                        │system prompt│
#                        │ + memory    │
#                        └──────┬──────┘
#                               ↓
#                        ┌─────────────┐
#                        │  PLAN_LLM   │
#                        │ call OpenAI │
#                        └──────┬──────┘
#                               │
#                        Plan returned?
#                          /        \
#                        NO         YES
#                         ↓           │
#                  ┌─────────────┐    │
#                  │PLAN_FALLBACK│    │
#                  │keyword match│    │
#                  └──────┬──────┘    │
#                         └─────┬─────┘
#                               ↓
#                     Final answer or tool call?
#                        /                  \
#                 FINAL_ANSWER             TOOL
#                       │                    ↓
#                       │             ┌─────────────┐
#                       │             │EXECUTE_TOOL │
#                       │             └──────┬──────┘
#                       │                    │
#                       │             Tools available:
#                       │             • find_top_country_by_wins()
#                       │             • find_top_country_by_goals()
#                       │             • compare_countries()
#                       │             • summarize_dataset()
#                       │             • get_country_stats()
#                       │                    │
#                       └─────────┬──────────┘
#                                 ↓
#                          ┌─────────────┐
#                          │FORMAT_ANSWER│
#                          │optional LLM │
#                          │  rewrite    │
#                          └──────┬──────┘
#                                 ↓
#                          ┌─────────────┐
#                          │FINALIZE_HIST│
#                          │trim + build │
#                          │agent_state  │
#                          └──────┬──────┘
#                                 ↓
#                          ┌─────────────┐
#                          │RECORD_MEMORY│
#                          └──────┬──────┘
#                                 │
#                          Tools available:
#                          • add_episodic_memory()
#                          • add_semantic_memory()
#                                 │
#                                 ↓
#                          ┌─────────────┐
#                          │SAVE_AND_RESP│
#                          │save_state() │
#                          └──────┬──────┘
#                                 ↓
#                               END
#
# Note: unlike a support-ticket-style graph, there is no human-approval /
# high-risk escalation branch here — every turn runs straight through to
# SAVE_AND_RESPOND. If you want a REVIEW/ESCALATE gate (e.g. before a tool
# call that mutates data), add a node + conditional edge the same way
# plan_llm -> plan_fallback is wired below.
# ---------------------------------------------------------------------------

END = "__END__"


class StateGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, str] = {}
        self._conditional_edges: Dict[str, Any] = {}
        self._entry_point: Optional[str] = None

    def add_node(self, name: str, fn: Any) -> "StateGraph":
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry_point = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> "StateGraph":
        self._edges[from_node] = to_node
        return self

    def add_conditional_edges(self, from_node: str, router: Any) -> "StateGraph":
        # router(state) -> next node name (or END)
        self._conditional_edges[from_node] = router
        return self

    def compile(self) -> "CompiledGraph":
        if self._entry_point is None:
            raise ValueError("StateGraph has no entry point set")
        return CompiledGraph(self._nodes, self._edges, self._conditional_edges, self._entry_point)


class CompiledGraph:
    def __init__(self, nodes, edges, conditional_edges, entry_point) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point

    def invoke(self, state: Dict[str, Any], max_steps: int = 50) -> Dict[str, Any]:
        current = self._entry_point
        trace: List[str] = []
        steps = 0
        while current != END:
            if steps >= max_steps:
                raise RuntimeError(f"State graph exceeded {max_steps} steps (possible cycle): {trace}")
            if current not in self._nodes:
                raise ValueError(f"Unknown node '{current}'")
            trace.append(current)
            state = self._nodes[current](state) or state
            if current in self._conditional_edges:
                current = self._conditional_edges[current](state)
            elif current in self._edges:
                current = self._edges[current]
            else:
                current = END
            steps += 1
        state["_graph_trace"] = trace
        return state


# ---------------------------------------------------------------------------
# Multi-agent implementation — supervisor pattern.
#
# Instead of one planner deciding both "which tool" and "how to answer", a
# Supervisor agent decides, turn by turn, which specialist should act next:
# the Dataset agent (FIFA World Cup stats) or the Memory agent (storing and
# recalling facts about the user). Each specialist owns its own scoped
# system prompt and its own tool set. The supervisor loops — after each
# dispatch it re-evaluates whether more work is needed — until it decides
# the turn is done, at which point a final answer is assembled from
# whatever the specialists produced.
#
#   update_memory -> supervisor_decide -+-> dispatch_dataset -+
#                           ^            |                    |
#                           |            +-> dispatch_memory -+
#                           +--------------------(loop back)--+
#                           |
#                           +-> finalize_answer -> finalize_history
#                               -> record_memory -> save_and_respond
# ---------------------------------------------------------------------------

MAX_SUPERVISOR_HOPS = 4

DATASET_TOOL_NAMES = {
    "get_country_data", "get_country_stats", "find_top_country_by_wins",
    "find_top_country_by_goals", "find_top_country_by_hosts", "get_average_goals",
    "get_countries_with_min_wins", "compare_countries", "get_top_n_countries",
    "summarize_dataset",
}
MEMORY_TOOL_NAMES = {
    "add_episodic_memory", "get_episodic_memory", "search_episodic_memory",
    "add_semantic_memory", "get_semantic_memory", "search_semantic_memory",
}

MEMORY_STORE_KEYWORDS = ("remember", "i prefer", "my name is")
MEMORY_RECALL_KEYWORDS = ("recall", "what do you know", "what did i tell", "remind me")
DATASET_KEYWORDS = ("win", "goal", "host", "compare", "summar", "top", "most", "average", "stats", "data")


def node_update_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    session_state = state["session_state"]
    session_state.setdefault("history", []).append({"role": "user", "content": state["user_prompt"]})
    return state


# --- Dataset agent (scoped to FIFA World Cup tools only) -------------------

def dataset_agent_system_prompt() -> str:
    tools = [t for t in get_tool_catalog() if t["name"] in DATASET_TOOL_NAMES]
    tool_lines = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    return (
        "You are the Dataset agent. You answer questions about FIFA World Cup "
        "winners using ONLY the tools below. You have no memory of past "
        "conversations and no knowledge outside this dataset. Respond with "
        'strict JSON: either {"tool": "<name>", "arguments": {...}} or '
        '{"final_answer": "..."}.\n\n'
        f"Tools:\n{tool_lines}"
    )


def node_dispatch_dataset(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    plan = call_llm(dataset_agent_system_prompt(), user_prompt) or fallback_plan(user_prompt)

    if "final_answer" in plan:
        text = plan["final_answer"]
        tool_name = None
    else:
        tool_name = plan.get("tool")
        result = execute_tool(tool_name, plan.get("arguments", {}))
        text = format_result(tool_name, result)

    state.setdefault("agent_outputs", []).append({"agent": "dataset", "text": text})
    if tool_name:
        state["dataset_tool_used"] = tool_name
    state.setdefault("dispatched", set()).add("dataset")
    return state


# --- Memory agent (scoped to episodic/semantic memory tools only) ----------

def memory_agent_system_prompt() -> str:
    tools = [t for t in get_tool_catalog() if t["name"] in MEMORY_TOOL_NAMES]
    tool_lines = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    return (
        "You are the Memory agent. You store and recall facts the user "
        "shares about themselves, using ONLY the tools below. You know "
        "nothing about the FIFA World Cup dataset.\n\n"
        f"Tools:\n{tool_lines}"
    )


def node_dispatch_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    lower_prompt = user_prompt.lower()
    session_state = state["session_state"]

    if any(keyword in lower_prompt for keyword in MEMORY_STORE_KEYWORDS):
        before = len(session_state.get("facts", []))
        update_memory(session_state, user_prompt, state["session_id"])
        stored = len(session_state.get("facts", [])) > before
        text = "Got it — I'll remember that." if stored else "I heard you, but couldn't pull out a specific fact to store."
        state["memory_action"] = "store"
    elif any(keyword in lower_prompt for keyword in MEMORY_RECALL_KEYWORDS):
        hits = search_semantic_memory(user_prompt) + search_episodic_memory(user_prompt)
        text = ("Here's what I recall: " + "; ".join(h.get("content", "") for h in hits[:3])) if hits \
            else "I don't have anything stored about that yet."
        state["memory_action"] = "recall"
    else:
        text = "No memory action was needed for that."
        state["memory_action"] = "none"

    state.setdefault("agent_outputs", []).append({"agent": "memory", "text": text})
    state.setdefault("dispatched", set()).add("memory")
    return state


# --- Supervisor agent --------------------------------------------------------

def supervisor_system_prompt(dispatched: set) -> str:
    return (
        "You are the Supervisor agent coordinating two specialists: 'dataset' "
        "(FIFA World Cup statistics) and 'memory' (storing/recalling facts "
        f"about the user). Already dispatched this turn: {sorted(dispatched) or ['none']}. "
        'Decide the next step. Respond with strict JSON: {"route": "dataset"}, '
        '{"route": "memory"}, or {"route": "final"} once no more specialists are needed.'
    )


def supervisor_fallback_route(user_prompt: str, dispatched: set) -> str:
    lower_prompt = user_prompt.lower()
    wants_memory = any(k in lower_prompt for k in MEMORY_STORE_KEYWORDS + MEMORY_RECALL_KEYWORDS)
    wants_dataset = any(k in lower_prompt for k in DATASET_KEYWORDS)

    if wants_memory and "memory" not in dispatched:
        return "memory"
    if wants_dataset and "dataset" not in dispatched:
        return "dataset"
    return "final"


def node_supervisor_decide(state: Dict[str, Any]) -> Dict[str, Any]:
    state["hops"] = state.get("hops", 0) + 1
    dispatched = state.setdefault("dispatched", set())

    if state["hops"] > MAX_SUPERVISOR_HOPS:
        state["route"] = "final"
        state.setdefault("route_history", []).append("final (max hops)")
        return state

    plan = call_llm(supervisor_system_prompt(dispatched), state["user_prompt"])
    route = plan.get("route") if isinstance(plan, dict) else None
    if route not in {"dataset", "memory", "final"}:
        route = supervisor_fallback_route(state["user_prompt"], dispatched)

    state["route"] = route
    state.setdefault("route_history", []).append(route)
    return state


def route_from_supervisor(state: Dict[str, Any]) -> str:
    route = state.get("route")
    if route == "dataset":
        return "dispatch_dataset"
    if route == "memory":
        return "dispatch_memory"
    return "finalize_answer"


# --- Finalize, memory recording, and response assembly ---------------------

def node_finalize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    outputs = state.get("agent_outputs", [])
    if not outputs:
        state["answer"] = (
            "I can help you compare teams, find the leader in wins or goals, "
            "summarize the dataset, or remember facts about you. Try asking: "
            "'Which country has the most wins?'"
        )
    else:
        state["answer"] = " ".join(o["text"] for o in outputs)
    return state


def node_finalize_history(state: Dict[str, Any]) -> Dict[str, Any]:
    session_state = state["session_state"]
    session_state.setdefault("history", []).append({"role": "assistant", "content": state["answer"]})
    trim_history(session_state)

    dispatched = state.get("dispatched", set())
    if dispatched:
        plan_summary = f"Supervisor dispatched: {', '.join(sorted(dispatched))} agent(s)."
    else:
        plan_summary = "Supervisor answered directly without dispatching a specialist."

    facts = session_state.get("facts", [])
    state["agent_state"] = {
        "plan": plan_summary,
        "tool_used": state.get("dataset_tool_used"),
        "route_history": state.get("route_history", []),
        "memory_facts": facts[-3:],
        "episodic_count": len(STATE.get("episodes", [])),
        "semantic_count": len(STATE.get("semantic_memories", [])),
    }
    return state


def node_record_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt, answer, session_id = state["user_prompt"], state["answer"], state["session_id"]
    try:
        if state.get("dataset_tool_used") in DATASET_TOOL_NAMES:
            add_semantic_memory(f"User asked: {user_prompt} -> Answer: {answer}", topic="dataset_fact", session_id=session_id)
    except Exception:
        pass
    try:
        add_episodic_memory(f"User: {user_prompt} -> Assistant: {answer}", session_id)
    except Exception:
        pass
    return state


def node_save_and_respond(state: Dict[str, Any]) -> Dict[str, Any]:
    save_state()
    session_state = state["session_state"]
    state["response"] = {
        "reply": state["answer"],
        "session_id": state["session_id"],
        "memory": session_state.get("facts", []),
        "state": {"history": session_state.get("history", [])},
        "tools": get_tool_catalog(),
        "agent_state": state["agent_state"],
    }
    return state


def build_agent_graph() -> CompiledGraph:
    graph = StateGraph()
    graph.add_node("update_memory", node_update_memory)
    graph.add_node("supervisor_decide", node_supervisor_decide)
    graph.add_node("dispatch_dataset", node_dispatch_dataset)
    graph.add_node("dispatch_memory", node_dispatch_memory)
    graph.add_node("finalize_answer", node_finalize_answer)
    graph.add_node("finalize_history", node_finalize_history)
    graph.add_node("record_memory", node_record_memory)
    graph.add_node("save_and_respond", node_save_and_respond)

    graph.set_entry_point("update_memory")
    graph.add_edge("update_memory", "supervisor_decide")
    graph.add_conditional_edges("supervisor_decide", route_from_supervisor)
    graph.add_edge("dispatch_dataset", "supervisor_decide")
    graph.add_edge("dispatch_memory", "supervisor_decide")
    graph.add_edge("finalize_answer", "finalize_history")
    graph.add_edge("finalize_history", "record_memory")
    graph.add_edge("record_memory", "save_and_respond")
    graph.add_edge("save_and_respond", END)

    return graph.compile()


AGENT_GRAPH = build_agent_graph()


def handle_prompt(user_prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    resolved_session_id = session_id or str(uuid4())
    session_state = ensure_session(resolved_session_id)

    initial_state: Dict[str, Any] = {
        "user_prompt": user_prompt,
        "session_id": resolved_session_id,
        "session_state": session_state,
    }
    final_state = AGENT_GRAPH.invoke(initial_state)
    return final_state["response"]


def run_agent() -> None:
    print("Agentic FIFA World Cup Assistant")
    print("Open http://localhost:8000 in your browser")
    server = HTTPServer(("127.0.0.1", 8000), AgentHandler)
    server.serve_forever()


if __name__ == "__main__":
    run_agent()
