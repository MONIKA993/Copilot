import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

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
STATE: Dict[str, Dict[str, Any]] = {}


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


def save_state() -> None:
    with MEMORY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(STATE, handle, indent=2)


load_state()


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
}


def ensure_session(session_id: str) -> Dict[str, Any]:
    if session_id not in STATE:
        STATE[session_id] = {"history": [], "facts": []}
    return STATE[session_id]


def update_memory(session_state: Dict[str, Any], user_prompt: str) -> None:
    prompt = user_prompt.strip()
    if not prompt:
        return

    lower_prompt = prompt.lower()
    if "remember" in lower_prompt:
        fact = prompt.split("remember", 1)[1].strip().strip(".?!")
        if fact and fact not in session_state.get("facts", []):
            session_state.setdefault("facts", []).append(fact)
    elif "i prefer" in lower_prompt:
        fact = prompt.strip().rstrip(".?!")
        if fact and fact not in session_state.get("facts", []):
            session_state.setdefault("facts", []).append(fact)
    elif "my name is" in lower_prompt:
        fact = prompt.strip().rstrip(".?!")
        if fact and fact not in session_state.get("facts", []):
            session_state.setdefault("facts", []).append(fact)

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

    return " ".join(memory_lines)


def make_system_prompt(session_state: Optional[Dict[str, Any]] = None) -> str:
    tool_names = ", ".join(TOOLS.keys())
    memory_context = ""
    if session_state:
        memory_context = build_memory_context(session_state)
        if memory_context:
            memory_context = f" Session memory: {memory_context}."

    return (
        "You are an agentic FIFA World Cup assistant. "
        "Use the available tools whenever the user asks about countries, winners, goals, or comparisons. "
        "Keep short-term state across this chat and remember simple user preferences when they are stated. "
        f"Available tools: {tool_names}."
        f"{memory_context}"
        "Return valid JSON only with one of these shapes: "
        '{"tool": "tool_name", "arguments": {...}} or {"final_answer": "..."}.'
    )


def call_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    payload = {
        "model": "gpt-4o-mini",
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
    return tool()


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
        if self.path == "/":
            with open("index.html", "r", encoding="utf-8") as handle:
                content = handle.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/ask":
            self.send_response(404)
            self.end_headers()
            return

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
        self.wfile.write(reply)

    def log_message(self, format, *args):
        return


def handle_prompt(user_prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    resolved_session_id = session_id or str(uuid4())
    session_state = ensure_session(resolved_session_id)
    session_state.setdefault("history", []).append({"role": "user", "content": user_prompt})
    update_memory(session_state, user_prompt)

    system_prompt = make_system_prompt(session_state)
    plan = call_llm(system_prompt, user_prompt)
    if not plan:
        plan = fallback_plan(user_prompt)

    if "final_answer" in plan:
        answer = plan["final_answer"]
    else:
        tool_name = plan.get("tool")
        arguments = plan.get("arguments", {})
        result = execute_tool(tool_name, arguments)
        answer = format_result(tool_name, result)

        if os.getenv("OPENAI_API_KEY"):
            final_prompt = (
                f"The user asked: {user_prompt}\n"
                f"The tool used was: {tool_name}\n"
                f"The tool result was: {json.dumps(result, ensure_ascii=False)}\n"
                "Write a short, natural-language answer."
            )
            llm_answer = call_llm(system_prompt, final_prompt)
            if llm_answer and "final_answer" in llm_answer:
                answer = llm_answer["final_answer"]

    session_state.setdefault("history", []).append({"role": "assistant", "content": answer})
    save_state()
    return {"reply": answer, "session_id": resolved_session_id}


def run_agent() -> None:
    print("Agentic FIFA World Cup Assistant")
    print("Open http://localhost:8000 in your browser")
    server = HTTPServer(("127.0.0.1", 8000), AgentHandler)
    server.serve_forever()


if __name__ == "__main__":
    run_agent()
