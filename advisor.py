import requests
import config


def generate_tips(days_out, appliances, weather_daily_summary, background_load_kW):
    """
    Rule-based fallback advisor -- used if no Groq API key is set, or if
    the API call fails for any reason (network issue, rate limit, etc).
    This keeps the Advisor page working even without internet access to
    Groq, or before a key is configured.
    """
    tips = []

    poor_days = [d for d in days_out if d["summary"]["solar_quality"] == "poor"]
    good_days = [d for d in days_out if d["summary"]["solar_quality"] == "good"]

    if len(poor_days) == len(days_out):
        tips.append(
            "Every day in this outlook shows low solar output. Lean on the grid for "
            "power-hungry appliances and save solar for your smallest loads instead "
            "of trying to run everything off panels."
        )
    elif good_days:
        best = good_days[0]
        tips.append(
            f"{best['label']} ({best['date']}) looks like the best solar day in this "
            f"window -- shift your most power-hungry appliance there if you can."
        )

    if appliances:
        heaviest = max(appliances, key=lambda a: a["typical_power_kW"])
        if heaviest["typical_power_kW"] > 2.0:
            tips.append(
                f"'{heaviest['name']}' draws {heaviest['typical_power_kW']}kW, which is "
                f"large relative to typical home solar surplus -- it may rarely show as "
                f"fully covered. That's expected, not a bug."
            )

    tips.append(
        f"Your background load is set to {background_load_kW}kW -- this is a fixed "
        f"assumption, adjust it on Live Mode if it doesn't match your real household."
    )

    rainy_days = [d for d in weather_daily_summary if d["total_precipitation_mm"] > 2]
    if rainy_days:
        names = ", ".join(f"{d['label']} ({d['date']})" for d in rainy_days)
        tips.append(f"Rain is expected on {names} -- charge devices earlier in the day.")

    if not tips:
        tips.append("Conditions look reasonably steady. Check each day's summary for the best windows.")

    return tips[:6]


def _build_context_summary(days_out, appliances, weather_daily_summary, background_load_kW):
    """
    Compresses the app's real computed data into a compact text block for
    the AI prompt -- keeps token usage low and keeps the model grounded in
    actual numbers instead of guessing.
    """
    lines = [f"Background load assumption: {background_load_kW} kW (fixed, not measured)."]

    lines.append("\nAppliances:")
    if appliances:
        for a in appliances:
            lines.append(f"- {a['name']}: {a['typical_power_kW']}kW, runs {a['duration_minutes']} min")
    else:
        lines.append("- none configured yet")

    lines.append("\nForecast:")
    for day, weather in zip(days_out, weather_daily_summary):
        s = day["summary"]
        lines.append(
            f"- {day['label']} ({day['date']}): {s['solar_quality']} solar, "
            f"{s['total_pv_kWh_today']}kWh generated, peak temp {weather['max_temp_C']}C, "
            f"cloud cover {weather['avg_cloudcover_pct']}%, rain {weather['total_precipitation_mm']}mm. "
            f"{s['appliances_covered']}/{s['appliances_total']} appliances fully solar-covered."
        )
        for name, w in day["recommendations"].items():
            if w:
                lines.append(f"    {name}: best window {w['start_time'][-8:-3]}-{w['end_time'][-8:-3]}, {w['coverage_pct']}% solar covered")

    return "\n".join(lines)


def get_ai_advice(days_out, appliances, weather_daily_summary, background_load_kW, user_question=None, groq_api_key=None):
    """
    Calls Groq's free API (Llama 3.3 70B) grounded in the app's real
    forecast data. Falls back to the rule-based tips if no API key is
    configured or the request fails for any reason.

    Returns (text_or_list, source) where source is "ai" or "offline".
    """
    api_key = groq_api_key or config.GROQ_API_KEY
    if not api_key:
        return generate_tips(days_out, appliances, weather_daily_summary, background_load_kW), "offline"

    context = _build_context_summary(days_out, appliances, weather_daily_summary, background_load_kW)

    system_prompt = (
        "You are a friendly solar energy advisor helping a homeowner understand their "
        "solar forecast and plan appliance usage. Use ONLY the data provided below -- "
        "never invent numbers. Be concise, practical, and plain-spoken (avoid jargon). "
        "If asked something the data can't answer, say so honestly.\n\n" + context
    )

    user_message = user_question or (
        "Give me a short, friendly summary of my solar outlook and the best way to use "
        "my appliances over the next few days. Keep it under 150 words."
    )

    try:
        resp = requests.post(
            config.GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.5,
                "max_tokens": 400,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text, "ai"
    except Exception as e:
        fallback = generate_tips(days_out, appliances, weather_daily_summary, background_load_kW)
        return fallback, "offline"


def process_copilot_chat(messages, context_str, groq_api_key=None):
    import json
    api_key = groq_api_key or config.GROQ_API_KEY
    if not api_key:
        return {"error": "Groq API key required for Copilot."}

    system_prompt = (
        "You are an expert Solar Energy AI Copilot embedded inside a professional SaaS dashboard. "
        "You help users analyze their data, configure their home battery, and schedule appliances. "
        "Format your responses using Markdown. You can output tables, bold text, and lists. "
        "If the user asks to add an appliance, return a valid JSON object wrapped in ```json block with an action. "
        "JSON SCHEMA: {\"action\": \"ADD_APPLIANCE\", \"appliance\": {\"name\": str, \"duration_minutes\": int, \"typical_power_kW\": float}}\n"
        "Otherwise, just respond normally in Markdown.\n\n"
        f"{context_str}"
    )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    # Keep only the last 10 messages to save context limit
    for msg in messages[-10:]:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        resp = requests.post(
            config.GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": config.GROQ_MODEL,
                "messages": formatted_messages,
                "temperature": 0.4,
                "max_tokens": 800,
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        
        # Check if the AI outputted a JSON action block
        action = None
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                action = data.get("action")
                if action == "ADD_APPLIANCE":
                    return {"message": content.replace(f"```json\n{json_str}\n```", ""), "action": action, "appliance": data.get("appliance")}
            except:
                pass
                
        return {"message": content, "action": action}
    except Exception as e:
        return {"error": str(e), "message": "Connection to AI Engine failed."}