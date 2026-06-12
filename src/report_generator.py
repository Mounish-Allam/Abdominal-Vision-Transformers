import os

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

ORGAN_NAMES = {1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}

_SYSTEM = (
    "You are a radiologist assistant. Based on 2D abdominal MRI segmentation results, "
    "write a concise clinical impression (under 180 words). Use formal radiology language. "
    "Comment on organ presence, relative coverage compared to typical anatomy, "
    "and any notable asymmetries between paired structures."
)


def generate_report(organ_stats: dict, api_key: str = "") -> str:
    key = api_key.strip() or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "No Groq API key provided. Get a free key at console.groq.com and enter it above."
    if not GROQ_AVAILABLE:
        return "Run: pip install groq"

    total = organ_stats.get("total", 1)
    bg_pct = organ_stats.get(0, {}).get("pct", 0.0)

    lines = [
        f"- {ORGAN_NAMES[cls_id]}: {stats['pixels']:,} px  ({stats['pct']:.1f}% of slice)"
        for cls_id, stats in organ_stats.items()
        if isinstance(cls_id, int) and cls_id != 0
    ]

    prompt = (
        "Single 2D abdominal MRI slice — automated segmentation output:\n\n"
        + "\n".join(lines)
        + f"\n- Background: {bg_pct:.1f}% of slice"
        + f"\n- Total pixels: {total:,}\n\n"
        "Generate a brief radiologist-style clinical impression."
    )

    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=400,
    )
    return response.choices[0].message.content
