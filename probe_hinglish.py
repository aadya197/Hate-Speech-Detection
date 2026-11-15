# Colab cell 1:  !pip -q install datasets huggingface_hub
from huggingface_hub import HfApi
api = HfApi()
seen = set()
for q in ["hinglish hate", "hinglish offensive", "hinglish abusive", "hasoc", "code-mixed hate", "hindi english hate speech"]:
    try:
        for d in api.list_datasets(search=q, limit=15):
            if d.id not in seen:
                seen.add(d.id); print(f"{d.id:60s} downloads={d.downloads}")
    except Exception as e:
        print("search failed:", q, str(e)[:80])
