import numpy as np
import json
import os
from collections import defaultdict
from pathlib import Path


def load_cluster_names(cache_path: str) -> dict[int, str] | None:
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert string keys back to integers
            return {int(k): v for k, v in data.items()}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}")
        return None


def save_cluster_names(cluster_names: dict[int, str], cache_path: str) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_names, f, indent=2, ensure_ascii=False)

    print(f"Saved cluster names to {cache_path}")


def generate_name_for_clusters(
        nodes: list[str],
        cluster_labels: np.ndarray,
        max_names_per_cluster: int = 500,
        model: str = "claude-sonnet-4-20250514",
        cache_path: str | None = None,
        force_regenerate: bool = False
) -> dict[int, str]:

    if cache_path and not force_regenerate:
        cached = load_cluster_names(cache_path)
        if cached is not None:
            print(f"Loaded {len(cached)} cluster names from cache: {cache_path}")
            return cached

    import anthropic

    # Group nodes by cluster
    clusters = defaultdict(list)
    for name, label in zip(nodes, cluster_labels):
        clusters[int(label)].append(name)

    client = anthropic.Anthropic()

    cluster_names = {}

    rng = np.random.default_rng(seed=42)

    for cluster_id, members in sorted(clusters.items()):
        # Sample members if cluster is too large
        if len(members) > max_names_per_cluster:
            sampled_members = rng.choice(members, max_names_per_cluster, replace=False).tolist()
            member_note = f"(showing {max_names_per_cluster} of {len(members)} total members)"
        else:
            sampled_members = members
            member_note = f"({len(members)} members)"

        prompt = f"""You are analyzing a cluster of actors and directors who have worked together professionally.

Based on the following names, determine what connects these people and provide a SHORT, descriptive name for this cluster (max 5-7 words).

Think about:
- Common film franchises or series they've worked on
- TV shows they've been part of
- A famous director and their frequent collaborators
- A specific genre or era they're known for
- Production companies or studios they're associated with

Cluster members {member_note}:
{chr(10).join(f"- {name}" for name in sampled_members)}

Respond with ONLY the cluster name, nothing else. Be specific if you recognize a pattern, or use a general descriptor if the connection isn't clear.

Examples of good cluster names:
- "Marvel Cinematic Universe Cast"
- "Quentin Tarantino Collaborators"
- "Breaking Bad / Better Call Saul Cast"
- "1990s Action Film Stars"
- "British Period Drama Actors"
- "Mixed Actors and Directors" (if no clear pattern)"""

        # Call the LLM
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        cluster_name = response.content[0].text.strip()
        cluster_names[cluster_id] = cluster_name

        print(f"Cluster {cluster_id}: {cluster_name} ({len(members)} members)")

    # Save to cache if path provided
    if cache_path:
        save_cluster_names(cluster_names, cache_path)

    return cluster_names