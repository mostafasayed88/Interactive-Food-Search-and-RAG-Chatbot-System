"""
Streamlit UI for the Interactive Food Search & RAG Chatbot System.

Converted from the original notebook `Interactive_Food_Search_and_RAG_Chatbot_System.ipynb`
which implemented a CLI chatbot over ChromaDB + Sentence-Transformers.

Run:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DATA_FILE = "FoodDataSet.json"
COLLECTION_NAME = "interactive_food_search"
EMBEDDING_MODEL = "paraphrase-MiniLM-L12-v2"


# -----------------------------------------------------------------------------
# Data loading & ChromaDB helpers  (ported from the notebook)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Create a single ChromaDB client (cached for the whole session)."""
    import chromadb
    return chromadb.Client()


def load_food_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and normalize food data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        food_data = json.load(f)

    for i, item in enumerate(food_data):
        item["food_id"] = str(item.get("food_id", i + 1))
        item.setdefault("food_ingredients", [])
        item.setdefault("food_description", "")
        item.setdefault("cuisine_type", "Unknown")
        item.setdefault("food_calories_per_serving", 0)

        if "food_features" in item and isinstance(item["food_features"], dict):
            taste_features = [str(v) for v in item["food_features"].values() if v]
            item["taste_profile"] = ", ".join(taste_features)
        else:
            item["taste_profile"] = ""

    return food_data


def build_search_collection(client, food_items: List[Dict[str, Any]]):
    """Create (or reset) the ChromaDB collection and populate it with food items."""
    from chromadb.utils import embedding_functions

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Interactive food search collection"},
        configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef},
    )

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    used_ids: set[str] = set()

    for i, food in enumerate(food_items):
        name = food.get("food_name", "").lower().strip()
        description = food.get("food_description", "").lower().strip()
        ingredients = ", ".join(food.get("food_ingredients", [])).lower().strip()
        cuisine_type = food.get("cuisine_type", "").lower().strip()
        cooking_method = food.get("cooking_method", "").lower().strip()

        # Apply synonym expansion (kept identical to the notebook)
        ingredient_synonyms = {
            "sugar": ["sugar", "sweetener"],
            "salt": ["salt", "flour"],
            "flour": ["flour", "baking powder"],
        }
        for key, syns in ingredient_synonyms.items():
            ingredients = re.sub(r"\b" + re.escape(key) + r"\b", ", ".join(syns), ingredients)

        text = (
            f"Name: {name}. Description: {description}. "
            f"Ingredients: {ingredients}. Cuisine: {cuisine_type}. "
            f"Cooking method: {cooking_method}. "
        )
        taste_profile = food.get("taste_profile", "")
        if taste_profile:
            text += f"Taste and features: {taste_profile}. "
        health_benefits = food.get("food_health_benefits", "")
        if health_benefits:
            text += f"Health benefits: {health_benefits}. "
        nutrition = food.get("food_nutritional_factors")
        if isinstance(nutrition, dict):
            text += "Nutrition: " + ", ".join(f"{k}: {v}" for k, v in nutrition.items()) + "."

        base_id = str(food.get("food_id", i))
        unique_id = base_id
        counter = 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)

        documents.append(text)
        ids.append(unique_id)
        metadatas.append({
            "name": food.get("food_name", "Unknown"),
            "cuisine_type": food.get("cuisine_type", "Unknown"),
            "ingredients": ", ".join(food.get("food_ingredients", [])),
            "calories": int(food.get("food_calories_per_serving", 0) or 0),
            "description": food.get("food_description", ""),
            "cooking_method": food.get("cooking_method", ""),
            "health_benefits": food.get("food_health_benefits", ""),
            "taste_profile": food.get("taste_profile", ""),
        })

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Plain semantic similarity search."""
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

    if not results or not results["ids"] or len(results["ids"][0]) == 0:
        return []

    formatted: List[Dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        formatted.append({
            "food_id": results["ids"][0][i],
            "food_name": meta["name"],
            "food_description": meta["description"],
            "cuisine_type": meta["cuisine_type"],
            "food_calories_per_serving": meta["calories"],
            "ingredients": meta.get("ingredients", ""),
            "cooking_method": meta.get("cooking_method", ""),
            "health_benefits": meta.get("health_benefits", ""),
            "taste_profile": meta.get("taste_profile", ""),
            "similarity_score": 1 - results["distances"][0][i],
            "distance": results["distances"][0][i],
        })
    return formatted


def perform_filtered_similarity_search(
    collection,
    query: str,
    cuisine_filter: Optional[str] = None,
    max_calories: Optional[int] = None,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """Similarity search with optional cuisine / calorie filters."""
    filters: List[Dict[str, Any]] = []
    if cuisine_filter and cuisine_filter != "All":
        filters.append({"cuisine_type": cuisine_filter})
    if max_calories:
        filters.append({"calories": {"$lte": max_calories}})

    where_clause: Optional[Dict[str, Any]] = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    try:
        results = collection.query(query_texts=[query], n_results=n_results, where=where_clause)
    except Exception as e:
        st.error(f"Filtered search error: {e}")
        return []

    if not results or not results["ids"] or len(results["ids"][0]) == 0:
        return []

    formatted: List[Dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        formatted.append({
            "food_id": results["ids"][0][i],
            "food_name": meta["name"],
            "food_description": meta["description"],
            "cuisine_type": meta["cuisine_type"],
            "food_calories_per_serving": meta["calories"],
            "ingredients": meta.get("ingredients", ""),
            "cooking_method": meta.get("cooking_method", ""),
            "health_benefits": meta.get("health_benefits", ""),
            "taste_profile": meta.get("taste_profile", ""),
            "similarity_score": 1 - results["distances"][0][i],
            "distance": results["distances"][0][i],
        })
    return formatted


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def match_score_bar(score: float) -> None:
    """Render a colored progress bar for the similarity score."""
    pct = max(0.0, min(1.0, score)) * 100
    if pct >= 70:
        color = "#22c55e"
    elif pct >= 45:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    st.markdown(
        f"""
        <div style="background:#e5e7eb;border-radius:6px;height:8px;overflow:hidden;">
          <div style="background:{color};height:100%;width:{pct:.1f}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(idx: int, r: Dict[str, Any]) -> None:
    """Render a single food result as a styled card."""
    pct = r["similarity_score"] * 100
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {idx}. {r['food_name']}")
            st.caption(
                f"{r['cuisine_type']}  •  {r['cooking_method'] or '—'}  •  "
                f"{r['food_calories_per_serving']} cal/serving"
            )
        with col2:
            st.metric("Match", f"{pct:.1f}%")
        match_score_bar(r["similarity_score"])

        st.markdown(r["food_description"] or "_No description._")

        if r.get("taste_profile"):
            st.markdown(f"**Taste & features:** {r['taste_profile']}")
        if r.get("ingredients"):
            st.markdown(f"**Ingredients:** {r['ingredients']}")
        if r.get("health_benefits"):
            st.markdown(f"**Health benefits:** {r['health_benefits']}")


def suggest_related_searches(results: List[Dict[str, Any]]) -> List[str]:
    """Build a list of suggested follow-up queries based on the results."""
    if not results:
        return []
    suggestions: List[str] = []
    cuisines = list({r["cuisine_type"] for r in results if r.get("cuisine_type")})
    for c in cuisines[:3]:
        suggestions.append(f"{c} dishes")

    avg_cal = sum(r["food_calories_per_serving"] for r in results) / max(len(results), 1)
    if avg_cal > 350:
        suggestions.append("low calorie")
    else:
        suggestions.append("hearty meal")
    return suggestions


# -----------------------------------------------------------------------------
# Page config & init
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Interactive Food Search",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍽️ Interactive Food Recommendation System")
st.caption("Semantic search over a food database powered by ChromaDB + Sentence-Transformers.")

# -----------------------------------------------------------------------------
# Load data & build collection (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading food database & building embeddings…")
def init_collection():
    client = get_chroma_client()
    items = load_food_data(DATA_FILE)
    coll = build_search_collection(client, items)
    return coll, items


try:
    collection, food_items = init_collection()
except Exception as e:
    st.error(f"Failed to initialize the system: {e}")
    st.stop()

# Sidebar — filters & options
with st.sidebar:
    st.header("⚙️ Filters & Options")

    cuisine_options = ["All"] + sorted({
        f.get("cuisine_type", "Unknown") for f in food_items if f.get("cuisine_type")
    })
    selected_cuisine = st.selectbox("Cuisine", cuisine_options, index=0)

    max_cal = st.slider("Max calories per serving", 0, 800, 800, step=10)
    use_calorie_filter = st.checkbox("Enable calorie cap", value=False)

    n_results = st.slider("Number of results", 1, 10, 4)

    st.divider()
    st.markdown("#### 💡 Example queries")
    for q in [
        "chocolate dessert",
        "Italian food",
        "sweet treats",
        "spicy soup",
        "low calorie",
        "fried rice",
    ]:
        if st.button(q, key=f"example_{q}", use_container_width=True):
            st.session_state["query"] = q

    st.divider()
    st.caption(f"📚 {len(food_items)} food items indexed.")

# -----------------------------------------------------------------------------
# Main search
# -----------------------------------------------------------------------------
query = st.text_input(
    "🔍 Search for food",
    value=st.session_state.get("query", ""),
    placeholder="e.g. creamy Italian pasta, spicy Thai soup, low calorie salad…",
)

search_btn = st.button("Search", type="primary", use_container_width=False)

if search_btn or query:
    if not query.strip():
        st.warning("Please enter a search term.")
    else:
        with st.spinner("Searching…"):
            results = perform_filtered_similarity_search(
                collection,
                query.strip(),
                cuisine_filter=selected_cuisine,
                max_calories=max_cal if use_calorie_filter else None,
                n_results=n_results,
            )

        if not results:
            st.error("No matching foods found. Try different keywords like 'Italian', 'spicy', 'sweet', 'low calorie'.")
        else:
            st.success(f"Found {len(results)} recommendation{'s' if len(results) > 1 else ''} for '{query}'.")
            for i, r in enumerate(results, 1):
                render_result_card(i, r)

            suggestions = suggest_related_searches(results)
            if suggestions:
                st.markdown("#### 💡 Related searches you might like")
                cols = st.columns(len(suggestions))
                for col, s in zip(cols, suggestions):
                    if col.button(s, key=f"sug_{s}"):
                        st.session_state["query"] = s
                        st.rerun()
