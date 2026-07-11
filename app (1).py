"""
Interactive Food Search & RAG Chatbot System - Streamlit UI
=============================================================

A web interface for semantic food search backed by ChromaDB and
sentence-transformers embeddings.

Run with:
    streamlit run app.py

Requires a food dataset JSON file to be uploaded via the sidebar before any
search can be performed. See `requirements.txt` for dependencies.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
COLLECTION_NAME = "interactive_food_search"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L12-v2"
DEFAULT_RESULT_COUNT = 4

INGREDIENT_SYNONYMS = {
    "sugar": ["sugar", "sweetener"],
    "salt": ["salt", "seasoning"],
    "flour": ["flour", "baking powder"],
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("food_search")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_food_data(uploaded_file) -> List[Dict[str, Any]]:
    """Parse and normalize food data from an uploaded JSON file.

    Args:
        uploaded_file: A Streamlit `UploadedFile` object.

    Returns:
        A list of normalized food-item dictionaries. Empty on failure.
    """
    try:
        food_data = json.load(uploaded_file)
    except json.JSONDecodeError as error:
        st.error(f"Could not parse the uploaded file as JSON: {error}")
        return []

    if not isinstance(food_data, list):
        st.error("The uploaded JSON must be a list of food items.")
        return []

    for index, item in enumerate(food_data):
        item["food_id"] = str(item.get("food_id", index + 1))
        item.setdefault("food_ingredients", [])
        item.setdefault("food_description", "")
        item.setdefault("cuisine_type", "Unknown")
        item.setdefault("food_calories_per_serving", 0)

        features = item.get("food_features")
        if isinstance(features, dict):
            item["taste_profile"] = ", ".join(str(v) for v in features.values() if v)
        else:
            item["taste_profile"] = ""

    return food_data


# --------------------------------------------------------------------------
# Vector store setup
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Create (once per session) the shared ChromaDB client."""
    return chromadb.Client()


def create_similarity_search_collection(
    collection_name: str,
    collection_metadata: Optional[Dict[str, Any]] = None,
    model_name: str = EMBEDDING_MODEL_NAME,
):
    """Create (or recreate) a ChromaDB collection for semantic search."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    return client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": embedding_function,
        },
    )


def _expand_ingredient_synonyms(ingredients_text: str) -> str:
    """Insert known synonyms into an ingredient string to improve recall."""
    for term, synonyms in INGREDIENT_SYNONYMS.items():
        pattern = r"\b" + re.escape(term) + r"\b"
        ingredients_text = re.sub(pattern, ", ".join(synonyms), ingredients_text)
    return ingredients_text


def _build_embedding_text(food: Dict[str, Any]) -> str:
    """Compose a single descriptive string used to embed a food item."""
    name = food.get("food_name", "").lower().strip()
    description = food.get("food_description", "").lower().strip()
    ingredients = _expand_ingredient_synonyms(
        ", ".join(food.get("food_ingredients", [])).lower().strip()
    )
    cuisine_type = food.get("cuisine_type", "").lower().strip()
    cooking_method = food.get("cooking_method", "").lower().strip()

    parts = [
        f"Name: {name}.",
        f"Description: {description}.",
        f"Ingredients: {ingredients}.",
        f"Cuisine: {cuisine_type}.",
        f"Cooking method: {cooking_method}.",
    ]

    taste_profile = food.get("taste_profile", "")
    if taste_profile:
        parts.append(f"Taste and features: {taste_profile}.")

    health_benefits = food.get("food_health_benefits", "")
    if health_benefits:
        parts.append(f"Health benefits: {health_benefits}.")

    nutrition = food.get("food_nutritional_factors")
    if isinstance(nutrition, dict) and nutrition:
        nutrition_text = ", ".join(f"{key}: {value}" for key, value in nutrition.items())
        parts.append(f"Nutrition: {nutrition_text}.")

    return " ".join(parts)


def populate_similarity_collection(collection, food_items: List[Dict[str, Any]]) -> None:
    """Embed and insert every food item into a ChromaDB collection."""
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    used_ids = set()

    for index, food in enumerate(food_items):
        documents.append(_build_embedding_text(food))

        base_id = str(food.get("food_id", index))
        unique_id, suffix = base_id, 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(unique_id)
        ids.append(unique_id)

        metadatas.append(
            {
                "name": food.get("food_name", "Unknown"),
                "cuisine_type": food.get("cuisine_type", "Unknown"),
                "ingredients": ", ".join(food.get("food_ingredients", [])),
                "calories": food.get("food_calories_per_serving", 0),
                "description": food.get("food_description", ""),
                "cooking_method": food.get("cooking_method", ""),
                "health_benefits": food.get("food_health_benefits", ""),
                "taste_profile": food.get("taste_profile", ""),
            }
        )

    collection.add(documents=documents, metadatas=metadatas, ids=ids)


# --------------------------------------------------------------------------
# Search functions
# --------------------------------------------------------------------------
def _format_query_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a raw ChromaDB query response into a list of clean dicts."""
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    formatted = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
        formatted.append(
            {
                "food_id": results["ids"][0][i],
                "food_name": metadata["name"],
                "food_description": metadata["description"],
                "cuisine_type": metadata["cuisine_type"],
                "food_calories_per_serving": metadata["calories"],
                "similarity_score": 1 - distance,
                "distance": distance,
            }
        )
    return formatted


def perform_filtered_similarity_search(
    collection,
    query: str,
    cuisine_filter: Optional[str] = None,
    max_calories: Optional[int] = None,
    n_results: int = DEFAULT_RESULT_COUNT,
) -> List[Dict[str, Any]]:
    """Run a similarity search, optionally constrained by metadata filters."""
    filters = []
    if cuisine_filter and cuisine_filter != "All":
        filters.append({"cuisine_type": cuisine_filter})
    if max_calories is not None:
        filters.append({"calories": {"$lte": max_calories}})

    if len(filters) > 1:
        where_clause = {"$and": filters}
    elif filters:
        where_clause = filters[0]
    else:
        where_clause = None

    try:
        results = collection.query(
            query_texts=[query], n_results=n_results, where=where_clause
        )
        return _format_query_results(results)
    except Exception as error:
        logger.error("Search failed for query %r: %s", query, error)
        st.error(f"Search failed: {error}")
        return []


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def render_result_card(result: Dict[str, Any]) -> None:
    """Render a single search result as a styled card."""
    match_pct = result["similarity_score"] * 100
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 🍽️ {result['food_name']}")
            st.caption(f"🏷️ {result['cuisine_type']}  •  🔥 {result['food_calories_per_serving']} cal/serving")
            st.write(result["food_description"] or "_No description available._")
        with col2:
            st.metric("Match", f"{match_pct:.1f}%")


def main() -> None:
    st.set_page_config(
        page_title="Food Search & RAG Chatbot",
        page_icon="🍽️",
        layout="centered",
    )
    st.title("🍽️ Interactive Food Search")
    st.caption("Semantic search over a food dataset, powered by ChromaDB + sentence-transformers.")

    # ---- Required file upload ------------------------------------------------
    st.sidebar.header("📁 Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a food dataset (JSON) — required",
        type=["json"],
        help="A JSON list of food items. See README for the expected schema.",
    )

    if uploaded_file is None:
        st.warning(
            "⬅️ Please upload a `FoodDataSet.json` file in the sidebar to get started. "
            "No search can be performed until a dataset is loaded."
        )
        with st.expander("Expected file format"):
            st.code(
                """[
  {
    "food_name": "Margherita Pizza",
    "food_description": "Classic pizza with tomato, mozzarella, and basil",
    "food_ingredients": ["tomato", "mozzarella", "basil", "flour"],
    "cuisine_type": "Italian",
    "cooking_method": "baked",
    "food_calories_per_serving": 285,
    "food_health_benefits": "Good source of calcium",
    "food_nutritional_factors": {"protein": "12g", "fat": "10g"},
    "food_features": {"taste": "savory", "texture": "crispy"}
  }
]""",
                language="json",
            )
        st.stop()

    # ---- Build (or reuse) the search index -----------------------------------
    file_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("file_signature") != file_signature:
        with st.spinner("Loading dataset and building search index..."):
            food_items = load_food_data(uploaded_file)
            if not food_items:
                st.stop()
            collection = create_similarity_search_collection(
                COLLECTION_NAME,
                {"description": "A collection for interactive food search"},
            )
            populate_similarity_collection(collection, food_items)

        st.session_state["collection"] = collection
        st.session_state["food_items"] = food_items
        st.session_state["file_signature"] = file_signature
        st.toast(f"Loaded {len(food_items)} food items ✅")

    collection = st.session_state["collection"]
    food_items = st.session_state["food_items"]
    st.sidebar.success(f"{len(food_items)} items indexed")

    # ---- Filters ---------------------------------------------------------------
    st.sidebar.header("🔧 Filters")
    cuisine_options = ["All"] + sorted(
        {item.get("cuisine_type", "Unknown") for item in food_items}
    )
    cuisine_filter = st.sidebar.selectbox("Cuisine type", cuisine_options)
    max_calories = st.sidebar.slider(
        "Max calories per serving", min_value=0, max_value=2000, value=2000, step=50
    )
    n_results = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=4)

    # ---- Search ------------------------------------------------------------
    with st.expander("💡 Search examples"):
        st.write(
            "- `chocolate dessert` — find chocolate desserts\n"
            "- `Italian food` — find Italian cuisine\n"
            "- `sweet treats` — find sweet desserts\n"
            "- `low calorie` — find lower-calorie options"
        )

    query = st.text_input("🔍 Search for food", placeholder="e.g. spicy noodle soup")
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked or query:
        if not query.strip():
            st.info("Type a search term above, then press Search.")
            return

        with st.spinner(f"Searching for '{query}'..."):
            results = perform_filtered_similarity_search(
                collection,
                query,
                cuisine_filter=cuisine_filter,
                max_calories=max_calories if max_calories < 2000 else None,
                n_results=n_results,
            )

        if not results:
            st.error("❌ No matching foods found. Try different keywords or loosen the filters.")
            return

        st.subheader(f"✅ Found {len(results)} recommendations")
        for result in results:
            render_result_card(result)

        cuisines = list({r["cuisine_type"] for r in results})
        if cuisines:
            st.caption(
                "💡 Related: " + ", ".join(f"`{c} dishes`" for c in cuisines[:3])
            )


if __name__ == "__main__":
    main()
