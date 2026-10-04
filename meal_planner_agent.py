"""
🍱 Smart Meal Planner Agent
---------------------------
A tool-using agent built on the Anthropic API.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key"      # Windows: set ANTHROPIC_API_KEY=your-key
    python meal_planner_agent.py

Then chat, e.g.:
    "Plan 5 days of vegetarian lunches and dinners, ~1800 kcal/day,
     high protein, no more than 30 min cooking, for 2 people."
"""

import json
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gemini-3.5-flash-lite"
MAX_TURNS = 8            # safety limit on tool-call loops per user message

# ---------------------------------------------------------------------------
# Recipe database (edit/extend freely). ingredients: name -> (quantity, unit)
# Quantities are per 1 serving.
# ---------------------------------------------------------------------------
RECIPES = [
    {"name": "Vegetable Poha", "meal": "breakfast", "diet": "vegan", "minutes": 20,
     "calories": 320, "protein": 8,
     "ingredients": {"flattened rice (poha)": (80, "g"), "onion": (50, "g"), "peanuts": (15, "g"),
                     "potato": (50, "g"), "peas": (30, "g"), "lemon": (0.25, "pc")}},
    {"name": "Moong Dal Chilla", "meal": "breakfast", "diet": "vegan", "minutes": 20,
     "calories": 280, "protein": 16,
     "ingredients": {"moong dal": (70, "g"), "onion": (30, "g"), "tomato": (30, "g"),
                     "green chilli": (5, "g"), "coriander": (5, "g")}},
    {"name": "Masala Omelette with Toast", "meal": "breakfast", "diet": "non-veg", "minutes": 15,
     "calories": 350, "protein": 20,
     "ingredients": {"eggs": (2, "pc"), "onion": (30, "g"), "tomato": (30, "g"),
                     "bread slices": (2, "pc"), "green chilli": (5, "g")}},
    {"name": "Idli with Sambar", "meal": "breakfast", "diet": "vegan", "minutes": 25,
     "calories": 300, "protein": 11,
     "ingredients": {"idli batter": (150, "g"), "toor dal": (30, "g"), "mixed vegetables": (80, "g"),
                     "sambar powder": (5, "g")}},
    {"name": "Dal Tadka with Jeera Rice", "meal": "lunch", "diet": "vegan", "minutes": 35,
     "calories": 520, "protein": 18,
     "ingredients": {"toor dal": (60, "g"), "basmati rice": (70, "g"), "onion": (40, "g"),
                     "tomato": (60, "g"), "garlic": (5, "g"), "cumin seeds": (3, "g")}},
    {"name": "Rajma Chawal", "meal": "lunch", "diet": "vegan", "minutes": 45,
     "calories": 580, "protein": 22,
     "ingredients": {"rajma (kidney beans)": (70, "g"), "basmati rice": (70, "g"), "onion": (50, "g"),
                     "tomato": (80, "g"), "ginger": (5, "g"), "garlic": (5, "g")}},
    {"name": "Paneer Bhurji with Roti", "meal": "lunch", "diet": "vegetarian", "minutes": 25,
     "calories": 560, "protein": 28,
     "ingredients": {"paneer": (120, "g"), "whole wheat flour": (80, "g"), "onion": (50, "g"),
                     "tomato": (60, "g"), "capsicum": (40, "g")}},
    {"name": "Chicken Curry with Rice", "meal": "lunch", "diet": "non-veg", "minutes": 40,
     "calories": 620, "protein": 38,
     "ingredients": {"chicken": (150, "g"), "basmati rice": (70, "g"), "onion": (60, "g"),
                     "tomato": (80, "g"), "ginger": (5, "g"), "garlic": (5, "g"), "curd": (30, "g")}},
    {"name": "Chana Masala with Roti", "meal": "dinner", "diet": "vegan", "minutes": 35,
     "calories": 540, "protein": 21,
     "ingredients": {"chickpeas": (80, "g"), "whole wheat flour": (80, "g"), "onion": (50, "g"),
                     "tomato": (80, "g"), "chole masala": (5, "g")}},
    {"name": "Palak Paneer with Roti", "meal": "dinner", "diet": "vegetarian", "minutes": 30,
     "calories": 560, "protein": 27,
     "ingredients": {"paneer": (100, "g"), "spinach": (150, "g"), "whole wheat flour": (80, "g"),
                     "onion": (40, "g"), "tomato": (50, "g"), "cream": (15, "g")}},
    {"name": "Vegetable Khichdi with Curd", "meal": "dinner", "diet": "vegetarian", "minutes": 30,
     "calories": 450, "protein": 17,
     "ingredients": {"rice": (50, "g"), "moong dal": (40, "g"), "mixed vegetables": (100, "g"),
                     "curd": (100, "g"), "ghee": (5, "g")}},
    {"name": "Egg Curry with Roti", "meal": "dinner", "diet": "non-veg", "minutes": 30,
     "calories": 520, "protein": 27,
     "ingredients": {"eggs": (3, "pc"), "whole wheat flour": (80, "g"), "onion": (60, "g"),
                     "tomato": (80, "g"), "ginger": (5, "g"), "garlic": (5, "g")}},
    {"name": "Sprouts Salad", "meal": "snack", "diet": "vegan", "minutes": 10,
     "calories": 180, "protein": 11,
     "ingredients": {"mixed sprouts": (100, "g"), "onion": (20, "g"), "tomato": (30, "g"),
                     "lemon": (0.25, "pc")}},
    {"name": "Roasted Makhana", "meal": "snack", "diet": "vegetarian", "minutes": 10,
     "calories": 150, "protein": 5,
     "ingredients": {"makhana (fox nuts)": (30, "g"), "ghee": (3, "g")}},
]

# Diets are nested: vegan ⊂ vegetarian ⊂ non-veg (anything goes).
ALLOWED = {"vegan": {"vegan"}, "vegetarian": {"vegan", "vegetarian"},
           "non-veg": {"vegan", "vegetarian", "non-veg"}}

RECIPE_INDEX = {r["name"].lower(): r for r in RECIPES}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def find_recipes(diet="non-veg", meal_type=None, max_minutes=None,
                 min_protein=None, exclude_ingredients=None):
    exclude = [e.lower() for e in (exclude_ingredients or [])]
    results = []
    for r in RECIPES:
        if r["diet"] not in ALLOWED.get(diet, ALLOWED["non-veg"]):
            continue
        if meal_type and r["meal"] != meal_type:
            continue
        if max_minutes and r["minutes"] > max_minutes:
            continue
        if min_protein and r["protein"] < min_protein:
            continue
        if any(ex in ing.lower() for ex in exclude for ing in r["ingredients"]):
            continue
        results.append({k: r[k] for k in ("name", "meal", "diet", "minutes", "calories", "protein")})
    return {"count": len(results), "recipes": results}


def nutrition_summary(plan):
    """plan: {"Day 1": [{"recipe": str, "servings": number}, ...], ...}"""
    out, missing = {}, []
    for day, items in plan.items():
        cal = pro = 0
        for item in items:
            r = RECIPE_INDEX.get(item["recipe"].lower())
            if not r:
                missing.append(item["recipe"])
                continue
            s = item.get("servings", 1)
            cal += r["calories"] * s
            pro += r["protein"] * s
        out[day] = {"calories": round(cal), "protein_g": round(pro)}
    return {"per_day": out, "unknown_recipes": missing}


def grocery_list(items):
    """
    Create a categorized grocery list from the recipes in the plan.

    items:
    [
        {"recipe": "Vegetable Poha", "servings": 2},
        ...
    ]
    """

    categories = {
        "🥦 Vegetables & Fruits": defaultdict(lambda: [0, ""]),
        "🌾 Grains & Cereals": defaultdict(lambda: [0, ""]),
        "🥛 Dairy & Alternatives": defaultdict(lambda: [0, ""]),
        "🥚 Protein & Legumes": defaultdict(lambda: [0, ""]),
        "🥜 Nuts & Seeds": defaultdict(lambda: [0, ""]),
        "🧂 Spices & Other": defaultdict(lambda: [0, ""])
    }

    missing = []

    grain_words = [
        "rice", "flour", "poha", "bread", "oats",
        "idli batter", "roti"
    ]

    dairy_words = [
        "paneer", "curd", "cream", "milk", "tofu",
        "soy yogurt"
    ]

    protein_words = [
        "dal", "rajma", "chickpeas", "chana",
        "sprouts", "eggs", "chicken", "soy"
    ]

    nut_words = [
        "peanuts", "almonds", "makhana"
    ]

    vegetable_words = [
        "onion", "tomato", "potato", "peas", "spinach",
        "capsicum", "lemon", "ginger", "garlic",
        "coriander", "vegetables", "green chilli"
    ]

    for item in items:

        recipe_name = item["recipe"]
        servings = item.get("servings", 1)

        recipe = RECIPE_INDEX.get(recipe_name.lower())

        if not recipe:
            missing.append(recipe_name)
            continue

        for ingredient, (quantity, unit) in recipe["ingredients"].items():

            total = quantity * servings
            name = ingredient.lower()

            if any(x in name for x in vegetable_words):
                category = "🥦 Vegetables & Fruits"

            elif any(x in name for x in grain_words):
                category = "🌾 Grains & Cereals"

            elif any(x in name for x in dairy_words):
                category = "🥛 Dairy & Alternatives"

            elif any(x in name for x in protein_words):
                category = "🥚 Protein & Legumes"

            elif any(x in name for x in nut_words):
                category = "🥜 Nuts & Seeds"

            else:
                category = "🧂 Spices & Other"

            categories[category][ingredient][0] += total
            categories[category][ingredient][1] = unit

    result = {}

    for category, ingredients in categories.items():

        if not ingredients:
            continue

        result[category] = [
            f"{ingredient}: {round(quantity, 2)} {unit}"
            for ingredient, (quantity, unit)
            in sorted(ingredients.items())
        ]

    return {
        "categories": result,
        "unknown_recipes": missing
    }

SUBSTITUTIONS = {
    "paneer": [
        {
            "replacement": "tofu",
            "reason": "Similar texture and a good vegetarian protein source."
        },
        {
            "replacement": "soy chunks",
            "reason": "Higher-protein alternative."
        }
    ],

    "curd": [
        {
            "replacement": "soy yogurt",
            "reason": "Dairy-free alternative."
        }
    ],

    "rice": [
        {
            "replacement": "millet",
            "reason": "Whole-grain alternative with a different texture."
        },
        {
            "replacement": "quinoa",
            "reason": "Higher-protein grain alternative."
        }
    ],

    "chickpeas": [
        {
            "replacement": "black chana",
            "reason": "Similar legume with comparable protein."
        },
        {
            "replacement": "rajma",
            "reason": "Another protein-rich legume."
        }
    ],

    "spinach": [
        {
            "replacement": "methi",
            "reason": "Leafy-green alternative."
        },
        {
            "replacement": "mixed leafy vegetables",
            "reason": "Easy general replacement."
        }
    ],

    "peanuts": [
        {
            "replacement": "roasted chana",
            "reason": "Crunchy, affordable protein alternative."
        },
        {
            "replacement": "almonds",
            "reason": "Nut-based alternative."
        }
    ],

    "moong dal": [
        {
            "replacement": "masoor dal",
            "reason": "Similar cooking use and protein profile."
        },
        {
            "replacement": "toor dal",
            "reason": "Common Indian dal alternative."
        }
    ]
}


def suggest_substitutions(ingredient, reason=""):
    """
    Suggest practical ingredient substitutions.
    """

    key = ingredient.lower().strip()

    options = SUBSTITUTIONS.get(key)

    if not options:

        # Try partial matching
        for original, replacements in SUBSTITUTIONS.items():
            if original in key or key in original:
                options = replacements
                break

    if not options:
        return {
            "ingredient": ingredient,
            "message": "No predefined substitution was found. Choose a similar ingredient based on the recipe."
        }

    return {
        "ingredient": ingredient,
        "reason": reason,
        "substitutions": options
    }

def get_recipe_details(recipe):
    """
    Return complete information about a recipe.
    """

    r = RECIPE_INDEX.get(recipe.lower())

    if not r:
        return {
            "error": f"Recipe '{recipe}' was not found."
        }

    return {
        "name": r["name"],
        "meal": r["meal"],
        "diet": r["diet"],
        "cooking_time_minutes": r["minutes"],
        "calories": r["calories"],
        "protein_g": r["protein"],
        "ingredients": [
            f"{name}: {quantity} {unit}"
            for name, (quantity, unit)
            in r["ingredients"].items()
        ]
    }

def create_meal_prep_schedule(plan):
    """
    Create a practical meal-preparation schedule from a meal plan.
    """

    schedule = {}

    for day, meals in plan.items():

        day_tasks = []

        total_time = 0

        for item in meals:

            recipe_name = item["recipe"]
            recipe = RECIPE_INDEX.get(recipe_name.lower())

            if not recipe:
                continue

            total_time += recipe["minutes"]

            if recipe["minutes"] <= 15:
                task = f"Quickly prepare {recipe_name} ({recipe['minutes']} min)."

            elif recipe["minutes"] <= 30:
                task = f"Prepare {recipe_name} ({recipe['minutes']} min)."

            else:
                task = (
                    f"Prepare {recipe_name} ({recipe['minutes']} min). "
                    "Consider batch-preparing ingredients earlier."
                )

            day_tasks.append(task)

        schedule[day] = {
            "estimated_active_time_minutes": total_time,
            "tasks": day_tasks
        }

    return {
        "schedule": schedule,
        "tip": (
            "For longer recipes, batch-cook grains, dals or basic "
            "vegetable preparations to reduce daily cooking time."
        )
    }

def analyze_plan(plan, calorie_target=None, protein_target=None):
    """
    Analyze a generated plan for nutrition, repetition and cooking time.
    """

    nutrition = nutrition_summary(plan)

    warnings = []
    total_days = len(plan)

    recipe_counts = defaultdict(int)
    longest_meal = 0

    for day, meals in plan.items():

        for item in meals:

            recipe_name = item["recipe"]
            recipe_counts[recipe_name] += 1

            recipe = RECIPE_INDEX.get(recipe_name.lower())

            if recipe:
                longest_meal = max(
                    longest_meal,
                    recipe["minutes"]
                )

    # Repetition check
    repeated = [
        recipe
        for recipe, count in recipe_counts.items()
        if count > 2
    ]

    if repeated:
        warnings.append(
            "These recipes are repeated more than twice: "
            + ", ".join(repeated)
        )

    # Calorie analysis
    if calorie_target:

        for day, data in nutrition["per_day"].items():

            calories = data["calories"]

            difference = calories - calorie_target

            if abs(difference) > calorie_target * 0.15:

                warnings.append(
                    f"{day}: calories are {calories}, "
                    f"target is {calorie_target}."
                )

    # Protein analysis
    if protein_target:

        for day, data in nutrition["per_day"].items():

            protein = data["protein_g"]

            if protein < protein_target:

                warnings.append(
                    f"{day}: protein is {protein}g, "
                    f"target is {protein_target}g."
                )

    return {
        "days": total_days,
        "nutrition": nutrition,
        "longest_single_recipe_minutes": longest_meal,
        "repeated_recipes": repeated,
        "warnings": warnings,
        "status": "Needs improvement" if warnings else "Looks good"
    }

def save_plan(filename, content):
    # Always save in a "meal_plans" folder next to this script (not the terminal's cwd)
    folder = Path(__file__).resolve().parent / "meal_plans"
    folder.mkdir(exist_ok=True)
    path = folder / (Path(filename).stem + ".md")
    path.write_text(content, encoding="utf-8")
    print(f"\n  ✅ Plan saved to: {path}\n")
    return {"saved_to": str(path)}


TOOL_FUNCS = {
    "find_recipes": find_recipes,
    "nutrition_summary": nutrition_summary,
    "grocery_list": grocery_list,
    "save_plan": save_plan,

    # Phase 2 tools
    "get_recipe_details": get_recipe_details,
    "suggest_substitutions": suggest_substitutions,
    "create_meal_prep_schedule": create_meal_prep_schedule,
    "analyze_plan": analyze_plan,
}

TOOLS = [
    {"name": "find_recipes",
     "description": "Search the recipe database. Always use this to pick meals; only use recipes it returns.",
     "input_schema": {"type": "object", "properties": {
         "diet": {"type": "string", "enum": ["vegan", "vegetarian", "non-veg"]},
         "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"]},
         "max_minutes": {"type": "integer"},
         "min_protein": {"type": "integer", "description": "Minimum grams of protein per serving"},
         "exclude_ingredients": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "nutrition_summary",
     "description": "Total calories and protein per day for a draft plan.",
     "input_schema": {"type": "object", "properties": {
         "plan": {"type": "object", "description": "Map of day label -> list of {recipe, servings}",
                  "additionalProperties": {"type": "array", "items": {"type": "object", "properties": {
                      "recipe": {"type": "string"}, "servings": {"type": "number"}},
                      "required": ["recipe"]}}}},
         "required": ["plan"]}},
    {"name": "grocery_list",
     "description": "Combined shopping list for every meal in the plan. servings = number of people eating.",
     "input_schema": {"type": "object", "properties": {
         "items": {"type": "array", "items": {"type": "object", "properties": {
             "recipe": {"type": "string"}, "servings": {"type": "number"}}, "required": ["recipe"]}}},
         "required": ["items"]}},
    {"name": "save_plan",
     "description": "Save the final meal plan (markdown) to a file. Call when the user says to save/export the plan (e.g. 'save it'), and tell them the returned path.",
     "input_schema": {"type": "object", "properties": {
         "filename": {"type": "string"}, "content": {"type": "string"}},
         "required": ["filename", "content"]}},
]

GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[

            types.FunctionDeclaration(
                name="find_recipes",
                description="Search the recipe database. Always use this to pick meals; only use recipes it returns.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "diet": {
                            "type": "string",
                            "enum": ["vegan", "vegetarian", "non-veg"]
                        },
                        "meal_type": {
                            "type": "string",
                            "enum": ["breakfast", "lunch", "dinner", "snack"]
                        },
                        "max_minutes": {
                            "type": "integer"
                        },
                        "min_protein": {
                            "type": "integer",
                            "description": "Minimum grams of protein per serving"
                        },
                        "exclude_ingredients": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["diet", "meal_type"]
                }
            ),
        
            types.FunctionDeclaration(
                name="nutrition_summary",
                description="Calculate total calories and protein per day for a draft meal plan.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "object",
                            "description": "Map of day label to list of recipe and servings."
                        }
                    },
                    "required": ["plan"]
                }
            ),

            types.FunctionDeclaration(
                name="grocery_list",
                description="Create a combined grocery shopping list for all meals in the plan.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "recipe": {
                                        "type": "string"
                                    },
                                    "servings": {
                                        "type": "number"
                                    }
                                },
                                "required": ["recipe"]
                            }
                        }
                    },
                    "required": ["items"]
                }
            ),

            types.FunctionDeclaration(
                name="save_plan",
                description="Save the final meal plan as a markdown file. Only call this after the user approves.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string"
                        },
                        "content": {
                            "type": "string"
                        }
                    },
                    "required": ["filename", "content"]
                }
            ),

            types.FunctionDeclaration(
                name="get_recipe_details",
                description="Get detailed information about a recipe.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "recipe": {
                            "type": "string"
                        }
                    },
                    "required": ["recipe"]
                }
            ),

            types.FunctionDeclaration(
                name="suggest_substitutions",
                description="Suggest alternatives for an ingredient.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "ingredient": {
                            "type": "string"
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": ["ingredient"]
                }
            ),

            types.FunctionDeclaration(
                name="create_meal_prep_schedule",
                description="Create a meal preparation schedule.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "object"
                        }
                    },
                    "required": ["plan"]
                }
            ),

            types.FunctionDeclaration(
                name="analyze_plan",
                description="Analyze calories, protein, repetition and cooking time.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "object"
                        },
                        "calorie_target": {
                            "type": "number"
                        },
                        "protein_target": {
                            "type": "number"
                        }
                    },
                    "required": ["plan"]
                }
            )
        ]
    )
]

SYSTEM_PROMPT = """
You are Smart Meal Planner, an intelligent nutrition-aware meal planning agent.

Your job is to create practical, personalized meal plans using the available tools.

WORKFLOW:

1. Understand the user's requirements:
   - diet
   - number of days
   - number of people
   - calorie target
   - protein target
   - allergies
   - disliked ingredients
   - maximum cooking time

2. Use find_recipes to select meals.
   Never invent recipes outside the recipe database.

3. Create variety.
   Do not repeat the same recipe more than twice unless necessary.

4. Use nutrition_summary to calculate calories and protein.

5. Use analyze_plan to check the draft.

6. If the plan has nutritional or variety problems:
   - change recipes
   - use different meals
   - call analyze_plan again

7. Use grocery_list when the user asks for shopping requirements
   or when a complete plan has been generated.

8. The grocery list should be presented in categories.

9. If the user says an ingredient is unavailable:
   use suggest_substitutions.

10. If the user asks how to prepare a recipe:
    use get_recipe_details.

11. If the user asks for meal preparation help:
    use create_meal_prep_schedule.

12. If the user asks to save the plan:
    use save_plan.

13. Always explain important changes briefly.

FINAL RESPONSE:

Present the result clearly using:

🍱 MEAL PLAN

Day-by-day meals

📊 NUTRITION

Calories and protein

🔍 PLAN CHECK

Mention whether the plan meets the requested requirements.

🛒 GROCERY LIST

Categorized ingredients when requested.

🍳 MEAL PREP

Give practical preparation suggestions when requested.

🔄 SUBSTITUTIONS

Show alternatives when ingredients are unavailable.

Be concise, friendly and practical.

Do not claim medical treatment or diagnosis.
For medical conditions, suggest consulting a qualified healthcare professional.
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_turn(client, messages, activity_callback=None, state=None):
    """Run Gemini until it finishes using the available meal-planning tools."""

    contents = []

    # Convert our simple conversation history into Gemini format
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(
                        text=msg["content"]
                    )
                ]
            )
        )

    for _ in range(MAX_TURNS):

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=GEMINI_TOOLS,
                max_output_tokens=2000,
            ),
        )

        # Safety check
        if not response.candidates:
            return "Sorry, I couldn't generate a response."

        model_content = response.candidates[0].content

        # Add Gemini's response to conversation
        contents.append(model_content)

        # Did Gemini request any tools?
        function_calls = response.function_calls

        if not function_calls:
            reply = response.text or "Sorry, I couldn't generate a response."

            messages.append({
                "role": "assistant",
                "content": reply
            })

            return reply

        # Execute requested tools
        function_response_parts = []

        for function_call in function_calls:

            tool_name = function_call.name
            tool_args = dict(function_call.args or {})

            activity = {
                "tool": tool_name,
                "label": {
                    "find_recipes": "Searching suitable recipes",
                    "nutrition_summary": "Checking nutrition",
                    "grocery_list": "Preparing grocery list",
                    "save_plan": "Saving meal plan",
                    "get_recipe_details": "Getting recipe details",
                    "suggest_substitutions": "Finding substitutions",
                    "create_meal_prep_schedule": "Creating meal-prep schedule",
                    "analyze_plan": "Analyzing and improving the plan"
                }.get(tool_name, "Using planning tool")
            }

            if activity_callback:
                activity_callback(activity)

            print(
                f"🔧 {tool_name}({json.dumps(tool_args)[:150]}...)"
            )

            try:
                if tool_name not in TOOL_FUNCS:
                    result = {
                        "error": f"Unknown tool: {tool_name}"
                    }
                else:
                    result = TOOL_FUNCS[tool_name](**tool_args)

            except Exception as e:
                result = {
                    "error": str(e)
                }

            function_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={
                        "result": result
                    }
                )
            )

        # Send tool results back to Gemini
        contents.append(
            types.Content(
                role="user",
                parts=function_response_parts
            )
        )

    return "Sorry, I reached the maximum number of planning steps. Please try a simpler request."

def main():
    client = genai.Client()  # reads GEMINI_API_KEY from environment
    messages = []
    print("🍱 Smart Meal Planner Agent  (type 'quit' to exit)\n")
    print("Tell me about your goals, e.g. '5-day vegetarian plan, 1800 kcal, 2 people'.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "q"}:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        print("\nAgent:", run_turn(client, messages), "\n")


if __name__ == "__main__":
    main()
