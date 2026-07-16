"""
Menu vocabulary for the synthetic dataset. Deliberately bigger and more
varied than a single fixed menu - the more distinct text strings and price
formats the synthesis model renders, the less the fine-tuned OCR model can
get away with memorizing a handful of shapes instead of learning to read.

get_menu_items() returns a flat list of (category, name, price) tuples.
format_price() renders a price as text the way it might realistically be
handwritten - varying decimal separator, decimal places, and whether a
currency letter is scrawled next to it - so the "price" crops aren't all
one rigid format either.
"""
import random

CATEGORIES = {
    "Coffee": [
        "Espresso", "Double Espresso", "Americano", "Cappuccino", "Latte",
        "Flat White", "Mocha", "Macchiato", "Cold Brew", "Iced Latte",
        "Affogato", "Turkish Coffee", "Cortado", "Ristretto", "Café au Lait",
    ],
    "Tea": [
        "Mint Tea", "Green Tea", "Black Tea", "Chamomile", "Earl Grey",
        "Ginger Tea", "Jasmine Tea", "Herbal Infusion", "Chai Latte", "Iced Tea",
    ],
    "Pastries": [
        "Croissant", "Pain au Chocolat", "Blueberry Muffin", "Cinnamon Roll",
        "Almond Tart", "Banana Bread", "Cheese Danish", "Scone",
        "Chocolate Brownie", "Apple Turnover", "Baklava", "Makroudh",
    ],
    "Breakfast": [
        "Avocado Toast", "Scrambled Eggs", "Eggs Benedict", "Granola Bowl",
        "Breakfast Burrito", "Smoked Salmon Bagel", "Fruit Salad", "Pancakes",
        "French Toast", "Omelette", "Shakshuka", "Labneh Plate",
    ],
    "Sandwiches": [
        "Club Sandwich", "Grilled Cheese", "Tuna Melt", "Chicken Panini",
        "Falafel Wrap", "Caprese Sandwich", "BLT", "Veggie Wrap",
        "Turkey Sandwich", "Kefta Sandwich",
    ],
    "Salads": [
        "Caesar Salad", "Greek Salad", "Quinoa Salad", "Tuna Salad",
        "Caprese Salad", "Lentil Salad", "Chicken Salad", "Mechouia Salad",
    ],
    "Desserts": [
        "Tiramisu", "Cheesecake", "Chocolate Cake", "Lemon Tart",
        "Panna Cotta", "Crème Brûlée", "Fruit Tart", "Kunafa",
        "Ice Cream Scoop", "Rice Pudding",
    ],
    "Juices & Smoothies": [
        "Orange Juice", "Lemon Mint", "Mango Smoothie", "Strawberry Smoothie",
        "Green Detox", "Watermelon Juice", "Avocado Smoothie", "Carrot Ginger",
    ],
    "Soft Drinks": [
        "Sparkling Water", "Still Water", "Cola", "Lemonade", "Iced Chocolate",
        "Ginger Beer",
    ],
}


def get_menu_items(seed: int = None) -> list:
    """Returns a flat list of (category, name, price) tuples covering
    every item in CATEGORIES, each assigned a plausible random price."""
    rng = random.Random(seed)
    items = []
    for category, names in CATEGORIES.items():
        for name in names:
            base = rng.uniform(2.0, 12.0)
            price = round(base, 2)
            items.append((category, name, price))
    return items


def format_price(value: float, rng: random.Random) -> str:
    """Render a numeric price as text the way it might actually be
    handwritten on a menu - mixing formats on purpose:
        - three-decimal TND-style: "12.500"
        - two-decimal EUR-style with comma: "4,50"
        - two-decimal with dot: "4.50"
        - bare integer, no decimals: "5"
        - occasionally with a trailing currency letter: "4.50 DT" / "3,50 €"
    """
    style = rng.choice(["tnd3", "eur_comma", "eur_dot", "integer"])

    if style == "tnd3":
        text = f"{value:.3f}"
    elif style == "eur_comma":
        text = f"{value:.2f}".replace(".", ",")
    elif style == "eur_dot":
        text = f"{value:.2f}"
    else:
        text = str(int(round(value)))

    if rng.random() < 0.25:
        suffix = rng.choice([" DT", " dt", " TND", " €", ""])
        text = f"{text}{suffix}"

    return text
