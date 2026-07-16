import os
import json
from demo import Hand

os.makedirs('dataset', exist_ok=True)
hand = Hand()

menu = {
    "Coffee": [
        ("Espresso", 2.50), ("Americano", 3.00), ("Cappuccino", 3.75),
        ("Latte", 4.00), ("Flat White", 4.25), ("Mocha", 4.50),
        ("Macchiato", 3.50), ("Cold Brew", 4.75), ("Iced Latte", 4.50),
        ("Affogato", 5.25),
    ],
    "Pastries": [
        ("Croissant", 3.00), ("Pain au Chocolat", 3.50), ("Blueberry Muffin", 3.25),
        ("Cinnamon Roll", 4.00), ("Almond Tart", 4.50), ("Banana Bread", 3.75),
        ("Cheese Danish", 3.90), ("Scone", 3.20), ("Chocolate Brownie", 4.10),
        ("Apple Turnover", 3.80),
    ],
    "Breakfast": [
        ("Avocado Toast", 6.50), ("Scrambled Eggs", 5.00), ("Eggs Benedict", 7.50),
        ("Granola Bowl", 5.50), ("Breakfast Burrito", 7.00), ("Smoked Salmon Bagel", 8.25),
        ("Fruit Salad", 4.50), ("Pancakes", 6.75), ("French Toast", 6.90),
        ("Omelette", 6.25),
    ],
}

STYLE = 12
BIAS = 0.85

# build flattened lines: category header + items with dotted leaders
lines = []
labels = []

for category, items in menu.items():
    lines.append(category.upper())
    labels.append({"type": "category", "text": category})

    for name, price in items:
        dots_needed = max(3, 40 - len(name) - len(f"{price:.2f}"))
        line = f"{name} {'.' * dots_needed} {price:.2f}"
        lines.append(line)
        labels.append({"type": "item", "name": name, "price": price})

biases = [BIAS] * len(lines)
styles = [STYLE] * len(lines)

hand.write(
    filename='dataset/full_menu.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_widths=[1.2] * len(lines),
)

with open('dataset/full_menu_labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

print(f"Generated {len(lines)} lines -> dataset/full_menu.svg")
print("Labels saved -> dataset/full_menu_labels.json")