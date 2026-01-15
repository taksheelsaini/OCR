"""
My Custom Grocery Items Dictionary for Spell Correction!
I manually curated this list of common grocery items, vegetables, fruits, dairy, and meat 
so my hybrid OCR engine actually knows what to look for when correcting spelling mistakes.
"""

GROCERY_ITEMS = [
    # Vegetables
    "cabbage", "carrots", "carrot", "onion", "onions", "tomatoes", "tomato",
    "potato", "potatoes", "cucumber", "lettuce", "iceberg lettuce", "spinach",
    "broccoli", "cauliflower", "green beans", "beans", "peas", "corn",
    "bell pepper", "red pepper", "green pepper", "capsicum", "celery",
    "mushrooms", "mushroom", "zucchini", "eggplant", "garlic", "ginger",
    "beetroot", "radish", "turnip", "pumpkin", "squash", "asparagus",
    "artichoke", "kale", "brussels sprouts", "leek", "shallots",
    
    # Fruits
    "apple", "apples", "banana", "bananas", "orange", "oranges", "grapes",
    "strawberry", "strawberries", "blueberry", "blueberries", "mango", "mangoes",
    "pineapple", "watermelon", "melon", "kiwi", "peach", "peaches", "pear", "pears",
    "plum", "plums", "cherry", "cherries", "lemon", "lemons", "lime", "limes",
    "papaya", "guava", "pomegranate", "coconut", "avocado", "fig", "dates",
    
    # Dairy
    "milk", "cheese", "butter", "yogurt", "yoghurt", "greek yogurt", "cream",
    "heavy cream", "sour cream", "cottage cheese", "cream cheese", "mozzarella",
    "cheddar", "parmesan", "feta cheese", "paneer", "ghee", "buttermilk",
    "whipped cream", "ice cream", "eggs", "egg",
    
    # Meat & Poultry
    "chicken", "chicken breast", "chicken thighs", "beef", "pork", "lamb",
    "mutton", "fish", "salmon", "tuna", "shrimp", "prawns", "crab", "lobster",
    "bacon", "ham", "sausage", "sausages", "turkey", "duck", "mince", "ground beef",
    
    # Grains & Cereals
    "rice", "brown rice", "basmati rice", "white rice", "sona masoori rice",
    "wheat", "flour", "all purpose flour", "wheat flour", "bajra flour",
    "bread", "whole wheat bread", "tortillas", "low carb tortillas", "naan",
    "pasta", "spaghetti", "noodles", "oats", "oatmeal", "quinoa", "barley",
    "corn flakes", "cereal", "muesli", "granola",
    
    # Pulses & Legumes
    "lentils", "dal", "toor dal", "moong dal", "chana dal", "masoor dal",
    "chickpeas", "kidney beans", "rajma", "black beans", "lobia", "pinto beans",
    "soybeans", "peanuts", "almonds", "cashews", "walnuts", "pistachios",
    
    # Spices & Condiments
    "salt", "pepper", "black pepper", "turmeric", "turmeric powder", "cumin",
    "coriander", "chili powder", "red chili", "paprika", "smoked paprika",
    "cinnamon", "cardamom", "cloves", "nutmeg", "oregano", "basil", "thyme",
    "rosemary", "bay leaves", "curry powder", "garam masala", "mustard",
    "yellow mustard", "ketchup", "mayonnaise", "mayo", "light mayo",
    "soy sauce", "oyster sauce", "bbq sauce", "hot sauce", "vinegar",
    "olive oil", "vegetable oil", "coconut oil", "sesame oil",
    "honey", "sugar", "brown sugar", "jaggery", "maple syrup",
    "tomato paste", "tomato sauce", "garlic powder", "onion powder",
    
    # Beverages
    "water", "juice", "orange juice", "apple juice", "coffee", "tea",
    "green tea", "milk tea", "soda", "cola", "energy drink", "smoothie",
    
    # Snacks & Others
    "chips", "biscuits", "cookies", "crackers", "popcorn", "nuts",
    "chocolate", "candy", "jam", "peanut butter", "nutella",
    
    # Household quantities
    "packet", "packets", "bottle", "bottles", "can", "cans", "box", "boxes",
    "bag", "bags", "piece", "pieces", "pcs", "dozen", "kg", "g", "gram", "grams",
    "ml", "liter", "litre", "L",
]

# I realized OCR engines often mess up capitalization, so I generate a completely lowercase version 
# of my dictionary here to make the fuzzy matching much more reliable.
GROCERY_ITEMS_LOWER = [item.lower() for item in GROCERY_ITEMS]
