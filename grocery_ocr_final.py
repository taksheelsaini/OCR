"""
This is my Final Hybrid OCR Solution for Handwritten Grocery Lists!
I built this engine to combine EasyOCR's text detection with my intelligent spell correction dictionary.
"""
import easyocr
import os
import re
from rapidfuzz import fuzz, process
from grocery_dictionary import GROCERY_ITEMS_LOWER

class GroceryOCR:
    def __init__(self):
        print("🔄 Loading OCR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.grocery_items = GROCERY_ITEMS_LOWER
        print("✅ Ready!\n")
    
    def find_best_match(self, text, threshold=50):
        """
        This is my fuzzy matching function. It takes whatever messy text the OCR found 
        and tries to find the closest matching real grocery item from my dictionary.
        """
        # First I clean up the text by removing weird punctuation to give the matcher a fair chance
        text_clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        if len(text_clean) < 3:
            return None, 0
        
        # If it's an exact match right away, awesome! I return a 100% confidence score.
        if text_clean in self.grocery_items:
            return text_clean, 100
        
        # If not, I use RapidFuzz to do a fuzzy string comparison against everything in my dictionary
        match = process.extractOne(
            text_clean,
            self.grocery_items,
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if match:
            return match[0], match[1]
        
        return None, 0
    
    def extract_grocery_list(self, image_path, min_confidence=0.40, min_match_score=75):
        """
        This is the main function I use to actually extract the list of grocery items from an image.
        """
        if not os.path.exists(image_path):
            print(f"❌ Error: File not found - {image_path}")
            return []
        
        # I fire up EasyOCR here to read all the raw text from the image
        results = self.reader.readtext(
            image_path,
            paragraph=False,
            min_size=10,
            text_threshold=0.4,
            low_text=0.3,
        )
        
        # EasyOCR sometimes reads things out of order, so I explicitly sort the text boxes top-to-bottom 
        # using their Y-coordinates to make sure the list stays in order
        results_sorted = sorted(results, key=lambda x: x[0][0][1])
        
        # Now I process the results and filter out the junk
        items = []
        seen = set()
        
        for bbox, text, confidence in results_sorted:
            if confidence < min_confidence:
                continue
            
            # I pass the messy text into my fuzzy matcher to see if it's a real grocery item
            match, score = self.find_best_match(text)
            
            if match and match not in seen:
                seen.add(match)
                items.append({
                    'original': text,
                    'item': match,
                    'confidence': confidence,
                    'match_score': score
                })
        
        return items
    
    def process_image(self, image_path):
        """
        I use this function to wrap everything up nicely and print a formatted summary to the terminal.
        """
        print(f"\n{'='*60}")
        print(f"📷 {os.path.basename(image_path)}")
        print('='*60)
        
        items = self.extract_grocery_list(image_path)
        
        if not items:
            print("No grocery items detected")
            return []
        
        print(f"\n📋 DETECTED GROCERY ITEMS:")
        print("-" * 45)
        print(f"{'#':<4} {'Item':<20} {'OCR Conf':<10} {'Match':<8}")
        print("-" * 45)
        
        for i, item in enumerate(items, 1):
            conf = item['confidence']
            match = item['match_score']
            
            # I built this little quality indicator so I can easily spot which items I might need to double-check
            if conf > 0.6 and match > 70:
                indicator = "✅"
            elif conf > 0.3 or match > 60:
                indicator = "⚠️"
            else:
                indicator = "❓"
            
            print(f"{i:<4} {item['item']:<20} {conf:.0%}{'':>4} {match}%{'':>2} {indicator}")
        
        print("-" * 45)
        print(f"Total: {len(items)} items")
        
        return [item['item'] for item in items]


def main():
    ocr = GroceryOCR()
    
    # Test images
    test_images = [
        os.path.expanduser('~/Downloads/1.jpeg'),
        os.path.expanduser('~/Downloads/2.jpg')
    ]
    
    all_items = []
    for img_path in test_images:
        if os.path.exists(img_path):
            items = ocr.process_image(img_path)
            all_items.extend(items)
    
    # Just printing out my nice final formatted list
    print("\n" + "=" * 60)
    print("🛒 COMPLETE SHOPPING LIST")
    print("=" * 60)
    
    unique_items = list(dict.fromkeys(all_items))  # Remove duplicates, preserve order
    
    for i, item in enumerate(unique_items, 1):
        print(f"  {i:2}. {item.title()}")
    
    print(f"\n📝 Total unique items: {len(unique_items)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
