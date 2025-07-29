import re
from collections import Counter

"""
This script defines a function `clean_transcript` that processes a given text transcript.
It removes specific phrases in parentheses that indicate audience reactions like applause or laughter,
and counts how many times each type of reaction occurs. 
The function returns the cleaned text along with a summary of counts for each reaction type.
"""
def clean_transcript(text):
    # Keywords to look for in parentheses, grouped by category (lowercase for case-insensitive)
    categories = {
        'applause': ['applause', 'applauds', 'clapping', 'cheer', 'cheers', 'cheering'],
        'laughter': ['laugh', 'laughs', 'laughter', 'laughing'],
    }

    # Flatten all keywords to one big pattern for searching inside parentheses
    all_keywords = [kw for kws in categories.values() for kw in kws]
    # Create a regex pattern that matches any of those keywords (word boundaries, ignore case)
    keywords_pattern = re.compile(r'\b(' + '|'.join(all_keywords) + r')\b', re.IGNORECASE)

    # Pattern to find parentheses and their content
    parentheses_pattern = re.compile(r'\([^()]*\)')

    counts = Counter()

    def replacer(match):
        content = match.group(0)  # e.g. "(audience applause)"
        # Check if content contains any keyword
        if keywords_pattern.search(content):
            # Determine which category it belongs to (first matched category)
            content_lower = content.lower()
            for cat, kws in categories.items():
                for kw in kws:
                    if kw in content_lower:
                        counts[cat] += 1
                        break
                else:
                    continue
                break
            return ''  # remove this parentheses and its content
        else:
            return content  # keep as is

    cleaned_text = parentheses_pattern.sub(replacer, text)

    # Add counts summary if any
    if counts:
        for cat, count in counts.items():
            cleaned_text += f' [{cat.upper()}_COUNT={count}]'

    return cleaned_text