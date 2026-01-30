"""Label mappings for mental health classification."""

from typing import Dict


# Label mapping: topic -> label_value
LABEL_MAPPING: Dict[str, int] = {
    # Normal (0)
    "jokes": 0,
    "fitness": 0,
    "relationships": 0,
    "teaching": 0,
    # Mental Health Conditions (1-6)
    "depression": 1,
    "autism": 2,
    "anxiety": 3,
    "ptsd": 4,
    "addiction": 5,
    "alcoholism": 6,
}

# Label names for display
LABEL_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Depression",
    2: "Autism",
    3: "Anxiety",
    4: "PTSD",
    5: "Addiction",
    6: "Alcoholism",
}
