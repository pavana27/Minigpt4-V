#The program reads a JSON file containing video engagement predictions, processes the predictions to determine engagement levels, and writes the results to a new JSON file. It also compares the processed results to ground truth data and generates a comparison report. Using Pydantic in this program provides several benefits, including data validation, type safety, default values, data transformation, and structured data handling. 
# These features help ensure that the input data conforms to the expected schema
import json
from pydantic import BaseModel, Field
from typing import List

# Define the Pydantic schema for the input JSON
class PredictionSchema(BaseModel):
    video_name: str
    pred: List[str] = Field(default_factory=list)
    pred1: List[str] = Field(default_factory=list)
    pred2: List[str] = Field(default_factory=list)

class InputSchema(BaseModel):
    entries: List[PredictionSchema]

# Define the Pydantic schema for the output JSON
class OutputSchema(BaseModel):
    video_name: str
    label: str

# Define the mapping of engagement levels
engagement_levels = ["Not Engaged", "Barely Engaged", "Engaged", "Highly Engaged"]

# Function to determine the engagement level from predictions
def determine_engagement(predictions):
    for pred in predictions:
        if "fully engaged" in pred.lower() or "focused" in pred.lower():
            return "Highly Engaged"
        if "engaging" in pred.lower():
            return "Engaged"
    for level in engagement_levels:
        for pred in predictions:
            if level.lower() in pred.lower():
                return level
    return "Unknown"

# Read the JSON file
with open('/home/pavana/Desktop/minigpt4_finetune_test_config_eval.json', 'r') as file:
    data = json.load(file)

# Parse the input JSON using Pydantic schema
input_data = InputSchema(entries=data)

# Transform the data to the desired output format
output_data = []
for entry in input_data.entries:
    video_name = entry.video_name
    predictions = entry.pred + entry.pred1 + entry.pred2
    activity_type = determine_engagement(predictions)
    output_entry = OutputSchema(video_name=video_name, label=activity_type)
    output_data.append(output_entry)

# Write the output JSON to a file
with open('/home/pavana/Desktop/parsed_engagement_levels.json', 'w') as file:
    json.dump([entry.dict() for entry in output_data], file, indent=4)

print("New JSON file created: parsed_engagement_levels.json")

"""
# Read the ground truth data
with open('path_to_ground_truth_file.json', 'r') as file:
    ground_truth_data = json.load(file)

# Compare the parsed output to the ground truth
matches = 0
mismatches = 0
for parsed_entry in output_data:
    for ground_truth_entry in ground_truth_data:
        if parsed_entry.video_name == ground_truth_entry['video_name']:
            if parsed_entry.label == ground_truth_entry['label']:
                matches += 1
            else:
                mismatches += 1
            break

# Generate a comparison report
print(f"Total matches: {matches}")
print(f"Total mismatches: {mismatches}")
"""