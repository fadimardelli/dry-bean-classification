import json

def classification_report_to_file(report, file_name):
    # Use .get() to safely retrieve the accuracy key, 
    # defaulting to None if the key is missing.
    accuracy_value = report.get("accuracy")

    # If accuracy is missing, you might set it to a placeholder (e.g., 'N/A')
    # or raise a more informative error, but for saving, we'll try to use the key.
    if accuracy_value is None:
        print(f"WARNING: 'accuracy' key missing from classification report for {file_name}. "
              "Check scikit-learn version or data integrity.")
        # We can still proceed by excluding the accuracy key, or use a placeholder
        accuracy_to_save = "N/A"
    else:
        accuracy_to_save = accuracy_value

    # We save the overall 'accuracy' and the 'weighted avg' for the other metrics
    results = {
        # Use the safely retrieved accuracy value
        "accuracy": accuracy_to_save, 
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"]
    }

    # 4. Save to a file
    # We remove the hardcoded folder path for simplicity unless you ensure the folder exists
    try:
        with open(f"classification_report/{file_name}.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to 'classification_report/{file_name}.json'")
    except FileNotFoundError:
        # Fallback if the 'classification_report/' directory doesn't exist
        with open(f"{file_name}.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to '{file_name}.json' (Note: Directory 'classification_report/' was not found)")