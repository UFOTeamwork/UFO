def validate_aeu(obj):
    required = ["question", "modality", "weight"]

    if not isinstance(obj, dict):
        return False

    for r in required:
        if r not in obj:
            return False

    if len(obj["question"]) < 5:
        return False

    if obj["modality"] not in ["text", "image", "joint"]:
        return False

    return True