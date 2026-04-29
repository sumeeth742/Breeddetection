def calculate_score(length, height):
    score = 0

    if length > 140:
        score += 5
    elif length > 120:
        score += 3

    if height > 130:
        score += 5
    elif height > 110:
        score += 3

    return score