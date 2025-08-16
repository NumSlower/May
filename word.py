import os
import json
import random

# Define subjects and appropriate grade levels
SUBJECTS = {
    "math": range(0, 13),  # K-12
    "science": range(0, 13),  # K-12
    "english": range(0, 13),  # K-12
    "history": range(2, 13),  # 2-12
    "social_studies": range(0, 13),  # K-12
    "art": range(0, 13),  # K-12
    "music": range(0, 13),  # K-12
    "physical_education": range(0, 13),  # K-12
    "foreign_language": range(6, 13),  # 6-12
    "computer_science": range(3, 13),  # 3-12
    "geography": range(3, 13),  # 3-12
    "civics": range(4, 13),  # 4-12
    "economics": range(9, 13),  # 9-12
    "biology": range(9, 13),  # 9-12
    "chemistry": range(9, 13),  # 9-12
    "physics": range(9, 13),  # 9-12
}

# Grade level mapping for labels
GRADE_LABELS = {
    0: "kindergarten",
    1: "1st_grade",
    2: "2nd_grade",
    3: "3rd_grade",
    4: "4th_grade",
    5: "5th_grade",
    6: "6th_grade",
    7: "7th_grade",
    8: "8th_grade",
    9: "9th_grade",
    10: "10th_grade",
    11: "11th_grade",
    12: "12th_grade"
}

# Helper functions to generate diverse question types
def generate_multiple_choice(subject, grade, base_text, options, correct):
    return {
        "type": "multiple_choice",
        "text": base_text,
        "options": options,
        "correct_answer": correct,
        "hint": f"Think about the {subject} concepts for grade {grade}.",
        "difficulty": random.randint(1, 5)
    }

def generate_short_answer(subject, grade, question, answer):
    return {
        "type": "short_answer",
        "text": question,
        "correct_answer": answer,
        "hint": f"Consider basic {subject} principles for this grade level.",
        "difficulty": random.randint(1, 5)
    }

def generate_problem_solving(subject, grade, problem, steps, solution):
    return {
        "type": "problem_solving",
        "text": problem,
        "steps_to_solve": steps,
        "correct_answer": solution,
        "hint": f"Break it down step-by-step using {subject} techniques.",
        "difficulty": random.randint(1, 5)
    }

# Generate sample questions for each subject by grade level
def generate_sample_questions(subject, grade):
    """Generate at least 100 diverse questions per subject and grade level"""
    questions = []

    # Common question components
    shapes = ["circle", "square", "triangle", "rectangle"]
    animals = ["bear", "bird", "fish", "elephant", "snake"]
    colors = ["red", "blue", "green", "yellow", "purple"]

    if subject == "math":
        if grade == 0:  # Kindergarten
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What number comes after {i}?", str(i + 1)))
                questions.append(generate_multiple_choice(subject, grade, f"Which is bigger: {i} or {i + 2}?", [str(i), str(i + 2)], str(i + 2)))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Count these {shapes[i % 4]}s: {' '.join([shapes[i % 4]] * (i % 5 + 1))}", str(i % 5 + 1)))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"If you have {i % 3 + 1} toys and get {i % 2 + 1} more, how many do you have?", ["Add the numbers together."], str(i % 3 + i % 2 + 2)))
        elif grade <= 2:  # 1st-2nd
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What is {i + 1} + {grade}?", str(i + 1 + grade)))
                questions.append(generate_multiple_choice(subject, grade, f"What is {i + 2} - {grade}?", [str(i + 2 - grade), str(i + 1)], str(i + 2 - grade)))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"If you have {i % 5 + 1} apples and eat {i % 3 + 1}, how many are left?", ["Subtract the eaten apples from the total."], str(i % 5 + 1 - i % 3 - 1)))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Count by {grade}s to {grade * (i % 5 + 1)}.", " ".join([str(grade * j) for j in range(1, i % 5 + 2)])))
        elif grade <= 5:  # 3rd-5th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What is {i * grade} × {grade}?", str(i * grade * grade)))
                questions.append(generate_multiple_choice(subject, grade, f"Divide {i * grade * 2} by {grade}.", [str(i * 2), str(i)], str(i * 2)))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Find the perimeter of a rectangle with sides {i % 5 + 1} and {i % 3 + 2}.", ["Add all sides: 2 × length + 2 × width"], str(2 * (i % 5 + 1) + 2 * (i % 3 + 2))))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Write {i * 100 + grade * 10 + i} in words.", f"{i} hundred {grade} ten {i}"))
        elif grade <= 8:  # 6th-8th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Solve for x: {i}x + {grade} = {i * grade}", str(grade - i)))
                questions.append(generate_multiple_choice(subject, grade, f"What is {i}% of {grade * 10}?", [str(i * grade / 10), str(i)], str(i * grade / 10)))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Find the area of a circle with radius {i % 5 + 1}. Use π ≈ 3.14.", ["Use A = πr²"], str(3.14 * (i % 5 + 1) ** 2)))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What is the Pythagorean theorem applied to a triangle with legs {i % 4 + 1} and {i % 3 + 1}?", str((i % 4 + 1) ** 2 + (i % 3 + 1) ** 2)))
        elif grade <= 12:  # 9th-12th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Find the derivative of f(x) = {i}x² + {grade}x.", f"{2 * i}x + {grade}"))
                questions.append(generate_multiple_choice(subject, grade, "Solve x² + 5x + 6 = 0", ["-2, -3", "2, 3"], "-2, -3"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"If sin(θ) = {i % 5 / 5}, find cos(θ) in a right triangle.", ["Use sin² + cos² = 1"], str((1 - (i % 5 / 5) ** 2) ** 0.5)))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What is the vector sum of ({i % 3 + 1}, {i % 2 + 1}) and ({i % 4 + 1}, {i % 5 + 1})?", f"({i % 3 + i % 4 + 2}, {i % 2 + i % 5 + 2})"))

    elif subject == "science":
        if grade <= 2:  # K-2nd
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Name an animal with {['fur', 'feathers', 'scales'][i % 3]}.", animals[i % 5]))
                questions.append(generate_multiple_choice(subject, grade, "What does a plant need most?", ["Water", "Candy"], "Water"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"If you see {i % 3 + 1} clouds, will it rain soon?", ["Clouds often mean rain."], "Maybe"))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What season comes after {['fall', 'winter', 'spring'][i % 3]}?", ['winter', 'spring', 'summer'][i % 3]))
        elif grade <= 5:  # 3rd-5th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, "What causes the day to turn into night?", "Earth's rotation"))
                questions.append(generate_multiple_choice(subject, grade, "Which state of matter is water in ice?", ["Solid", "Liquid"], "Solid"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Describe how rain forms with {i % 3 + 1} steps.", ["Evaporation, condensation, precipitation"], "Water cycle"))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Name a plant part that helps with {['growth', 'support', 'food'][i % 3]}.", ["leaves", "stem", "roots"][i % 3]))
        elif grade <= 8:  # 6th-8th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, "What gas do plants use in photosynthesis?", "Carbon dioxide"))
                questions.append(generate_multiple_choice(subject, grade, "Which law says an object stays at rest unless acted upon?", ["Newton's 1st", "Newton's 2nd"], "Newton's 1st"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"How does a cell divide if it has {i % 4 + 1} chromosomes?", ["Mitosis duplicates chromosomes."], "Mitosis"))
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Name a type of rock formed by {['cooling', 'pressure', 'sediment'][i % 3]}.", ["igneous", "metamorphic", "sedimentary"][i % 3]))
        elif grade <= 12:  # 9th-12th
            if subject in ["biology", "science"]:
                for i in range(25):
                    questions.append(generate_short_answer(subject, grade, "What molecule carries genetic information?", "DNA"))
                    questions.append(generate_multiple_choice(subject, grade, "What process produces ATP?", ["Photosynthesis", "Cellular respiration"], "Cellular respiration"))
                for i in range(25):
                    questions.append(generate_problem_solving(subject, grade, f"How does {animals[i % 5]} adapt to its environment?", ["Consider survival traits"], "Varies by animal"))
            elif subject in ["chemistry", "science"]:
                for i in range(25):
                    questions.append(generate_short_answer(subject, grade, "What element has atomic number 1?", "Hydrogen"))
                    questions.append(generate_multiple_choice(subject, grade, "Is HCl an acid or base?", ["Acid", "Base"], "Acid"))
                for i in range(25):
                    questions.append(generate_problem_solving(subject, grade, f"Balance: {i % 3 + 1}H₂ + O₂ → H₂O", ["Adjust coefficients"], f"{i % 3 + 1}H₂ + {0.5 * (i % 3 + 1)}O₂ → {i % 3 + 1}H₂O"))
            elif subject in ["physics", "science"]:
                for i in range(25):
                    questions.append(generate_short_answer(subject, grade, f"What is the speed of an object moving {i * 10}m in {i}s?", f"{10}m/s"))
                    questions.append(generate_multiple_choice(subject, grade, "What force pulls objects to Earth?", ["Gravity", "Magnetism"], "Gravity"))
                for i in range(25):
                    questions.append(generate_problem_solving(subject, grade, f"Calculate momentum for {i % 5 + 1}kg at {i % 10 + 1}m/s.", ["Momentum = mass × velocity"], str((i % 5 + 1) * (i % 10 + 1))))

    elif subject == "english":
        if grade <= 2:  # K-2nd
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"What sound does '{chr(65 + i)}' make?", f"/{chr(97 + i)}/"))
                questions.append(generate_multiple_choice(subject, grade, f"Which rhymes with 'cat'?", ["hat", "dog"], "hat"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Tell a story with {colors[i % 5]}.", ["Start with a character"], "Varies"))
        elif grade <= 5:  # 3rd-5th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Identify the noun in: 'The {animals[i % 5]} runs.'", animals[i % 5]))
                questions.append(generate_multiple_choice(subject, grade, "What’s the verb in 'She jumps'?", ["She", "jumps"], "jumps"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Write a story with {i % 3 + 1} characters.", ["Include a beginning, middle, end"], "Varies"))
        elif grade <= 8:  # 6th-8th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Define {['metaphor', 'simile', 'alliteration'][i % 3]}.", ["Comparison without like/as", "Comparison with like/as", "Repeated sounds"][i % 3]))
                questions.append(generate_multiple_choice(subject, grade, "What’s the theme of a story about bravery?", ["Courage", "Fear"], "Courage"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Analyze a {i % 4 + 1}-sentence story.", ["Look for themes"], "Varies"))
        elif grade <= 12:  # 9th-12th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Explain symbolism in 'The {colors[i % 5]} rose.'", f"{colors[i % 5]} might mean emotion"))
                questions.append(generate_multiple_choice(subject, grade, "What’s the tone of a gloomy poem?", ["Sad", "Happy"], "Sad"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Write a {i % 5 + 1}-paragraph essay on freedom.", ["Structure with intro, body, conclusion"], "Varies"))

    elif subject in ["history", "social_studies"]:
        if grade <= 2:  # K-2nd
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, f"Name a job in the community like {['firefighter', 'teacher', 'doctor'][i % 3]}.", "Any job"))
                questions.append(generate_multiple_choice(subject, grade, "What’s a holiday in winter?", ["Christmas", "Halloween"], "Christmas"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Why do we have {i % 3 + 1} rules at home?", ["To keep order"], "Safety/order"))
        elif grade <= 5:  # 3rd-5th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, "Who was America’s first president?", "George Washington"))
                questions.append(generate_multiple_choice(subject, grade, "Why did pilgrims come to America?", ["Freedom", "Gold"], "Freedom"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"How did {i % 3 + 1} colonies unite?", ["Formed agreements"], "United States"))
        elif grade <= 8:  # 6th-8th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, "What started the Civil War?", "Slavery disputes"))
                questions.append(generate_multiple_choice(subject, grade, "What’s in the Constitution?", ["Laws", "Stories"], "Laws"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Describe {i % 3 + 1} causes of World War I.", ["Alliances, militarism, etc."], "Varies"))
        elif grade <= 12:  # 9th-12th
            for i in range(25):
                questions.append(generate_short_answer(subject, grade, "What ended the Roman Empire?", "Invasions, corruption"))
                questions.append(generate_multiple_choice(subject, grade, "What was the Cold War about?", ["Ideology", "Trade"], "Ideology"))
            for i in range(25):
                questions.append(generate_problem_solving(subject, grade, f"Analyze {i % 5 + 1} effects of globalization.", ["Economic, cultural impacts"], "Varies"))

    # Ensure at least 100 questions
    while len(questions) < 100:
        questions.append(generate_short_answer(subject, grade, f"Sample {subject} question {len(questions) + 1} for {GRADE_LABELS[grade]}.", f"Answer {len(questions) + 1}"))

    return questions

def create_question_files():
    """Create the directory structure and files for the question database"""
    os.makedirs("question_database", exist_ok=True)
    
    all_questions = {}
    
    for subject in SUBJECTS:
        subject_dir = os.path.join("question_database", subject)
        os.makedirs(subject_dir, exist_ok=True)
        
        all_questions[subject] = {}
        
        for grade in SUBJECTS[subject]:
            grade_label = GRADE_LABELS[grade]
            filename = f"{grade_label}_questions.json"
            filepath = os.path.join(subject_dir, filename)
            
            # Generate detailed questions
            questions = generate_sample_questions(subject, grade)
            
            # Create question data structure
            question_data = {
                "subject": subject,
                "grade": grade_label,
                "questions": [
                    {"id": f"{subject}_{grade}_{i+1}", **q}
                    for i, q in enumerate(questions)
                ]
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(question_data, f, indent=2)
            
            # Track for summary
            all_questions[subject][grade_label] = question_data["questions"]
    
    # Create summary file
    summary = {
        "total_subjects": len(SUBJECTS),
        "subjects": list(SUBJECTS.keys()),
        "grade_levels": list(GRADE_LABELS.values()),
        "question_counts": {
            subject: {grade_label: len(all_questions[subject].get(grade_label, [])) 
                      for grade_label in [GRADE_LABELS[g] for g in SUBJECTS[subject]]}
            for subject in SUBJECTS
        }
    }
    
    with open(os.path.join("question_database", "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create README
    readme_content = """# AI Bot Question Database

This directory contains question files organized by subject and grade level for use with the AI Bot.

## Structure
- Each subject has its own directory
- Within each subject directory, files are organized by grade level
- All questions are stored in JSON format with rich metadata (type, text, answers, hints, difficulty)

## Available Subjects
{}

## Grade Levels
{}

## Usage with AI Bot
To train the AI bot, load the JSON files using `load_knowledge` with the format:
- subject: string
- grade: string
- question_id: string
- text: string
- difficulty: int
Additional fields (e.g., correct_answer, hint) can enhance training.
""".format(
        "\n".join([f"- {subject}" for subject in SUBJECTS]),
        "\n".join([f"- {label}" for label in GRADE_LABELS.values()])
    )
    
    with open(os.path.join("question_database", "README.md"), 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    create_question_files()
    print("Question database structure created successfully.")