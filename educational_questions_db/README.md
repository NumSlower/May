# Educational Question Database

This database contains educational questions organized by subject and grade level for AI chatbot training.

## Database Structure

```
educational_questions_db/
├── database_summary.json          # Summary statistics
├── README.md                      # This file
├── mathematics/                   # Math questions
│   ├── kindergarten_questions.json
│   ├── first_grade_questions.json
│   └── ...
├── science/                       # Science questions
├── english_language_arts/         # ELA questions
├── social_studies/               # Social studies questions
├── computer_science/             # CS questions
└── foreign_languages/            # Language questions
```

## Available Subjects

- **Mathematics**: Grades 0-12
- **Science**: Grades 0-12
- **English Language Arts**: Grades 0-12
- **Social Studies**: Grades 2-12
- **Computer Science**: Grades 3-12
- **Foreign Languages**: Grades 6-12

## Question Format

Each question file contains:
```json
{
  "subject": "mathematics",
  "grade": "fifth_grade",
  "question_count": 150,
  "questions": [
    {
      "id": "mathematics_5_1234",
      "subject": "mathematics",
      "grade": "fifth_grade",
      "topic": "arithmetic",
      "difficulty": 3,
      "type": "generated",
      "question": "What is 15 + 27?",
      "answer": "42",
      "variables": {"a": 15, "b": 27, "result": 42}
    }
  ]
}
```

## Integration with AI Bot

To use with the C AI bot:
1. Load JSON files using a JSON parser (jansson recommended)
2. Map to KnowledgeEntry structure:
   - subject → subject
   - grade → grade  
   - id → question_id
   - question → text
   - difficulty → difficulty

## Generated Content

- **Total Questions**: ~15,000+ across all subjects and grades
- **Question Types**: Multiple choice, short answer, problem solving
- **Difficulty Levels**: 1 (easiest) to 5 (hardest)
- **Topics**: Subject-specific topics with grade-appropriate content

Generated on: 1755347208.9249165
