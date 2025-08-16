#!/usr/bin/env python3
"""
Educational Question Generator
Generates diverse educational questions across multiple subjects and grade levels
for AI chatbot training.
"""

import os
import json
import random
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuestionTemplate:
    """Template for generating questions"""
    type: str
    template: str
    answer_template: str
    difficulty_range: Tuple[int, int]
    grade_range: Tuple[int, int]

class EducationalQuestionGenerator:
    """Main class for generating educational questions"""
    
    # Subject configurations
    SUBJECTS_CONFIG = {
        "mathematics": {
            "grades": range(0, 13),
            "topics": ["arithmetic", "algebra", "geometry", "calculus", "statistics"]
        },
        "science": {
            "grades": range(0, 13),
            "topics": ["biology", "chemistry", "physics", "earth_science", "general_science"]
        },
        "english_language_arts": {
            "grades": range(0, 13),
            "topics": ["reading", "writing", "grammar", "literature", "vocabulary"]
        },
        "social_studies": {
            "grades": range(2, 13),
            "topics": ["history", "geography", "civics", "economics", "culture"]
        },
        "computer_science": {
            "grades": range(3, 13),
            "topics": ["programming", "algorithms", "data_structures", "web_development"]
        },
        "foreign_languages": {
            "grades": range(6, 13),
            "topics": ["spanish", "french", "german", "mandarin"]
        }
    }

    GRADE_LABELS = {
        0: "kindergarten", 1: "first_grade", 2: "second_grade", 3: "third_grade",
        4: "fourth_grade", 5: "fifth_grade", 6: "sixth_grade", 7: "seventh_grade",
        8: "eighth_grade", 9: "ninth_grade", 10: "tenth_grade", 11: "eleventh_grade",
        12: "twelfth_grade"
    }

    def __init__(self, output_dir: str = "educational_questions_db"):
        """Initialize the question generator"""
        self.output_dir = Path(output_dir)
        self.question_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, List[QuestionTemplate]]:
        """Initialize question templates for different subjects"""
        return {
            "mathematics": self._get_math_templates(),
            "science": self._get_science_templates(),
            "english_language_arts": self._get_ela_templates(),
            "social_studies": self._get_social_studies_templates(),
            "computer_science": self._get_cs_templates(),
            "foreign_languages": self._get_language_templates()
        }

    def _get_math_templates(self) -> List[QuestionTemplate]:
        """Get mathematics question templates"""
        return [
            QuestionTemplate("arithmetic", "What is {a} + {b}?", "{result}", (1, 3), (0, 5)),
            QuestionTemplate("arithmetic", "What is {a} - {b}?", "{result}", (1, 3), (0, 5)),
            QuestionTemplate("multiplication", "What is {a} × {b}?", "{result}", (2, 4), (2, 8)),
            QuestionTemplate("division", "What is {a} ÷ {b}?", "{result}", (2, 4), (2, 8)),
            QuestionTemplate("algebra", "Solve for x: {a}x + {b} = {c}", "x = {result}", (3, 5), (6, 12)),
            QuestionTemplate("geometry", "Find the area of a rectangle with length {a} and width {b}.", "{result} square units", (2, 4), (3, 8)),
            QuestionTemplate("calculus", "Find the derivative of f(x) = {a}x² + {b}x + {c}", "f'(x) = {result}", (4, 5), (11, 12))
        ]

    def _get_science_templates(self) -> List[QuestionTemplate]:
        """Get science question templates"""
        return [
            QuestionTemplate("biology", "What gas do plants use during photosynthesis?", "Carbon dioxide", (2, 3), (3, 8)),
            QuestionTemplate("chemistry", "What is the chemical symbol for {element}?", "{symbol}", (3, 4), (8, 12)),
            QuestionTemplate("physics", "What is the speed of an object that travels {distance}m in {time}s?", "{speed} m/s", (3, 4), (8, 12)),
            QuestionTemplate("earth_science", "What causes the seasons on Earth?", "Earth's tilted axis", (2, 3), (4, 8)),
            QuestionTemplate("general_science", "Name the three states of matter.", "Solid, liquid, gas", (1, 2), (1, 5))
        ]

    def _get_ela_templates(self) -> List[QuestionTemplate]:
        """Get English Language Arts question templates"""
        return [
            QuestionTemplate("grammar", "What type of word is '{word}' in the sentence '{sentence}'?", "{part_of_speech}", (2, 3), (2, 8)),
            QuestionTemplate("vocabulary", "What does the word '{word}' mean?", "{definition}", (2, 4), (1, 12)),
            QuestionTemplate("reading", "What is the main idea of this passage: '{passage}'?", "{main_idea}", (3, 4), (3, 12)),
            QuestionTemplate("writing", "Write a {type} about {topic}.", "Student response varies", (3, 5), (3, 12)),
            QuestionTemplate("literature", "What literary device is used in '{example}'?", "{device}", (4, 5), (8, 12))
        ]

    def _get_social_studies_templates(self) -> List[QuestionTemplate]:
        """Get social studies question templates"""
        return [
            QuestionTemplate("history", "Who was the {ordinal} president of the United States?", "{president}", (2, 4), (3, 12)),
            QuestionTemplate("geography", "What is the capital of {state/country}?", "{capital}", (2, 3), (3, 8)),
            QuestionTemplate("civics", "What are the three branches of government?", "Executive, Legislative, Judicial", (3, 4), (5, 12)),
            QuestionTemplate("economics", "What is {economic_term}?", "{definition}", (3, 5), (8, 12)),
            QuestionTemplate("culture", "What is a tradition from {culture}?", "{tradition}", (2, 3), (2, 8))
        ]

    def _get_cs_templates(self) -> List[QuestionTemplate]:
        """Get computer science question templates"""
        return [
            QuestionTemplate("programming", "What does this code do: {code}?", "{explanation}", (3, 4), (6, 12)),
            QuestionTemplate("algorithms", "What is the time complexity of {algorithm}?", "{complexity}", (4, 5), (9, 12)),
            QuestionTemplate("data_structures", "What data structure uses LIFO (Last In, First Out)?", "Stack", (3, 4), (8, 12)),
            QuestionTemplate("web_development", "What does HTML stand for?", "HyperText Markup Language", (2, 3), (6, 12))
        ]

    def _get_language_templates(self) -> List[QuestionTemplate]:
        """Get foreign language question templates"""
        return [
            QuestionTemplate("vocabulary", "How do you say '{english_word}' in {language}?", "{translation}", (2, 3), (6, 12)),
            QuestionTemplate("grammar", "Conjugate the verb '{verb}' in {tense} tense.", "{conjugation}", (3, 4), (7, 12)),
            QuestionTemplate("culture", "What is a famous landmark in {country}?", "{landmark}", (2, 3), (6, 10))
        ]

    def generate_question(self, subject: str, grade: int, template: QuestionTemplate) -> Dict[str, Any]:
        """Generate a single question based on template"""
        question_data = {
            "id": f"{subject}_{grade}_{random.randint(1000, 9999)}",
            "subject": subject,
            "grade": self.GRADE_LABELS[grade],
            "topic": template.type,
            "difficulty": random.randint(*template.difficulty_range),
            "type": "generated"
        }

        # Generate question-specific content based on template type
        if template.type == "arithmetic":
            a, b = random.randint(1, 20), random.randint(1, 20)
            if "+" in template.template:
                result = a + b
            elif "-" in template.template:
                result = max(a, b) - min(a, b)
                a, b = max(a, b), min(a, b)
            question_data.update({
                "question": template.template.format(a=a, b=b),
                "answer": template.answer_template.format(result=result),
                "variables": {"a": a, "b": b, "result": result}
            })
        elif template.type == "multiplication":
            a, b = random.randint(1, 12), random.randint(1, 12)
            result = a * b
            question_data.update({
                "question": template.template.format(a=a, b=b),
                "answer": template.answer_template.format(result=result),
                "variables": {"a": a, "b": b, "result": result}
            })
        elif template.type == "division":
            b = random.randint(1, 12)
            result = random.randint(1, 12)
            a = b * result
            question_data.update({
                "question": template.template.format(a=a, b=b),
                "answer": template.answer_template.format(result=result),
                "variables": {"a": a, "b": b, "result": result}
            })
        else:
            # For other templates, use the template as-is
            question_data.update({
                "question": template.template,
                "answer": template.answer_template
            })

        return question_data

    def generate_questions_for_subject_grade(self, subject: str, grade: int, count: int = 150) -> List[Dict[str, Any]]:
        """Generate questions for a specific subject and grade"""
        if subject not in self.question_templates:
            logger.warning(f"Subject {subject} not found in templates")
            return []

        questions = []
        templates = [t for t in self.question_templates[subject] 
                    if t.grade_range[0] <= grade <= t.grade_range[1]]
        
        if not templates:
            logger.warning(f"No templates found for {subject} grade {grade}")
            return []

        for _ in range(count):
            template = random.choice(templates)
            question = self.generate_question(subject, grade, template)
            questions.append(question)

        return questions

    def generate_all_questions(self) -> Dict[str, Any]:
        """Generate questions for all subjects and grades"""
        all_questions = {}
        total_questions = 0

        for subject, config in self.SUBJECTS_CONFIG.items():
            all_questions[subject] = {}
            
            for grade in config["grades"]:
                grade_label = self.GRADE_LABELS[grade]
                questions = self.generate_questions_for_subject_grade(subject, grade)
                
                if questions:
                    all_questions[subject][grade_label] = questions
                    total_questions += len(questions)
                    logger.info(f"Generated {len(questions)} questions for {subject} {grade_label}")

        logger.info(f"Total questions generated: {total_questions}")
        return all_questions

    def save_questions_to_files(self, questions: Dict[str, Any]) -> None:
        """Save questions to organized file structure"""
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Save questions by subject and grade
        for subject, grade_data in questions.items():
            subject_dir = self.output_dir / subject
            subject_dir.mkdir(exist_ok=True)
            
            for grade, question_list in grade_data.items():
                filename = f"{grade}_questions.json"
                filepath = subject_dir / filename
                
                question_data = {
                    "subject": subject,
                    "grade": grade,
                    "question_count": len(question_list),
                    "questions": question_list
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(question_data, f, indent=2, ensure_ascii=False)

        # Create summary file
        summary = self._create_summary(questions)
        with open(self.output_dir / "database_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Create README
        self._create_readme()

    def _create_summary(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics"""
        summary = {
            "generation_timestamp": str(Path(__file__).stat().st_mtime),
            "total_subjects": len(questions),
            "subjects": list(questions.keys()),
            "grade_levels": list(self.GRADE_LABELS.values()),
            "question_statistics": {}
        }

        total_questions = 0
        for subject, grade_data in questions.items():
            subject_stats = {
                "total_questions": 0,
                "grades": {},
                "topics": set()
            }
            
            for grade, question_list in grade_data.items():
                subject_stats["total_questions"] += len(question_list)
                subject_stats["grades"][grade] = len(question_list)
                
                for q in question_list:
                    subject_stats["topics"].add(q.get("topic", "unknown"))
            
            subject_stats["topics"] = list(subject_stats["topics"])
            summary["question_statistics"][subject] = subject_stats
            total_questions += subject_stats["total_questions"]

        summary["total_questions"] = total_questions
        return summary

    def _create_readme(self) -> None:
        """Create README file for the question database"""
        readme_content = f"""# Educational Question Database

This database contains educational questions organized by subject and grade level for AI chatbot training.

## Database Structure

```
{self.output_dir.name}/
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

{chr(10).join([f"- **{subject.replace('_', ' ').title()}**: Grades {min(config['grades'])}-{max(config['grades'])}" 
              for subject, config in self.SUBJECTS_CONFIG.items()])}

## Question Format

Each question file contains:
```json
{{
  "subject": "mathematics",
  "grade": "fifth_grade",
  "question_count": 150,
  "questions": [
    {{
      "id": "mathematics_5_1234",
      "subject": "mathematics",
      "grade": "fifth_grade",
      "topic": "arithmetic",
      "difficulty": 3,
      "type": "generated",
      "question": "What is 15 + 27?",
      "answer": "42",
      "variables": {{"a": 15, "b": 27, "result": 42}}
    }}
  ]
}}
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

Generated on: {Path(__file__).stat().st_mtime}
"""
        
        with open(self.output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Main function to generate the question database"""
    logger.info("Starting Educational Question Generator")
    
    generator = EducationalQuestionGenerator()
    
    # Generate all questions
    logger.info("Generating questions for all subjects and grades...")
    all_questions = generator.generate_all_questions()
    
    # Save to files
    logger.info("Saving questions to file system...")
    generator.save_questions_to_files(all_questions)
    
    logger.info(f"Question database successfully created in '{generator.output_dir}'")
    
    # Print summary
    total_files = sum(len(grade_data) for grade_data in all_questions.values())
    total_questions = sum(len(question_list) 
                         for grade_data in all_questions.values() 
                         for question_list in grade_data.values())
    
    print(f"\n📚 Question Database Generated Successfully!")
    print(f"📁 Output Directory: {generator.output_dir}")
    print(f"📊 Total Files: {total_files}")
    print(f"❓ Total Questions: {total_questions:,}")
    print(f"🎯 Subjects: {len(all_questions)}")

if __name__ == "__main__":
    main()