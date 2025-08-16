# AI Bot Question Database

This directory contains question files organized by subject and grade level for use with the AI Bot.

## Structure
- Each subject has its own directory
- Within each subject directory, files are organized by grade level
- All questions are stored in JSON format with rich metadata (type, text, answers, hints, difficulty)

## Available Subjects
- math
- science
- english
- history
- social_studies
- art
- music
- physical_education
- foreign_language
- computer_science
- geography
- civics
- economics
- biology
- chemistry
- physics

## Grade Levels
- kindergarten
- 1st_grade
- 2nd_grade
- 3rd_grade
- 4th_grade
- 5th_grade
- 6th_grade
- 7th_grade
- 8th_grade
- 9th_grade
- 10th_grade
- 11th_grade
- 12th_grade

## Usage with AI Bot
To train the AI bot, load the JSON files using `load_knowledge` with the format:
- subject: string
- grade: string
- question_id: string
- text: string
- difficulty: int
Additional fields (e.g., correct_answer, hint) can enhance training.
