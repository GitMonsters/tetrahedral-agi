# GAIA TRAINING DATA DIAGRAMS

## Data Structure Overview

```
gaia_data/
├── 2023/
│   ├── test/               # Test set (301 questions)
│   │   ├── metadata.parquet
│   │   ├── metadata.level1.parquet
│   │   ├── metadata.level2.parquet
│   │   ├── metadata.level3.parquet
│   │   └── [301 supporting files]
│   │       ├── PDFs (research papers)
│   │       ├── MP3s (audio files)
│   │       ├── PNGs (images)
│   │       ├── JPGs (photos)
│   │       ├── XLSXs (spreadsheets)
│   │       ├── CSVs (data tables)
│   │       ├── PPTXs (presentations)
│   │       ├── ZIPs (archives)
│   │       ├── JSONs (structured data)
│   │       ├── XMLs (markup)
│   │       ├── TXTs (text)
│   │       ├── PYs (code)
│   │       ├── PDBs (protein structures)
│   │       └── MOVs (videos)
│   │
│   └── validation/        # Validation set (165 questions)
│       ├── metadata.parquet        # All 165 questions
│       ├── metadata.level1.parquet  # Level 1 (53 questions)
│       ├── metadata.level2.parquet  # Level 2 (86 questions)
│       ├── metadata.level3.parquet  # Level 3 (26 questions)
│       └── [43 supporting files]
│           ├── XLSX files
│           ├── MP3 files
│           ├── PDF files
│           ├── PNG files
│           ├── JPG files
│           ├── CSV files
│           ├── TXT files
│           ├── PPTX files
│           ├── ZIP files
│           ├── JSON files
│           ├── PDB files
│           └── PY files
│
└── README.md               # Dataset documentation
```

## Dataset Schema

### Main Metadata Columns
```
task_id                # Unique question identifier
Question               # Full question text
Level                  # Difficulty (1=Easy, 2=Medium, 3=Hard)
Final answer           # Ground truth answer
file_name             # Supporting filename (optional)
file_path             # Full path to supporting file
Annotator Metadata     # Additional metadata (dict)
```

### Question Level Distribution

```
Validation Set (165 questions):
├── Level 1: 53 questions (32.1%)
│   ├── Target: "breakable by very good LLMs"
│   ├── Requires: Basic reasoning, simple patterns
│   └── Example: "What is 2 + 2?"
│
├── Level 2: 86 questions (52.1%)
│   ├── Target: "requires reasoning and tools"
│   ├── Requires: Complex reasoning, web search, calculations
│   └── Example: Research about invasive species
│
└── Level 3: 26 questions (15.8%)
    ├── Target: "indicates strong jump in capabilities"
    ├── Requires: Advanced reasoning, multiple tools, synthesis
    └── Example: Complex multi-step reasoning
```

### Supporting Files Distribution

```
Validation Set Files (43 files):
├── XLSX (Spreadsheet)      13 files  (Excel data tables)
├── MP3 (Audio)              5 files   (Audio recordings)
├── PDF (Documents)           8 files   (Research papers)
├── PNG (Images)              3 files   (Charts, diagrams)
├── JPG (Images)              2 files   (Photos)
├── CSV (Data)               1 file    (Tabular data)
├── TXT (Text)                1 file    (Raw text)
├── PPTX (Presentations)       1 file    (PowerPoint)
├── ZIP (Archives)            1 file    (Compressed data)
├── JSON (Structured Data)      1 file    (JSON-LD format)
├── PDB (Protein)            1 file    (Molecular structure)
└── PY  (Code)                6 files   (Python scripts)

Total: 43 files
Total Size: ~20MB (validation set)
```

## Sample Questions by Level

### Level 1 Examples (53 questions)
```
Question 1 (Level 1):
  Task ID: e1fc63a2-da7a-432f-be78-7c4a95598703
  Question: "If Eliud Kipchoge could maintain his record-making marathon pace 
            indefinitely, how many thousand hours would it take to complete 
            a marathon of 42.195 kilometers?"
  Answer: "17"
  File: None
  Type: Mathematical calculation
  Difficulty: Easy

---

Question 2 (Level 1):
  Task ID: 42b8257e-f47b-4dcb-8599-459c329ac153.mp3
  Question: [Audio file - speech/question about time]
  Answer: [Answer derived from audio]
  File: Audio file
  Type: Audio understanding
  Difficulty: Easy
```

### Level 2 Examples (86 questions)
```
Question 1 (Level 2):
  Task ID: c61d22de-5f6c-4958-a7f6-5e9707bd3466
  Question: "A paper about AI regulation that was originally submitted to arXiv.org 
            in June 2022 shows a figure with..."
  Answer: "egalitarian"
  File: None
  Type: Research understanding
  Difficulty: Medium
  Requires: Document analysis, reasoning about regulation concepts

---

Question 2 (Level 2):
  Task ID: 17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc
  Question: "I'm researching species that became invasive after people who kept them 
            as pets released them. There are..."
  Answer: "34689"
  File: PDF document
  Type: Document analysis + calculation
  Difficulty: Medium
  Requires: PDF parsing, data extraction, calculation
```

### Level 3 Examples (26 questions)
```
Question 1 (Level 3):
  Task ID: [Complex reasoning task]
  Question: [Multi-step problem requiring advanced reasoning]
  Answer: [Derived answer]
  File: [Multiple supporting files]
  Type: Complex synthesis
  Difficulty: Hard
  Requires: Advanced reasoning, tool use, synthesis
```

## Data Flow Diagram

```
GAIA Dataset Download
    ↓
├── Test Set (301 questions)
│   ├── Level 1: 93 questions
│   ├── Level 2: 159 questions
│   └── Level 3: 49 questions
│
└── Validation Set (165 questions)
    ├── Level 1: 53 questions
    ├── Level 2: 86 questions
    └── Level 3: 26 questions
    ↓
Load via PyArrow
    ↓
├── metadata.parquet (165 rows × 7 columns)
├── metadata.level1.parquet (53 rows)
├── metadata.level2.parquet (86 rows)
└── metadata.level3.parquet (26 rows)
    ↓
Process with GAIADataset Class
    ↓
├── Extract questions
├── Extract levels
├── Extract answers
├── Load supporting files (if present)
└── Return to PyTorch DataLoader
    ↓
Train/Validate Model
    ↓
Calculate Metrics
    ↓
Submit to Hugging Face
```

## File Size Analysis

```
Test Set (301 questions):
├── Parquet files: ~50MB
├── Supporting files: ~100MB
└── Total: ~150MB

Validation Set (165 questions):
├── Parquet files: ~3MB
├── Supporting files: ~20MB
└── Total: ~23MB

Complete Dataset:
├── Total questions: 466
├── Total files: ~344
├── Total size: ~173MB
└── Compression: Parquet (efficient storage)
```

## Question Type Analysis

Based on sample questions, GAIA requires:

1. **Mathematical Reasoning**
   - Arithmetic calculations
   - Formula applications
   - Unit conversions
   - Time calculations

2. **Document Understanding**
   - PDF parsing
   - Excel data extraction
   - Image analysis
   - Audio transcription
   - Code execution

3. **Research/Information Retrieval**
   - Web search required
   - Knowledge from documents
   - Fact verification
   - Cross-referencing sources

4. **Logical Reasoning**
   - Pattern recognition
   - Deductive reasoning
   - Inductive reasoning
   - Logical puzzles

5. **Multimodal Processing**
   - Text + Image
   - Text + Audio
   - Text + Video
   - Text + Document

## Training Pipeline Integration

```
GAIA Dataset
    ↓
1. Data Loading
   ├── GAIADataset class
   ├── Parquet file reading
   └── Supporting file handling
    ↓
2. Data Processing
   ├── Question encoding
   ├── Answer tokenization
   ├── Level classification
   └── File type detection
    ↓
3. Model Training
   ├── ProductionTetrahedralModel
   ├── GAIATrainer class
   ├── Optuna-optimized parameters
   └── Multi-task learning
    ↓
4. Evaluation
   ├── GAIABenchmarkEvaluator
   ├── 165-question evaluation
   ├── Level-specific scoring
   └── Metrics calculation
    ↓
5. Submission
   ├── HuggingFaceSubmissionGuide
   ├── Model card generation
   └── Leaderboard submission
```

---

**GAIA Training Data Structure Complete** 📊

Total: 466 questions (test: 301, validation: 165)
Levels: 3 difficulty tiers
Files: 344+ supporting documents
Format: Parquet + various file types
