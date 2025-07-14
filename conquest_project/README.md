# Project Setup

### Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run the script: `uv run main`


### Code Organization

- The [data/](./data/) directory contains the input csv files that you'll need to use.
- The `main` method in [drug_discovery/main.py](./drug_discovery/main.py) is the entry point for your code.
    - The pyproject.toml file is already configured to call this method with `uv run main`
- The [tests/](./tests/) directory contains the unit tests.
    - Tests can be run with `uv run pytest`
- Type-checking with mypy can be run with `uv run mypy`


# Instructions

You’re building a tool to process a large dataset of chemical compounds for drug discovery. These compounds will be represented by [SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System) strings. Assume that you're working with a team, and your code will need to be well tested and easily understood by your teammates.

Your script should:

1. Find the usable compounds contained in [data/compound_scores.csv](data/compound_scores.csv), keeping in mind that this file may be very large.
    - For this exercise, usable compounds are determined by the following:
        - **activity score greater than or equal to a customizable threshold**
        - **fewer than six carbon atoms**
    - Your script should support an optional `--threshold` argument for customizing the threshold. If no value is provided, the threshold should default to 0.5.
2. Save the usable compounds to `output/usable_compounds.csv`.
    - In order to produce this file, you will need to calculate the number of carbon (C), nitrogen (N), and oxygen (O) atoms in each compound.
    - This file should contain the following columns:
        - "smiles"
        - "activity_score"
        - "c" (number of carbon atoms)
        - "n" (number of nitrogen atoms)
        - "o" (number of oxygen atoms)
3. Print the top five usable compounds sorted by their activity score to the console. Include the compound's name, SMILES string, and activity score in the string that is printed.
    - Each of the top compounds should be printed in the following format (replace "#" with 1-5):
        ```
        #. name - activity_score
           smiles
        ```
    - The name for each compound can be found in [data/compound_names.csv](data/compound_names.csv).
    

---

# Limitations and LLM Uage

### Allowed

- Web search
- Using an LLM as an alternative to web search:
    - "How is an enum defined in Python?"
    - "How can I read data from a file using Python?"
    - "What is the syntax for passing a callback in Python?"

### Not allowed

- Asking an LLM for a full solution or problem-solving suggestions:
    - "Please write code that solves the following prompt..."
    - "I'm being asked to solve this problem .... What is the best way to do this?"
- LLM-driven auto-complete features
    - If your IDE is configured to leverage an LLM for its auto-complete, you will need to disable this prior to coding your solution.


