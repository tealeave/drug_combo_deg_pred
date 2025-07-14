import pandas as pd
from rdkit import Chem
from collections import Counter
from drug_discovery.const import FILEPATH_COMPOUND_NAMES, FILEPATH_COMPOUND_SCORES, DEFAULT_THRESHOLD, CARBON_ATOM_LIMIT
import ast

def extract_cno(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (0, 0, 0)
    cnt = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    return (
        cnt.get("C", 0),
        cnt.get("N", 0),
        cnt.get("O", 0),
    )


def main() -> None:
    """
    Instructions Recap:
    1. Find usable compounds in `data/compound_scores.csv`:
        - `activity_score >= threshold`
        - `carbon count < 6`
    2. Save usable compounds to `output/usable_compounds.csv`
        - Columns: `smiles`, `activity_score`, `c`, `n`, `o`
    3. Print the top 5 usable compounds with their name from `data/compound_names.csv`
    """
    print("Hello, world!")
    print('cpd_names:')


    cpd_names_df = pd.read_csv(FILEPATH_COMPOUND_NAMES)

    # Turn string lit to list
    cpd_names_df["smile_as_list"] = cpd_names_df["smiles"].apply(ast.literal_eval)

    # There could be more than 1 element in the smile list, we'd like 1:1 in the match for dict creation
    cpd_names_exploded = cpd_names_df.explode("smile_as_list").reset_index(drop=True)

    # Create dict for cpd name look up from smile
    smile_to_name_dict = cpd_names_exploded.set_index("smile_as_list")["name"].to_dict()


    print('\ncpd_scores')
    cpd_scores_df = pd.read_csv(FILEPATH_COMPOUND_SCORES)

    # Look up the cpd name from the dict
    cpd_scores_df['name'] = cpd_scores_df['smiles'].map(smile_to_name_dict)

    # Find the C, N, O count with RDkit
    cpd_scores_df[["C_count","N_count","O_count"]] = cpd_scores_df["smiles"].apply(
        lambda s: pd.Series(extract_cno(s), index=["C_count","N_count","O_count"])
        )
    filtered_cpd_scores_df = (
        cpd_scores_df[
            (cpd_scores_df['activity_score'].astype(float) >= DEFAULT_THRESHOLD) & 
            (cpd_scores_df['C_count'].astype(int) < 6)
            ]
            .sort_values('activity_score', ascending=False)
    )
    
    print(filtered_cpd_scores_df.shape)
    print(filtered_cpd_scores_df.head())


    n = 0
    for i, row in filtered_cpd_scores_df.iterrows():
        print(f'{n+1}. {row['name']} - activity_score: {row['activity_score']}')
        print(f'            smiles: {row['smiles']}')
        n += 1
        if n > 4:
            break

