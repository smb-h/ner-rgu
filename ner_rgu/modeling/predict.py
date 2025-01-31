import typer
import re
import spacy
from tqdm import tqdm
from loguru import logger

from ner_rgu.dataset import get_data


app = typer.Typer()

PRE_PATTERNS_FOR_INTRO = [
    "my name is",
    "i am",
    "i’m",
    "i'm",
    "they call me",
    "i’m called",
    "i'm called",
    "i am called",
]
POST_PATTERNS_FOR_INTRO = [
    "that is me",
    "that’s me",
    "that's me",
    "that is me",
    "that would be me",
]

def extract_speaker_pattern(text: str) -> list:
    p = re.compile('(Speaker\d*)')
    speaker_list = p.findall(text)
    unique_speakers = list(set(speaker_list))
    return unique_speakers

def extract_person_entities(doc) -> list:
    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)
    names = list(set(names))

    full_names = [x.lower() for x in names if " " in x]
    single_names = [x.lower() for x in names if x not in full_names]

    # double cross names to use full names
    filtered_single_names = []
    for sn in single_names:
        found_it = False
        for fn in full_names:
            if sn in fn:
                found_it = True
                break
        if not found_it:
            filtered_single_names.append(sn)

    single_names = filtered_single_names
    del filtered_single_names
    names = [
        *full_names,
        *single_names
    ]
    return names

def link_entities(doc_raw_lines, entities, labels) -> dict:
    context = {}
    for ln in doc_raw_lines:
        this_name = None
        this_label = None
        if not ln:
            continue
        tokens = ln.lower()
        # check pre-patterns
        for pattern in PRE_PATTERNS_FOR_INTRO:
            if pattern in tokens:
                # idx = tokens.index(pattern)
                # remaining = tokens[idx:]
                remaining = tokens.split(pattern)[1]
                potentiel_name = remaining.split(" ")[:4]
                potentiel_name = " ".join(potentiel_name)
                # find name 
                for n in entities:
                    if n in potentiel_name:
                        this_name = n
                        this_name = this_name.split(" ")
                        this_name = [(tmp_char[0].upper() + tmp_char[1:]) for tmp_char in this_name]
                        this_name = " ".join(this_name)
                        del entities[entities.index(n)]
                        break
                # find label
                for l in labels:
                    if l.lower() in tokens:
                        this_label = l
                        this_label = this_label.replace("speaker", "Speaker")
                        del labels[labels.index(l)]
                        break
                if this_label and this_name:
                    context[this_label] = this_name
                    this_name = None
                    this_label = None
                break
        # check post-patterns
        for pattern in POST_PATTERNS_FOR_INTRO:
            if pattern in tokens:
                remaining = tokens.split(pattern)[0]
                potentiel_name = remaining.split(" ")[-4:-1]
                potentiel_name = " ".join(potentiel_name)
                # find name 
                for n in entities:
                    if n in potentiel_name:
                        this_name = n
                        this_name = this_name.split(" ")
                        this_name = [(tmp_char[0].upper() + tmp_char[1:]) for tmp_char in this_name]
                        this_name = " ".join(this_name)
                        del entities[entities.index(n)]
                        break
                # find label
                for l in labels:
                    if l.lower() in tokens:
                        this_label = l
                        this_label = this_label.replace("speaker", "Speaker")
                        del labels[labels.index(l)]
                        break
                if this_label and this_name:
                    context[this_label] = this_name
                    this_name = None
                    this_label = None
                break
    
    return context
    
def replace_speakers_via_entities(doc_raw, ref) -> str:
    for k, v in ref.items():
        doc_raw = doc_raw.replace(k, v)
    return doc_raw

def compare_results(ground_truth, predicted) -> None:
    speakers = list(ground_truth.keys())
    correct = 0
    incorrect = 0
    missing = 0
    for speaker in speakers:
        if speaker in predicted:
            if ground_truth[speaker] == predicted[speaker]:
                correct += 1
            else:
                incorrect += 1
        else:
            missing += 1
    acc = correct / (correct + incorrect + missing)
    print("===" * 4)
    print(f"Achieved accuracy: {acc}")

@app.command()
def main(datafile_name: str):
    logger.info("Performing inference for model...")

    spacy_model = spacy.load("en_core_web_lg")
    # Read datA
    data_dict = get_data(datafile_name)
    doc = spacy_model(data_dict["raw_data"])
    # Extract speaker pattern
    unique_speakers = extract_speaker_pattern(data_dict["raw_data"])
    # Extract PERSON entities
    full_names = extract_person_entities(doc)
    # Link entities to speakers
    linked_entities = link_entities(
        data_dict["data_splitted_lines"], 
        full_names, 
        unique_speakers
    )

    # Replace speakers with entities
    doc_raw = replace_speakers_via_entities(data_dict["raw_data"], linked_entities)

    print("###" * 10)
    print("Ground truth:")
    print(data_dict["labels"])
    print("---" * 6)
    print("Predicted:")
    print(linked_entities)
    print("---" * 6)
    print("Final result:")
    print(doc_raw)

    compare_results(data_dict["labels"], linked_entities)
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
