import os


def extract_sentence_pairs(lines):
    qa_pairs = []
    for i in range(len(lines) - 1):
        input_line = lines[i].strip()
        target_line = lines[i + 1].strip()
        if input_line and target_line:
            qa_pairs.append([input_line, target_line])
    return qa_pairs


def load_sentence_pairs(corpus_name):
    path = os.path.join('data', corpus_name)
    files = os.listdir(path)
    all_pairs = []
    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            lines = f.readlines()
            all_pairs.extend(extract_sentence_pairs(lines))
    return all_pairs
