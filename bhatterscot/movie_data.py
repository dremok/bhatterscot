import os
from io import open


def load_movie_sentence_pairs(corpus):
    # Splits each line of the file into a dictionary of fields
    def loadLines(fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines

    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def loadConversations(fileName, lines, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj['utteranceIDs'] == '['L598485', 'L598486', ...]')
                lineIds = eval(convObj['utteranceIDs'])
                # Reassemble lines
                convObj['lines'] = []
                for lineId in lineIds:
                    convObj['lines'].append(lines[lineId])
                conversations.append(convObj)
        return conversations

    def extract_sentence_pairs(conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation['lines']) - 1):  # We ignore the last line (no answer for it)
                input_line = conversation['lines'][i]['text'].strip()
                target_line = conversation['lines'][i + 1]['text'].strip()
                # Filter wrong samples (if one of the lists is empty)
                if input_line and target_line:
                    qa_pairs.append([input_line, target_line])
        return qa_pairs

    # Initialize lines dict, conversations list, and field ids
    MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
    MOVIE_CONVERSATIONS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']

    # Load lines and process conversations
    lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines,
                                      MOVIE_CONVERSATIONS_FIELDS)
    sentence_pairs = extract_sentence_pairs(conversations)
    return sentence_pairs
