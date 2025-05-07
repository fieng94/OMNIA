from sklearn.metrics import confusion_matrix

def clean_score(list_score) -> list[float]:
    list_score = [extract_score(score) for score in list_score]
    list_score = [item.rstrip('.') if item.endswith('.') else item for item in list_score]
    list_score = [0 if item == '' else item for item in list_score]
    score_bin = [1 if item == '1'  else 0 for item in list_score]
    return score_bin


def extract_score(text:str)->str:
    # Function to extract score
    # Find the starting index of "Score: "
    start_index = text.find("Score: ") + len("Score: ")

    # Find the end of the number, which could be marked by a non-digit character
    end_index = start_index
    while end_index < len(text) and (text[end_index].isdigit() or text[end_index] == '.'):
        end_index += 1

    # Extract and return the number using slicing
    return text[start_index:end_index]

def compute_score(prediction, ground_truth):
    cm = confusion_matrix(ground_truth, prediction)
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return accuracy, f1_score, recall, precision

def get_gt_pred(df, score_list):
    prediction = clean_score(score_list)
    ground_truth = df['Missing']
    return prediction, ground_truth
