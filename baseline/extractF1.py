import re
import matplotlib.pyplot as plt

def extract_f1_scores(log_content):
    # Regular expression to match the desired section in the logs
    event_based_pattern = re.compile(
        r"INFO - evaluation_measures - Event based metrics \(onset-offset\).*?Class-wise average metrics \(macro-average\).*?F-measure \(F1\)\s+:\s+(\d+\.\d+)\s*%", 
        re.DOTALL
    )
    matches = event_based_pattern.findall(log_content)
    return [float(match) for match in matches]

def process_log_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return extract_f1_scores(content)

if __name__ == "__main__":
    import os

    log_directory = "baseline/logs"
    
    all_f1_scores = []

    for filename in os.listdir(log_directory):
        #if filename.endswith(".log"):
        if filename == "test_14.log":
            file_path = os.path.join(log_directory, filename)
            f1_scores = process_log_file(file_path)
            all_f1_scores.extend(f1_scores)

    for i, score in enumerate(all_f1_scores, 1):
        print(f"F1 Score {i}: {score}%")
        
    max_f1_score = max(all_f1_scores)
    max_f1_index = all_f1_scores.index(max_f1_score)
    print(f"Max F1: {max_f1_score}% at epoch {max_f1_index}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_f1_scores) + 1), all_f1_scores, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('F1 Score (%)')
    plt.title('F1 Scores')
    plt.axvline(x=max_f1_index + 1, color='r', linestyle='--', label=f'Max F1: {max_f1_score}%')
    plt.legend()
    plt.grid(True)
    plt.show()
    