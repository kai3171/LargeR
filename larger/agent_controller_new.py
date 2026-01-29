import torch
import torch.nn as nn
import torch.nn.functional as F
from llm_tools_new import (
    ask_llm_stage, 
    deeplearning_build, 
    result_code_generation, 
    get_reference_code,
    SmartCodeFixer,
    debug_agent,
    check_and_fix_model_code,
    auto_fix_and_retry
)
from data_utils_new import check_and_process_data
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score, accuracy_score

def predict(model, sample, features):
    input_tensors = []
    for feature in features:
        input_tensors.append(torch.from_numpy(sample[feature]))
    return model(*input_tensors)

def evaluate_result(model, dataset, features):
    model.eval()
    scores = []
    predictions = []
    labels = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            
            output = predict(model, row, features)

            if output.numel() == 1:
                score = output.cpu().item()
                pred = 1 if score > 0.5 else 0
            elif len(output.shape) == 1 and output.shape[0] == 2:
                probs = F.softmax(output, dim=0)
                score = probs[1].cpu().item()
                pred = torch.argmax(probs).cpu().item()
            else:
                score = output.cpu().mean().item()
                pred = 1 if score > 0.5 else 0
            
            scores.append(score)
            predictions.append(pred)
            labels.append(int(row['label']))
    
    return labels, predictions, scores

def clean_generated_code(code):
    code, _ = SmartCodeFixer.auto_fix(code)
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        if stripped and not stripped.startswith('#'):
            has_code_chars = any(c in stripped for c in ['=', '(', ')', ':', '[', ']', '{', '}', ',', '.', '+', '-', '*', '/', '"', "'"])
            is_import = stripped.startswith('import ') or stripped.startswith('from ')
            is_keyword = any(stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'def ', 'class ', 'return ', 'print(', 'try:', 'except', 'with ', 'else:', 'elif '])
            
            if not has_code_chars and not is_import and not is_keyword:
                cleaned_lines.append('# ' + line)
                continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

class AgentController:

    def __init__(self):
        print("Initializing RLAgent Controller...")
        self.model_code_path = "./generated_model.py"

    def run_pipeline(self):  
        msg = ask_llm_stage(
            "Explain to the user how to prepare their training dataset. "
            "Ask user to prepare input CSV in datasets/train.csv with correct format", 
            ""
        )

        ready = False
        while not ready:
            user_input = input("\n[Your reply (yes/no)] → ").strip().lower()
            
            if user_input == 'yes':
                ready = True
                print("\nUnderstood. Starting to process your training data, please wait...")
            elif user_input == 'no':
                print("Please prepare your data first, then type 'yes' when ready.")
            else:
                print("Invalid input. Please type 'yes' or 'no'.")

        data, feature, label = check_and_process_data("./datasets/train.csv", "./datasets/demo_processed.csv")
        if data is None:
            print("Data processing failed. Please check your CSV file and try again.")
            return
        else:
            print("Data processing completed. Saved as ./datasets/demo_processed.csv.")
 
        model_code = None
        if os.path.exists(self.model_code_path):
            print(f"Found saved model code: {self.model_code_path}")

            with open(self.model_code_path, 'r') as f:
                saved_code = f.read()
            
            use_saved = input("\nUse saved model code? (yes/no/edit) → ").strip().lower()
            
            if use_saved == 'yes':
                model_code = saved_code
            elif use_saved == 'edit':
                print(f"\nPlease manually edit the file: {self.model_code_path}")
                input("Press Enter once you have finished editing...")
                with open(self.model_code_path, 'r') as f:
                    model_code = f.read()
                print("Edited model code loaded successfully.")
            else:
                print("Rebuilding model...")
                model_code = None

        if model_code is None:
            model_code = deeplearning_build(data, feature, label)
            
            model_code, _ = SmartCodeFixer.auto_fix(model_code)

            with open(self.model_code_path, 'w') as f:
                f.write(model_code)
            print(f"\nModel code saved to: {self.model_code_path}")
        
        print("\nValidating model code...")
        model_code, validation_success = check_and_fix_model_code(model_code, feature, data, max_retries=50)
        
        if validation_success:
            with open(self.model_code_path, 'w') as f:
                f.write(model_code)
            print("Model code validated and saved.")
        else:
            print("Warning: Model code validation failed. You may need to manually fix the code.")
        
        continue_train = input("Continue to training? (yes/no/edit) → ").strip().lower()
        if continue_train == 'edit':
            print(f"\nPlease manually edit the file: {self.model_code_path}")
            input("Press Enter once you have finished editing...")
            with open(self.model_code_path, 'r') as f:
                model_code = f.read()
        elif continue_train != 'yes':
            print("Training cancelled.")
            return
        
        houxu = f'''
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = sum_model()
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {{total_params}}")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
epochs = 100
batch_size = 32

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train set size: {{len(train_data)}}, Test set size: {{len(test_data)}}")

def predict(model, sample):
    inputs = []
    for feature_name in {feature}:
        if feature_name in sample.index:
            tensor = torch.from_numpy(sample[feature_name])
            inputs.append(tensor.to(device))
    return model(*inputs)

def evaluate(model, dataset):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            output = predict(model, row)
            
            # Handle different output shapes
            if output.numel() == 1:
                pred = 1 if output.item() > 0.5 else 0
            elif len(output.shape) > 0 and output.shape[-1] == 2:
                pred = torch.argmax(output, dim=-1).item()
            else:
                pred = 1 if output.mean().item() > 0.5 else 0
            
            predictions.append(pred)
            labels.append(int(row['{label}']))

    return accuracy_score(labels, predictions)

print("Start training...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    for idx in range(0, len(train_data), batch_size):
        batch = train_data.iloc[idx:idx+batch_size]
        
        optimizer.zero_grad()
        batch_loss = 0
        
        for _, row in batch.iterrows():
            output = predict(model, row)
            
            # BCELoss needs float target and matching shapes
            target = torch.tensor([float(row['{label}'])], dtype=torch.float32).to(device)
            
            # Flatten output to match target shape [1]
            output = output.view(-1)
            
            loss = criterion(output, target)
            batch_loss += loss
        
        batch_loss = batch_loss / len(batch)
        
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item()
        num_batches += 1

    avg_train_loss = running_loss / num_batches

    train_acc = evaluate(model, train_data)
    test_acc = evaluate(model, test_data)
    
    print(f'Epoch {{epoch+1}}/{{epochs}} | Loss: {{avg_train_loss:.4f}} | Train Acc: {{train_acc:.4f}} | Test Acc: {{test_acc:.4f}}')

print("-" * 50)
print('Training finished!')
'''
        namespace = {}
        namespace['data'] = data
        namespace['feature'] = feature
        namespace['label'] = label
        
        try:
            full_code = model_code + houxu

            try:
                exec(full_code, namespace)
            except Exception as e:
                error_msg = str(e)
                print(f"\nTraining error: {error_msg}")
                print("Attempting automatic fix...")

                success, fixed_model_code, _ = auto_fix_and_retry(
                    model_code, 
                    "RNA-Ligand model with sigmoid output",
                    {'torch': torch, 'nn': nn},
                    max_retries=350
                )
                
                if success:
                    with open(self.model_code_path, 'w') as f:
                        f.write(fixed_model_code)
                    print(f"Fixed model code saved to: {self.model_code_path}")

                    namespace = {}
                    namespace['data'] = data
                    namespace['feature'] = feature
                    namespace['label'] = label
                    exec(fixed_model_code + houxu, namespace)
                else:
                    raise Exception(f"Auto-fix failed after multiple attempts. Last error: {error_msg}")
            
            model = namespace['model']
            test_data = namespace['test_data']

            labels, predictions, scores = evaluate_result(model, test_data, feature)
            result_df = pd.DataFrame({
                'labels': labels,
                'predictions': predictions,
                'score': scores
            })
            
            print("Model trained successfully!")
            print("\nPrediction Results Preview:")
            print(result_df.head(10))
            
            print("""
Available commands:
    - Calculate AUC value
    - Plot and save ROC curve
    - Plot and save Confusion Matrix
    - Display Classification Report
    - Calculate Accuracy
    - exit (to quit)
            """)
            
            exec_namespace = {
                'result': result_df,
                'labels': labels,
                'predictions': predictions,
                'scores': scores,
                'model': model,
                'test_data': test_data,
                'train_data': namespace.get('train_data'),
                'data': data,
                'feature': feature,
                'label': label,
                'torch': torch,
                'np': np,
                'pd': pd,
                'plt': plt,
                'roc_curve': roc_curve,
                'auc': auc,
                'roc_auc_score': roc_auc_score,
                'confusion_matrix': confusion_matrix,
                'classification_report': classification_report,
                'accuracy_score': accuracy_score,
            }

            while True:
                user_input = input("\n[Your request, 'exit' to end] → ").strip()
                if user_input.lower() == 'exit':
                    break
                
                result = result_code_generation(user_input)
                code = result['code']
                
                code = clean_generated_code(code)
                
                try:
                    exec(code, exec_namespace)
                except Exception as e:
                    print(f"Execution Error: {e}")
                    print("Please try re-describing your request.")
        
        except Exception as e:
            print(f"\nError during training: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"\nModel code saved at: {self.model_code_path}")
            print("You can manually edit this file to fix issues, then restart and select 'yes' to use the saved code.")
            return