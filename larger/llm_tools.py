import httpx
from typing import Optional, Tuple
import asyncio
import json
import numpy as np
import torch
import re

client = httpx.AsyncClient(timeout=60.0)

REFERENCE_MODEL_CODE = '''
import torch
import torch.nn as nn
from models.mamba import mambalayer

class sum_model(nn.Module):
    def __init__(self):
        super(sum_model, self).__init__()
        print('model building')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mamba_rna = mambalayer(embedding_dim=641).to(self.device)
        
        self.mamba_ligand = mambalayer(embedding_dim=30).to(self.device)
        
        self.fc = nn.Linear(671, 1).to(self.device)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, rna_feature, ligand_feature):
        rna_feature = rna_feature.float().to(self.device)
        ligand_feature = ligand_feature.float().to(self.device)
        
        rna_out = self.mamba_rna(rna_feature) 
        ligand_out = self.mamba_ligand(ligand_feature)
        
        rna_pooled = torch.mean(rna_out, dim=0)
        ligand_pooled = torch.mean(ligand_out, dim=0)
        
        combined = torch.cat([rna_pooled, ligand_pooled], dim=-1)
        
        output = self.fc(combined)
        output = self.sigmoid(output)
        
        return output
'''

def get_reference_code():
    """Return the reference model code"""
    return REFERENCE_MODEL_CODE


class SmartCodeFixer:
    
    TYPO_FIXES = {
        r'\bland_feature\b': 'ligand_feature',
        r'\bligan_feature\b': 'ligand_feature',
        r'\blignad_feature\b': 'ligand_feature',
        r'\bligand_featuer\b': 'ligand_feature',
        r'\bligand_featrue\b': 'ligand_feature',
        r'\bligand_featur\b': 'ligand_feature',
        r'\bliganf_feature\b': 'ligand_feature',
        r'\bland_projected\b': 'ligand_projected',
        
        r'\brna_featuer\b': 'rna_feature',
        r'\brna_featrue\b': 'rna_feature',
        r'\brna_featur\b': 'rna_feature',
        r'\bran_feature\b': 'rna_feature',
        
        r'\bembeding_dim\b': 'embedding_dim',
        r'\bdevcie\b': 'device',
    }
    
    @classmethod
    def fix_typos(cls, code):
        if not isinstance(code, str):
            return code
        
        fixed_code = code
        for pattern, replacement in cls.TYPO_FIXES.items():
            fixed_code = re.sub(pattern, replacement, fixed_code)
        
        return fixed_code
    
    @classmethod
    def fix_undefined_variable(cls, code, error_msg):
        match = re.search(r"name '(\w+)' is not defined", error_msg)
        if not match:
            return code, False
        
        undefined_var = match.group(1)
        
        var_map = {
            'land_feature': 'ligand_feature',
            'ligan_feature': 'ligand_feature',
            'lignad_feature': 'ligand_feature',
            'rna_featuer': 'rna_feature',
            'rna_featrue': 'rna_feature',
            'land_projected': 'ligand_projected',
        }
        
        if undefined_var in var_map:
            correct_var = var_map[undefined_var]
            fixed_code = re.sub(r'\b' + undefined_var + r'\b', correct_var, code)
            print(f" auto-repair: '{undefined_var}' -> '{correct_var}'")
            return fixed_code, True
        
        return code, False
    
    @classmethod
    def fix_shape_mismatch(cls, code, error_msg, variation=None):
        if "shapes cannot be multiplied" not in error_msg:
            return code, False
        
        shape_match = re.search(r'\((\d+)x(\d+)\) and \((\d+)x(\d+)\)', error_msg)
        if not shape_match:
            return code, False
        
        m1_rows, m1_cols, m2_rows, m2_cols = map(int, shape_match.groups())
        
        if "final_layer" in code or "Linear" in code:
            old_pattern = r'self\.final_layer\s*=\s*(?:torch\.nn\.)?Linear\s*\(\s*(\d+)'
            match = re.search(old_pattern, code)
            if match:
                old_dim = int(match.group(1))
                if old_dim != m1_cols:
                    new_code = re.sub(
                        old_pattern,
                        f'self.final_layer = nn.Linear({m1_cols}',
                        code
                    )
                    print(f" Automatically fix the input dimension of final_layer: {old_dim} -> {m1_cols}")
                    return new_code, True
        
        return code, False
    
    @classmethod
    def auto_fix(cls, code, error_msg="", variation=None):
        original_code = code
        
        code = cls.fix_typos(code)
        
        if error_msg:
            code, fixed = cls.fix_undefined_variable(code, error_msg)
            if fixed:
                return code, True
            
            if variation:
                code, fixed = cls.fix_shape_mismatch(code, error_msg, variation)
                if fixed:
                    return code, True
        
        was_fixed = (code != original_code)
        
        return code, was_fixed


def run_LLM(prompt: str) -> str:
    api_key = ""
    api_url = "http://192.168.0.108:1025/v1/chat/completions"
    try:
        response = asyncio.run(client.post(
            api_url,
            json={
                "model": "llama_70b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.7
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
        ))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].split('</think>')[-1] if response.text else "no useful message return"
    except httpx.RequestError as e:
        print(f"requests false: {str(e)}")
        return "requests false"
    except Exception as e:
        print(f"error: {str(e)}")
        return "system error, please try later."


def answer2json(string):
    start_index = string.find('{')
    end_index = string.rfind('}') + 1
    json_string = string[start_index:end_index]
    return json.loads(json_string)   


def run_LLM_json_auto_retry(full_prompt):
    answer = run_LLM(full_prompt)
    try:
        result_json = answer2json(answer)
    except Exception as e:
        result_json = run_LLM_json_auto_retry(full_prompt)
    return result_json


def ask_llm_stage(stage_prompt: str, user_context: str = "") -> str:
    """
    Generate agent reply for current stage.
    """
    full_prompt = f"""
        You are RLAgent, an interactive assistant for RNA-Ligand interaction modeling.
        
        Your task is to guide the user step by step through the modeling pipeline.
        
        Current task: {stage_prompt}
        
        Constraints:
        - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
        - Only provide clear and concise instructions to the user.
        - Do not apologize or explain your capabilities.
        - Wait for user reply before continuing.
        
        Now, explain to the user how to prepare their training dataset, using the following information:
        
        ---
        
        You need to prepare a dataset in CSV format.
        
        Each row should contain:
        - ligand: Ligand name
        - label: 1 (positive) or 0 (negative), indicating interaction
        - rna_sequence: RNA sequence (string of A/U/G/C)
        - region_mask: A list of 0/1 values of the same length as RNA sequence, where 1 indicates the region to be predicted
        
        Here is an example:
        
        | ligand | label | rna_sequence       | region_mask                            |
        |--------|-------|--------------------|----------------------------------------|
        | CIR    | 1     | ACGGUUAGGUCGCU     | [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] |
        
        Once your data is ready, please save it to:
        `datasets/train.csv`  
        with the correct format as shown above.
        
        ---
        
        After presenting this to the user, do not output any additional content. Wait for the user to reply.
        Please prompt the user that if the data is ready, Please press yes when data is ready...
    """

    response = run_LLM(full_prompt)
    print(response)
    return response


def judge_user_satisfied(user_reply: str, original_question: str) -> bool:
    """
    Use LLM to judge whether the user is satisfied with the plan.
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The agent ask the user:
    "{original_question}"
    
    The user replied:
    "{user_reply}"

    Please determine whether the user's response indicates that they are satisfied.
    
    Constraints:
    - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
    - Only provide clear and concise instructions.
    - Do not apologize or explain your capabilities.

    Does this shows that the user is satisfied?

    Please answer only "YES" or "NO". Do NOT provide any explanation or additional text.
    """

    llm_reply = run_LLM(full_prompt)

    return "YES" in str(llm_reply)


def generate_model(user_input):
    """
    Generate model code based on user input and reference code.
    """
    reference_code = get_reference_code()
    
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    User proposed their ideas for modifying the model.

    Please modify the reference code based on the user's requirements to generate a new model.

    Reference code:
    ```python
    {reference_code}
    ```

    User requirement:
    "{user_input}"

    Available modifications include:
    1. Add layers (e.g., "add dropout layer", "add attention mechanism", "add more linear layers")
    2. Change architecture (e.g., "use LSTM instead of Mamba", "add batch normalization")
    3. Modify dimensions (e.g., "change hidden dimension to 512", "add intermediate layer with 256 units")
    4. Change activation (e.g., "use ReLU activation", "use tanh instead of sigmoid")
    5. Add regularization (e.g., "add dropout with rate 0.3", "add layer normalization")
    6. Modify pooling (e.g., "use max pooling instead of mean pooling", "use attention pooling")

    IMPORTANT:
    - Keep the same input/output interface (rna_feature, ligand_feature as inputs, sigmoid output in [0,1])
    - Keep 'from models.mamba import mambalayer' if using Mamba
    - The model class must be named 'sum_model'
    - Output should be compatible with BCELoss (single value with sigmoid)

    Return the content in JSON format, including the following fields:

    - code: the complete modified model code (must be valid Python code that can be executed directly)
    - explain: explanation of the changes made in simple terms

    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    return run_LLM_json_auto_retry(full_prompt)


def deeplearning_build(data, features, label):
    """
    Build deep learning model interactively with auto-fix capability.
    Returns the model code.
    """
    reference_code = get_reference_code()
    
    while True:
        asking = "\n[Describe modifications you want (or 'ok' to use reference code)] → "
        response = input(asking).strip()
        
        if response.lower() == 'ok':
            print("\nUsing reference code as-is.")
            print("\nValidating model...")
            fixed_code, success = check_and_fix_model_code(reference_code, features, data)
            if success:
                return fixed_code
            else:
                print("Warning: Model validation failed, but returning code anyway.")
                return fixed_code
        
        if judge_user_satisfied(response, asking):
            print("\nUsing reference code as-is.")
            return reference_code
        
        print("\nGenerating modified model based on your request...")

        try:
            result = generate_model(response)
            generated_code = result['code']

            generated_code, _ = SmartCodeFixer.auto_fix(generated_code)
            

            print("\nExplanation:", result.get('explain', 'No explanation provided.'))
            
            while True:
                confirm = input("\n[Accept this model? (yes/no/modify)] → ").strip().lower()
                
                if confirm == 'yes':
                    print("\nModel accepted! Validating...")

                    fixed_code, success = check_and_fix_model_code(generated_code, features, data)
                    
                    if success:
                        print("Model validation successful!")
                        return fixed_code
                    else:
                        print("\nModel validation failed. Options:")
                        print("  1. 'retry' - Try to fix again")
                        print("  2. 'accept' - Accept anyway (may cause training errors)")
                        print("  3. 'modify' - Make more modifications")
                        
                        choice = input("\n[Your choice] → ").strip().lower()
                        if choice == 'retry':
                            fixed_code, success = check_and_fix_model_code(fixed_code, features, data, max_retries=5)
                            if success:
                                return fixed_code
                        elif choice == 'accept':
                            return fixed_code
                        elif choice == 'modify':
                            break
                        
                elif confirm == 'modify':
                    further = input("\n[Describe additional modifications] → ").strip()
                    if further:
                        result = generate_model(f"Based on this code:\n{generated_code}\n\nMake these changes: {further}")
                        generated_code = result['code']
                        generated_code, _ = SmartCodeFixer.auto_fix(generated_code)
                        
                        print("\nExplanation:", result.get('explain', 'No explanation provided.'))
                        
                elif confirm == 'no':
                    print("\nLet's try a different modification...")
                    break
                else:
                    print("Please enter 'yes', 'no', or 'modify'.")
                    
        except Exception as e:
            print(f"\nError generating model: {e}")
            print("Please try describing your modification differently.")

def result_code_generation(user_input):
    """
    Generate code for result analysis and visualization.
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    In pandas, the three columns in the 'result' DataFrame—labels, predictions, and score—record the true labels, predicted results, and model scores for each sample in the test set during the model evaluation (which can be used to calculate the ROC curve and AUC value).

    Please refer to the user's requirements to generate the code for evaluation 'model' or visualize the result.

    User requirement:
    "{user_input}"

    Return the content in JSON format, including the following fields:
    
    - code: the code to calculate and print the result.

    In the code, use print to output results to the user, while images should be saved to disk.
    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    return run_LLM_json_auto_retry(full_prompt)

def init_column_process(column_data, new_column_name):
    """Initialize column processing plan"""
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with data processing.
    
    Please create a plan to process the following column data and create a new column named '{new_column_name}'.
    
    Sample data (first 5 rows):
    {column_data.head().tolist()}
    
    Return the content in JSON format:
    - description: explanation of the processing plan
    - code: Python code to process the column
    
    Please respond with JSON only.
    """
    return run_LLM_json_auto_retry(full_prompt)

def process_absolute_exec(code, description, new_column_name, data):
    try:
        local_vars = {'data': data}
        exec(code, local_vars)
        return local_vars.get('data', data)
    except Exception as e:
        print(f"Error executing code: {e}")
        return data

def column_process_adjust(column_data, code, description, new_column_name):
    """Adjust column processing based on feedback"""
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with data processing.
    
    The previous plan was rejected. Please create a new plan.
    
    Previous code:
    {code}
    
    Previous description:
    {description}
    
    Sample data:
    {column_data.head().tolist()}
    
    New column name: {new_column_name}
    
    Return the content in JSON format:
    - description: new explanation
    - code: new Python code
    
    Please respond with JSON only.
    """
    return run_LLM_json_auto_retry(full_prompt)


def Feature_Recognition(data):
    """Recognize features in the data"""
    columns = list(data.columns)
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.
    
    Please identify which columns in the dataset should be used as features for the model.
    
    Available columns: {columns}
    
    Return the content in JSON format:
    - features: list of column names to use as features
    - label: column name to use as label
    - explain: explanation
    
    Please respond with JSON only.
    """
    return run_LLM_json_auto_retry(full_prompt)


def Feature_adaption(data, features):
    """Adapt features for the model"""
    return features


def absolute_exec(code, namespace):
    """Execute code in the given namespace"""
    try:
        exec(code, namespace)
        return True
    except Exception as e:
        print(f"Execution error: {e}")
        return False


def debug_agent(code, query, error, variation=None):
    """
    Use LLM to debug and fix code errors automatically.
    """
    if variation is not None:
        new_variation = {}
        for q, k in variation.items():
            if hasattr(k, 'shape'):
                new_variation[q] = k.shape
            else:
                new_variation[q] = str(type(k))
        variation = new_variation
    
    fix_hints = []
    
    if "is not defined" in error:
        match = re.search(r"name '(\w+)' is not defined", error)
        if match:
            undefined_var = match.group(1)
            fix_hints.append(f"variable '{undefined_var}' Undefined. Check for spelling errors")
    
    if "shapes cannot be multiplied" in error:
        fix_hints.append("Tensor shape mismatch. Check whether the input and output dimensions of the Linear layer are correct")
        shape_match = re.search(r'\((\d+)x(\d+)\) and \((\d+)x(\d+)\)', error)
        if shape_match:
            dims = shape_match.groups()
            fix_hints.append(f"current shape: ({dims[0]}x{dims[1]}) and ({dims[2]}x{dims[3]})")
            fix_hints.append(f"The input dimension of the Linear layer should be {dims[1]}, not {dims[2]}")
    
    hints_text = "\n".join(f"- {h}" for h in fix_hints) if fix_hints else "No specific suggestion"
    
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.
    The code has been generated according to the user's request or describe. However, an error occurred during execution. Please correct the code.
    
    Code:
    "{code}"
    User's query or code description:
    "{query}"
    Error or Dissatisfied needs:
    "{error}"
    
    SPECIFIC FIX HINTS:
    {hints_text}
    Current variable and their shape (which may sometimes be provided):
    "{variation}"
    The following content may be related to undefined functions:
    "
from models.mamba import mambalayer
device = 'cuda:2'
embedding_dim = 50
input = torch.randn(100, embedding_dim).float().to(device)
layer = mambalayer(embedding_dim=embedding_dim).float().to(device)
output = layer(input)
# IMPORTANT: Mamba does NOT support .double() or float64, always use .float()
# IMPORTANT: mambalayer does NOT have num_head parameter
# IMPORTANT: Mamba MUST run on CUDA, always define device='cuda:2' and use .to(device)
# IMPORTANT: Always include 'from models.mamba import mambalayer' when using Mamba
from models.self_attention import bertlayer
embedding_dim = 50
input = torch.randn(100, embedding_dim)
layer = bertlayer(embedding_dim, num_head=5)
output = layer(input)
    "
    
    CRITICAL RULES:
    1. 'land_feature' is WRONG, use 'ligand_feature'
    2. Check all variable names for typos
    3. Ensure Linear layer dimensions match tensor shapes
    4. For LSTM: use batch_first=False and unsqueeze(1) to add batch dimension
    5. For LSTM pooling: use hidden state rna_hidden[-1].squeeze(0), NOT mean(dim=1)
    
    Try to avoid naming new variables in most case and use the same variable name to demonstrate the flow of data.
    Since all data has been provided, please strictly ensure that the code does not generate sample data or use variables for demonstration, such as "feature_1 = torch.randn(42, embedding_dim)"
    Return the content in JSON format, including the following fields:
    - code: The corrected code when it does not meet the user's request, optional.
    All variables have been defined; please do NOT generate sample data.
    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    result = run_LLM_json_auto_retry(full_prompt)
    return result['code']


def check_and_fix_model_code(code, features, data, max_retries=50):
    """
    Check if model code can be executed and auto-fix if there are errors.
    
    Args:
        code: The model code to check
        features: List of feature names
        data: The dataset
        max_retries: Maximum number of fix attempts
        
    Returns:
        tuple: (fixed_code, success)
    """
    sample = data.iloc[0]
    
    for attempt in range(max_retries):
        try:
            namespace = {}
            for feature in features:
                namespace[feature] = torch.from_numpy(sample[feature])

            exec(code, namespace)

            model = namespace['sum_model']()
            inputs = [namespace[f] for f in features]
            output = model(*inputs)
            
            print(f"Model code validation successful!")
            return code, True
            
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1}/{max_retries}: Error - {error_msg[:100]}")

            fixed_code, was_fixed = SmartCodeFixer.auto_fix(code, error_msg)
            
            if was_fixed:
                code = fixed_code
                print("  -> Applied SmartCodeFixer")
                continue

            print("  -> Using debug_agent for fix...")
            
            variation = {}
            for feature in features:
                variation[feature] = torch.from_numpy(sample[feature])
            
            code = debug_agent(
                code=code,
                query="Build a deep learning model for RNA-Ligand interaction prediction",
                error=error_msg,
                variation=variation
            )

            code, _ = SmartCodeFixer.auto_fix(code)
    
    print(f"Failed to fix model code after {max_retries} attempts")
    return code, False


def debug_agent(code, query, error, variation=None):
    """
    Auto-debug function to fix code errors using LLM.
    """
    if variation is not None:
        new_variation = {}
        for q, k in variation.items():
            if hasattr(k, 'shape'):
                new_variation[q] = k.shape
            else:
                new_variation[q] = str(type(k))
        variation = new_variation
    
    fix_hints = []
    
    if "is not defined" in error:
        match = re.search(r"name '(\w+)' is not defined", error)
        if match:
            undefined_var = match.group(1)
            fix_hints.append(f"Variable '{undefined_var}' is undefined. Check for spelling errors")
    
    if "shapes cannot be multiplied" in error:
        fix_hints.append("Tensor shape mismatch. Check whether the input and output dimensions of the Linear layer are correct")
        shape_match = re.search(r'\((\d+)x(\d+)\) and \((\d+)x(\d+)\)', error)
        if shape_match:
            dims = shape_match.groups()
            fix_hints.append(f"Current shape: ({dims[0]}x{dims[1]}) and ({dims[2]}x{dims[3]})")
            fix_hints.append(f"The input dimension of the Linear layer should be {dims[1]}, not {dims[2]}")
    
    if "mat1 and mat2" in error:
        fix_hints.append("Matrix multiplication dimension mismatch")
        shape_match = re.search(r'\((\d+)x(\d+)\) and \((\d+)x(\d+)\)', error)
        if shape_match:
            dims = shape_match.groups()
            fix_hints.append(f"Current shape: ({dims[0]}x{dims[1]}) and ({dims[2]}x{dims[3]})")
            fix_hints.append(f"Need to change Linear input dimension from {dims[2]} to {dims[1]}")
    
    if "expected" in error.lower() and "got" in error.lower():
        fix_hints.append("Input/output shape or type mismatch")
    
    hints_text = "\n".join(f"- {h}" for h in fix_hints) if fix_hints else "No specific suggestion"
    
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.
    The code has been generated according to the user's request. However, an error occurred during execution. Please correct the code.
    
    Code:
    "{code}"
    
    User's query or code description:
    "{query}"
    
    Error message:
    "{error}"
    
    SPECIFIC FIX HINTS:
    {hints_text}
    
    Current variables and their shapes:
    "{variation}"
    
    IMPORTANT NOTES about model architecture:
    - Input data format: rna_feature shape is (seq_len, 641), ligand_feature shape is (seq_len, 30)
    - Data is processed ONE SAMPLE AT A TIME (not in batches)
    - For LSTM: use batch_first=False and unsqueeze(1) to add batch dimension
    - For pooling: use hidden state from LSTM, not mean over wrong dimension
    
    Reference for Mamba usage:
    ```
    from models.mamba import mambalayer
    device = 'cuda:2'
    embedding_dim = 50
    input = torch.randn(100, embedding_dim).float().to(device)
    layer = mambalayer(embedding_dim=embedding_dim).float().to(device)
    output = layer(input)
    # IMPORTANT: Mamba does NOT support .double() or float64, always use .float()
    # IMPORTANT: mambalayer does NOT have num_head parameter
    # IMPORTANT: Mamba MUST run on CUDA
    ```
    
    Reference for LSTM usage:
    ```
    # For single sample input (seq_len, input_size):
    rna_feature = rna_feature.unsqueeze(1)  # (seq_len, 1, input_size)
    rna_out, (rna_hidden, _) = self.rnn_rna(rna_feature)
    rna_pooled = rna_hidden[-1].squeeze(0)  # Use last hidden state
    ```
    
    CRITICAL RULES:
    1. 'land_feature' is WRONG, use 'ligand_feature'
    2. Check all variable names for typos
    3. Ensure Linear layer dimensions match tensor shapes
    4. For LSTM with single sample: add batch dim with unsqueeze(1), use hidden state for pooling
    5. Output must be compatible with BCELoss (sigmoid output in [0,1])
    
    Return the content in JSON format:
    - code: The corrected complete model code
    
    Please respond with JSON only. Do NOT provide any other explanation.
    """
    result = run_LLM_json_auto_retry(full_prompt)
    return result['code']


def check_and_fix_model_code(code, features, data, max_retries=50):
    """
    Check if model code can be executed and auto-fix if there are errors.
    
    Args:
        code: The model code to check
        features: List of feature names
        data: The dataset
        max_retries: Maximum number of fix attempts
        
    Returns:
        tuple: (fixed_code, success)
    """
    sample = data.iloc[0]
    
    for attempt in range(max_retries):
        try:
            namespace = {}

            exec(code, namespace)

            model = namespace['sum_model']()

            inputs = []
            for feature in features:
                inputs.append(torch.from_numpy(sample[feature]))
            
            output = model(*inputs)

            if output.numel() != 1:
                raise ValueError(f"Output should be single value, got shape {output.shape}")
            
            return code, True
            
        except Exception as e:
            error_msg = str(e)
            print(f"\nAttempt {attempt + 1}/{max_retries} - Error: {error_msg}")
            
            if attempt < max_retries - 1:
                print("Attempting auto-fix...")

                code, was_fixed = SmartCodeFixer.auto_fix(code, error_msg)
                
                if not was_fixed:
                    variation = {}
                    for feature in features:
                        variation[feature] = torch.from_numpy(sample[feature])
                    
                    code = debug_agent(
                        code=code,
                        query="RNA-Ligand binding prediction model",
                        error=error_msg,
                        variation=variation
                    )
                    code, _ = SmartCodeFixer.auto_fix(code)
                
                print("Fix applied, retrying...")
            else:
                print(f"Failed to fix after {max_retries} attempts")
                return code, False
    
    return code, False


def debug_agent(code, query, error, variation=None):
    """
    Use LLM to automatically debug and fix code errors.
    
    Args:
        code: The code that has errors
        query: User's original request or code description
        error: Error message from execution
        variation: Dictionary of variable names and their shapes
        
    Returns:
        Fixed code string
    """
    if variation is not None:
        new_variation = {}
        for q, k in variation.items():
            if hasattr(k, 'shape'):
                new_variation[q] = k.shape
            else:
                new_variation[q] = str(type(k))
        variation = new_variation
    
    fix_hints = []

    if "is not defined" in error:
        match = re.search(r"name '(\w+)' is not defined", error)
        if match:
            undefined_var = match.group(1)
            fix_hints.append(f"Variable '{undefined_var}' is undefined. Check for spelling errors.")
    
    if "shapes cannot be multiplied" in error:
        fix_hints.append("Tensor shape mismatch. Check whether the input and output dimensions of the Linear layer are correct.")
        shape_match = re.search(r'\((\d+)x(\d+)\) and \((\d+)x(\d+)\)', error)
        if shape_match:
            dims = shape_match.groups()
            fix_hints.append(f"Current shape: ({dims[0]}x{dims[1]}) and ({dims[2]}x{dims[3]})")
            fix_hints.append(f"The input dimension of the Linear layer should be {dims[1]}, not {dims[2]}")
    
    if "expected" in error.lower() and "got" in error.lower():
        fix_hints.append("Data type or dimension mismatch. Check tensor shapes and types.")
    
    if "CUDA" in error or "cuda" in error:
        fix_hints.append("CUDA error. Ensure all tensors are on the same device.")
    
    if "batch" in error.lower():
        fix_hints.append("Batch dimension issue. Check if unsqueeze(1) is needed for single sample input.")
    
    hints_text = "\n".join(f"- {h}" for h in fix_hints) if fix_hints else "No specific suggestion"
    
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.
    The code has been generated according to the user's request. However, an error occurred during execution. Please correct the code.
    
    Code:
    "{code}"
    
    User's query or code description:
    "{query}"
    
    Error message:
    "{error}"
    
    SPECIFIC FIX HINTS:
    {hints_text}
    
    Current variables and their shapes (if provided):
    "{variation}"
    
    REFERENCE FOR LAYERS:
    ```python
    # Mamba layer usage:
    from models.mamba import mambalayer
    device = 'cuda:2'
    embedding_dim = 50
    input = torch.randn(100, embedding_dim).float().to(device)
    layer = mambalayer(embedding_dim=embedding_dim).float().to(device)
    output = layer(input)
    # IMPORTANT: Mamba does NOT support .double() or float64, always use .float()
    # IMPORTANT: mambalayer does NOT have num_head parameter
    # IMPORTANT: Mamba MUST run on CUDA, always define device='cuda:2' and use .to(device)
    # IMPORTANT: Always include 'from models.mamba import mambalayer' when using Mamba
    
    # Self-attention layer usage:
    from models.self_attention import bertlayer
    embedding_dim = 50
    input = torch.randn(100, embedding_dim)
    layer = bertlayer(embedding_dim, num_head=5)
    output = layer(input)
    
    # LSTM usage (for single sample, not batch):
    # Input shape: (seq_len, input_size) e.g., (42, 641)
    # Need to add batch dimension: unsqueeze(1) -> (seq_len, 1, input_size)
    rna_feature = rna_feature.unsqueeze(1)  # Add batch dim
    rna_out, (rna_hidden, _) = self.rnn_rna(rna_feature)
    rna_pooled = rna_hidden[-1].squeeze(0)  # Get last hidden state
    ```
    
    CRITICAL RULES:
    1. 'land_feature' is WRONG, use 'ligand_feature'
    2. Check all variable names for typos
    3. Ensure Linear layer dimensions match tensor shapes
    4. For LSTM with single sample input, use batch_first=False and unsqueeze(1)
    5. Use hidden state (not output) for pooling in LSTM: rna_hidden[-1].squeeze(0)
    
    Try to avoid naming new variables in most cases and use the same variable name to demonstrate the flow of data.
    Since all data has been provided, please strictly ensure that the code does not generate sample data.
    
    Return the content in JSON format, including the following fields:
    - code: The corrected complete model code
    
    All variables have been defined; please do NOT generate sample data.
    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    result = run_LLM_json_auto_retry(full_prompt)
    return result['code']

def auto_fix_and_retry(code, query, namespace, max_retries=3):
    """
    Automatically fix code errors and retry execution.
    
    Args:
        code: Code to execute
        query: User's original request
        namespace: Execution namespace
        max_retries: Maximum number of retry attempts
        
    Returns:
        tuple: (success, final_code, namespace)
    """
    current_code = code
    
    for attempt in range(max_retries):
        try:
            current_code, _ = SmartCodeFixer.auto_fix(current_code)

            exec(current_code, namespace)
            print(f"Code executed successfully!")
            return True, current_code, namespace
            
        except Exception as e:
            error_msg = str(e)
            print(f"\nAttempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                print("Attempting automatic fix...")

                fixed_code, was_fixed = SmartCodeFixer.auto_fix(current_code, error_msg, namespace)
                
                if was_fixed:
                    print("SmartCodeFixer applied a fix.")
                    current_code = fixed_code
                else:
                    print("Using LLM debug agent...")
                    current_code = debug_agent(current_code, query, error_msg, namespace)
                    current_code, _ = SmartCodeFixer.auto_fix(current_code)
            else:
                print(f"Max retries reached. Last error: {error_msg}")
                return False, current_code, namespace
    
    return False, current_code, namespace
