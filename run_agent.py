import os

# 检查并删除 PYTHONPATH
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH'] 
    
from larger.agent_controller import AgentController

if __name__ == "__main__":
    print("=== Welcome to LargeR ===")
    print("This is an interactive agent for RNA-Ligand modeling.\n")

    agent = AgentController()
    agent.run_pipeline()
