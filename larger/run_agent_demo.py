from agent_controller_new import AgentController

if __name__ == "__main__":
    print("=== Welcome to RLAgent ===")
    print("This is an interactive agent for RNA-Ligand modeling.\n")

    agent = AgentController()
    agent.run_pipeline()
