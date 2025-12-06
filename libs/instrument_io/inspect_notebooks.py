
from pathlib import Path
from rich.console import Console
from rich.tree import Tree

def inspect_notebooks():
    console = Console()
    notebooks_dir = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab\Notebooks")
    
    # Get all notebook folders
    users = [d for d in notebooks_dir.iterdir() if d.is_dir()]
    
    for user in users:
        # Create a tree for each user
        tree = Tree(f"[bold blue]{user.name}[/bold blue]")
        
        # Get all files, relative to user dir
        # Limit depth to avoid huge trees if they have thousands of files
        # We'll just walk top 2-3 levels
        
        has_files = False
        for path in user.rglob('*'):
            if path.name.startswith(".") or path.name.startswith("~"): continue
            
            # Calculate depth relative to user dir
            rel_path = path.relative_to(user)
            parts = rel_path.parts
            
            if len(parts) > 3: continue # Skip too deep
            
            if path.is_file():
                has_files = True
                # finding/creating the branch
                current_branch = tree
                for part in parts[:-1]:
                    # Simple search for existing branch (not perfect but okay for visualization)
                    found = False
                    for child in current_branch.children:
                        if str(child.label) == part:
                            current_branch = child
                            found = True
                            break
                    if not found:
                        current_branch = current_branch.add(part)
                
                # Add file
                filename = parts[-1]
                if filename.lower().endswith(('.xlsx', '.xls')):
                    icon = "📊"
                    style = "green"
                elif filename.lower().endswith(('.docx', '.doc')):
                    icon = "📝"
                    style = "blue"
                elif filename.lower().endswith('.pdf'):
                    icon = "📕"
                    style = "red"
                elif filename.lower().endswith(('.py', '.r', '.m')):
                    icon = "💻"
                    style = "yellow"
                else:
                    icon = "📄"
                    style = "white"
                    
                current_branch.add(f"{icon} [{style}]{filename}[/{style}]")

        if has_files:
            console.print(tree)
            console.print("")

if __name__ == "__main__":
    inspect_notebooks()
