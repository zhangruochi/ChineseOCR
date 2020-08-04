from pathlib import Path
root = Path("data")
for direc in root.glob("*"):
    if direc.is_dir():
        source = direc / "0"
        target = direc / "1"

        
        target_set = set()
        for t_char in target.glob("*.png"):
            target_set.add(t_char.name[0])
        
        for s_char in source.glob("*.png"):
            if s_char.name[0] not in target_set:
                print(direc.name)
                print(s_char.name[0])