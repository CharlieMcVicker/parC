import os
import sys
import dotenv
import uvicorn
from constants import get_yaml_dir

dotenv.load_dotenv("parC.env")

if not os.environ.get("get_yaml_dir()"):
    from tkinter import filedialog, Tk

    def pick_directory():
        root = Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()
        with open(".env", "w") as f:
            f.write(f"get_yaml_dir()={folder_selected}")

    pick_directory()

if not get_yaml_dir():
    sys.exit("No config directory selected.")

if __name__ == "__main__":
    uvicorn.run("parC.api:app", host="127.0.0.1", port=8000, reload=True)
