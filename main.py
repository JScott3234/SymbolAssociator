import tkinter as tk
import drawingWindow as dw
import numpy as np
from brian2 import *


if __name__ == "__main__":
    root = tk.Tk()
    app = dw.drawingWindow(root)
    root.mainloop()
