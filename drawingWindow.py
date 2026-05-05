import numpy as np
import tkinter as tk
import identifier as id
from brian2 import ms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Constants
gridSize = 10
cellSize = 20

class drawingWindow:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Izhikevich Letter Recognizer")
        
        self.grid = np.zeros((gridSize, gridSize))
        
        # Canvas
        self.canvas = tk.Canvas(root, width=gridSize*cellSize, height=gridSize*cellSize, bg='white')
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        # Graph tracking variables
        self.graphWindow = None
        self.canvasPlot = None
        self.ax = None
        
        btnFrame = tk.Frame(root)
        btnFrame.pack()
        
        # Clear Screen
        self.clearBtn = tk.Button(btnFrame, text="Clear", command=self.clearCanvas)
        self.clearBtn.pack(side=tk.LEFT, padx=5)
        
        # Identification Button
        self.recogBtn = tk.Button(btnFrame, text="Identify", command=self.recognize)
        self.recogBtn.pack(side=tk.LEFT, padx=5)
        
        # Letter Dropdown
        self.targetLetter = tk.StringVar(root)
        self.targetLetter.set(id.LETTERS[0]) # default A
        self.dropdown = tk.OptionMenu(btnFrame, self.targetLetter, *id.LETTERS)
        self.dropdown.pack(side=tk.LEFT, padx=5)
        
        # Training Button
        self.trainingBtn = tk.Button(btnFrame, text="Train", command=self.train)
        self.trainingBtn.pack(side=tk.LEFT, padx=5)
        
        self.resultLabel = tk.Label(root, text="Draw a letter and click Recognize", font=("Arial", 12))
        self.resultLabel.pack(pady=10)
        
        self.drawGrid()
        
    def drawGrid(self):
        for i in range(gridSize):
            self.canvas.create_line(i*cellSize, 0, i*cellSize, gridSize*cellSize, fill='gray')
            self.canvas.create_line(0, i*cellSize, gridSize*cellSize, i*cellSize, fill='gray')
      
    def paint(self, event):
        x, y = event.x, event.y
        col = x // cellSize
        row = y // cellSize
        
        if 0 <= row < gridSize and 0 <= col < gridSize:
            if self.grid[row, col] == 0:
                self.grid[row, col] = 1
                self.canvas.create_rectangle(col*cellSize, row*cellSize, (col+1)*cellSize, (row+1)*cellSize, fill='black')
                
    def clearCanvas(self):
        self.canvas.delete("all")
        self.drawGrid()
        self.grid = np.zeros((gridSize, gridSize))
        self.resultLabel.config(text="Cleared!")
        
    def recognize(self):
        self.resultLabel.config(text="Running nodes...")
        self.root.update()
        
        flatGrid = self.grid.flatten()
        resultText, spikes = id.letterIdentification(flatGrid)
        
        self.resultLabel.config(text=f"Result: {resultText}")
        if spikes is not None:
            self.updateGraph(spikes)

    def train(self):
        self.resultLabel.config(text="Training...")
        self.root.update()
        
        flatGrid = self.grid.flatten()
        targetLetter = self.targetLetter.get()
        targetIndex = id.LETTERS.index(targetLetter)
        
        resultText, spikes = id.letterIdentification(flatGrid, train_target=targetIndex)
        
        self.resultLabel.config(text=f"Trained {targetLetter}! {resultText}")
        if spikes is not None:
            self.updateGraph(spikes)

    def updateGraph(self, spikes):
        # Create the window if it doesn't exist
        if self.graphWindow is None or not tk.Toplevel.winfo_exists(self.graphWindow):
            self.graphWindow = tk.Toplevel(self.root)
            self.graphWindow.title("Spike Raster Plot")
            
            fig = Figure(figsize=(6, 4), dpi=100)
            self.ax = fig.add_subplot(111)
            
            self.canvasPlot = FigureCanvasTkAgg(fig, master=self.graphWindow)
            self.canvasPlot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        # Update the plot
        self.ax.clear()
        
        # Plot times vs indices
        times = np.array(spikes.t / ms)
        indices = np.array(spikes.i)
        
        # Graph Settings
        self.ax.scatter(times, indices, color='black', marker='|', s=100)
        self.ax.set_xlim(0, 1000) # Runtime duration, I really need to make this an exportable constant
        self.ax.set_ylim(-0.5, len(id.LETTERS) - 0.5)
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Output Neuron')
        self.ax.set_yticks(range(len(id.LETTERS)))
        self.ax.set_yticklabels(id.LETTERS)
        self.ax.set_title("Lateral Inhibition Spike Plot")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvasPlot.draw()