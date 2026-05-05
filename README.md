<h1>Symbol Recognition System</h1>

<h3>Components</h3>
Python - typical numpy
Drawing Window - tkinter UI, matplotlib
Izhikevich STDP- brian2
Model State - weights.npy

<h3>Setup:</h3>
Should be easy! Just install listed software in the requirements.txt
' pip install -r requirements.txt '

<h3>How to Run:</h3>
In terminal at project folder, ' python3 main.py '

<h3>How to Train:</h3>
After getting a result that you find unfitting, select the preferred symbol in the dropdown, and press the "train" button.
Fires the synapses for selected symbol several times, run the recognize button a couple times to relax plasticity.
<br/>
To get rid of or make new weights, all that has to be done is remove the weights.npy from the folder