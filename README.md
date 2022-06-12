# Modeling learning behavior of Learnnavi users

*Authors: Marie Biolková, Ghali Chraibi*

### Abstract

Knowledge tracing is used widely for study learning behavior and is important for improving educational technologies (EdTechs), which are becoming increasingly popular. We conduct an extensive study and compare various knowledge tracing models using data collected from an EdTech start-up, Lernnavi. Focusing on their Mathematics track only, we found that GRU with 16 recurrent units can best model the mastery. It achieves an AUC of 0.677 and RMSE of 0.476 on the task of predicting whether the student will answer correctly or not. We also analyzed the learning curves and discussed possible improvements.

### Requirements

We used Python 3.7. To install all dependencies, run `pip install -r requirements.txt`. 

### File structure

- `learnnavi_knowledge_tracing_analysis.ipynb`: Jupyter notebook with all our pre-processing, modeling and evaluation. 

- `helpers.py`: Useful functions called from the notebook.

- `img/`: Directory for exported images.

- `results/`: Contains logs from optimization. 

  
