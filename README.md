# dist-framework
Distance(s) and metric(s) calculation framework. A framework in python for discovering and testing new distances and metrics. This is an experimental program that permit to test for distance and metrics. You need to configure the script to make it works if not it will fail. 

## Model of calculation of new formulas
Sum up : 
![Phases]([[http://url/to/img.png](https://raw.githubusercontent.com/Lombard-Web-Services/dist-framework/refs/heads/main/phases.png)


## Install
Load libraries :
```
pip3 install --upgrade scikit-learn numpy pandas matplotlib seaborn scipy POT sympy umap-learn ripser persim torch
```

Execute : 
```
chmod +x dist-framework.py
```

## Convert the results to pdf :
```
pdflatex distance_formulas.tex
```

## Usage 
Create a folder named documents, paste the data you want to find distances with, text files IE. 
Configure and execute the script.
