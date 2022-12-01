
# Intro
```
do data science work here
```

### Installation 
#### &#x23f5; pyhhmm 
```python 
# for apple silicon to work with HMM
pip install pyhhmm 
```

### Configuration
#### &#x23f5; Jupyter notebook default browser 
```
$ cd ~/.jupyter
$ jupyter notebook --generate-config 
$ vim jupyter_notebook_config.py 

# seach for /browser keyword 
# modify this line: c.NotebookApp.browser = 'open -a /Applications/Google\ Chrome.app %s'
```

#### &#x23f5; Jupyter Notebook Layout 
```
$ cd ~/.jupyter 
$ mkdir custom 
$ cd custom 
$ vim custom.css
```
##### add the .css file
```css 
/* default cell width */
.container {
    width: 80% !important;
}
```
