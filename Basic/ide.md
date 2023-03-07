

## &#x03e8; Usage

#### &#x21f0; Run Shell Command in Jupyter
```
# add exclamation mark before the command
!pwd

# list theme 
!pip install jupyterthemes
!jt -l

# change theme
!jt -t <name of theme>

# revert change 
!jt -r
```

#### &#x21f0; Jupyter notebook default browser 
```
$ cd ~/.jupyter
$ jupyter notebook --generate-config 
$ vim jupyter_notebook_config.py 

# seach for /browser keyword 
# modify this line: c.NotebookApp.browser = 'open -a /Applications/Google\ Chrome.app %s'
```
#### &#x21f0; NBextension

##### &#x2570; collapse/folding
```
$ conda activate base
$ conda install -c conda-forge jupyter_contrib_nbextensions
$ pip install jupyter-contrib-nbextensions
$ jupyter contrib nbextension install --user

# 1) re-open jupyter notebook 
# 2) toolbar
# 3) Nbextension 
# 4) uncheck disable configuration for nbextension 
# 5) check Collapsible headings 
# 6) over
# 7) other option: Code Folding, Code Folding in Editor => click on them you will see how it works

# if lots of yellow error occurs during execution of jupyter notebook: 
#   -> check if duplicate Jupyter directory exist inside ~/Library/
```

## &#x03e8; UI Configuration

#### &#x21f0; Notebook Layout
```
$ cd ~/.jupyter 
$ mkdir custom 
$ cd custom 
$ vim custom.css
```
##### &#x2570; add .css file
```css 
/* default cell width */
/* adding the !important constraint, you won't be able to change config inside cell */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code&family=Montserrat+Alternates&family=Poiret+One&family=Ubuntu&display=swap');


.container {
    width: 85% !important;
}

.CodeMirror, .CodeMirror pre, .CodeMirror-dialog, 
.CodeMirror-dialog .CodeMirror-search-field, 
.terminal-app .terminal {
    font-size: 101%;
    line-height: 2rem;
    font-family: 'Ubuntu', 'Fira Code', 'Monaco', 'Poiret One';
}
```


#### &#x21f0; Test Config
`: For other people to see your configuration, add the code on the top cell`
##### &#x2570; width 
```python
from IPython.core.display import display, HTML                         
display(HTML("<style>.container { width:85% !important; }</style>"))
```

##### &#x2570; Font
```python
%%html                      
<style type='text/css'>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code&family=Montserrat+Alternates&family=Poiret+One&family=Ubuntu&display=swap');
.CodeMirror{
    font-size: 101%;
    line-height: 2rem;
    font-size: 'Ubuntu';
</style>
```






