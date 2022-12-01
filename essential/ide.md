

### Configuration

#### &#x0275; Run Shell Command in Jupyter
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

#### &#x0275; Jupyter notebook default browser 
```
$ cd ~/.jupyter
$ jupyter notebook --generate-config 
$ vim jupyter_notebook_config.py 

# seach for /browser keyword 
# modify this line: c.NotebookApp.browser = 'open -a /Applications/Google\ Chrome.app %s'
```

#### &#x0275; test config in the cell 
```
1) width 
-----------------------------------------------------------------------------
|                                                                           |   
|   from IPython.core.display import display, HTML                          |
|   display(HTML("<style>.container { width:85% !important; }</style>"))    |
|                                                                           |
-----------------------------------------------------------------------------

2) font
---------------------------------
|                               |
|   %%html                      |
|   <style type='text/css'>     |
|   .CodeMirror{                |
|   font-size: 14px;            |
|   </style>                    |
|                               |
---------------------------------
```

#### &#x0275; Jupyter Notebook Layout 
```
$ cd ~/.jupyter 
$ mkdir custom 
$ cd custom 
$ vim custom.css
```

##### &#x2192; add the .css file
```css 
/* default cell width */
.container {
    width: 85% !important;
}

.CodeMirror, .CodeMirror pre, .CodeMirror-dialog, 
.CodeMirror-dialog .CodeMirror-search-field, 
.terminal-app .terminal {
    font-size: 101% !important;
    line-height: 2rem;
    font-family: 'Ubuntu', 'Monaco' !important;
}
```


