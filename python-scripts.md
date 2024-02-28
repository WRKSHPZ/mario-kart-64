In de [Tensorkart](https://github.com/WRKSHPZ/Tensorkart) repository vind je een aantal python scripts:

| Naam         | Stap | Beschrijving |
| ------------ | ---- | ------------ |
| [flip-data.py](#flip-datapy) | 2    | Script om trainingsrun te spiegelen |
| [manual.py](#manualpy)    | 5    | Script om resultaat van model te toetsen aan de trainingsdata |
| play.py      | 5    | **Ongebruikt** - Gebruik gym-mypen64plus
| [record.py](#recordpy)   | 1    | Script om de mupen64plus N64 emulator op te nemen en tagging data te genereren |
| [train.py](#trainpy)     | 4    | Script om middels Tensorflow het model daadwerkelijk te trainen |
| [utils.py](#utilspy)    | 3    | Script om de trainingsdata geschikt te maken voor het trainen van het model |

#### flip-data.py
Script om trainingsrun data te spiegelen

Syntax
```python ./flip-data.py [folder]```
- [folder] = Full name of folder 
-----
#### manual.py
Script om resultaat van model te toetsen aan de trainingsdata

Syntax
```python ./manual.py [folder]```
- [folder] = Full name of folder 
-----
#### record.py 
Script om de mupen64plus N64 emulator op te nemen en tagging data te genereren

Syntax
```python ./record.py```

-----
#### train.py 
Script om middels Tensorflow het model daadwerkelijk te trainen

Syntax
```python ./train.py```

-----
#### utils.py 
Script om de trainingsdata geschikt te maken voor het trainen van het model

Syntax per folder ```python ./utils.py prepare [folder]```

- [folder] = Full name of folder 

Syntax alle subfolders in samples folder ```python ./utils.py prepare samples/*```

-----