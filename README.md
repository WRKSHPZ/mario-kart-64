We gaan deze workshop aan de slag met machine learning in Python en Tensorflow om een agent te trainen om Mario Kart 64 te spelen.

Als basis voor deze workshop is het project [Tensorkart](https://github.com/kevinhughes27/TensorKart) van **Kevin Hughes** gebruikt. Dit project was echter al 8 jaar oud en gebaseerd op Python 2.7. De requirements die nodig zijn voor dit project zijn daarmee niet eenvoudig uit de standaard package management sources te vinden. We hebben zijn project geforkt en geschikt gemaakt voor Python 3.11.

De fork van deze repository vind je hier: https://github.com/WRKSHPZ/Tensorkart

## Prerequisites
We hebben [Python 3.11](https://www.python.org/downloads/release/python-3110/) nodig. Dit is qua versie precies de sweet spot dat de meeste zaken in het TensorKart project werken, op de **play.py** na die een dependency heeft op XFVB (virual framebuffer) die ik op Windows 11 niet werkend kreeg.

Daarnaast hebben we [Docker Desktop](https://www.docker.com/products/docker-desktop/) nodig voor het alternatief wat we hebben voor de **play.py** om je eigen model in actie te zien.

## Workshop intro
Aangezien we een machine learning model gaan trainen om Mario Kart 64 te spelen, doorlopen we de 'standaard' stappen die horen bij het trainen van een ML model:
- Trainingsdata verzamelen en taggen
- Trainingsdata geschikt maken voor verwerking
- Trainen van het model
- Verifiëren van de performance van het model

Dit is overigens geen kinderachtig model, maar een [Tensorflow implementatie](https://github.com/SullyChen/Autopilot-TensorFlow) van een model wat door Nvidia is ontwikkeld voor de aansturing van zelfrijdende auto's. Meer over dit model lees je in deze [whitepaper](https://arxiv.org/pdf/1604.07316.pdf)

## Stap 0: Installeren prerequisites
Uitgebreide info over het installeren van de prerequisites vind je hier: [link](prerequisites.md)

TLDR:
- Installeren Python 3.11 (vergeet je PATH environment variable niet aan te passen)
- Docker Desktop
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
- [Mupen64plus](https://github.com/mupen64plus/mupen64plus-core/releases/tag/2.5.9)
- [Mario kart 64 ROM](https://mariokart.blob.core.windows.net/data/mario-kart-64.z64)
- Repositories downloaden
https://github.com/WRKSHPZ/Tensorkart, https://github.com/WRKSHPZ/gym-mupen64plus

## Stap 1: Trainingsdata verzamelen en taggen
In de [Tensorkart](https://github.com/WRKSHPZ/Tensorkart) repository vind je een aantal [python scripts](python-scripts.md). Een van die scripts is **record.py** die we in de eerste stap gaan gebruiken voor het genereren van onze training data:

1. Sluit een USB gamepad aan en controleer of deze herkend wordt in het systeem.
1. Open de Tensorkart repository in Visual Studio Code
1. Open een terminal en start record.py middels: ```python .\record.py```
1. Er start nu een GUI. Linksboven zie je 640x480 pixels uit de linkse bovenhoek van je scherm. Rechts zie je de input van de USB gamepad. Onderin een map waar de screenshots (frames) en joystick input naar wordt weggeschreven en een 'Record' knop.
1. Kopieer nu de [Mario kart 64 ROM](https://mariokart.blob.core.windows.net/data/mario-kart-64.z64) naar de mupen64plus map en start in een nieuwe terminal in vs code de mupen64plus emulator met de ROM middels: ```.\mupen64plus-ui-console.exe  mario-kart-64.z64```
1. Mario kart 64 start nu in de emulator en je ziet een draaiend, gouden, Nintendo logo. Deze emulator moet je zo goed mogelijk in de linker bovenhoek van je scherm positioneren zodat de GUI van **record.py** zoveel mogelijk data kan verzamelen.
1. Ga nu met de USB controller in Mario kart 64 naar de Luigi Raceway time trials met Mario als coureur. Hiervoor druk je op start tot je het 'Game select' scherm ziet. Start een 1p game, kies voor T.Trials, klik begin en klik Ok. Selecteer Mario als coureur, ga naar de Mushroom cup en kies voor Luigi raceway.
1. Als Lakitu (de schildpad die het startsignaal verzorgt) is verdwenen, kan je gaan opnemen. Klik op de 'Record' knop in de **record.py** GUI en ga racen.
1. Je kan de race helemaal afmaken, of eerder stoppen, maar bij machine learning is het eigenlijk altijd: hoe meer data, des te beter het is. Om te stoppen, klik je op de 'Stop' knop in de **record.py** GUI.
1. De data is opgeslagen onder het samples mapje in het Tensorkart project onder de door jou gespecificeerde map in de GUI. Controleer of de screenshots er staan en open het **data.csv** bestand

### Trainingsdata
Het **record.py** script heeft voor elke screen capture van de emulator de stuurhoek van je joystick vastgelegd in de **data.csv**
In de data file vind je voor elke screenshot een rij met 6 kolommen met data, gescheiden door een komma:

```samples/2024-02-20_231645/img_0.png,0.00390625,0.00390625,0,0,0```

| Kolom | Voorbeeld data                      | Type                 | Waarde                                  |
| ----- | ----------------------------------- | -------------------- | --------------------------------------- |
| 1     | samples/2024-02-20_231645/img_0.png | String               | Bestandsnaam screenshot                 |
| 2     | 0.00390625                          | Float tussen -1 en 1 | Genormaliseerde X-as-waarde joystick    |
| 3     | 0.00390625                          | Float tussen -1 en 1 | Genormaliseerde Y-as-waarde joystick    |
| 4     | 0                                   | 0 of 1               | A-knop (gas)                            |
| 5     | 0                                   | 0 of 1               | RB-knop (ongebruikt)                    |
| 6     | 0                                   | 0 of 1               | RT-knop (ongebruikt)                    |

Het circuit Luigi raceway is voor 95% bochten naar links. Tussen de python scripts zit **'flip-data.py'**. Hiermee kan je screenshot en X-as-stuurwaarden spiegelen en verdubbelt dus je trainingsdata!

### Training tips
- Houd het gas continu ingedrukt voor een consistente training. Doel is om met name de X-as-waarde te trainen
- Maak soepele bochten. Je hebt vaak de neiging om schokkerig bij te sturen, maar dat betekent dat je op 1 of 2 frames en scherpe waarde hebt (-1 voor links of 1 voor rechts), maar de tussenliggende frames geen joystick waarde registreert. Het model gaat elke frame proberen  te relateren aan een stuurwaarde, dus als je daar vaker 0 traint dan een stuurwaarde, gaat Mario alleen maar rechtdoor.
- Houdt training data consistent. Er zijn een aantal visuele zaken die je kan aanpassen met de rechter analoge stick. Je kan verder uitzoomen, of een snelheidsmeter tonen, maar wat je ook kiest, zorg dat je consistentie hebt in al je trainingdata.
- Denk na over het effect van de GUI op de training. Heb je wellicht een uitsnede nodig om bepaalde elementen uit te sluiten?
- Denk na over welke circuits handig zijn om mee te trainen? Moeten deze juist op elkaar lijken, of is het juist handig om te trainen met hele andere circuits?
- Denk na over foutsituaties. Als Mario ooit tegen een muur aanrijdt, wie leert hem dan aan die situatie te ontsnappen?
- In hoeverre zijn we het model aan het trainen op de stand van de auto? Is het wellicht zuiverder de auto en de coureur uit het plaatje te verwijderen? Het model krijgt bij een screenshot van een auto die een bocht maakt, een bepaalde stuurhoek aangeboden. Als we het model dan trainen dat het gaat draaien als het model de auto ziet draaien, gaan we altijd alleen maar rechtdoor.
- De meeste Mario 64 tracks hebben meer bochten naar links, dan naar rechts. Moeten we hier nog rekening mee houden?
- Moet je juist in het midden van de track blijven, of juist de kanten opzoeken? Moet je beide trainen?
- Het rijden van rechte stukken, genereert veel trainingsdata over rechtdoor rijden. Beïnvloedt dat het resultaat wellicht teveel? Moet je de eerste en laatste trainingsdata frames wellicht uit de data.csv verwijderen?
11. Herhaal de stappen als je meer trainingsdata wilt genereren, of vul je trainingsdata aan met runs die je kan downloaden van [deze pagina](datasets.md)

## Stap 2: Trainingsdata geschikt maken voor verwerking
In de Tensorkart repository vind je een aantal python scripts. Een van die scripts is **utils.py** die we in de eerste stap gaan gebruiken voor het prepareren van onze training data:

1. Open de Tensorkart repository in Visual Studio Code
1. Open een terminal en start **utils.py** middels: ```python .\utils.py prepare samples/*```. Met dit commando worden alle folders onder samples verwerkt en toegevoegd aan de training data. Als je een specifieke folder wil training, gebruik je **samples/[foldernaam]**

Het verwerken van de trainingsdata maakt middels de NumPy library twee arrays van data. Eentje (**X.npy**) bevat de screenshots omgezet naar arrays van pixel waarden en de ander (**y.npy**) bevat de joystick waarden die we in stap 1 getraind hebben. Door ze in deze arrays te zetten, kan Tensorflow deze gebruiken om het model te trainen in stap 3. Een index op de array van screenshots in X komt overeen met de index van de joystick waarden in Y.

## Stap 3: Trainen van het model
Tensorflow wordt gebruikt om ons Mario model te trainen. Hier gebruiken we het script **train.py** voor: ```python .\train.py```
Het script pakt de npy-bestanden uit het data mapje op en gaat hiermee trainen.

Als je het **train.py** bestand opent, staan tussen regel 63 en 75 een aantal hele relevante waarden voor onze training job:

**epochs**: Het aantal training cycli van je data. In elke epoch wordt alle trainingsdata door je model gehaald voor de training en gevalideerd met het percentage van de validatie set (regel 73: validation_split=0.2).

**batch_size**: Het aantal datapoints wat verwerkt wordt, voordat je model weer wordt ge-update. Grote batch size kan leiden tot overfitting. Kleine batch_size zorgt voor langere trainingsduur.

**learning_rate**: Hoe snel je model 'leert' en wordt ge-update door je trainingsdata. Het is een parameter die bepaalt hoe snel je model kan veranderen. Grote waarde kan leiden tot niet stabiele trainingsruns en een te kleine waarde zorgt ervoor dat er onvoldoende geleerd wordt.

Als het trainen klaar is (alle N aantal epochs), heb je een HDF5 model (**model_weights.h5**). Dit is een legacy model wat in feite al is gedeprecate door Tensorflow, maar alle scripts in dit project werkten hier al mee.

## Stap 4: Verifiëren van de performance van het model
Maak je nog geen illusies. Als je één set trainingsdata hebt gemaakt en gespiegeld, heb je vermoedelijk nog niet voldoende trainingsdata voor een model wat het circuit rond komt. Machine learning heeft **veel** data nodig en dan vooral **veel, kwalitatief goede** data. 

Er zijn een aantal [datasets](datasets.md) te downloaden om in volgende runs aan je trainingsdata toe te voegen. **N.B. Let wel, dit is ook geen perfecte data**

We willen nu het model in actie zien. Helaas is het **play.py** script niet in een werkende staat. 

Hier hebben we een andere repo voor nodig: [gym-mupen64plus](https://github.com/WRKSHPZ/gym-mupen64plus). Gym slaat op de OpenAI Gym wrapper en is dus een wrapper voor de mupen64plus emulator en stelt modellen in staat om agents in de games aan te sturen. Dit is dus niet beperkt tot Mario Kart 64.

1. Ga naar de folder waar je de repo hebt gecloned.
1. Kopieer het gegenereerde model (**model_weights.h5**) naar de root. Deze wordt in de **example.py** gebruikt om de kart in de emulator in de Docker container aan te sturen.

De architectuur ziet er zo uit (dit komt uit de documentatie van gym-mupen64plus): 
![Gym-Mupen64Plus architecture](image.png)
De rode stippellijnen zijn de container boundaries.

3. Open nu een terminal en trap de containers af middels: ```docker-compose up --build -d```
1. De eerste build duurt wat langer door het ophalen van de packages en dependencies, maar als alle 4 de containers gestart zijn, open dan de RealVNC vieuwer en maak een connectie naar ```localhost:5900```
1. Via de VNC viewer kunnen we bekijken welke info er over de Framebuffer container wordt gestuurd.

## Stap 5: Lather, rinse, repeat
Nu we de training pipeline hebben staan, kunnen we het model gaan fine-tunen, zodat deze een ronde rond het circuit kan rijden. Check hiervoor ook nog eens de [training tips](#training-tips) hierboven.
