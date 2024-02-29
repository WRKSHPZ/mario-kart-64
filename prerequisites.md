## Prerequisites

#### Hardware
 - Xbox-like USB controller
 - Maximale resolutie van je 'main screen' van 1920x1080 pixels. Het spelen en capturen van de Mario Kart emulator gebeurt altijd vanaf Display:0 = je main screen

#### Software
- [Python 3.11](https://www.python.org/downloads/release/python-3110/) (vergeet je [PATH](#Path) environment variable niet aan te passen)
- Pip behorende bij de juiste Python versie (3.11) (vergeet je [PATH](#Path) environment variable voor pip niet aan te passen, deze staat vaak in een Scripts submap bij de Python installatie)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
- [Mupen64plus](https://github.com/mupen64plus/mupen64plus-core/releases/tag/2.5.9)

#### Path
Als er meerdere Python versies zijn geïnstalleerd, kan het onduidelijk zijn welke Python versie en welke package manager (pip) je gebruikt. Dit kan je in de command line controleren met de commandos ```python -V``` en ```pip -V```. Op Windows ga je naar je **System properties** en kies je voor **Environment variables**. Hier kan je de Path (system) variabele aanpassen. Zorg ervoor dat de paden voor Python 3.11 en de pip versie die hoort bij Python 3.11 hoger staan dan de andere Python versies. Pip staat veelal in een subfolder 'Scripts' onder de Python installatie folder.

#### Files
- [Mario kart 64 ROM](https://mariokart.blob.core.windows.net/data/mario-kart-64.z64)
- [Training data](datasets.md)


#### Repositories
- https://github.com/WRKSHPZ/Tensorkart
- https://github.com/WRKSHPZ/gym-mupen64plus

#### Dependencies (PIP)
- https://github.com/mupen64plus/mupen64plus-core/releases/tag/2.5.9
