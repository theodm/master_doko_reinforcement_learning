# Adaption von MCTS und AlphaZero für Spiele mit imperfekter Information am Beispiel Doppelkopf
Hier befindet sich der Source Code, der im Rahmen der Abschlussarbeit "Adaption von MCTS und AlphaZero für Spiele mit imperfekter Information am Beispiel Doppelkopf" an der Fernuniversität in Hagen unter Betreuung von [Dr. Fabio Valdés](https://www.fernuni-hagen.de/mi/fakultaet/lehrende/valdes/index.shtml) enstand.

Die schriftliche Ausarbeitung ist im Repository unter [/docs/final.pdf](/docs/final.pdf) zu finden. Die zugehörige Abschlusspräsentation ist unter [/docs/final.pptx](/docs/final.pptx).

## Zusammenfassung der Arbeit
Das automatische Spielen von Brett- und Computerspielen dient traditionell
als wichtiges Testfeld und Anwendungsgebiet für Fortschritte in der künstli-
chen Intelligenz (KI). Während für Spiele mit perfekter Information, wie Go
oder Schach, bereits herausragende Erfolge erzielt wurden, stellen Spiele mit
imperfekter Information, bei denen Spieler nur einen Teil des Spielzustands
beobachten können, aufgrund der daraus resultierenden Unsicherheiten nach
wie vor eine große Herausforderung dar.

Diese Arbeit untersucht, wie das etablierte Suchverfahren Monte Carlo Tree
Search (MCTS) sowie das Lernverfahren AlphaZero, die in ihrer Grundform
nur auf Spiele mit perfekter Information anwendbar sind, so erweitert werden
können, dass sie auch auf Spiele mit imperfekter Information angewendet wer-
den können. Hierfür wird der Ansatz der Determinisierung verfolgt: Aus der
unvollständigen Beobachtung eines Spielers werden zunächst ein oder meh-
rere plausible vollständige Spielzustände rekonstruiert. Auf diesen rekonstru-
ierten Zuständen werden anschließend der MCTS und AlphaZero ausgeführt.
Für die Zustandsrekonstruktion kommen sowohl ein statischer, regelbasierter
Algorithmus als auch ein trainierbares autoregressives Modell, basierend auf
der Transformer-Architektur, zum Einsatz.

Der vorgestellte Ansatz wird exemplarisch am deutschen Kartenspiel Dop-
pelkopf evaluiert. Als Stichspiel mit vier Spielern, die in teils unbekannten
Partnerschaften agieren, bietet Doppelkopf strategische Tiefe und dient als
geeignetes Testfeld für die entwickelten Methoden. Die Evaluierung der resul-
tierenden Spielstärke anhand mehrerer spielspezifischer Metriken zeigt, dass
das autoregressive Modell zuverlässig plausible Zustände generiert und der
Determinisierungsansatz zu einer kompetenten, wenn auch nicht optimalen,
Spielstrategie führt. Die Arbeit beleuchtet abschließend die konzeptionellen
und praktischen Limitationen dieses Ansatzes und die Anwendbarkeit auf
andere Spiele.

## Abschlusspräsentation
![Folie 1](docs/Slide1.jpg)
![Folie 2](docs/Slide2.jpg)
![Folie 3](docs/Slide3.jpg)
![Folie 4](docs/Slide4.jpg)
![Folie 5](docs/Slide5.jpg)
![Folie 6](docs/Slide6.jpg)
![Folie 7](docs/Slide7.jpg)
![Folie 8](docs/Slide8.jpg)
![Folie 9](docs/Slide9.jpg)
![Folie 10](docs/Slide10.jpg)
![Folie 11](docs/Slide11.jpg)
![Folie 12](docs/Slide12.jpg)
![Folie 13](docs/Slide13.jpg)
![Folie 14](docs/Slide14.jpg)
![Folie 15](docs/Slide15.jpg)
![Folie 16](docs/Slide16.jpg)
![Folie 17](docs/Slide17.jpg)
![Folie 18](docs/Slide18.jpg)
![Folie 19](docs/Slide19.jpg)
![Folie 20](docs/Slide20.jpg)
![Folie 21](docs/Slide21.jpg)
![Folie 22](docs/Slide22.jpg)
![Folie 23](docs/Slide23.jpg)
![Folie 24](docs/Slide24.jpg)
![Folie 25](docs/Slide25.jpg)
![Folie 26](docs/Slide26.jpg)
![Folie 27](docs/Slide27.jpg)
![Folie 28](docs/Slide28.jpg)
![Folie 29](docs/Slide29.jpg)
![Folie 30](docs/Slide30.jpg)
![Folie 31](docs/Slide31.jpg)
![Folie 32](docs/Slide32.jpg)
![Folie 33](docs/Slide33.jpg)
![Folie 34](docs/Slide34.jpg)
![Folie 35](docs/Slide35.jpg)
![Folie 36](docs/Slide36.jpg)
![Folie 37](docs/Slide37.jpg)
![Folie 38](docs/Slide38.jpg)
![Folie 39](docs/Slide39.jpg)
![Folie 40](docs/Slide40.jpg)
![Folie 41](docs/Slide41.jpg)
![Folie 42](docs/Slide42.jpg)
![Folie 43](docs/Slide43.jpg)
![Folie 44](docs/Slide44.jpg)
![Folie 45](docs/Slide45.jpg)
![Folie 46](docs/Slide46.jpg)
![Folie 47](docs/Slide47.jpg)
![Folie 48](docs/Slide48.jpg)
![Folie 49](docs/Slide49.jpg)
![Folie 50](docs/Slide50.jpg)
![Folie 51](docs/Slide51.jpg)
![Folie 52](docs/Slide52.jpg)
![Folie 53](docs/Slide53.jpg)
![Folie 54](docs/Slide54.jpg)
![Folie 55](docs/Slide55.jpg)