## À quoi sert cette partie

Tu vas amener un réseau de neurones d'un fichier vide à un modèle entraîné et **vérifié**, sans framework : passe avant, fonction de perte, rétropropagation avec toutes les formes, boucle d'entraînement, et preuve (face aux différences finies et à l'autograd de PyTorch) que tes gradients sont justes. Le but n'est pas d'utiliser des bibliothèques, mais de comprendre ce qu'elles font, pour que, quand l'entraînement échoue, tu saches où regarder.

**Prérequis :** la partie Mathématiques, surtout les leçons 5 et 6 (gradients, jacobiennes). Le panneau « Requires » de chaque leçon nomme l'idée précédente exacte dont elle a besoin, et l'application t'avertit quand tes résultats aux quiz indiquent qu'un prérequis est fragile.

## Le parcours

| # | Leçon | La question à laquelle elle répond | Prérequis (maths / ml) |
|---|---|---|---|
| 7 | Descente de gradient | Comment utilise-t-on le gradient pour progresser, et quand échoue-t-elle ? | M5 |
| 8 | Fonctions de perte | Qu'est-ce qui est « faux », et pourquoi ces formules ? | M1 |
| 9 | Couches et passe avant | Que calcule un réseau ? | M3, 8 |
| 10 | Rétropropagation | Comment obtient-on tous les gradients d'un coup ? | M6, 9 |
| 11 | Entraînement à partir de zéro | Comment le faire réellement apprendre, et le déboguer ? | 7, 10 |
| 12 | Autodiff et PyTorch | Mon implémentation est-elle prouvablement juste ? | 11 |

## Comment étudier une leçon

La même boucle que dans la partie Mathématiques, avec un ajout : **écris toujours la forme de chaque tenseur avant d'écrire du code.** Dans cette partie, presque tous les bugs sont des bugs de forme, et un gradient qui n'a pas la forme de son paramètre est faux avant même que tu exécutes quoi que ce soit.

Quand l'entraînement se comporte mal, suis dans l'ordre l'échelle de la leçon 11 : perte initiale ≈ ln K, vérification du gradient, surapprentissage de 8 exemples, balayage du taux d'apprentissage, et seulement ensuite l'entraînement complet.

## Preuve de maîtrise

Les quiz et les exercices suivent la maîtrise des concepts comme d'habitude. Pour cette partie, le vrai test est le **Projet de synthèse 1** (Capstone 1) : tu soumets tes chiffres de vérification de gradient, ta propre dérivation des quatre gradients, une expérience sur le taux d'apprentissage avec plusieurs graines, un échec que tu as diagnostiqué et les limites de ton implémentation. Rien ne le note à ta place ; l'idée est que tu puisses le défendre devant un relecteur.

## Et après

La bibliothèque de sources contient déjà les prochaines étapes, importées depuis tes dossiers : ML classique, réseaux convolutifs et récurrents, attention et transformer, et construction d'un GPT. Ce sont des textes sources avec quiz, pas encore des leçons rédigées avec exercices ; la page Roadmap montre exactement ce qui est couvert et ce qui manque.
