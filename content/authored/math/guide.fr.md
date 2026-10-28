## À quoi sert cette partie

Le machine learning ressemble à de la magie, jusqu'à ce qu'on voie qu'il est construit à partir d'une poignée d'outils mathématiques utilisés encore et encore. Cette partie enseigne exactement ces outils, dans l'ordre où ils dépendent les uns des autres, et *uniquement* ceux-là. Quand tu auras terminé, tu pourras lire l'équation `∇_W L = Xᵀ(P − Y)/B` et dire ce que représente chaque symbole, quelle forme il a et d'où il vient.

Tu pars de zéro : si tu sais additionner, multiplier et lire une fonction comme `f(x) = x²`, tu peux commencer la leçon 1.

## Le parcours

| # | Leçon | La question à laquelle elle répond | Prérequis |
|---|---|---|---|
| 1 | Fonctions, sommes, logarithmes | Que signifient les symboles, et pourquoi calcule-t-on dans l'espace logarithmique ? | aucun |
| 2 | Vecteurs et produit scalaire | Comment mesurer à quel point deux listes de nombres se ressemblent ? | 1 |
| 3 | Matrices et applications linéaires | Qu'est-ce qu'une couche, géométriquement, et quelles sont ses formes ? | 2 |
| 4 | Dérivées | Comment la sortie réagit-elle à une toute petite variation de l'entrée ? | 1 |
| 5 | Gradients | Avec beaucoup d'entrées, quelle direction monte ? | 2, 4 |
| 6 | Règle de la chaîne, jacobiennes | Comment les taux de variation se propagent-ils à travers une composition ? | 3, 4, 5 |

Les leçons 1 à 3 relèvent de l'algèbre et de la géométrie (le *langage* des données) ; les leçons 4 à 6 relèvent du calcul différentiel (le *langage du changement*). La leçon 6 est le pont : ses trois formules de passe arrière constituent toute la rétropropagation, sur laquelle la partie Machine Learning s'appuie ensuite.

## Comment étudier une leçon (la boucle qui marche)

1. **Prédis (2 min).** Lis seulement les objectifs et écris une phrase qui devine l'idée principale. Une mauvaise hypothèse écrite t'apprend plus qu'une bonne impression.
2. **Lis pour l'intuition.** Saute la dérivation à la première lecture ; dis l'idée à voix haute, sans symboles.
3. **Dérive sur papier.** Ferme la page et reproduis chaque dérivation toi-même. Quand tu n'y arrives pas, trouve l'unique étape qui t'a échappé, referme la page et réessaie. C'est l'habitude qui rapporte le plus.
4. **Fais tourner les nombres.** Refais l'exemple résolu à la main, puis joue avec la figure interactive jusqu'à pouvoir prédire ce que fait un curseur *avant* de le bouger.
5. **Code-le.** Fais d'abord l'exercice sans indices. Les tests te disent *ce qui* échoue ; les formes (« quelle doit être la forme de ceci ? ») te disent *pourquoi*.
6. **Auto-évalue-toi**, puis **explique-le toi-même** avec les questions de la fin. Si tu n'y arrives pas, tu ne le maîtrises pas encore.
7. **Reviens.** Les flashcards le font resurgir à intervalles planifiés pour que ça dure des mois, pas des jours.

## Comment la progression est mesurée

Lire ne rapporte rien en soi. L'application enregistre la **maîtrise par concept** à partir de preuves : réponses aux quiz, exercices résolus, rappel des flashcards. Un concept est maîtrisé à 80 % avec au moins trois éléments de preuve, de sorte que la chance ne puisse pas t'y mener. Rater deux fois le même concept te renvoie à ses prérequis les plus faibles. Termine la partie par l'évaluation cumulative, puis passe au Machine Learning.
