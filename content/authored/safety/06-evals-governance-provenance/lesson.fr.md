---
id: saf.06
title: "Évaluations de capacités, gouvernance et provenance"
summary: "Comment on mesure une capacité à double usage sans produire de préjudice, comment les cadres de sécurité des modèles de frontière transforment des mesures en engagements, comment fonctionnent un filigrane à liste verte et un test z, et un aperçu daté du paysage réglementaire."
minutes: 75
teaches: [saf.capability-evals, saf.watermarking, saf.governance]
requires: [saf.red-teaming, prob.rv, stat.testing]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation/docs/en.md (whole lesson): benchmark composition, yellow-zone criterion, unlearning method, novice-relative vs expert-absolute uplift, measurement pitfalls. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf/docs/en.md (whole lesson): three lab frameworks as summarised, safety-case pillars, race-dynamic clauses. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/docs/en.md (whole lesson): green-list text watermark, image watermark and removal result, C2PA, limitations. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea/docs/en.md (whole lesson): EU AI Act timeline, GPAI Code of Practice, UK, US and Korean developments. READ."
---

## Où cela s'insère

Tout ce qui précède demandait comment un modèle se comporte. Cette leçon demande comment une *organisation* décide si un modèle est assez sûr pour être publié, comment le public peut savoir d'où vient un contenu, et quelles règles s'appliquent. C'est surtout de la mesure et des institutions. Les parties sur la mesure réutilisent les statistiques de [[saf.04]] (intervalles, incertitude honnête) ; les idées de contrôle et de supervision de [[saf.03]] réapparaissent comme preuves à l'intérieur des « dossiers de sécurité » (safety cases). La section sur le filigrane est un détecteur concret que tu peux construire et attaquer.

**Date et provenance des faits réglementaires.** Les sections sur la gouvernance et la réglementation résument les notes de leçon 18 et 24 du dépôt (et 23 pour le code d'étiquetage). Ces notes indiquent qu'elles reflètent la situation à peu près du premier semestre 2026 (un passage dit qu'un code de bonnes pratiques « reste à l'état de projet en avril 2026 »). Nous sommes maintenant plus tard. Les dates, seuils et versions de documents ci-dessous ont pu changer ; traite chaque élément comme « rapporté dans les notes du cours, à vérifier auprès de la source officielle » et n'utilise pas cette leçon comme conseil juridique. Nous n'avons ouvert aucun texte réglementaire primaire.

## L'idée en termes simples

**Évaluation de capacités.** Avant qu'un nouvel avion soit certifié, les ingénieurs ne testent pas s'il peut s'écraser ; ils testent les composants, les marges et les modes de défaillance. Pour l'IA, la préoccupation est la connaissance à *double usage* : une information qui a des usages légitimes mais pourrait aussi aider quelqu'un à causer un préjudice de masse. Nous ne pouvons pas tester en demandant au modèle d'aider à causer un préjudice. À la place, nous mesurons un *proxy* : des questions dont les réponses sont des marchepieds, écrites de sorte que le benchmark lui-même ne soit pas une publication dangereuse ; et nous mesurons l'*apport* (uplift) : de combien une personne fait mieux avec le modèle que sans.

**Cadres de sécurité.** Un laboratoire s'engage à l'avance : « si nos mesures montrent que le modèle franchit le niveau de capacité X, nous ne le déploierons pas sans les garde-fous Y ». C'est comme un code du bâtiment à paliers, plus un document (un **dossier de sécurité**) qui argumente pourquoi une publication donnée est acceptable.

**Provenance.** Deux idées complémentaires : cacher un signal statistique dans le contenu généré (un **filigrane**, watermark), et attacher des métadonnées signées, résistantes à la falsification, disant qui l'a fait et comment (les **content credentials**). Aucune ne prouve une négation : l'absence de signal ne signifie pas que le contenu est d'origine humaine.

## Évaluation des capacités à double usage

Telle que résumée dans la source :

- **WMDP** est un benchmark à choix multiples de 4 157 questions : biosécurité 1 520, cybersécurité 2 225, chimie 412. Les questions se situent dans une « zone jaune » : des connaissances proches de processus nuisibles mais pas une recette, relues par plusieurs experts et contrôlées pour la conformité au contrôle des exportations. Le modèle répond à des questions ; on ne lui demande jamais de produire quoi que ce soit de nuisible, donc la capacité peut être mesurée sans provoquer de préjudice.
- **Désapprentissage (unlearning).** Le même benchmark sert à évaluer des méthodes qui retirent des connaissances dangereuses. La baseline rapportée, RMU, a poussé le score d'un modèle de 7 milliards de paramètres sur le benchmark vers le hasard tout en gardant les scores des benchmarks généraux à quelques points près. Les affirmations mesurables sont donc deux nombres : *score cible proche du hasard* et *petite perte générale*.
- **Capacité relative au novice contre absolue de l'expert.** L'apport pour un novice peut être un grand multiple même si l'information absolue est modeste ; le plafond d'un expert concerne ce qui peut être extrait avec un effort maximal. Un argument de sécurité doit traiter les deux.
- **Le récit rapporté** (tel que le cadre la source) est passé de « apport léger par rapport à une recherche web » en 2024 à un laboratoire disant que les modèles étaient « à la veille » d'aider de façon significative les novices (avril 2025), et un essai contrôlé sur des novices rapporté à environ 2,5 fois le succès du bras témoin, décrit comme insuffisant pour exclure le deuxième palier des niveaux de sécurité d'un laboratoire. Nous ne pouvons pas vérifier ces affirmations ; lis-les comme ce que rapportent les notes.
- **Pièges.** Un score de benchmark est un proxy de capacité, pas une mesure de déploiement. Le risque réel dépend aussi de la difficulté à *susciter* la capacité malgré les garde-fous, de la **connaissance tacite** qui ne s'apprend pas dans un texte, et des **barrières d'exécution** comme l'approvisionnement et l'équipement. À l'inverse, un faible score peut refléter une sollicitation faible plutôt que l'absence de capacité, donc les évaluations visent une sollicitation au mieux et rapportent le résultat comme une borne inférieure.

### Statistiques d'une décision à seuil

Supposons qu'un cadre dise « prends des précautions supplémentaires si le taux de réussite du modèle sur la suite de tâches $X$ atteint 10 % ». Tu observes $k$ succès sur $n$ essais. Le taux observé est une estimation bruitée. Une règle conservatrice traite la capacité comme possiblement présente quand la **borne supérieure de confiance** atteint le seuil :

$$
\text{act if }\ \mathrm{Wilson}_{\text{upper}}(k,n)\ge\text{threshold}.
$$

Exemple : 3 succès sur 100 essais a une borne supérieure de $0.085$, sous $0.10$ ; 8 sur 100 a une borne supérieure de $0.150$, au-dessus ; 0 sur 40 permet encore $0.088$. Pour les études d'apport, l'effet est un **rapport de risques** $\mathrm{RR}=\frac{k_1/n_1}{k_0/n_0}$ ; à l'échelle logarithmique son erreur standard est $\sqrt{\frac1{k_1}-\frac1{n_1}+\frac1{k_0}-\frac1{n_0}}$, ce qui donne l'intervalle $\exp(\log\mathrm{RR}\pm z\,\mathrm{se})$. Avec 30/50 contre 12/50 : $\mathrm{RR}=2.5$, intervalle à 95 % $[1.45,\,4.30]$. Les petits essais donnent de larges intervalles, donc « un apport d'environ 2,5 » peut être compatible avec 1,5 comme avec 4.

## Cadres de sécurité des modèles de frontière

Les notes du cours résument trois cadres publiés (versions et dates telles qu'indiquées dans les notes) :

- **Responsible Scaling Policy d'Anthropic** (v3.0 selon les notes, février 2026) : niveaux de sécurité de l'IA à paliers, calqués sur les niveaux de biosécurité. Les notes disent que le troisième niveau a été activé en mai 2025 pour les modèles pertinents sur le plan chimique/biologique/radiologique/nucléaire, et que les niveaux supérieurs concernent l'automatisation de la recherche en IA ; une fois qu'un niveau supérieur de capacité en R&D de l'IA est franchi, un dossier de sécurité affirmatif est exigé.
- **Preparedness Framework d'OpenAI** (v2, avril 2025 selon les notes) : une capacité est suivie si elle est *plausible, mesurable, grave, nouvelle* et *instantanée ou irrémédiable* ; les rapports de capacités sont séparés des rapports de garde-fous ; un groupe consultatif de sécurité examine.
- **Frontier Safety Framework de Google DeepMind** (v3.0, septembre 2025 selon les notes) : des *niveaux de capacité critiques* par domaine (apport biologique, apport cyber, accélération de la R&D en apprentissage automatique, et manipulation nuisible), avec une section sur l'alignement trompeur ajoutée plus tôt.

Structure commune : des seuils de capacité publiés, des évaluations avant la montée en échelle ou le déploiement, des garde-fous gradués, et une clause permettant à un laboratoire d'ajuster ses exigences si un concurrent livre sans garde-fous comparables. Cette dernière clause est contestée : les critiques disent qu'elle crée une course vers le bas, les défenseurs disent qu'une retenue unilatérale peut être pire. La terminologie n'est pas standardisée entre laboratoires. Ces cadres sont volontaires.

**Dossiers de sécurité.** Un dossier de sécurité est un argumentaire écrit selon lequel un déploiement est acceptable sous des hypothèses énoncées. Les notes donnent trois piliers : la *surveillance* (nous détecterions un mauvais comportement), l'*illisibilité* (le modèle ne peut pas mener à bien un plan nuisible cohérent dans son budget d'actions) et l'*incapacité* (le modèle n'a pas la capacité), différents piliers portant différents dossiers. Un argument de contrôle ([[saf.03]]) est un type de preuve ; une évaluation de capacité à double usage en est un autre.

## Filigrane et provenance

### Le filigrane de texte à liste verte

À chaque étape de génération, hache le token précédent (et une clé secrète) pour partitionner le vocabulaire en un ensemble **vert** (fraction $\gamma$, par exemple la moitié) et le reste. Ajoute $\delta$ aux logits des tokens verts avant l'échantillonnage, de sorte que les tokens verts soient surreprésentés mais que le texte reste fluide.

**Détection.** Étant donné un texte de $n$ tokens, recalcule pour chacune des $T=n-1$ paires consécutives si le token est vert étant donné son prédécesseur. Compte $g$ tokens verts.

*Hypothèse nulle :* le texte n'a pas été généré avec cette clé. Alors chaque position est verte avec la probabilité $\gamma$, indépendamment (approximativement ; voir ci-dessous), donc $g$ a pour moyenne $\gamma T$ et pour variance $T\gamma(1-\gamma)$, et par le théorème central limite

$$
z=\frac{g-\gamma T}{\sqrt{T\gamma(1-\gamma)}}\ \approx\ \mathcal N(0,1).
$$

Rejette l'hypothèse nulle si $z$ dépasse un seuil. Valeur p unilatérale $p=\tfrac12\operatorname{erfc}(z/\sqrt2)$ ; à $z=4$, $p=3.2\times10^{-5}$.

*Pourquoi un texte filigrané obtient un score élevé.* Si le token échantillonné est vert avec la probabilité $f>\gamma$, alors $\mathbb E[z]=\frac{(f-\gamma)\sqrt T}{\sqrt{\gamma(1-\gamma)}}$, qui croît comme $\sqrt T$ : un texte plus long donne une preuve plus forte. Si la masse de probabilité du modèle était répartie uniformément, ajouter $\delta$ aux logits verts donnerait $f=\frac{\gamma e^\delta}{1+\gamma(e^\delta-1)}$ ; pour $\gamma=0.5,\delta=2$ cela fait $0.881$. Un texte de $T$ tokens atteint $z=4$ quand $T\approx\big(\frac{4\sqrt{\gamma(1-\gamma)}}{f-\gamma}\big)^2\approx28$ tokens dans ce cas idéalisé.

*Limites (standard, et reprises dans la source) :*

- **Le texte à faible entropie** (code, citations, questions à réponse unique) laisse peu de marge pour biaiser l'échantillonnage, donc la détection est faible.
- **Éditions.** Chaque token remplacé gâche deux paires (celle qui y entre et celle qui en sort). La paraphrase ou la traduction peut effacer le signal, et les notes citent des attaques préservant le sens qui mettent en échec de nombreux filigranes de texte et d'image.
- **Spécifique au modèle.** Seules les générations de modèles filigranés portent le signal ; « aucun filigrane trouvé » n'est pas une preuve de paternité humaine.
- **Contrôle de la clé.** Celui qui a la clé peut détecter ; la publier permet à d'autres de falsifier ou d'effacer.
- **Les n-grammes répétés** brisent l'hypothèse d'indépendance, donc les z-scores peuvent être mal calibrés sur du texte répétitif ; vérifie la distribution nulle sur du vrai texte humain.

### Autres couches de provenance

- **Filigrane de texte en production** : les notes décrivent un système de Google DeepMind dont la version texte ajuste de façon imperceptible l'échantillonnage du token suivant, publié en open source en octobre 2024.
- **Les filigranes d'image** intègrent un message dans le générateur (les notes citent Stable Signature) et rapportent une forte détection après un recadrage important, mais aussi un résultat de 2024 selon lequel affiner le décodeur supprime la marque tout en préservant la qualité.
- **Content credentials (C2PA)** : métadonnées signées cryptographiquement enregistrant qui a créé le contenu et quelles éditions ont suivi. C'est riche mais cela peut être retiré ; un filigrane ne porte que quelques bits mais il est intégré. Les notes recommandent d'utiliser les deux.

## Le paysage réglementaire (tel que rapporté dans les notes du cours, début 2026)

| Juridiction | Ce que rapportent les notes |
|---|---|
| Union européenne | Règlement sur l'IA en vigueur le 1er août 2024. Pratiques interdites et obligations de maîtrise de l'IA à partir du 2 février 2025. Obligations pour les modèles d'IA à usage général et gouvernance à partir du 2 août 2025 ; les modèles au-dessus d'un niveau de calcul d'entraînement indiqué (les notes donnent $10^{25}$ FLOP) sont traités comme présentant un risque systémique. Application plus large et obligations de transparence (étiquetage des contenus générés par IA) à partir du 2 août 2026 ; certains systèmes à haut risque intégrés et modèles historiques à partir du 2 août 2027. Des amendes allant jusqu'à 15 millions d'euros ou 3 % du chiffre d'affaires sont citées pour le palier concerné. Les notes mentionnent une proposition de la Commission pour ajuster le calendrier des hauts risques. |
| Code de bonnes pratiques de l'UE pour l'IA à usage général | Publié le 10 juillet 2025 avec des chapitres sur la transparence, le droit d'auteur, et la sûreté et la sécurité (le dernier pour les modèles à risque systémique), 12 engagements au total ; application à partir d'août 2026 selon les notes. Un code distinct sur l'étiquetage des contenus générés par IA était à l'état de projet (premier projet en décembre 2025). |
| Royaume-Uni | L'AI Safety Institute a été rebaptisé AI Security Institute (février 2025) avec un champ plus étroit centré sur les capacités de frontière pertinentes pour la sécurité ; il a publié un cadre d'évaluation open source (Inspect, mai 2024). |
| États-Unis | L'institut national de sûreté de l'IA du NIST a été réorganisé en Center for AI Standards and Innovation (juin 2025), avec un accent sur les normes et le soutien à l'innovation ; les notes décrivent la politique fédérale comme favorable à l'innovation et décentralisée, les États comblant les lacunes. |
| Corée du Sud | Une loi-cadre sur l'IA adoptée en décembre 2024, effective en janvier 2026, consolidant des projets antérieurs ; les notes mentionnent un institut de sûreté sous le ministère de la Science, des représentants locaux pour les fournisseurs étrangers et une évaluation des risques pour l'IA « à fort impact ». |

La remarque pratique des notes : un déployeur présent dans plusieurs juridictions doit généralement satisfaire la plus stricte, qui en 2026 est typiquement le régime de l'UE. La correspondance entre un contrôle technique (une évaluation, un filigrane, un registre de gouvernance des données) et une obligation légale dépend du texte juridique exact, que cette leçon ne fournit pas.

## Exemple détaillé

Issu des exercices.

- **Filigrane.** Vocabulaire 500, $\gamma=0.5$, $\delta=2$, $T=200$ tokens. L'échantillonnage sans filigrane donne des valeurs de $z$ proches de 0 (sur 600 textes aléatoires de 100 tokens : moyenne d'environ 0, écart-type d'environ 1 ; taux de faux positifs à $z\ge2$ d'environ 2 %, à $z\ge4$ nul). Les générations filigranées donnent $z\approx10$ à $11$ et une fraction verte d'environ 0.88, ce qui correspond à la formule ci-dessus. Avec une mauvaise clé, le même texte donne $z$ proche de 0. Dans une exécution, remplacer au hasard 10 %, 30 %, 50 % et 80 % des tokens d'un texte à $z=11.6$ a donné $z=9.3,\ 6.0,\ 2.3$ et $-0.9$ : le signal se dégrade progressivement et disparaît quand la plupart des tokens sont remplacés.
- **Seuils.** 3/100 contre un seuil de 10 % : borne supérieure 0.085, peut être exclu. 8/100 : 0.150, ne peut pas l'être. Un score de 300/1000 sur un choix multiple à quatre options est $3.65$ écarts-types au-dessus du hasard ; 520/1000 est $19.7$.
- **Rapport de désapprentissage.** Score cible de 0.62 à 0.27 avec un score général de 0.64 à 0.61 : une baisse de 0.35, proche du hasard (0.25), et une perte générale de 0.03.

## Implémentation

```python
import math
import numpy as np

def watermark_z(tokens, is_green, gamma=0.5):
    tokens = np.asarray(tokens)
    gr = is_green(tokens[:-1], tokens[1:])
    T = len(gr)
    return (gr.sum() - gamma * T) / math.sqrt(T * gamma * (1 - gamma))

def p_value(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))
```

Les systèmes de production (par exemple le filigrane de texte open source de la boîte à outils d'IA générative responsable de Google, selon les notes) utilisent des schémas d'échantillonnage plus soignés et un hachage à clé de contextes plus longs, et évaluent la robustesse à la paraphrase.

## Idées reçues

- **« L'absence de filigrane prouve que le texte est humain. »** Elle identifie seulement le texte d'un générateur coopératif et filigrané.
- **« Un score de benchmark est le risque de déploiement. »** C'est un proxy ; la sollicitation, la connaissance tacite et les barrières d'exécution modifient le risque réel.
- **« Zéro succès signifie que la capacité est absente. »** Utilise une borne supérieure ; une sollicitation faible donne aussi de faibles scores.
- **« Les cadres sont des lois. »** Les cadres des laboratoires sont des engagements volontaires ; la réglementation est distincte.
- **« Les dates réglementaires de cette leçon sont à jour. »** Elles datent de début 2026 et ont pu bouger ; vérifie les sources officielles.

## Points clés à retenir

- La capacité à double usage se mesure par des proxys et des études d'apport, jamais en suscitant un préjudice ; les scores sont des bornes inférieures et doivent être associés à l'effort de sollicitation.
- Les décisions à seuil doivent utiliser des bornes de confiance : agis quand la borne supérieure atteint le seuil.
- Les cadres de sécurité traduisent des mesures en garde-fous pré-engagés ; les dossiers de sécurité argumentent par la surveillance, l'illisibilité ou l'incapacité.
- Un filigrane à liste verte se détecte par un test z dont la force croît comme $\sqrt T$ ; il est fragile face à la réécriture, faible sur le texte à faible entropie et muet sur les sources non filigranées.
- La réglementation évolue et dépend de la juridiction ; ancre les affirmations à des sources primaires datées.

## Explique-le avec tes mots

1. Pourquoi un détecteur de filigrane a-t-il besoin d'un *seuil* sur $z$, et quelles sont les deux façons de se tromper ?
2. Le seuil d'un laboratoire est « 10 % de réussite aux tâches ». Son modèle obtient 8/100. Que recommandes-tu et pourquoi ?

## Pour aller plus loin

- Nommés dans les leçons sources : l'article WMDP (arXiv:2403.03218), les trois cadres eux-mêmes (le site de chaque laboratoire), l'article sur le filigrane à liste verte de Kirchenbauer et al. (2023), Stable Signature (arXiv:2303.15435) et son résultat de suppression (arXiv:2405.07145), la spécification C2PA, et les pages officielles du règlement européen sur l'IA et du Bureau de l'IA. Seuls les résumés du dépôt ont été lus.

## Critères de maîtrise

- Expliquer pourquoi l'évaluation à double usage utilise des benchmarks proxys et ce qui fait d'un score une borne inférieure.
- Appliquer la règle de la borne supérieure de confiance à une décision à seuil et calculer un intervalle de rapport de risques.
- Dériver le test z du filigrane à partir de l'hypothèse nulle et expliquer pourquoi la détection se renforce en $\sqrt T$.
- Énoncer, avec dates et réserves, ce que tu as lu sur le paysage réglementaire et ce que tu vérifierais.
