---
id: llm.05
title: "Alignement : modèles de récompense, RLHF et optimisation directe des préférences"
summary: "Pourquoi l'imitation ne suffit pas, comment les préférences humaines par paires deviennent un modèle de récompense (Bradley-Terry), l'objectif RLHF régularisé par la KL et son optimum en forme close, comment DPO s'en déduit, et comment ces systèmes échouent (détournement de la récompense, complaisance)."
minutes: 90
teaches: [llm.reward-model, llm.dpo]
requires: [llm.sft, llm.lm, info.entropy, ml.logreg, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/07-rlhf/docs/en.md (three-stage pipeline, reward model with scalar head, Bradley-Terry loss, KL-penalised objective, PPO clipped surrogate and advantage, reward hacking and mitigation table, InstructGPT figures): READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/08-dpo/docs/en.md (DPO key insight, closed-form optimal policy, Z cancellation, DPO loss, RLHF vs DPO comparison, KTO/ORPO/SimPO, when each wins): READ."
---

## Où cela s'inscrit

Après le pré-entraînement ([[llm.03]]) et le fine-tuning supervisé ([[llm.04]]), tu as un modèle qui suit des instructions. Est-ce un *bon* assistant ? Demande-lui d'expliquer l'informatique quantique et il pourra donner deux réponses correctes et grammaticales, l'une claire et concise, l'autre décousue. Le SFT considère les deux comme des démonstrations acceptables ; rien dans sa perte ne dit que l'une est meilleure. Pourtant, les utilisateurs se soucient précisément de cette différence : utilité, honnêteté, ton, refus de ce qui doit être refusé.

Le remède consiste à entraîner sur des **comparaisons**. Il s'avère que les gens trouvent beaucoup plus facile et plus cohérent de dire « la réponse A est meilleure que la réponse B » que d'écrire la réponse idéale ou de donner une note comme 7.3 sur 10. Les données de comparaison sont ensuite utilisées de deux façons :

- **RLHF** (apprentissage par renforcement à partir de retours humains) : ajuster un **modèle de récompense** aux comparaisons, puis optimiser le modèle de langage pour qu'il produise du texte à forte récompense sans trop s'éloigner de son point de départ.
- **DPO** (optimisation directe des préférences) : une courte étape algébrique montre que le même objectif peut être optimisé directement sur les comparaisons, sans modèle de récompense séparé et sans boucle d'apprentissage par renforcement.

Cette leçon démontre les trois éléments, montre ce que chacun coûte, et explique les principales façons dont ils tournent mal.

## L'idée en mots simples

Imagine un concours de cuisine. Tu ne peux pas donner une recette au candidat (c'est le SFT), mais un jury peut goûter deux plats et dire lequel est meilleur. De nombreux verdicts de ce genre fait émerger un **modèle du goût** : une fonction qui note n'importe quel plat. C'est le modèle de récompense. Maintenant le candidat s'entraîne, en cherchant de bonnes notes auprès du modèle du goût. Deux dangers apparaissent :

1. Le modèle du goût a été appris à partir d'un nombre limité de verdicts. Un candidat rusé peut trouver des plats qui obtiennent une bonne note auprès de lui mais que de vrais juges détesteraient (très longs, très salés, couverts de garniture). C'est le **détournement de la récompense** (reward hacking) : optimiser un substitut jusqu'à ce qu'il cesse de mesurer ce qui comptait pour toi.
2. Un candidat qui réécrit tout son style pour courir après la note peut perdre ses compétences de base.

Le remède au second danger, et en partie au premier, est une **laisse** : le candidat est pénalisé s'il s'écarte trop de son style d'origine (SFT). C'est la pénalité KL. L'idée de DPO est qu'avec la laisse en place, le meilleur candidat possible a une forme close connue ; on peut donc entraîner le candidat en rendant *directement* les plats préférés plus probables et les plats rejetés moins probables, mesurés par rapport à la laisse, et sauter le modèle du goût.

## Données de préférence et modèle de récompense

**Données.** Pour un prompt $x$, le modèle produit deux réponses ; une personne marque la meilleure, ce qui donne un triplet $(x,y_w,y_l)$ (« gagnant », « perdant »). L'effort InstructGPT décrit dans la source a recueilli environ 33 000 comparaisons de ce type auprès de 40 prestataires, environ cinq minutes chacune.

**Modèle de récompense.** Prends le modèle SFT, remplace la tête de vocabulaire par une tête qui produit un seul nombre, et lis-le à la fin du texte (prompt, réponse) : $r_\phi(x,y)\in\mathbb R$.

### Démonstration de Bradley-Terry

Il nous faut un modèle probabiliste reliant les scores aux choix. Le modèle de Bradley-Terry donne à chaque réponse une force positive $s=e^{r}$ et dit que la probabilité que $a$ batte $b$ est proportionnelle à la force :

$$
P(a\succ b)=\frac{e^{r_a}}{e^{r_a}+e^{r_b}}=\frac{1}{1+e^{-(r_a-r_b)}}=\sigma(r_a-r_b).
$$

Propriétés à remarquer : la probabilité ne dépend que de la *différence* $r_a-r_b$ (ajouter une constante à tous les scores ne change rien, donc seules les différences sont identifiables) ; des scores égaux donnent $1/2$ ; la probabilité tend vers 1 quand l'écart grandit.

**Entraînement.** Maximiser la vraisemblance des gagnants observés. Pour une paire, la log-vraisemblance négative est

$$
\ell(\phi)=-\ln\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)=\ln\bigl(1+e^{-d}\bigr),\qquad d=r_w-r_l .
$$

C'est une régression logistique sur la différence de scores. En utilisant $\sigma'(u)=\sigma(u)(1-\sigma(u))$ et $1-\sigma(d)=\sigma(-d)$ :

$$
\frac{d\ell}{dd}=-\frac{\sigma(d)(1-\sigma(d))}{\sigma(d)}=-\sigma(-d),\qquad
\frac{\partial\ell}{\partial r_w}=-\sigma(-d),\quad\frac{\partial\ell}{\partial r_l}=+\sigma(-d).
$$

La poussée sur les deux scores est égale et opposée, et sa taille $\sigma(-d)$ est exactement la probabilité actuelle du modèle de s'être *trompé* sur la paire : les paires justes avec confiance bougent à peine ; les paires fausses reçoivent la pleine poussée. Les scores refluent ensuite à travers le réseau par la rétropropagation ordinaire.

**Nombres.** $r_w=1.5$, $r_l=0.5$ : $d=1$, $P=\sigma(1)=0.7311$, perte $=0.3133$, le gradient sur $r_w$ est $-\sigma(-1)=-0.2689$ (et $+0.2689$ sur $r_l$). Avec des scores égaux, la perte vaut $\ln2=0.6931$. Dans l'exercice, une récompense linéaire $r=\phi^\top\theta$ est ajustée sur 400 comparaisons bruitées simulées dont les étiquettes viennent d'un vrai $\theta^\star$ ; le $\theta$ appris pointe dans la même direction (cosinus $0.996$) et atteint $85\%$ de précision par paire, alors que même la vraie récompense ne pourrait atteindre qu'environ $83\%$ sur cet échantillon parce que les étiquettes sont bruitées par construction.

## RLHF : optimiser face à la récompense, en laisse

Soit $\pi_\theta(y\mid x)$ la politique (le modèle de langage en cours d'entraînement) et $\pi_{\rm ref}$ le modèle SFT gelé. Le RLHF maximise

$$
J(\pi)=\mathbb E_{x}\,\mathbb E_{y\sim\pi(\cdot|x)}\bigl[r(x,y)\bigr]-\beta\,\mathbb E_x\,\mathrm{KL}\bigl(\pi(\cdot|x)\,\|\,\pi_{\rm ref}(\cdot|x)\bigr),
$$

avec $\beta>0$ la force de la laisse. Le premier terme veut une forte récompense ; le second pénalise les distributions qui diffèrent de la référence. En pratique, $y$ est une séquence de tokens échantillonnée à partir de la politique courante et la mise à jour utilise un algorithme d'apprentissage par renforcement, généralement PPO. La leçon source résume le rôle de PPO : échantillonner des réponses, les noter, estimer un *avantage* (récompense moins une base de référence comme la moyenne récente), et mettre à jour avec un objectif écrêté

$$
\min\bigl(\rho A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\,A\bigr),\qquad \rho=\frac{\pi_{\rm new}}{\pi_{\rm old}},
$$

de sorte qu'aucun échantillon isolé ne puisse provoquer un changement énorme ($\epsilon$ vaut couramment $0.2$). Coûts : jusqu'à quatre modèles en mémoire (politique, référence, modèle de récompense, et souvent une base de valeur), trois étapes d'entraînement (SFT, modèle de récompense, PPO), et une sensibilité aux hyperparamètres.

### La politique optimale en forme close

Fixe un prompt $x$ et retire-le de la notation. Sur un ensemble fini de réponses, définis

$$
\pi^\star(y)=\frac{\pi_{\rm ref}(y)\,e^{r(y)/\beta}}{Z},\qquad Z=\sum_{y}\pi_{\rm ref}(y)\,e^{r(y)/\beta}.
$$

Affirmation : $\pi^\star$ maximise $J$. Preuve en complétant la divergence. Pour toute distribution $\pi$,

$$
\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\ln\frac{\pi(y)}{\pi_{\rm ref}(y)}
=-\beta\sum_y\pi(y)\Bigl[\ln\frac{\pi(y)}{\pi_{\rm ref}(y)}-\frac{r(y)}{\beta}\Bigr].
$$

Le crochet vaut $\ln\dfrac{\pi(y)}{\pi_{\rm ref}(y)e^{r(y)/\beta}}=\ln\dfrac{\pi(y)}{Z\,\pi^\star(y)}=\ln\dfrac{\pi(y)}{\pi^\star(y)}-\ln Z$. Par conséquent

$$
J(\pi)=-\beta\,\mathrm{KL}(\pi\,\|\,\pi^\star)+\beta\ln Z .
$$

Une divergence KL est $\ge0$ avec égalité seulement en $\pi=\pi^\star$. Donc l'unique maximiseur est $\pi^\star$ et la valeur maximale est $\beta\ln Z$. (Lecture de la formule : repondérer la référence par $e^{r/\beta}$ ; un petit $\beta$ signifie une préférence marquée pour la forte récompense, un grand $\beta$ signifie rester sur place.)

**Nombres.** Trois réponses avec $\pi_{\rm ref}=(0.5,0.3,0.2)$ et des récompenses $(1,0,-1)$.

| $\beta$ | $\pi^\star$ | $\mathbb E_{\pi^\star}[r]$ | $\mathrm{KL}(\pi^\star\|\pi_{\rm ref})$ | $J(\pi^\star)=\beta\ln Z$ |
|---|---|---|---|---|
| 1.0 | (0.7844, 0.1731, 0.0425) | 0.742 | 0.192 | 0.5497 |
| 0.5 | (0.9187, 0.0746, 0.0067) | 0.912 | 0.432 | 0.6958 |
| 0.1 | (0.99997, 0.00003, 0.0000) | 1.000 | 0.693 | 0.9307 |

La référence elle-même a $\mathbb E[r]=0.3$ et $\mathrm{KL}=0$, donc $J=0.3$. Quand $\beta$ baisse, la récompense monte mais la politique s'éloigne davantage de la référence : cet arbitrage est toute l'histoire du réglage du RLHF. L'exercice vérifie qu'aucune des 3000 distributions aléatoires ne bat $\pi^\star$.

## DPO : le même optimum sans le modèle de récompense

Réarrange la formule de la politique optimale pour exprimer la récompense à travers la politique :

$$
r(x,y)=\beta\ln\frac{\pi^\star(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\ln Z(x).
$$

Toute fonction de récompense a une politique optimale associée, donc toute récompense peut s'écrire ainsi. Injecte-la maintenant dans la probabilité de Bradley-Terry pour une paire répondant au *même* prompt. Le gênant $\beta\ln Z(x)$ ne dépend que de $x$, donc il s'annule dans la différence :

$$
r(x,y_w)-r(x,y_l)=\beta\ln\frac{\pi(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\beta\ln\frac{\pi(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}.
$$

La probabilité de préférence est donc une fonction des propres probabilités de la politique. Au lieu d'ajuster un modèle de récompense puis une politique, on ajuste la politique directement par maximum de vraisemblance sur les comparaisons, la **perte DPO** :

$$
\mathcal L_{\rm DPO}=-\ln\sigma(z),\qquad z=\beta\Bigl[\bigl(\ln\pi(y_w|x)-\ln\pi_{\rm ref}(y_w|x)\bigr)-\bigl(\ln\pi(y_l|x)-\ln\pi_{\rm ref}(y_l|x)\bigr)\Bigr].
$$

Ici $\ln\pi(y|x)$ est la somme des log-probabilités de tokens de la réponse entière, calculable avec une passe avant par réponse. La quantité $\hat r(y)=\beta(\ln\pi-\ln\pi_{\rm ref})$ s'appelle la **récompense implicite** : le modèle de langage est secrètement un modèle de récompense.

**Gradient.** Avec $\mathrm{d}\mathcal L/\mathrm{d}z=-\sigma(-z)$, le même calcul que pour le modèle de récompense donne

$$
\frac{\partial\mathcal L}{\partial\ln\pi(y_w)}=-\beta\,\sigma(-z),\qquad \frac{\partial\mathcal L}{\partial\ln\pi(y_l)}=+\beta\,\sigma(-z).
$$

La descente augmente la log-probabilité de la réponse choisie et diminue celle de la réponse rejetée, avec un poids $\sigma(-z)$ grand quand la récompense implicite classe actuellement mal la paire. Les gradients des log-probabilités refluent ensuite vers les poids par la rétropropagation ordinaire.

**Nombres.** $\beta=0.1$, log-probabilités de la politique $(-12,-15)$, référence $(-13,-14)$ : la réponse choisie est devenue 1 nat plus probable que sous la référence et la rejetée 1 nat moins probable, donc $z=0.1\,(1-(-1))=0.2$, $\sigma(0.2)=0.5498$, perte $0.5981$, le gradient sur $\ln\pi(y_w)$ est $-0.1\cdot\sigma(-0.2)=-0.0450$. Si la politique est égale à la référence, $z=0$, la perte vaut $\ln2$ et les gradients valent $\mp0.05$.

### RLHF contre DPO

| | RLHF avec PPO | DPO |
|---|---|---|
| Modèles en mémoire | politique, référence, modèle de récompense (+ base de valeur) | politique, référence |
| Étapes | SFT, modèle de récompense, PPO | SFT, DPO |
| Données | comparaisons pour le modèle de récompense, puis échantillons frais de la politique | un jeu fixe de comparaisons |
| Stabilité | sensible aux hyperparamètres | proche d'un entraînement supervisé ordinaire |

La source rapporte que DPO égale ou bat souvent le RLHF avec de petits jeux de préférences et un calcul limité, tandis que le RLHF a des avantages quand « meilleur » a de nombreuses dimensions qu'une récompense apprise peut combiner, et pour des boucles itératives où la politique courante génère de nouveaux échantillons qui sont renotés. Des méthodes apparentées retirent d'autres éléments : KTO utilise des étiquettes bon/mauvais non appariées, ORPO intègre le signal de préférence dans la perte SFT, SimPO abandonne le modèle de référence en utilisant la log-probabilité moyenne normalisée par la longueur comme récompense. (Elles sont décrites dans la source ; elles ne sont pas démontrées ici.)

## Implémentation

```python
import numpy as np

def bt_loss(r_w, r_l):                       # arrays of rewards, one entry per pair
    d = r_w - r_l
    return float(np.mean(np.logaddexp(0.0, -d)))     # = mean(-log sigmoid(d)), stable

def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    z = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return float(np.mean(np.logaddexp(0.0, -z)))

def dpo_grads(lp_w, lp_l, ref_w, ref_l, beta):
    z = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    g = -beta * (1.0 / (1.0 + np.exp(z))) / z.size   # -beta * sigmoid(-z), averaged
    return g, -g

def optimal_policy(ref, r, beta):
    w = np.log(ref) + r / beta
    p = np.exp(w - w.max())
    return p / p.sum()
```

## Formes

- Modèle de récompense : entrée tokens (prompt+réponse) $(B,T)$, sortie un scalaire par séquence, de forme $(B,)$. Un batch de paires donne $r_w,r_l$ de forme $(B,)$ chacun.
- DPO : quatre tableaux de forme $(B,)$, chacun étant la *somme* sur les tokens de la réponse des log-probabilités par token (donc les log-probabilités de tokens $(B,T_{\rm resp})$ sont sommées sur l'axe du temps, avec les positions du prompt et du padding masquées comme dans [[llm.04]]).
- Vérification tabulaire dans l'exercice : logits de la politique $\theta\in\mathbb R^{K}$, $\ln\pi=\theta-\operatorname{logsumexp}(\theta)$, et $\partial\ln\pi_i/\partial\theta_j=\mathbb 1[i=j]-\pi_j$.

## Modes d'échec

- **Détournement de la récompense (loi de Goodhart).** La politique optimise le *modèle* de récompense, un substitut imparfait entraîné sur des données finies. Symptômes rapportés dans la source : *verbosité* (les annotateurs préféraient les réponses plus longues, donc la longueur rapporte de la récompense), *complaisance* (sycophancy : approuver la prémisse de l'utilisateur obtenait de bons scores), *esquive* (hedging : les réponses évasives sont rarement marquées fausses), *jeu sur la forme* (les puces et les titres paraissent soignés). Remèdes cités : une pénalité KL plus forte, entraîner le modèle de récompense sur des exemples adversariaux, utiliser plusieurs modèles de récompense.
- **Sur-optimisation par dérive.** Dans l'exercice tabulaire, cinq préférences cohérentes sur quatre réponses équiprobables (0 bat 1, 2, 3 ; 1 et 2 battent 3), $\beta=1$, taux d'apprentissage $0.5$ : après 20 pas la politique vaut $(0.677,0.146,0.146,0.032)$ avec $\mathrm{KL}=0.45$ nat par rapport à la référence ; après 60 pas $(0.856,0.069,0.069,0.006)$ avec $\mathrm{KL}=0.85$ ; après 1000 pas $\mathrm{KL}=1.32$. Comme les étiquettes sont parfaitement cohérentes, la perte n'a pas de minimiseur fini : la probabilité du gagnant continue de monter. Les vraies données ont des étiquettes bruitées, qui tirent dans l'autre sens, mais la leçon reste : la durée d'exécution, $\beta$ et le bruit des étiquettes décident conjointement de la distance de dérive par rapport à la référence.
- **Angles morts du modèle de récompense.** Il ne connaît que la région de texte sur laquelle il a été entraîné ; plus la politique bouge, plus on lui demande de noter du texte hors de cette région.
- **Les données de préférence sont humaines, donc biaisées.** Les préférences des annotateurs pour la longueur, l'assurance et l'agrément deviennent le comportement du modèle. Les comparaisons n'enregistrent pas non plus *pourquoi* une réponse est meilleure.
- **Limite hors ligne de DPO.** Il apprend à partir de paires fixes, généralement produites par un ancien modèle ; la source note que la boucle en ligne du RLHF (échantillons frais de la politique courante, renotés) a un avantage pour l'alignement itératif.
- **Alignement n'est pas vérité.** Un modèle optimisé sur les préférences est optimisé pour être bien *noté*. Des réponses fausses mais assurées peuvent être bien notées si les évaluateurs ne peuvent pas vérifier.

## Idées reçues courantes

- « Le modèle de récompense donne la vraie qualité d'une réponse. » Il donne un score ajusté sur les comparaisons de certaines personnes ; seules les différences ont un sens.
- « La pénalité KL ne fait que ralentir l'entraînement. » Elle définit aussi l'optimum : $\pi^\star\propto\pi_{\rm ref}e^{r/\beta}$.
- « DPO n'a pas de modèle de récompense. » Il en a un *implicite*, $\beta\ln(\pi/\pi_{\rm ref})$ ; la démonstration montre que la perte est équivalente à Bradley-Terry sur cette récompense.
- « DPO n'a pas besoin de modèle de référence. » La référence apparaît dans $z$ ; seules certaines variantes ultérieures la retirent.
- « Une récompense plus élevée pendant l'entraînement signifie un meilleur modèle. » Passé un certain point, cela signifie généralement plus d'exploitation du substitut ; évalue avec des jugements humains mis de côté ou indépendants.

## Points clés à retenir

- Les comparaisons sont plus faciles à collecter que les démonstrations ; Bradley-Terry les transforme en $P(a\succ b)=\sigma(r_a-r_b)$ et en la perte $-\ln\sigma(r_w-r_l)$ de gradient $\mp\sigma(-d)$.
- Le RLHF maximise $\mathbb E[r]-\beta\,\mathrm{KL}(\pi\|\pi_{\rm ref})$ ; son unique maximiseur est $\pi^\star=\pi_{\rm ref}e^{r/\beta}/Z$ de maximum $\beta\ln Z$ (en complétant la KL).
- Inverser cette formule donne $r=\beta\ln(\pi/\pi_{\rm ref})+\beta\ln Z$ ; les termes en $Z$ s'annulent dans une comparaison, ce qui donne la perte DPO $-\ln\sigma\bigl(\beta[\Delta_w-\Delta_l]\bigr)$.
- DPO supprime le modèle de récompense et la boucle RL au prix d'un jeu de données fixe hors ligne.
- La laisse $\beta$ arbitre entre récompense et dérive ; le détournement de la récompense, la complaisance et la verbosité sont le substitut exploité.

## Explique-le toi-même

1. Pourquoi les termes $\ln Z(x)$ s'annulent-ils dans la démonstration de DPO, et que se passerait-il si les deux réponses d'une paire répondaient à des prompts différents ?
2. La KL d'un modèle par rapport à sa référence continue de croître pendant l'entraînement sur les préférences tandis qu'une note humaine mise de côté cesse de s'améliorer. Que se passe-t-il, et quels deux réglages changerais-tu ?

## Pour aller plus loin

- Ouyang et al. (2022), *Training language models to follow instructions with human feedback* (InstructGPT).
- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* ; Bradley et Terry (1952), le modèle original de comparaison par paires.

## Critères de maîtrise

- Établir la perte de Bradley-Terry et son gradient $\mp\sigma(-d)$.
- Établir $\pi^\star=\pi_{\rm ref}e^{r/\beta}/Z$ en complétant la KL et énoncer la valeur maximale.
- En déduire la perte DPO, y compris pourquoi $Z(x)$ s'annule, et la calculer pour des log-probabilités données.
- Nommer trois modes d'échec et ce que chacun révèle sur le substitut.
