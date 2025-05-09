# Predikcia prítomnosti a typu synkopy pomocou strojového učenia

## Úvod

Tento projekt sa zaoberá analýzou a predikciou **synkopy** (krátkodobá strata vedomia) pomocou metód **strojového učenia**.  Vstupné dáta pochádzajú z **lekárskych správ a dotazníkov vyplnených pacientmi**, ktoré boli manuálne spracované do jednotného dátového súboru. Na ich základe sa vytvárajú modely, ktoré umožňujú:

- predikovať, či pacient trpí synkopou (áno/nie),
- ak áno, určiť typ synkopy podľa klasifikácie **VASIS (I, II, III)**.

Projekt je vytvorený ako súčasť bakalárskej práce a slúži ako základ pre budúce využitie v klinickej diagnostike.

---
## Ciele projektu
- **Predspracovať** rozsiahly dotazníkový dataset (textové a číselné údaje, dátumy, odpovede typu áno/nie),

- Predikovať prítomnosť synkopy (**binárna klasifikácia**),

- Predikovať typ synkopy, ak bola diagnostikovaná (**multiclass klasifikácia**),

- Zabezpečiť vyváženie dát a výber najlepších príznakov (**RFECV, SMOTE**),

- Vizualizovať výsledky pre **klinickú interpretáciu** (matíce zámien, význam príznakov).

---

## Ukážka výstupu
Projekt generuje rôzne vizualizácie a hodnotenia modelov, ktoré sú použiteľné pre medicínsku interpretáciu:

- Confusion matrix pre binárnu aj viactriednu klasifikáciu

- Dôležitosť príznakov z modelov (Random Forest, XGBoost)

- Zhrnutie metrík modelov: Accuracy, Precision, Recall, F1 Score, AUC

> Tento projekt môže slúžiť ako nástroj na podporu diagnostiky synkopy v klinickej praxi, ako aj ako podklad pre ďalší výskum v oblasti medicínskeho strojového učenia.


