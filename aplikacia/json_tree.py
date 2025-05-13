# Načítame potrebné knižnice
from sklearn.tree import _tree
import json
import numpy as np
# Vytvoríme slovník, do ktorého sa načíta globálny priestor zo súboru s rozhodovacím stromom
globals_dict = {}
exec(open("aplikacia/DT bez HUT.py", encoding="utf-8").read(), globals_dict)
# Načítame natrénovaný orezaný rozhodovací strom a názvy vstupných príznakov
model = globals_dict['best_pruned_dt']
features = list(globals_dict['X'].columns)
# Slovník prekladajúci kódy premenných na zrozumiteľné otázky (atribúty dotazníka)
labels = {
    'B2': 'Bol/bola vyšetrovaný pre pocity hroziacej straty vedomia(B2)',
    'B3': 'Bol/bola vyšetrovaný pre stav po resuscitácii(B3)',
    'B4': 'Bol/bola vyšetrovaný pre stav po epileptickom záchvate',
    'B5': 'Bol/bola vyšetrovaný pre opakované pády',
    'C3': 'Vek pri poslednom odpadnutí',
    'C4': 'Vek pri najhoršom stave',
    'C1': 'Vek pri začiatku ťažkostí ',
    'D1': 'Tažkosti / V akej situácii vznikli: Strata vedomia pri státí',
    'D2': 'Tažkosti / V akej situácii vznikli: Strata vedomia do 1 minúty po postavení sa',
    'D3': 'Tažkosti / V akej situácii vznikli: Strata vedomia pri chôdzi',
    'D4': 'Tažkosti / V akej situácii vznikli: Strata vedomia pri fyzickej námahe',
    'D5': 'Tažkosti / V akej situácii vznikli: Strata vedomia v sede',
    'D6': 'Tažkosti / V akej situácii vznikli: Strata vedomia poležiačky',
    'E1': 'Čo viedlo k strate vedomia: Preľudnené priestory', 'E2': 'Čo viedlo k strate vedomia: Dusné prostredie', 'E3': 'Čo viedlo k strate vedomia: Teplé prostredie',
    'E4': 'Čo viedlo k strate vedomia: Pohľad na krv', 'E5': 'Čo viedlo k strate vedomia: Nepríjemné emócie', 'E6': 'Čo viedlo k strate vedomia: Medicínsky výkon',
    'E7': 'Čo viedlo k strate vedomia: Bolesť', 'E8': 'Čo viedlo k strate vedomia: Dehydratácia', 'E9': 'Čo viedlo k strate vedomia: Menštruácia', 'E10': 'Čo viedlo k strate vedomia: Strata krvi',
    'H1': 'Cítili tesne pred stratou vedomia: Pocit na zvracanie', 'H2': 'Cítili tesne pred stratou vedomia: Pocit tepla', 'H3': 'Cítili tesne pred stratou vedomia: Potenie',
    'H4': 'Cítili tesne pred stratou vedomia: Zahmlievanie pred očami', 'H5': 'Cítili tesne pred stratou vedomia: Hučanie v ušiach', 'H6': 'Cítili tesne pred stratou vedomia: Búšenie srdca',
    'H8': 'Cítili tesne pred stratou vedomia: Bolesť na hrudi', 'H9': 'Cítili tesne pred stratou vedomia: Neobvyklý zápach', 'H10': 'Cítili tesne pred stratou vedomia: Neobvyklé zvuky',
    'H11': 'Cítili tesne pred stratou vedomia: Poruchy reči / slabosť tela', 'H12': 'Cítili tesne pred stratou vedomia: Nepociťoval som nič zvláštne', 'H13': 'Cítili tesne pred stratou vedomia: Nepamätám sa',
    'I1': 'Ako dlho trvali tieto pocity pred stratou vedomia: Niekoľko sekúnd (pred stratou vedomia)', 'I2': 'Ako dlho trvali tieto pocity pred stratou vedomia: Do 1 minúty',
    'I3': 'Ako dlho trvali tieto pocity pred stratou vedomia: Do 5 minút', 'I4': 'Ako dlho trvali tieto pocity pred stratou vedomia: Viac ako 5 minút',
    'J1': 'Čo ste urobili pri hroziacej strate vedomia: Sadol som si', 'J2': 'Čo ste urobili pri hroziacej strate vedomia: Ľahol som si', 'K4': 'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie: Bezvedomie trvalo viac ako 5 minút',
    'L': 'Kŕče počas bezvedomia', 'M': 'Odišla stolica alebo moč počas bezvedomia?',
    'N2': 'Pamätáte si na udalosti po strate vedomia: Poranenie pri páde', 'N3': 'Pamätáte si na udalosti po strate vedomia: Dezorientácia > 30 minút',
    'N5': 'Pamätáte si na udalosti po strate vedomia: Nevoľnosť po prebratí', 'N6': 'Pamätáte si na udalosti po strate vedomia: Cítil/a sa normálne', 'N7': 'Pamätáte si na udalosti po strate vedomia: Nepamätám sa',
    'P1': 'Na aké ochorenia ste sa doteraz liečili: Ochorenie srdca_3',
    'P2': 'Na aké ochorenia ste sa doteraz liečili: úzkostný stav',
    'P3': 'Na aké ochorenia ste sa doteraz liečili: Ochorenie chlopní',
    'P4': 'Na aké ochorenia ste sa doteraz liečili: Srdcová slabosť',
    'P5': 'Na aké ochorenia ste sa doteraz liečili: Koronárna chorova srdca',
    'P6': 'Na aké ochorenia ste sa doteraz liečili: Srdcové arytmie',
    'P7': 'Na aké ochorenia ste sa doteraz liečili: Búšenie srdca',
    'P9': 'Na aké ochorenia ste sa doteraz liečili: Bolesti na hrudníku',
    'P10': 'Na aké ochorenia ste sa doteraz liečili: Vysoký tlak krvi',
    'P11': 'Na aké ochorenia ste sa doteraz liečili: Nízky tlak krvi',
    'P12': 'Na aké ochorenia ste sa doteraz liečili: Závraty',
    'P13': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia obličiek',
    'P14': 'Na aké ochorenia ste sa doteraz liečili: Diabetes (cukrovka)',
    'P15': 'Na aké ochorenia ste sa doteraz liečili: Anémia',
    'P16': 'Na aké ochorenia ste sa doteraz liečili: Astma',
    'P17': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia pľúc',
    'P18': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia priedušiek',
    'P19': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia žalúdka',
    'P20': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia čreva',
    'P21': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia štítnej žľazy',
    'P22': 'Na aké ochorenia ste sa doteraz liečili: Endokrinologické ochorenia',
    'P23': 'Na aké ochorenia ste sa doteraz liečili: Bolesti hlavy',
    'P24': 'Na aké ochorenia ste sa doteraz liečili: Neurologické ochorenia',
    'P25': 'Na aké ochorenia ste sa doteraz liečili: Parkinsonová choroba',
    'P26': 'Na aké ochorenia ste sa doteraz liečili: Psychiatrické ochorenia ',
    'P27': 'Na aké ochorenia ste sa doteraz liečili: Depresia',
    'P28': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia krčnej chrbtice',
    'P29': 'Na aké ochorenia ste sa doteraz liečili: Bolesti chrbta',
    'P30': 'Na aké ochorenia ste sa doteraz liečili: Reumatologické ochorenia',
    'P31': 'Na aké ochorenia ste sa doteraz liečili: Nádorové ochorenie',
    'P32': 'Na aké ochorenia ste sa doteraz liečili: Prekonané operácie',
    'P33': 'Na aké ochorenia ste sa doteraz liečili: Prekonané úrazy',
    'P34': 'Na aké ochorenia ste sa doteraz liečili: Alergie',
    'Q1': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Kardiologické vyšetrenia',
    'Q2': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Záťažový test (bicyklová ergometria)',
    'Q3': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Koronografické vyšetrenie',
    'Q4': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: HUT test',
    'Q5': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Pažerákovú stimuláciu',
    'Q6': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Invazívne vyšetrenie arytmií (EFV)',
    'Q7': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Nukleárne vyšetrenie srdca (SPECT)',
    'Q8': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: CT srdca',
    'Q9': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: MRI srdca',
    'Q10': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Neurologické vyšetrenia',
    'Q11': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: USG mozgových ciev',
    'Q12': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: CT alebo MRI mozgu',
    'Q13': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: RTG, CT alebo MRI krčnej chrbtice',
    'Q14': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Elektromyografia (EMG)',
    'Q15': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Psychiatrické vyšetrenie',
    'Q16': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Endokrinologické vyšetrenie',
    'Q17': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Odber krvi',
    'Q18': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Iné',
    'R1': 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov): Proti HPV (rakovina krčka maternice)',
    'R2': 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov): Proti chrípke',
    'R3': 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov): Iné'
}

MAX_DEPTH = None

numeric_features = {"Vek", "C1", "C3", "C4"}

# Funkcia na výpočet istoty (confidence) rozhodnutia na danom uzle stromu
def compute_confidence(value):
    total = np.sum(value)
    if total == 0:
        return 0.0
    confidence = np.max(value) / total
    return round(confidence * 100, 2)
# Rekurzívna funkcia, ktorá konvertuje rozhodovací strom na zrozumiteľnú JSON štruktúru
def tree_to_json_limited(tree, feature_names, node=0, depth=0):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    if MAX_DEPTH is not None and depth >= MAX_DEPTH or tree_.feature[node] == _tree.TREE_UNDEFINED:
        value = tree_.value[node][0]
        class_idx = int(np.argmax(value))
        return {"result": "Pozitívny HUT" if class_idx == 1 else "Negatívny HUT", "confidence": compute_confidence(value)}

    name = feature_name[node]
    label = labels.get(name, name)
    threshold = tree_.threshold[node]
    # Vytvárame otázku pre číselný alebo binárny atribút
    if name in numeric_features:
        return {
            "question": f"{label} > {threshold:.1f}?",
            "feature": name,
            "type": "numeric",
            "threshold": threshold,
            "answers": {
                "yes": tree_to_json_limited(tree, feature_names, tree_.children_right[node], depth + 1),
                "no": tree_to_json_limited(tree, feature_names, tree_.children_left[node], depth + 1)
            }
        }
    else:
        return {
            "question": f"{label} (áno/nie)",
            "feature": name,
            "type": "binary",
            "answers": {
                "1": tree_to_json_limited(tree, feature_names, tree_.children_right[node], depth + 1),
                "0": tree_to_json_limited(tree, feature_names, tree_.children_left[node], depth + 1)
            }
        }

# Konvertujeme celý strom na formát JSON
tree_json_limited = tree_to_json_limited(model, features)

# Uložíme štruktúru rozhodovacieho stromu do JSON súboru
json_path = "tree.json"
with open("tree.json", "w", encoding="utf-8") as f:
    json.dump(tree_json_limited, f, ensure_ascii=False, indent=2)

json_path
