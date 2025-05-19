import pandas as pd
df = pd.read_csv("data/data_full.csv")
df_full = df.copy()

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Predspracovanie: úprava číselných a binárnych hodnôt
if 'Pohlavie' in df.columns:
    df['Pohlavie'] = df['Pohlavie'].map({'M': 0, 'F': 1})

blood_pressure_cols = ['A2', 'A4', 'A6', 'A8']
pulse_cols = ['A3', 'A5', 'A7', 'A9']

for col in blood_pressure_cols:
    if col in df.columns:
        df[[f'{col}_systolic', f'{col}_diastolic']] = df[col].astype(str).str.extract(r'(\d+)/(\d+)').astype(float)
        df.drop(columns=[col], inplace=True)

for col in pulse_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

block_A_cols = [
    'A1', 'A9',
    'A2_systolic', 'A2_diastolic', 'A3',
    'A4_systolic', 'A4_diastolic', 'A5',
    'A6_systolic', 'A6_diastolic', 'A7',
    'A8_systolic', 'A8_diastolic'
]
# Odstránenie všetkých stĺpcov z bloku A (tlaky, pulzy, status HUT)
df.drop(columns=[col for col in block_A_cols if col in df.columns], inplace=True)

# Príprava datasetu
df = df.select_dtypes(include=[float, int])
df_clean = df[df["Synkopa"].isin([0, 1])].copy()
df_clean["Synkopa"] = df_clean["Synkopa"].astype(int)

X = df_clean.drop(columns=["Synkopa", "Typ Synkopy"], errors='ignore').copy()
y = df_clean["Synkopa"]
X = X.fillna(-1).astype(float)
# Rozdelenie dát na trénovaciu a testovaciu množinu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Parametre pre optimalizáciu rozhodovacieho stromu
param_grid_dt = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
# GridSearchCV pre výber najlepšej kombinácie parametrov (s F1 skóre)
grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid_dt,
    cv=StratifiedKFold(3),
    scoring='f1_weighted',
    n_jobs=-1
)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

binary_cols = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, -1})]
numeric_cols = [col for col in X.columns if col not in binary_cols]
# Funkcia pre ľudsky čitateľné exportovanie pravidiel zo stromu
def logical_export_text(decision_tree, feature_names, binary_cols):
    rules_text = export_text(decision_tree, feature_names=feature_names)
    lines = rules_text.split('\n')
    new_lines = []

    for line in lines:
        if "<=" in line or ">" in line:
            for col in binary_cols:
                if f"{col} <= 0.50" in line or f"{col} <= 0.5" in line:
                    line = line.replace(f"{col} <= 0.50", f"{col} = 0").replace(f"{col} <= 0.5", f"{col} = 0")
                elif f"{col} >  0.50" in line or f"{col} >  0.5" in line:
                    line = line.replace(f"{col} >  0.50", f"{col} = 1").replace(f"{col} >  0.5", f"{col} = 1")
        new_lines.append(line)
    return '\n'.join(new_lines)

# Vyhodnotenie výkonnosti modelu
y_pred = best_dt.predict(X_test)
y_proba = best_dt.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Decision Tree - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, AUC: {roc_auc:.3f}")

# Vizualizácia rozhodovacieho stromu
plt.figure(figsize=(50, 40))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=["Negatívny", "Pozitívny"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=None
)
plt.title("Vizualizácia rozhodovacieho stromu pre predikciu synkopy (HUT test)")
plt.tight_layout()
plt.savefig("strom.png", dpi=300)

# Orezávanie stromu pomocou Cost-Complexity Pruning (CCP)
from sklearn.tree import DecisionTreeClassifier
import numpy as np

path = best_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

dt_models = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        ccp_alpha=ccp_alpha
    )
    clf.fit(X_train, y_train)
    dt_models.append(clf)

from sklearn.metrics import f1_score

# Сохраняем метрики для каждого дерева
all_scores = []

for clf in dt_models:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    all_scores.append(scores)


best_idx = np.argmax([s['f1'] for s in all_scores])
best_pruned_dt = dt_models[best_idx]

print(f"\n Najlepší model po CCP pruning:")
print(f"ccp_alpha = {ccp_alphas[best_idx]:.5f}")
print(f"Accuracy: {all_scores[best_idx]['accuracy']:.3f}")
print(f"Precision: {all_scores[best_idx]['precision']:.3f}")
print(f"Recall: {all_scores[best_idx]['recall']:.3f}")
print(f"F1 Score: {all_scores[best_idx]['f1']:.3f}")
print(f"AUC: {all_scores[best_idx]['roc_auc']:.3f}")


# Vizualizácia orezaného stromu
plt.figure(figsize=(50, 40))
plot_tree(
    best_pruned_dt,
    feature_names=X.columns,
    class_names=["Negatívny", "Pozitívny"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Orezané rozhodovacie strom (post-pruning s CCP)")
plt.tight_layout()
plt.savefig("strom.png", dpi=300)

