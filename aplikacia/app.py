# Hlavný súbor Flask aplikácie pre rozhodovací strom na predikciu HUT testu
# Táto aplikácia načíta strom vo formáte JSON a krok po kroku vedie používateľa otázkami k predikcii

from flask import Flask, render_template, request, redirect, url_for, session
import json

app = Flask(__name__)
app.secret_key = 'tajne_kluc_slova'

# Načítanie rozhodovacieho stromu zo súboru
with open("tree.json", encoding="utf-8") as f:
    tree = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if "node" not in session:
        session["node"] = tree
        session["answers"] = []

    node = session["node"]

    if "result" in node:
        result = node["result"]
        confidence = node.get("confidence", 0)
        answers = session.get("answers", [])
        session.clear()

        # Úprava váženej istoty podľa metriky
        accuracy = 0.750
        if result == "Pozitívny HUT":
            confidence_weighted = round(confidence * accuracy, 2)
        else:
            confidence_weighted = round(confidence * accuracy, 2)

        return render_template("result.html", result=result, confidence=confidence_weighted, answers=answers)

    if request.method == "POST":
        if node.get("type", "binary") == "numeric":
            try:
                user_value = float(request.form.get("numeric_answer"))
                threshold = node["threshold"]
                direction = "yes" if user_value > threshold else "no"
                answer_label = str(user_value)  # zobrazíme priamo hodnotu
            except (ValueError, TypeError):
                return "Zadajte platné číslo.", 400
        else:
            direction = request.form.get("answer")
            answer_label = "Áno" if direction == "1" else "Nie"

        # Uchovanie otázky tak, ako ju videl používateľ
        if "answers" not in session:
            session["answers"] = []
        session["answers"].append({
            "question": node["question"],
            "odpoved": answer_label
        })

        if direction in node["answers"]:
            session["node"] = node["answers"][direction]
            return redirect(url_for("index"))
        else:
            return "Neplatná odpoveď", 400

    # Zobrazenie aktuálnej otázky
    return render_template(
        "question.html",
        question=node["question"],
        type=node.get("type", "binary")
    )

# Spustenie aplikácie
if __name__ == "__main__":
    app.run(debug=True)
