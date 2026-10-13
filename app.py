
import os
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Generate sample student data for training
np.random.seed(42)
n = 1500

data = pd.DataFrame({
    "attendance": np.random.uniform(30, 100, n),
    "assignment_avg": np.random.uniform(20, 100, n),
    "midterm_score": np.random.uniform(10, 100, n),
    "study_hours": np.random.uniform(0, 12, n),
    "previous_score": np.random.uniform(20, 100, n),
    "sleep_hours": np.random.uniform(3, 10, n)
})

data["final_score"] = np.clip(
    0.20 * data["attendance"]
    + 0.20 * data["assignment_avg"]
    + 0.25 * data["midterm_score"]
    + 1.5 * data["study_hours"]
    + 0.20 * data["previous_score"]
    + 1.0 * data["sleep_hours"]
    + np.random.normal(0, 5, n),
    0, 100
)

features = [
    "attendance", "assignment_avg", "midterm_score",
    "study_hours", "previous_score", "sleep_hours"
]

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data["final_score"],
    test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))


def predict_success(attendance, assignment, midterm,
                    study_hours, previous_score, sleep_hours):
    student = pd.DataFrame([{
        "attendance": attendance,
        "assignment_avg": assignment,
        "midterm_score": midterm,
        "study_hours": study_hours,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours
    }])[features]

    score = float(np.clip(model.predict(student)[0], 0, 100))
    result = f"### Predicted Final Score: {score:.1f}/100\n"

    if score >= 75:
        result += "\n**Performance outlook:** Good"
    elif score >= 50:
        result += "\n**Performance outlook:** Needs improvement"
    else:
        result += "\n**Performance outlook:** At risk — seek academic support"

    tips = []
    if attendance < 75:
        tips.append("Improve attendance and attend classes regularly.")
    if assignment < 70:
        tips.append("Practise assignments and submit them on time.")
    if midterm < 60:
        tips.append("Revise difficult topics and practise previous questions.")
    if study_hours < 2:
        tips.append("Start a daily study routine of at least 2 focused hours.")
    if sleep_hours < 7:
        tips.append("Aim for a regular, healthy sleep schedule.")
    if previous_score < 60:
        tips.append("Review earlier mistakes and strengthen the basics.")
    if not tips:
        tips.append("Maintain your routine and keep practising consistently.")

    result += "\n\n### Personalised Improvement Plan\n"
    result += "\n".join(f"- {tip}" for tip in tips)
    result += f"\n\n*Model test MAE: {mae:.1f} points. This is an educational estimate, not a guaranteed result.*"

    return result


with gr.Blocks(title="Student Success AI") as demo:
    gr.Markdown(
        "# 🎓 Student Success AI\n"
        "Predict your potential final score and receive a personalised "
        "academic improvement plan."
    )

    with gr.Row():
        attendance = gr.Slider(0, 100, value=80, label="Attendance (%)")
        assignment = gr.Slider(0, 100, value=75, label="Assignment Average")

    with gr.Row():
        midterm = gr.Slider(0, 100, value=65, label="Midterm Score")
        study = gr.Slider(0, 12, value=3, step=0.5, label="Daily Study Hours")

    with gr.Row():
        previous = gr.Slider(0, 100, value=70, label="Previous Score")
        sleep = gr.Slider(0, 12, value=8, step=0.5, label="Daily Sleep Hours")

    predict_button = gr.Button("Predict My Success", variant="primary")
    output = gr.Markdown()

    predict_button.click(
        fn=predict_success,
        inputs=[attendance, assignment, midterm, study, previous, sleep],
        outputs=output
    )

    gr.Markdown(
        "*Demo version using synthetic training data. "
        "Predictions are illustrative, not official academic results.*"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
