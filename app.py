
import os
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# 1. TRAIN THE DEMONSTRATION MODEL
# --------------------------------------------------

np.random.seed(42)
n = 1500

data = pd.DataFrame({
    "attendance": np.random.uniform(30, 100, n),
    "assignment_avg": np.random.uniform(20, 100, n),
    "midterm_score": np.random.uniform(10, 100, n),
    "study_hours": np.random.uniform(0, 12, n),
    "previous_score": np.random.uniform(20, 100, n),
    "sleep_hours": np.random.uniform(3, 10, n),
})

data["final_score"] = np.clip(
    0.20 * data["attendance"]
    + 0.20 * data["assignment_avg"]
    + 0.25 * data["midterm_score"]
    + 1.5 * data["study_hours"]
    + 0.20 * data["previous_score"]
    + 1.0 * data["sleep_hours"]
    + np.random.normal(0, 5, n),
    0,
    100,
)

features = [
    "attendance",
    "assignment_avg",
    "midterm_score",
    "study_hours",
    "previous_score",
    "sleep_hours",
]

X_train, X_test, y_train, y_test = train_test_split(
    data[features],
    data["final_score"],
    test_size=0.2,
    random_state=42,
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))


# --------------------------------------------------
# 2. PERSONALIZED PREDICTION AND RECOMMENDATIONS
# --------------------------------------------------

def predict_success(
    attendance,
    assignment,
    midterm,
    study_hours,
    previous_score,
    sleep_hours,
):
    student = pd.DataFrame([{
        "attendance": attendance,
        "assignment_avg": assignment,
        "midterm_score": midterm,
        "study_hours": study_hours,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
    }])[features]

    score = float(np.clip(model.predict(student)[0], 0, 100))

    if score >= 75:
        outlook = "Good performance outlook"
        status = "On track"
    elif score >= 50:
        outlook = "There is room to improve"
        status = "Needs attention"
    else:
        outlook = "Consider additional academic support"
        status = "Support recommended"

    tips = []

    if attendance < 75:
        tips.append(
            "**Attendance:** Aim for at least 75% attendance, "
            "subject to your college's rules."
        )

    if assignment < 70:
        tips.append(
            "**Assignments:** Practise the topics you find difficult "
            "and submit work on time."
        )

    if midterm < 60:
        tips.append(
            "**Midterm preparation:** Review incorrect answers and "
            "practise questions from weaker units."
        )

    if study_hours < 2:
        tips.append(
            "**Study routine:** Start with two focused study hours "
            "daily, with short breaks."
        )

    if previous_score < 60:
        tips.append(
            "**Fundamentals:** Revisit concepts from earlier topics "
            "before moving to advanced material."
        )

    if sleep_hours < 7:
        tips.append(
            "**Rest:** Try to maintain a regular sleep routine "
            "and allow enough time for rest."
        )

    if not tips:
        tips.append(
            "**Keep going:** Maintain your study routine, practise "
            "regularly, and review your progress each week."
        )

    recommendations = "\n".join(f"- {tip}" for tip in tips)

    result = f"""
<div class="result-summary">
  <div class="result-label">YOUR ESTIMATED FINAL SCORE</div>
  <div class="score-number">{score:.1f}<span>/100</span></div>
  <div class="score-status">{status}</div>
  <p class="outlook">{outlook}</p>
</div>

### Your personalised improvement plan

{recommendations}

<div class="note">
  <strong>About this prediction:</strong> This demonstration uses
  synthetic training data. Test-set MAE is {mae:.1f} points on
  synthetic data only. This does not establish accuracy on real
  students or guarantee an exam result.
</div>
"""

    return result


# --------------------------------------------------
# 3. PROFESSIONAL, RESPONSIVE DESIGN
# --------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    max-width: 1120px !important;
    margin: auto !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

body {
    background: #f4f7fc !important;
}

#hero {
    background: linear-gradient(125deg, #172554 0%, #3730a3 58%, #4f46e5 100%);
    color: white;
    border-radius: 24px;
    padding: 32px 30px;
    margin: 8px 0 22px 0;
    box-shadow: 0 12px 32px rgba(49, 46, 129, 0.18);
}

#hero h1 {
    color: white !important;
    font-size: clamp(27px, 4vw, 40px) !important;
    line-height: 1.2 !important;
    margin: 8px 0 12px 0 !important;
}

#hero p {
    color: #e0e7ff !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

.brand-tag {
    display: inline-block;
    padding: 6px 11px;
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 30px;
    font-size: 12px;
    letter-spacing: .7px;
    color: #eef2ff;
}

.panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 20px !important;
    box-shadow: 0 5px 20px rgba(15, 23, 42, 0.04);
}

.section-title h2 {
    color: #172554 !important;
    font-size: 21px !important;
}

#predict-btn {
    background: linear-gradient(100deg, #4338ca, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 50px;
    font-size: 16px !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, .2);
}

#predict-btn:hover {
    filter: brightness(1.07);
}

#result-panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px !important;
    min-height: 250px;
}

.result-summary {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 22px;
    margin: 4px 0 22px 0;
}

.result-label {
    color: #4338ca;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.1px;
}

.score-number {
    color: #1e1b4b;
    font-size: clamp(42px, 8vw, 60px);
    font-weight: 800;
    line-height: 1.25;
    margin-top: 8px;
}

.score-number span {
    color: #6366f1;
    font-size: 22px;
    font-weight: 600;
}

.score-status {
    display: inline-block;
    margin-top: 8px;
    padding: 6px 10px;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
}

.outlook {
    color: #334155;
    margin-bottom: 0;
}

#result-panel h3 {
    color: #172554 !important;
    margin-top: 18px !important;
}

#result-panel li {
    margin-bottom: 10px;
    line-height: 1.65;
}

.note {
    background: #f8fafc;
    border-left: 4px solid #818cf8;
    color: #475569;
    padding: 13px 15px;
    border-radius: 6px;
    margin-top: 20px;
    font-size: 12px;
    line-height: 1.7;
}

#footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 18px 8px;
}

@media (max-width: 640px) {
    .gradio-container {
        padding: 10px !important;
    }

    #hero {
        padding: 23px 18px;
        border-radius: 17px;
        margin-top: 4px;
    }

    .panel, #result-panel {
        padding: 15px !important;
        border-radius: 15px;
    }

    #predict-btn {
        min-height: 48px;
    }

    .result-summary {
        padding: 17px;
    }
}
"""


# --------------------------------------------------
# 4. BUILD THE WEBSITE
# --------------------------------------------------

with gr.Blocks(
    title="Student Success AI | Academic Planner",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as demo:

    gr.HTML("""
    <div id="hero">
      <div class="brand-tag">AI-POWERED LEARNING ASSISTANT</div>
      <h1>Student Success AI 🎓</h1>
      <p>
        Understand your academic progress, estimate your potential
        final score, and discover practical steps to improve your
        study routine.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown(
                "## 📚 Your academic profile\n"
                "Adjust the values to match your current progress.",
                elem_classes=["section-title"],
            )

            attendance = gr.Slider(
                0, 100, value=80, step=1,
                label="Attendance (%)",
            )
            assignment = gr.Slider(
                0, 100, value=75, step=1,
                label="Assignment average (%)",
            )
            midterm = gr.Slider(
                0, 100, value=65, step=1,
                label="Midterm score (%)",
            )
            previous = gr.Slider(
                0, 100, value=70, step=1,
                label="Previous score (%)",
            )

            gr.Markdown("### ⏱️ Daily routine")

            study = gr.Slider(
                0, 12, value=3, step=0.5,
                label="Study hours per day",
            )
            sleep = gr.Slider(
                0, 12, value=8, step=0.5,
                label="Sleep hours per day",
            )

            predict_button = gr.Button(
                "✨  Predict My Success",
                variant="primary",
                elem_id="predict-btn",
            )

        with gr.Column(scale=1, elem_id="result-panel"):
            gr.Markdown(
                "## 📈 Your success report\n\n"
                "Adjust your academic details and select "
                "**Predict My Success** to view your estimate "
                "and personalised recommendations."
            )

            output = gr.Markdown()

    gr.HTML("""
    <div id="footer">
      Student Success AI · Educational demonstration<br>
      Predictions are estimates, not official academic results.
    </div>
    """)

    predict_button.click(
        fn=predict_success,
        inputs=[
            attendance,
            assignment,
            midterm,
            study,
            previous,
            sleep,
        ],
        outputs=output,
    )


# --------------------------------------------------
# 5. START SERVER FOR RENDER
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
